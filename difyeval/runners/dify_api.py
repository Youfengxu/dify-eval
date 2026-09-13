"""dify-chatflow and dify-workflow runners — live Dify Service API, streaming.

The SSE consumer is ported from an earlier internal eval runner run_chatflow
(node_finished journaling against a `watch` node-id set, `message`
accumulation, workflow_finished / message_end status capture) and extended
with usage capture (message_end metadata.usage; per-node outputs.usage via
the aggregation helper) and workflow-run support (/v1/workflows/run, output
= workflow_finished data.outputs[report_key]).

Runner spec (both types)::

    runner:
      type: dify-workflow            # or dify-chatflow
      base: https://dify.example.com # or base_env: DIFY_BASE
      api_key_env: DIFY_API_KEY      # Service API key env var (default shown)
      app: my-app                    # label only — the API key selects the app
      input_map: {caption: post_text}  # sample input key -> Dify variable name
      watch: [merge_s4, sources]     # node ids to journal; omit = journal ALL
      report_key: report             # workflow only; default "report"
      query_key: query               # chatflow only; sample input holding the query
      run_timeout: 3900              # overall wall clock, seconds
      read_timeout: 2400             # per-chunk SSE read, seconds

Network access is lazy (requests imported inside the transport) and the
stream opener is injectable — tests feed a recorded SSE transcript, never a
socket.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable

from ..core import Run
from . import register_runner
from .usage import aggregate_usage

DEFAULT_RUN_TIMEOUT_S = 3900
DEFAULT_READ_TIMEOUT_S = 2400
DEFAULT_REPORT_KEY = "report"
DEFAULT_USER = "difyeval"


def _open_stream(url: str, headers: dict, payload: dict,
                 read_timeout: float) -> tuple[int, Iterable[str] | None, str | None]:
    """Real transport: returns (status, line_iterator, error_text). The only
    network touch-point of both runners — tests monkeypatch this."""
    import requests  # lazy by design

    resp = requests.post(url, headers=headers, json=payload, stream=True,
                         timeout=(30, read_timeout))
    if resp.status_code != 200:
        return resp.status_code, None, resp.text[:500]
    return 200, resp.iter_lines(decode_unicode=True), None


def sse_events(lines: Iterable[str]):
    """SSE 'data:' lines -> event dicts (ported parser; junk lines skipped)."""
    for raw in lines:
        if not raw or not raw.startswith("data:"):
            continue
        try:
            yield json.loads(raw[5:].strip())
        except (json.JSONDecodeError, ValueError):
            continue


def consume_stream(lines: Iterable[str], watch: set | None = None,
                   deadline: float | None = None,
                   clock: Callable[[], float] = time.time) -> dict:
    """Consume a Dify SSE stream (chatflow or workflow). Returns
    {nodes, answer, outputs, status, elapsed, top_usage, error}.

    watch=None journals ALL finished nodes (a universal eval tool wants the
    trace by default); a set journals only those ids (the run_eval behavior).
    """
    nodes: dict = {}
    answer: list[str] = []
    outputs: dict = {}
    status = "unknown"
    elapsed = None
    top_usage: dict = {}
    error = None
    for ev in sse_events(lines):
        if deadline is not None and clock() > deadline:
            status = "client_timeout"
            break
        et = ev.get("event")
        d = ev.get("data") or {}
        if et == "node_finished":
            nid = d.get("node_id")
            if nid and (watch is None or nid in watch):
                nodes[nid] = {"status": d.get("status"),
                              "elapsed": d.get("elapsed_time"),
                              "outputs": d.get("outputs") or {}}
        elif et == "message":
            answer.append(ev.get("answer", ""))
        elif et == "workflow_finished":
            status = d.get("status") or status
            if isinstance(d.get("outputs"), dict):
                outputs = d["outputs"]
            if d.get("elapsed_time") is not None:
                elapsed = d.get("elapsed_time")
            tt = d.get("total_tokens")
            if isinstance(tt, (int, float)) and not isinstance(tt, bool):
                top_usage.setdefault("workflow_tokens", int(tt))
        elif et == "message_end":
            status = d.get("status") or status
            u = (ev.get("metadata") or {}).get("usage") or {}
            if isinstance(u, dict) and u:
                mu: dict = {}
                tt = u.get("total_tokens")
                if isinstance(tt, (int, float)) and not isinstance(tt, bool):
                    mu["tokens"] = int(tt)
                price = u.get("total_price")
                try:
                    price = float(price) if price is not None else None
                except (TypeError, ValueError):
                    price = None
                if price is not None:
                    mu["cost"] = {str(u.get("currency") or "USD"): price}
                if mu:
                    top_usage["message_end"] = mu
        elif et == "error":
            status = "error"
            error = str(ev.get("message") or d.get("error") or ev)[:300]
    return {"nodes": nodes, "answer": "".join(answer), "outputs": outputs,
            "status": "succeeded" if status == "unknown" else status,
            "elapsed": elapsed, "top_usage": top_usage, "error": error}


def _resolve_base_key(spec: dict) -> tuple[str | None, str | None, str | None]:
    base = spec.get("base") or (os.environ.get(spec["base_env"]) if spec.get("base_env") else None)
    if not base:
        return None, None, "dify runner: no base URL (set runner.base or runner.base_env)"
    key_env = spec.get("api_key_env", "DIFY_API_KEY")
    key = os.environ.get(key_env)
    if not key:
        return None, None, f"dify runner: API key env var {key_env} is not set"
    return base.rstrip("/"), key, None


def _map_inputs(inputs: dict, input_map: dict | None) -> dict:
    """sample.inputs -> Dify start-node variables, mechanically; input_map
    renames keys (sample key -> Dify variable name)."""
    input_map = input_map or {}
    return {input_map.get(k, k): v for k, v in inputs.items()}


def _build_usage(nodes: dict, top_usage: dict) -> dict:
    """Node usage blocks + top-level usage, double-count safe: chatflow
    message_end usage is genuinely additional (the answer LLM), so it is
    summed in; the workflow_finished total_tokens is an AGGREGATE of the node
    totals, so it is used only when no node reported usage."""
    me = top_usage.get("message_end")
    usage = aggregate_usage(nodes, me)
    if not usage.get("tokens") and top_usage.get("workflow_tokens"):
        usage["tokens"] = top_usage["workflow_tokens"]
    return usage


def _run_streaming(spec: dict, url: str, key: str, payload: dict,
                   open_stream: Callable | None = None) -> Run:
    open_stream = open_stream or _open_stream
    run_timeout = spec.get("run_timeout", DEFAULT_RUN_TIMEOUT_S)
    read_timeout = spec.get("read_timeout", DEFAULT_READ_TIMEOUT_S)
    watch = set(spec["watch"]) if spec.get("watch") else None
    t0 = time.time()
    try:
        status_code, lines, err = open_stream(
            url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload, read_timeout)
    except Exception as e:  # noqa: BLE001 — network failure degrades
        return Run(error=f"stream open failed: {str(e)[:300]}",
                   meta={"status": "runner_error"})
    if status_code != 200:
        return Run(error=f"HTTP {status_code}: {err}",
                   meta={"status": f"http_{status_code}"})
    try:
        result = consume_stream(lines, watch, deadline=t0 + run_timeout)
    except Exception as e:  # noqa: BLE001 — mid-stream network failure degrades
        return Run(error=f"stream read failed: {str(e)[:300]}",
                   meta={"status": "runner_error"})
    elapsed = result["elapsed"] if result["elapsed"] is not None else round(time.time() - t0, 1)
    return Run(
        output="",  # caller fills per mode
        nodes=result["nodes"],
        usage=_build_usage(result["nodes"], result["top_usage"]),
        meta={"status": result["status"], "elapsed": elapsed,
              "outputs": result["outputs"]},
        error=result["error"],
    ), result


@register_runner("dify-chatflow")
def run_chatflow(spec: dict, sample, repeat: int, case, open_stream=None) -> Run:
    """POST /v1/chat-messages (streaming); output = accumulated `message`
    answer (the streamed report — ported from run_eval.run_chatflow)."""
    base, key, err = _resolve_base_key(spec)
    if err:
        return Run(error=err, meta={"status": "runner_error"})
    query_key = spec.get("query_key", "query")
    query = str(sample.inputs.get(query_key, ""))
    inputs = _map_inputs({k: v for k, v in sample.inputs.items() if k != query_key},
                         spec.get("input_map"))
    payload = {"inputs": inputs, "query": query, "response_mode": "streaming",
               "user": spec.get("user", DEFAULT_USER), "conversation_id": ""}
    out = _run_streaming(spec, f"{base}/v1/chat-messages", key, payload, open_stream)
    if isinstance(out, Run):
        return out
    run, result = out
    run.output = result["answer"]
    return run


@register_runner("dify-workflow")
def run_workflow(spec: dict, sample, repeat: int, case, open_stream=None) -> Run:
    """POST /v1/workflows/run (streaming); output = workflow_finished
    data.outputs[report_key] (JSON-dump of all outputs when the key is
    absent/non-string)."""
    base, key, err = _resolve_base_key(spec)
    if err:
        return Run(error=err, meta={"status": "runner_error"})
    payload = {"inputs": _map_inputs(sample.inputs, spec.get("input_map")),
               "response_mode": "streaming",
               "user": spec.get("user", DEFAULT_USER)}
    out = _run_streaming(spec, f"{base}/v1/workflows/run", key, payload, open_stream)
    if isinstance(out, Run):
        return out
    run, result = out
    report_key = spec.get("report_key", DEFAULT_REPORT_KEY)
    rep = (result["outputs"] or {}).get(report_key)
    if isinstance(rep, str):
        run.output = rep
    elif result["outputs"]:
        run.output = json.dumps(result["outputs"], ensure_ascii=False, sort_keys=True)
    return run
