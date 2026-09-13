"""Normalize the run-payload shapes difyeval accepts into the Run contract.

Recognized shapes (detection order matters — see docstrings):

1. native Run          — {"output": ..., "nodes": ..., "usage": ..., "meta": ..., "error": ...}
2. legacy archive  — {"report": str, "nodes": {...}, "elapsed": n, "status": s}
3. difyctl run -o json — {"data": {"outputs": {...}, ...}} (bare {"outputs": ...} tolerated)
"""
from __future__ import annotations

import json

from ..core import Run
from .usage import aggregate_usage

DEFAULT_REPORT_KEY = "report"


def normalize_run_payload(obj, report_key: str = DEFAULT_REPORT_KEY) -> Run:
    """Coerce any recognized payload into a Run. Unrecognized shapes degrade
    to a runner_error Run — never raise."""
    if not isinstance(obj, dict):
        return Run(error=f"run payload is {type(obj).__name__}, expected a JSON object",
                   meta={"status": "runner_error"})
    if "output" in obj:
        # native Run shape (what --save-fixture writes)
        return Run(output=str(obj.get("output") or ""),
                   nodes=obj.get("nodes") or {},
                   usage=obj.get("usage") or {},
                   meta=obj.get("meta") or {},
                   error=obj.get("error"))
    if "report" in obj:
        # legacy archive shape: {report, nodes, elapsed, status}
        nodes = obj.get("nodes") or {}
        return Run(output=str(obj.get("report") or ""),
                   nodes=nodes,
                   usage=aggregate_usage(nodes),
                   meta={"status": obj.get("status", "unknown"),
                         "elapsed": obj.get("elapsed")},
                   error=obj.get("error"))
    if "data" in obj or "outputs" in obj:
        return _normalize_difyctl(obj, report_key)
    if "error" in obj:
        return Run(error=str(obj["error"]), meta={"status": "runner_error"})
    return Run(error="unrecognized run payload shape (no output/report/outputs key)",
               meta={"status": "runner_error"})


def _normalize_difyctl(obj: dict, report_key: str) -> Run:
    """difyctl run -o json: outputs live at data.outputs (bare outputs
    tolerated). difyctl emits NO node trace, so Run.nodes is empty — node.*
    checks cannot be used against difyctl-acquired runs; the full outputs
    mapping is preserved at meta.outputs (reachable via `meta.outputs.<key>`
    selectors)."""
    data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
    outputs = data.get("outputs") if isinstance(data.get("outputs"), dict) else {}
    rep = outputs.get(report_key)
    if isinstance(rep, str):
        output = rep
    else:
        # report key absent/non-string: fall back to the whole outputs object
        output = json.dumps(outputs, ensure_ascii=False, sort_keys=True) if outputs else ""
    usage = {}
    tt = data.get("total_tokens")
    if isinstance(tt, (int, float)) and not isinstance(tt, bool):
        usage["tokens"] = int(tt)
    meta = {"status": data.get("status", "unknown"),
            "elapsed": data.get("elapsed_time"),
            "outputs": outputs,
            "node_trace": "none (difyctl does not emit one)"}
    err = data.get("error")
    return Run(output=output, nodes={}, usage=usage, meta=meta,
               error=str(err) if err else None)
