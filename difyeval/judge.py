"""Cross-vendor LLM judge panel — ported from an earlier internal eval runner.

Ported verbatim-in-spirit: JUDGE_BACKENDS (vendor list, activation by env-key
presence), the OpenAI-compatible /chat/completions call, per-dim median +
spread + n across the panel, independent_vendor = >=2 distinct vendors,
per-judge raw JSON + rationales retained, the weighted binary rubric with
majority vote per item, and the TRADECRAFT_DIMS wording (the three default
dims shipped here).

Consolidation additions (locked by design review):

* one retry (RETRY_BACKOFF_S backoff) on transport errors / non-200 ONLY —
  never on JSON-parse failure of a model reply;
* report truncation at REPORT_CAP chars is *recorded* (manifest + a warning
  line in the judge scorecard block), not silent;
* the judge NEVER gates — every judge-derived Score is advisory by
  construction (enforced in difyeval.validate + the engine);
* deterministic per-judge rubric item-order shuffle:
  ``random.Random(f"{case_id}:{judge_name}")`` — kills positional bias
  without sacrificing replayability;
* network access is lazy: ``requests`` is imported only inside the real
  transport, so --no-judge paths (and tests, which inject a stub transport)
  never touch the network.
"""
from __future__ import annotations

import json
import random
import re
import time
from statistics import median
from typing import Callable

REPORT_CAP = 40_000          # chars of report fed to a judge; overflow recorded
REFERENCE_CAP = 8_000        # chars of serialized reference/ground-truth
RETRY_BACKOFF_S = 5.0        # single retry backoff (transport/non-200 only)
JUDGE_TIMEOUT_S = 150

RUBRIC_WEIGHTS = {"essential": 3, "expected": 2, "bonus": 1}

# The three default judge.tradecraft dims — wording ported from an earlier internal pipeline
# run_eval.py TRADECRAFT_DIMS (ICD-203-adapted analytic-tradecraft standards).
TRADECRAFT_DIMS = {
    "uncertainty_expression": "Confidence language matches evidence strength; unverified/unresolved items "
        "are explicitly marked as such, not silently asserted; probability terms are used consistently.",
    "info_vs_judgment": "Underlying collected information is clearly distinguished from analyst/model "
        "inference and assumption (e.g. deterministic data vs LLM-inferred clustering).",
    "alternatives_considered": "Plausible alternative explanations are addressed — e.g. an alternative "
        "origin, or organic virality considered before asserting coordination.",
}

JUDGE_SYS = (
    "You are a strict evaluation judge for a court-citable analysis tool. Score a REPORT on each "
    "requested dimension 0.0-1.0 (1.0 = fully meets the criterion), following each dimension's "
    "CRITERION definition exactly. For every dimension, reason claim-by-claim over the report "
    "BEFORE scoring. Reference-comparison dimensions compare against GROUND TRUTH — if GROUND "
    "TRUTH is empty, return null for those. Tradecraft dimensions are judged from the report "
    "alone. Judge CORRECTNESS and GROUNDING ONLY — never reward length, fluency, or confidence; "
    "a concise correct report must outscore a longer one. Penalize over-claiming (asserting "
    "CONFIRMED where evidence supports only LIKELY). Be conservative. Return ONLY JSON: "
    '{"<dim>": {"score": <float 0-1 or null>, "rationale": "<=30 words"}}.'
)

RUBRIC_SYS = (
    "You are a strict evaluation judge. For EACH rubric item, check whether the REPORT satisfies "
    "it. Reason over the report first; verdicts must be evidence-based — include a short quote or "
    "pointer from the report as evidence (or state what is absent). Negative items ('does NOT ...') "
    "pass only when the offending content is truly absent. Return ONLY JSON: "
    '{"<item_id>": {"pass": true|false, "evidence": "<=25 words"}}.'
)

# Vendor-flexible judge backends (OpenAI-compatible /chat/completions). A backend
# is ACTIVE iff its key env var is present. Ported from an earlier internal eval runner.
JUDGE_BACKENDS = [
    {"name": "kimi-k2.6", "vendor": "moonshot", "key_env": "MOONSHOT_API_KEY",
     "base_env": "MOONSHOT_BASE_URL", "base_default": "https://api.moonshot.ai/v1",
     "model": "kimi-k2.6", "params": {"temperature": 0.6, "thinking": {"type": "disabled"}}},
    {"name": "openrouter-claude", "vendor": "anthropic", "key_env": "OPENROUTER_API_KEY",
     "base_env": "OPENROUTER_BASE_URL", "base_default": "https://openrouter.ai/api/v1",
     "model_env": "OPENROUTER_JUDGE_MODEL", "model": "anthropic/claude-3.7-sonnet",
     "params": {"temperature": 0}},
    {"name": "openai", "vendor": "openai", "key_env": "OPENAI_API_KEY",
     "base_env": "OPENAI_BASE_URL", "base_default": "https://api.openai.com/v1",
     "model_env": "OPENAI_JUDGE_MODEL", "model": "gpt-4o", "params": {"temperature": 0}},
    {"name": "glm-5.2", "vendor": "zhipu", "key_env": "GLM_API_KEY",
     "base_env": "GLM_BASE_URL", "base_default": "https://api.z.ai/api/coding/paas/v4",
     "model_env": "GLM_MODEL", "model": "glm-5.2",
     "params": {"temperature": 0, "thinking": {"type": "disabled"}}},
    {"name": "local", "vendor": "local", "key_env": "LOCAL_LLM_API_KEY",
     "base_env": "LOCAL_LLM_BASE_URL", "base_default": None,
     "model_env": "LOCAL_LLM_MODEL", "model": None, "params": {"temperature": 0}},
]


def active_judges(env: dict, only=None) -> list[dict]:
    """Backends whose key env var (and base + model) resolve. Order follows
    JUDGE_BACKENDS — deterministic."""
    out = []
    for b in JUDGE_BACKENDS:
        if only and b["name"] not in only:
            continue
        key = env.get(b["key_env"])
        base = env.get(b.get("base_env", "")) or b.get("base_default")
        model = env.get(b.get("model_env", "")) or b.get("model")
        if key and base and model:
            out.append({**b, "_key": key, "_base": base, "_model": model})
    return out


def redacted_backends(judges: list[dict]) -> list[dict]:
    """Manifest view of the panel — never includes keys."""
    return [{"name": j["name"], "vendor": j["vendor"], "model": j.get("_model") or j.get("model")}
            for j in judges]


# --------------------------------------------------------------------------
# transport (lazy requests import) + retry
# --------------------------------------------------------------------------

def _http_chat(backend: dict, body: dict, timeout: float) -> tuple[int, str]:
    """Real transport: POST /chat/completions. Returns (status, content) where
    content is the assistant message on 200 (raw text otherwise). Only place
    in the judge path that touches the network — requests imported lazily so
    --no-judge / stubbed-transport paths stay offline."""
    import requests  # lazy by design

    r = requests.post(
        f"{backend['_base']}/chat/completions",
        headers={"Authorization": f"Bearer {backend['_key']}",
                 "Content-Type": "application/json"},
        json=body, timeout=timeout)
    if r.status_code != 200:
        return r.status_code, r.text[:500]
    try:
        return 200, r.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        # a 200 with an unreadable body is a parse problem, not a transport
        # problem -> no retry; surface it as unparseable content
        return 200, f"<<unreadable 200 body: {e}>>"


def _call_judge(backend: dict, body: dict, transport: Callable | None) -> tuple[str | None, str | None]:
    """One judge call with a single retry on transport errors / non-200 only.
    Returns (content, error). JSON-parse failures of the reply are NOT
    retried (that happens in the caller, after a successful transport)."""
    transport = transport or _http_chat
    last = None
    for attempt in (1, 2):
        try:
            status, content = transport(backend, body, JUDGE_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — transport failure, retry once
            last = f"transport error: {str(e)[:120]}"
        else:
            if status == 200:
                return content, None
            last = f"HTTP {status}"
        if attempt == 1:
            time.sleep(RETRY_BACKOFF_S)
    return None, last


def _extract_json(text: str):
    """Ported: pull the first {...} blob out of a model reply."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


def truncate_report(report: str) -> tuple[str, bool, int]:
    """Cap the report fed to judges; the flag is recorded, never silent."""
    report = report or ""
    if len(report) > REPORT_CAP:
        return report[:REPORT_CAP], True, len(report)
    return report, False, len(report)


# --------------------------------------------------------------------------
# dimension panel (judge.tradecraft and friends)
# --------------------------------------------------------------------------

def _panel_user(report: str, reference, judge_notes: str, dims: dict) -> str:
    crit = "\n".join(f"- {d}: {c}" for d, c in dims.items())
    ref_json = json.dumps(reference or {}, ensure_ascii=False, sort_keys=True)[:REFERENCE_CAP]
    return (f"DIMENSIONS TO SCORE: {', '.join(dims)}\n\n"
            f"CRITERIA:\n{crit}\n\n"
            f"GROUND TRUTH:\n{ref_json}\n\n"
            f"RUBRIC / JUDGE NOTES:\n{judge_notes or '(none)'}\n\n"
            f"REPORT:\n{report}\n\nScore now, JSON only.")


def run_panel(report: str, dims: dict, judge_notes: str, reference,
              judges: list[dict], transport: Callable | None = None) -> dict:
    """Run K judges over scored dimensions; median + spread + n per dim.
    Ported from run_eval.judge_panel/judge_one. Returns {} when there is
    nothing to judge (no panel / empty report)."""
    if not judges or not (report or "").strip() or not dims:
        return {}
    text, truncated, original = truncate_report(report)
    user = _panel_user(text, reference, judge_notes, dims)
    per = {}
    for j in judges:
        body = {"model": j["_model"], "max_tokens": 2600,
                "messages": [{"role": "system", "content": JUDGE_SYS},
                             {"role": "user", "content": user}]}
        body.update(j.get("params", {}))
        content, err = _call_judge(j, body, transport)
        if err:
            per[j["name"]] = {d: {"score": None, "rationale": err} for d in dims}
            continue
        parsed = _extract_json(content)
        if not isinstance(parsed, dict):
            per[j["name"]] = {d: {"score": None, "rationale": "unparseable judge reply"}
                              for d in dims}
        else:
            per[j["name"]] = parsed
    med = {}
    for d in dims:
        xs = [per[jn][d]["score"] for jn in per
              if isinstance(per[jn].get(d), dict)
              and isinstance(per[jn][d].get("score"), (int, float))
              and not isinstance(per[jn][d].get("score"), bool)]
        if xs:
            med[d] = {"median": round(median(xs), 3),
                      "spread": round(max(xs) - min(xs), 3), "n": len(xs)}
    vendors = sorted({j["vendor"] for j in judges})
    return {"per_judge": per, "median": med, "judges": [j["name"] for j in judges],
            "vendors": vendors, "independent_vendor": len(vendors) >= 2,
            "truncated": truncated, "report_chars": original}


# --------------------------------------------------------------------------
# weighted binary rubric (judge.rubric / gt.derived_rubric)
# --------------------------------------------------------------------------

def _rubric_verdict(v) -> tuple[bool, str]:
    """Ported: coerce a judge's per-item verdict to (pass, evidence) across
    the shapes judges actually emit."""
    if isinstance(v, dict):
        p = v.get("pass")
        if isinstance(p, str):
            p = p.strip().lower() in ("true", "yes", "1", "pass")
        return bool(p), str(v.get("evidence", "") or "")
    if isinstance(v, bool):
        return v, ""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "pass"), ""
    return False, ""


def normalize_rubric_items(items: list) -> list[dict]:
    out = []
    for i, it in enumerate(items or [], 1):
        it = it if isinstance(it, dict) else {"item": str(it)}
        out.append({"id": str(it.get("id") or f"r{i}"), "item": str(it.get("item", "")),
                    "weight": str(it.get("weight", "expected"))})
    return out


def run_rubric(report: str, items: list, judges: list[dict], case_id: str,
               transport: Callable | None = None) -> dict:
    """Weighted binary rubric, each item judged by every panel member with an
    evidence quote, majority vote per item, weighted score. Ported from
    run_eval.judge_rubric + the deterministic per-judge item-order shuffle."""
    norm = normalize_rubric_items(items)
    if not norm or not judges or not (report or "").strip():
        return {}
    text, truncated, original = truncate_report(report)
    per = {}
    for j in judges:
        # deterministic positional-bias mitigation: each judge sees the items
        # in an order seeded by (case_id, judge name) — replayable forever
        order = list(norm)
        random.Random(f"{case_id}:{j['name']}").shuffle(order)
        listing = "\n".join(f"- {it['id']} [{it['weight']}]: {it['item']}" for it in order)
        user = f"RUBRIC ITEMS:\n{listing}\n\nREPORT:\n{text}\n\nEvaluate now, JSON only."
        body = {"model": j["_model"], "max_tokens": 2000,
                "messages": [{"role": "system", "content": RUBRIC_SYS},
                             {"role": "user", "content": user}]}
        body.update(j.get("params", {}))
        content, err = _call_judge(j, body, transport)
        if err:
            per[j["name"]] = {"_error": err}
            continue
        parsed = _extract_json(content)
        per[j["name"]] = parsed if isinstance(parsed, dict) else {}
    results, wsum, wpass = [], 0, 0
    essential_ok = True
    for it in norm:  # aggregation in authored order — shuffle never leaks out
        vs = [_rubric_verdict((per.get(jn) or {}).get(it["id"])) for jn in sorted(per)]
        votes = [p for p, _ in vs]
        passed = sum(votes) * 2 > len(votes) if votes else False  # strict majority
        ev = next((e for p, e in vs if e), "")
        w = RUBRIC_WEIGHTS.get(it["weight"], 2)
        wsum += w
        wpass += w if passed else 0
        if it["weight"] == "essential" and not passed:
            essential_ok = False
        results.append({**it, "pass": passed, "votes": f"{sum(votes)}/{len(votes)}",
                        "evidence": ev[:120]})
    vendors = sorted({j["vendor"] for j in judges})
    return {"items": results, "score": round(wpass / wsum, 3) if wsum else None,
            "essential_ok": essential_ok, "judges": sorted(per),
            "per_judge": per, "vendors": vendors,
            "independent_vendor": len(vendors) >= 2,
            "truncated": truncated, "report_chars": original}
