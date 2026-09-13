"""Shared eval.json loading + extraction for diff / audit / drift.

eval.json files handed to these commands are *config* (the user names them on
the CLI), so unreadable/undecodable files raise ConfigError — exit 2. Missing
optional structure INSIDE a loadable eval.json (no rubric detail, no usage) is
data-shaped and degrades to empty extractions instead.
"""
from __future__ import annotations

import json

from .core import ConfigError
from .judge import _rubric_verdict


def load_eval(path: str) -> dict:
    """Load an eval.json produced by difyeval. ConfigError on unreadable /
    undecodable / structurally-alien files."""
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except OSError as e:
        raise ConfigError(f"cannot read eval file {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"eval file {path} is not valid JSON: {e}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("results"), list):
        raise ConfigError(f"eval file {path} is not a difyeval eval.json "
                          f"(expected a mapping with a 'results' list)")
    return obj


def eval_case_id(ev: dict) -> str:
    case = ev.get("case")
    if isinstance(case, dict) and isinstance(case.get("case_id"), str):
        return case["case_id"]
    return "?"


def eval_profile(ev: dict) -> str:
    return str(ev.get("profile") or "?")


def iter_scores(ev: dict):
    """Yield (result_dict, score_dict) in stored order — deterministic."""
    for r in ev.get("results") or []:
        if not isinstance(r, dict):
            continue
        for s in r.get("scores") or []:
            if isinstance(s, dict):
                yield r, s


def check_order(ev: dict) -> list[str]:
    """Check ids in authored case.yml order, then any extras found in scores."""
    out, seen = [], set()
    case = ev.get("case") or {}
    for spec in (case.get("checks") or []) if isinstance(case, dict) else []:
        cid = spec.get("id") if isinstance(spec, dict) else None
        if isinstance(cid, str) and cid not in seen:
            seen.add(cid)
            out.append(cid)
    for _, s in iter_scores(ev):
        cid = s.get("check_id")
        if isinstance(cid, str) and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _judge_vote(per_judge: dict, judge_name: str, item_id: str):
    """A judge's binary vote on one rubric item — None when the judge errored
    or never addressed the item (distinct from an explicit False)."""
    pj = per_judge.get(judge_name)
    if not isinstance(pj, dict) or "_error" in pj:
        return None
    v = pj.get(item_id)
    if v is None:
        return None
    return _rubric_verdict(v)[0]


def rubric_entries(ev: dict) -> list[dict]:
    """Every rubric item verdict across all runs, in stored order. Each entry:
    {check, sample, repeat, item, weight, text, panel ('pass'|'fail'), votes,
    evidence, judge_votes: {judge_name: bool|None}}."""
    out = []
    for r, s in iter_scores(ev):
        detail = s.get("detail") or {}
        items = detail.get("items")
        if not isinstance(items, list) or not items:
            continue
        per_judge = detail.get("per_judge") or {}
        judge_names = sorted(per_judge) if isinstance(per_judge, dict) else []
        for it in items:
            if not isinstance(it, dict):
                continue
            out.append({
                "check": s.get("check_id"),
                "sample": r.get("sample_id"),
                "repeat": r.get("repeat"),
                "item": it.get("id"),
                "weight": it.get("weight"),
                "text": it.get("item"),
                "panel": "pass" if it.get("pass") else "fail",
                "votes": it.get("votes"),
                "evidence": it.get("evidence") or "",
                "judge_votes": {jn: _judge_vote(per_judge, jn, str(it.get("id")))
                                for jn in judge_names},
            })
    return out


def dim_entries(ev: dict) -> list[dict]:
    """Every tradecraft-dim panel median across all runs, in stored order
    (dims sorted within a score). Each entry: {check, sample, repeat, dim,
    median, spread, n, judge_scores: {judge_name: float|None}}."""
    out = []
    for r, s in iter_scores(ev):
        detail = s.get("detail") or {}
        med = detail.get("median")
        if not isinstance(med, dict) or not med:
            continue
        per_judge = detail.get("per_judge") or {}
        judge_names = sorted(per_judge) if isinstance(per_judge, dict) else []
        for dim in sorted(med):
            v = med[dim]
            if not isinstance(v, dict):
                continue
            scores = {}
            for jn in judge_names:
                pj = per_judge.get(jn)
                dv = pj.get(dim) if isinstance(pj, dict) else None
                sc = dv.get("score") if isinstance(dv, dict) else None
                scores[jn] = sc if isinstance(sc, (int, float)) and not isinstance(sc, bool) else None
            out.append({"check": s.get("check_id"), "sample": r.get("sample_id"),
                        "repeat": r.get("repeat"), "dim": dim,
                        "median": v.get("median"), "spread": v.get("spread"),
                        "n": v.get("n"), "judge_scores": scores})
    return out


def dim_medians(ev: dict) -> dict:
    """{dim: [per-run panel medians...]} across all runs, stored order."""
    dims: dict[str, list] = {}
    for _, s in iter_scores(ev):
        med = (s.get("detail") or {}).get("median") or {}
        if not isinstance(med, dict):
            continue
        for d in sorted(med):
            v = med[d]
            if isinstance(v, dict) and isinstance(v.get("median"), (int, float)) \
                    and not isinstance(v.get("median"), bool):
                dims.setdefault(d, []).append(v["median"])
    return dims


def sum_usage(ev: dict) -> dict:
    """Total tokens + per-currency cost across all runs: {tokens, cost}."""
    tokens = 0
    cost: dict[str, float] = {}
    for r in ev.get("results") or []:
        usage = (r.get("run") or {}).get("usage") or {}
        t = usage.get("tokens")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            tokens += int(t)
        for cur, amt in (usage.get("cost") or {}).items():
            if isinstance(amt, (int, float)) and not isinstance(amt, bool):
                cost[str(cur)] = round(cost.get(str(cur), 0.0) + float(amt), 6)
    return {"tokens": tokens, "cost": cost}
