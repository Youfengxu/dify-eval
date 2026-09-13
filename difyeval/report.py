"""Scorecard renderer + eval.json/1 persistence.

Determinism contract: rendering is a pure function of the Experiment except
for exactly ONE wall-clock line, ``- generated: <ts>`` (and eval.json's
manifest.scored_at). ``difyeval replay --no-judge`` twice must produce
byte-identical scorecards minus that line — enforced by a test.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .core import Experiment
from .verdict import failed_gate_lines, gated_check_ids

VERDICT_BADGE = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}
_OK = "✅"
_NO = "❌"
_NA = "–"  # en dash: advisory/skipped, no pass-fail notion


def _mark(passed) -> str:
    if passed is True:
        return _OK
    if passed is False:
        return _NO
    return _NA


def render_scorecard(exp: Experiment, generated_at: str | None = None) -> str:
    case = exp.case
    results = exp.results
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()
    n_samples = len(case.samples)
    n_runs = len(results)
    gated = gated_check_ids(case)
    gated_set = set(gated)

    L = [f"# difyeval scorecard — `{case.case_id}` × `{exp.profile}`", ""]
    L.append(f"- generated: {generated_at}")
    header_bits = f"- mode: {case.mode} · samples: {n_samples} · runs: {n_runs}"
    if case.expect_verdict:
        header_bits += f" · expect_verdict: {case.expect_verdict}"
    L.append(header_bits)
    dsl = exp.manifest.get("dsl_sha256")
    if dsl:
        L.append(f"- dsl_sha256: {dsl}")
    L.append("")

    L.append(f"## Verdict: {VERDICT_BADGE.get(exp.verdict, exp.verdict)}")
    L.append("")
    L.append(f"> Scope: {exp.scope}")
    if exp.verdict == "WARN" and exp.warn_reason:
        L.append(f"> ⚠ {exp.warn_reason}")
    if exp.verdict == "FAIL":
        for line in failed_gate_lines(case, results):
            L.append(f"> ❌ gate failed: {line}")
    L.append("")

    # ---- checks table: gated first, then advisory ----
    by_check: dict[str, list] = {}
    for r in results:
        for s in r.scores:
            by_check.setdefault(s.check_id, []).append(s)

    def _row(spec) -> str:
        cid = spec["id"]
        scores = by_check.get(cid, [])
        n = len(scores)
        ok = sum(1 for s in scores if s.passed is True)
        skipped = all(s.passed is None for s in scores) if scores else True
        marginal = any(s.marginal for s in scores)
        if skipped:
            cell = f"{_NA} advisory"
            if scores and all(s.detail.get("skipped") for s in scores):
                cell = f"{_NA} skipped"
        else:
            cell = f"{_OK if ok == n else _NO} {ok}/{n}"
        if marginal:
            cell += " (marginal)"
        return f"| {cid} | {spec['type']} | {cell} |"

    gated_specs = [c for c in case.checks if c["id"] in gated_set]
    advis_specs = [c for c in case.checks if c["id"] not in gated_set]
    if gated_specs:
        L += ["### Gated checks", "", "| check | type | result |", "|---|---|---|"]
        L += [_row(c) for c in gated_specs]
        L.append("")
    if advis_specs:
        L += ["### Advisory checks", "", "| check | type | result |", "|---|---|---|"]
        L += [_row(c) for c in advis_specs]
        L.append("")

    # ---- runs table ----
    L += ["## Runs", "", "| sample | repeat | status | elapsed | output chars | error |",
          "|---|---|---|---|---|---|"]
    for r in results:
        meta = r.run.meta or {}
        L.append(f"| {r.sample_id} | {r.repeat} | {meta.get('status', '?')} | "
                 f"{meta.get('elapsed') if meta.get('elapsed') is not None else _NA} | "
                 f"{len(r.run.output or '')} | {(r.run.error or _NA)} |")
    L.append("")

    # ---- per-check detail sections (case.yml order) ----
    L.append("## Check details")
    for spec in case.checks:
        cid = spec["id"]
        kind = "gate" if cid in gated_set else "advisory"
        L += ["", f"### {cid} — `{spec['type']}` [{kind}]"]
        for r in results:
            for s in r.scores:
                if s.check_id != cid:
                    continue
                val = "" if s.value is None else f" · value={s.value}"
                marg = " (marginal)" if s.marginal else ""
                L.append(f"- {r.sample_id} r{r.repeat}: {_mark(s.passed)}{marg} — "
                         f"{s.evidence}{val}")
                _render_rubric_detail(L, s)
                _render_panel_detail(L, s)

    # ---- judge panel block ----
    L += ["", "## Judge panel"]
    backends = exp.manifest.get("judge_backends") or []
    if not backends:
        L.append("- no active judge backends (judge skipped) — all judge checks advisory-skipped")
    else:
        names = ", ".join(f"{b['name']} ({b['vendor']})" for b in backends)
        vendors = sorted({b["vendor"] for b in backends})
        tag = ("cross-vendor panel" if len(vendors) >= 2 else
               "⚠ SAME-VENDOR panel — advisory only; add a cross-vendor judge key")
        L.append(f"- backends: {names} [{tag}]")
        for t in exp.manifest.get("truncation") or []:
            L.append(f"- ⚠ report truncated at {t.get('cap')} chars for judge check "
                     f"'{t.get('check_id')}' ({t.get('sample_id')}) — original "
                     f"{t.get('report_chars')} chars")
        dim_stats = _aggregate_dims(results)
        if dim_stats:
            L.append("- tradecraft dims (mean of per-run medians):")
            for d in sorted(dim_stats):
                xs = dim_stats[d]
                L.append(f"  - **{d}**: {round(sum(xs) / len(xs), 3)} (n={len(xs)})")
    L.append("")

    # ---- metrics block ----
    L += ["## Metrics", ""]
    pr = exp.metrics.get("pass_rate") or {}
    L.append(f"- gated pass-rate: {pr.get('passed')}/{pr.get('total')}"
             + (f" ({pr.get('rate')})" if pr.get("rate") is not None else "")
             + f" — {pr.get('basis')}")
    per_check = exp.metrics.get("per_check") or {}
    for spec in case.checks:
        cid = spec["id"]
        st = per_check.get(cid)
        if not st:
            continue
        if "mean" in st:
            L.append(f"- {cid}: mean {st['mean']} ± {st['stderr']} (stderr, n={st['n']}), "
                     f"range [{st['min']}, {st['max']}]")
        else:
            L.append(f"- {cid}: value {st['value']} (n=1 — no interval; single observation)")
    cons = exp.metrics.get("repeat_consistency") or {}
    for sid in sorted(cons):
        c = cons[sid]
        L.append(f"- repeat consistency {sid}: {c['all_repeats_agree']} of checks agree "
                 f"across {c['repeats']} repeats")
    return "\n".join(L) + "\n"


def _render_rubric_detail(L: list, s) -> None:
    items = s.detail.get("items")
    if not items:
        return
    L += ["", "  | id | w | verdict | item | evidence |", "  |---|---|---|---|---|"]
    for it in items:
        L.append(f"  | {it['id']} | {it['weight']} | "
                 f"{_OK if it['pass'] else _NO} ({it['votes']}) | "
                 f"{it['item'][:70]} | {it['evidence'][:60]} |")
    L.append("")


def _render_panel_detail(L: list, s) -> None:
    med = s.detail.get("median")
    if not med:
        return
    indep = s.detail.get("independent_vendor")
    tag = "cross-vendor" if indep else "same-vendor (advisory only)"
    L.append(f"  - panel [{tag}]: "
             + "; ".join(f"{d}: median {med[d]['median']} (spread {med[d]['spread']}, "
                         f"n={med[d]['n']})" for d in sorted(med)))


def _aggregate_dims(results) -> dict:
    dims: dict[str, list] = {}
    for r in results:
        for s in r.scores:
            for d, v in (s.detail.get("median") or {}).items():
                if isinstance(v, dict) and isinstance(v.get("median"), (int, float)):
                    dims.setdefault(d, []).append(v["median"])
    return dims


# --------------------------------------------------------------------------
# eval.json/1
# --------------------------------------------------------------------------

def experiment_to_dict(exp: Experiment) -> dict:
    return {
        "schema": "eval.json/1",
        "case": exp.case.raw,            # config echo — exactly what was authored
        "case_path": exp.case.path,
        "profile": exp.profile,
        "verdict": exp.verdict,
        "warn_reason": exp.warn_reason,
        "scope": exp.scope,
        "results": [
            {
                "sample_id": r.sample_id,
                "repeat": r.repeat,
                "run": r.run.to_dict(),
                # judge per-judge raw outputs + rationales live in each
                # judge-check Score's detail.per_judge — retained here
                "scores": [s.to_dict() for s in r.scores],
            }
            for r in exp.results
        ],
        "metrics": exp.metrics,
        "manifest": exp.manifest,
    }


def write_eval_json(exp: Experiment, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(experiment_to_dict(exp), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def strip_generated_line(scorecard: str) -> str:
    """The determinism-contract helper: everything but the single timestamp line."""
    return "\n".join(line for line in scorecard.split("\n")
                     if not line.startswith("- generated: "))
