"""Verdict assembly + honest metrics.

Tiers (FAIL > WARN > PASS):

* FAIL — any gated Score not passed. Gates are non-judge checks the case
  marked ``gate: true``; a gated Score whose passed is not True (including a
  degraded None) fails the gate: an unprovable gate is a failed gate.
* WARN — gates passed, but a rubric ESSENTIAL item failed (judge.rubric /
  gt.derived_rubric detail.essential_ok == False). The judge never FAILs a
  case, but a missed load-bearing claim must not silently PASS either.
* PASS — everything else.

Scope statement: rendered under the verdict, mode-aware — live-mode PASS is
process-soundness only; reference-mode PASS is scoped to the gated checks.

Honest-stats rule: ranges + explicit n, never CIs/stderr for n<2.
"""
from __future__ import annotations

from math import sqrt
from statistics import mean, stdev

from .core import Case, Experiment


def gated_check_ids(case: Case) -> list[str]:
    from . import checks as registry

    out = []
    for spec in case.checks:
        cd = registry.get(spec["type"])
        if spec.get("gate") and not (cd and cd.judge):
            out.append(spec["id"])
    return out


def scope_statement(case: Case) -> str:
    gated = gated_check_ids(case)
    gated_txt = ", ".join(gated) if gated else "none declared"
    if case.mode == "live":
        return (f"live mode — PASS = process-soundness only (gated checks: {gated_txt}); "
                f"output correctness is NOT established without a reference.")
    return (f"reference mode — PASS is scoped to the gated checks ({gated_txt}); "
            f"claims outside those checks are not asserted by this verdict.")


def compute_verdict(case: Case, results: list) -> tuple[str, str | None]:
    """(verdict, warn_reason) over all sample-repeat results."""
    gated = set(gated_check_ids(case))
    failed_gates = []
    for r in results:
        for s in r.scores:
            if s.check_id in gated and s.passed is not True:
                failed_gates.append(f"{s.check_id} ({r.sample_id} r{r.repeat}): {s.evidence}")
    if failed_gates:
        return "FAIL", None
    essential_failures = []
    for r in results:
        for s in r.scores:
            if s.detail.get("essential_ok") is False:
                essential_failures.append(f"{s.check_id} ({r.sample_id} r{r.repeat})")
    if essential_failures:
        return "WARN", ("rubric essential item(s) failed — hard gates passed but a "
                        "load-bearing claim was missed: " + "; ".join(essential_failures))
    return "PASS", None


def failed_gate_lines(case: Case, results: list) -> list[str]:
    gated = set(gated_check_ids(case))
    out = []
    for r in results:
        for s in r.scores:
            if s.check_id in gated and s.passed is not True:
                out.append(f"{s.check_id} ({r.sample_id} r{r.repeat}): {s.evidence}")
    return out


def compute_metrics(case: Case, results: list) -> dict:
    """pass-rate, per-check value stats (mean ± stderr only for n>=2),
    repeat-consistency per sample."""
    gated = set(gated_check_ids(case))
    total = len(results)
    passed_runs = 0
    for r in results:
        run_gates = [s for s in r.scores if s.check_id in gated]
        if all(s.passed is True for s in run_gates):
            passed_runs += 1
    metrics: dict = {
        "pass_rate": {"passed": passed_runs, "total": total,
                      "rate": round(passed_runs / total, 4) if total else None,
                      "basis": f"gated checks ({len(gated)}) all-pass per run"},
    }

    per_check: dict = {}
    for spec in case.checks:
        cid = spec["id"]
        xs = []
        for r in results:
            for s in r.scores:
                if s.check_id == cid and isinstance(s.value, (int, float)) \
                        and not isinstance(s.value, bool):
                    xs.append(float(s.value))
        if not xs:
            continue
        entry: dict = {"n": len(xs), "min": round(min(xs), 4), "max": round(max(xs), 4)}
        if len(xs) >= 2:
            entry["mean"] = round(mean(xs), 4)
            entry["stderr"] = round(stdev(xs) / sqrt(len(xs)), 4)
        else:
            entry["value"] = round(xs[0], 4)  # n=1: no mean±stderr, honest single value
        per_check[cid] = entry
    metrics["per_check"] = per_check

    consistency: dict = {}
    by_sample: dict = {}
    for r in results:
        by_sample.setdefault(r.sample_id, []).append(r)
    for sid in sorted(by_sample):
        reps = by_sample[sid]
        if len(reps) < 2:
            continue
        check_ids = [spec["id"] for spec in case.checks]
        agree = 0
        for cid in check_ids:
            outcomes = set()
            for r in reps:
                for s in r.scores:
                    if s.check_id == cid:
                        outcomes.add(s.passed)
            if len(outcomes) <= 1:
                agree += 1
        consistency[sid] = {"repeats": len(reps),
                            "all_repeats_agree": round(agree / len(check_ids), 4)
                            if check_ids else None}
    metrics["repeat_consistency"] = consistency
    return metrics
