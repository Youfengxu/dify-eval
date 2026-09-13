"""``difyeval drift`` — re-score anchor runs and compare against baselines.

anchors.yml lists anchors::

    anchors:                       # (a bare top-level list also works)
      - case: cases/demo/case.yml  # paths relative to the anchors.yml
        runs: [runs/s1.json]       # frozen run files (the SAME runs the
        baseline: base/eval.json   #   baseline was scored from)

For each anchor the runs are re-scored NOW (this is the one networked
difyeval command — the judge panel runs live unless --no-judge) and compared
against the baseline eval.json:

* deterministic (non-judge) checks must be IDENTICAL per (sample, repeat,
  check): any change is a loud DETERMINISM-BREAK — the code or config moved
  under you, which is a different failure class from judge drift;
* judge drift: any rubric item flip, or any per-dim panel-median delta
  > DIM_DRIFT_THRESHOLD, marks the anchor DRIFT.

Exit 1 on any DRIFT or DETERMINISM-BREAK; 0 when every anchor is clean.
Under --no-judge only the deterministic comparison runs (a deterministic-only
drift check), and the judge column reports "not assessed".
"""
from __future__ import annotations

import os
from statistics import median

import yaml

from .core import ConfigError, load_case
from .engine import build_context, score_results
from .evalio import eval_profile, iter_scores, load_eval
from .report import experiment_to_dict
from .runners.file import assign_run_paths
from .validate import prepare_case
from . import checks as registry

DIM_DRIFT_THRESHOLD = 0.15


# --------------------------------------------------------------------------
# anchors.yml
# --------------------------------------------------------------------------

def load_anchors(path: str, baseline_dir: str | None = None) -> list[dict]:
    """Parse anchors.yml; resolve every path (case/runs relative to the
    anchors file, baseline relative to --baseline-dir when given)."""
    if not os.path.isfile(path):
        raise ConfigError(f"anchors file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in anchors file {path}: {e}") from e
    entries = raw.get("anchors") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"anchors file {path} must contain a non-empty list "
                          f"(top-level or under 'anchors:')")
    base_dir = os.path.dirname(os.path.abspath(path))

    def _resolve(p: str, against: str) -> str:
        return p if os.path.isabs(p) else os.path.join(against, p)

    out = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ConfigError(f"anchors[{i}] must be a mapping")
        case = e.get("case")
        runs = e.get("runs")
        baseline = e.get("baseline")
        if not isinstance(case, str) or not case.strip():
            raise ConfigError(f"anchors[{i}]: 'case' is required (path to case.yml)")
        if not isinstance(runs, list) or not runs or not all(isinstance(r, str) for r in runs):
            raise ConfigError(f"anchors[{i}]: 'runs' must be a non-empty list of run-file paths")
        if not isinstance(baseline, str) or not baseline.strip():
            raise ConfigError(f"anchors[{i}]: 'baseline' is required (path to eval.json)")
        out.append({
            "case": _resolve(case, base_dir),
            "runs": [_resolve(r, base_dir) for r in runs],
            "baseline": _resolve(baseline, baseline_dir or base_dir),
        })
    return out


# --------------------------------------------------------------------------
# comparisons (all on eval.json-dict shape — the new Experiment is converted
# via experiment_to_dict so both sides go through identical extraction)
# --------------------------------------------------------------------------

def _det_score_map(ev: dict, det_ids: set) -> dict:
    """{(sample, repeat, check): (passed, value)} for deterministic checks."""
    out = {}
    for r, s in iter_scores(ev):
        cid = s.get("check_id")
        if cid in det_ids:
            out[(r.get("sample_id"), r.get("repeat"), cid)] = (s.get("passed"), s.get("value"))
    return out


def _rubric_pass_map(ev: dict) -> dict:
    """{(check, sample, repeat, item): pass} for every rubric item verdict."""
    out = {}
    for r, s in iter_scores(ev):
        for it in (s.get("detail") or {}).get("items") or []:
            if isinstance(it, dict):
                out[(s.get("check_id"), r.get("sample_id"), r.get("repeat"),
                     str(it.get("id")))] = bool(it.get("pass"))
    return out


def _dim_aggregate(ev: dict) -> dict:
    """{dim: median of all per-run panel medians}."""
    dims: dict[str, list] = {}
    for _, s in iter_scores(ev):
        med = (s.get("detail") or {}).get("median") or {}
        if isinstance(med, dict):
            for d in sorted(med):
                v = med[d]
                if isinstance(v, dict) and isinstance(v.get("median"), (int, float)) \
                        and not isinstance(v.get("median"), bool):
                    dims.setdefault(d, []).append(v["median"])
    return {d: round(median(xs), 4) for d, xs in dims.items()}


def compare_anchor(case, base: dict, new: dict, judge_assessed: bool) -> dict:
    """Compare a re-scored anchor against its baseline. Returns
    {status, breaks, flips, dim_drifts, notes}."""
    det_ids = set()
    for spec in case.checks:
        cd = registry.get(spec["type"])
        if cd is not None and not cd.judge:
            det_ids.add(spec["id"])

    breaks, flips, dim_drifts, notes = [], [], [], []

    bmap, nmap = _det_score_map(base, det_ids), _det_score_map(new, det_ids)
    for key in sorted(set(bmap) | set(nmap), key=str):
        sample, repeat, cid = key
        loc = f"{cid} ({sample} r{repeat})"
        if key not in bmap:
            breaks.append(f"{loc}: score present only in the re-score (baseline lacks it)")
        elif key not in nmap:
            breaks.append(f"{loc}: score present only in the baseline (re-score lacks it)")
        elif bmap[key] != nmap[key]:
            (bp, bv), (np_, nv) = bmap[key], nmap[key]
            breaks.append(f"{loc}: passed {bp} → {np_}, value {bv} → {nv}")

    if judge_assessed:
        brub, nrub = _rubric_pass_map(base), _rubric_pass_map(new)
        for key in sorted(set(brub) | set(nrub), key=str):
            cid, sample, repeat, item = key
            loc = f"{cid}/{item} ({sample} r{repeat})"
            if key not in brub:
                flips.append(f"{loc}: item judged only in the re-score")
            elif key not in nrub:
                flips.append(f"{loc}: item judged only in the baseline")
            elif brub[key] != nrub[key]:
                flips.append(f"{loc}: {'pass' if brub[key] else 'fail'} → "
                             f"{'pass' if nrub[key] else 'fail'}")
        bdim, ndim = _dim_aggregate(base), _dim_aggregate(new)
        for d in sorted(set(bdim) | set(ndim)):
            if d not in bdim:
                dim_drifts.append(f"{d}: scored only in the re-score ({ndim[d]})")
            elif d not in ndim:
                dim_drifts.append(f"{d}: scored only in the baseline ({bdim[d]})")
            else:
                delta = round(ndim[d] - bdim[d], 4)
                if abs(delta) > DIM_DRIFT_THRESHOLD:
                    dim_drifts.append(f"{d}: {bdim[d]} → {ndim[d]} "
                                      f"(Δ {'+' if delta > 0 else ''}{delta:g} "
                                      f"> {DIM_DRIFT_THRESHOLD})")
    else:
        notes.append("judge not assessed (--no-judge or no active judge backends) — "
                     "deterministic-only drift check")

    status = ("DETERMINISM-BREAK" if breaks
              else "DRIFT" if (flips or dim_drifts) else "OK")
    return {"status": status, "breaks": breaks, "flips": flips,
            "dim_drifts": dim_drifts, "notes": notes,
            "det_compared": len(set(bmap) & set(nmap))}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_drift(anchors_path: str, baseline_dir: str | None = None,
              no_judge: bool = False, judges_only: list | None = None,
              transport=None, env: dict | None = None) -> tuple[str, int]:
    """Re-score every anchor and compare. Returns (markdown, exit_code):
    0 clean, 1 on any DRIFT / DETERMINISM-BREAK. Config problems raise
    ConfigError (exit 2 in the CLI)."""
    anchors = load_anchors(anchors_path, baseline_dir)
    reports = []
    for a in anchors:
        case = load_case(a["case"])
        prepare_case(case)
        ctx = build_context(case, no_judge=no_judge, judges_only=judges_only,
                            env=env, transport=transport)
        results = assign_run_paths(case, a["runs"])
        base = load_eval(a["baseline"])
        exp = score_results(case, results, ctx, profile="drift-recheck")
        new = experiment_to_dict(exp)
        cmp_ = compare_anchor(case, base, new, judge_assessed=bool(ctx.judges))
        reports.append({"case_id": case.case_id, "baseline_path": a["baseline"],
                        "baseline_profile": eval_profile(base),
                        "baseline_verdict": base.get("verdict", "?"),
                        "new_verdict": exp.verdict, **cmp_})

    dirty = any(r["status"] != "OK" for r in reports)
    L = [f"# difyeval drift — {len(reports)} anchor(s) from `{anchors_path}`", "",
         "| case | baseline | deterministic | judge | status |",
         "|---|---|---|---|---|"]
    badge = {"OK": "✅ OK", "DRIFT": "🔴 DRIFT", "DETERMINISM-BREAK": "❌ DETERMINISM-BREAK"}
    for r in reports:
        det = (f"❌ {len(r['breaks'])} break(s)" if r["breaks"]
               else f"identical ({r['det_compared']} score(s))")
        if r["notes"]:
            jcell = "not assessed"
        elif r["flips"] or r["dim_drifts"]:
            jcell = f"{len(r['flips'])} flip(s), {len(r['dim_drifts'])} dim drift(s)"
        else:
            jcell = "stable"
        L.append(f"| {r['case_id']} | {r['baseline_profile']} "
                 f"({r['baseline_verdict']}) | {det} | {jcell} | "
                 f"{badge[r['status']]} |")
    L.append("")

    for r in reports:
        L += [f"## {r['case_id']} — {badge[r['status']]}", "",
              f"- baseline: `{r['baseline_path']}` (profile {r['baseline_profile']}, "
              f"verdict {r['baseline_verdict']})",
              f"- re-scored verdict: {r['new_verdict']}"
              + ("" if r["new_verdict"] == r["baseline_verdict"] else "  ⚠ changed"),
              f"- deterministic scores compared: {r['det_compared']}"]
        for b in r["breaks"]:
            L.append(f"- ❌ DETERMINISM-BREAK: {b}")
        for fl in r["flips"]:
            L.append(f"- 🔴 rubric flip: {fl}")
        for dd in r["dim_drifts"]:
            L.append(f"- 🔴 dim drift: {dd}")
        for n in r["notes"]:
            L.append(f"- note: {n}")
        L.append("")
    return "\n".join(L).rstrip("\n") + "\n", (1 if dirty else 0)


def drift_command(anchors_path: str, baseline_dir: str | None = None,
                  no_judge: bool = False, judges_only: list | None = None,
                  out: str | None = None) -> int:
    md, rc = run_drift(anchors_path, baseline_dir=baseline_dir,
                       no_judge=no_judge, judges_only=judges_only)
    print(md, end="")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"drift report written: {out}")
    return rc
