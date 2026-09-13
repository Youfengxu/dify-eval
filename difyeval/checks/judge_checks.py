"""Judge-backed checks: judge.rubric and judge.tradecraft.

Both are ADVISORY ALWAYS — the judge never gates. ``gate: true`` on a judge
check is a loud config error (difyeval.validate), and the engine forces
``advisory=True`` on every Score from a judge-registered check as a second
line of defense.

A failing *essential* rubric item still matters: it drives the WARN verdict
tier (gates pass, but a load-bearing claim was missed) — see difyeval.verdict.
"""
from __future__ import annotations

from .. import judge
from ..core import ConfigError, VALID_RUBRIC_WEIGHTS
from . import make_score, register


def _skipped(spec, why: str):
    s = make_score(spec, None, None, why, detail={"skipped": True})
    s.advisory = True
    return s


def _record_truncation(ctx, spec, sample, result: dict) -> None:
    if result.get("truncated"):
        ctx.truncations.append({"check_id": spec["id"], "sample_id": sample.id,
                                "report_chars": result.get("report_chars"),
                                "cap": judge.REPORT_CAP})


# --------------------------------------------------------------------------
# judge.rubric
# --------------------------------------------------------------------------

def validate_rubric_items(spec, items) -> None:
    if not isinstance(items, list) or not items:
        raise ConfigError(f"check '{spec['id']}': needs a non-empty 'items' list")
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not str(it.get("item", "")).strip():
            raise ConfigError(f"check '{spec['id']}': items[{i}] needs an 'item' text")
        w = it.get("weight", "expected")
        if w not in VALID_RUBRIC_WEIGHTS:
            raise ConfigError(f"check '{spec['id']}': items[{i}] weight {w!r} "
                              f"not in {VALID_RUBRIC_WEIGHTS}")


def _validate_rubric(spec, case):
    validate_rubric_items(spec, spec.get("items"))


@register("judge.rubric", judge=True, validator=_validate_rubric)
def judge_rubric_check(spec, sample, run, ctx):
    if not ctx.judges:
        return _skipped(spec, "judge skipped (--no-judge or no active judge backends)")
    result = judge.run_rubric(run.output, spec["items"], ctx.judges,
                              ctx.case.case_id, transport=ctx.transport)
    if not result:
        return _skipped(spec, "rubric skipped (empty report)")
    _record_truncation(ctx, spec, sample, result)
    s = make_score(spec, result.get("score"), result.get("essential_ok"),
                   f"rubric score {result.get('score')} — "
                   f"{sum(1 for it in result['items'] if it['pass'])}/{len(result['items'])} "
                   f"items passed"
                   + ("" if result.get("essential_ok") else " (essential item FAILED)"),
                   detail=result)
    s.advisory = True
    return s


# --------------------------------------------------------------------------
# judge.tradecraft
# --------------------------------------------------------------------------

def _resolve_dims(spec) -> dict:
    """Default = the three shipped TRADECRAFT_DIMS. A case may override with
    `dims:` — a list of built-in dim names and/or {id, criterion} mappings."""
    raw = spec.get("dims")
    if not raw:
        return dict(judge.TRADECRAFT_DIMS)
    dims = {}
    for d in raw:
        if isinstance(d, str):
            if d not in judge.TRADECRAFT_DIMS:
                raise ConfigError(f"check '{spec['id']}': unknown tradecraft dim '{d}' "
                                  f"(built-ins: {sorted(judge.TRADECRAFT_DIMS)}; custom dims "
                                  f"need {{id, criterion}})")
            dims[d] = judge.TRADECRAFT_DIMS[d]
        elif isinstance(d, dict) and d.get("id") and d.get("criterion"):
            dims[str(d["id"])] = str(d["criterion"])
        else:
            raise ConfigError(f"check '{spec['id']}': each dim must be a built-in name "
                              f"or {{id, criterion}}")
    return dims


def _validate_tradecraft(spec, case):
    _resolve_dims(spec)


@register("judge.tradecraft", judge=True, validator=_validate_tradecraft)
def judge_tradecraft_check(spec, sample, run, ctx):
    if not ctx.judges:
        return _skipped(spec, "judge skipped (--no-judge or no active judge backends)")
    dims = _resolve_dims(spec)
    result = judge.run_panel(run.output, dims, ctx.case.judge_notes,
                             sample.reference, ctx.judges, transport=ctx.transport)
    if not result:
        return _skipped(spec, "tradecraft panel skipped (empty report)")
    _record_truncation(ctx, spec, sample, result)
    medians = result.get("median") or {}
    scored = [medians[d]["median"] for d in sorted(medians)]
    value = round(sum(scored) / len(scored), 3) if scored else None
    ev = ("; ".join(f"{d}={medians[d]['median']}" for d in sorted(medians))
          if medians else "no dim received a numeric score from any judge")
    s = make_score(spec, value, None, ev, detail=result)
    s.advisory = True
    return s
