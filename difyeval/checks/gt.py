"""gt.derived_rubric — map a structured reference list into rubric items.

Generic ground-truth-to-rubric bridge: a case declares where the reference
list lives (``from: reference.events``) and a text template
(``template: "Report includes event: {date} — {description}"``); each list
entry becomes one weighted rubric item and the whole set feeds the SAME
judge-rubric machinery (majority vote, weighted score, essential -> WARN).

Judge-backed, therefore advisory always. Missing/odd reference data degrades
to a skipped Score — a sample without reference simply contributes nothing.
"""
from __future__ import annotations

import string

from .. import judge
from ..core import ConfigError, SAMPLE_ROOTS, VALID_RUBRIC_WEIGHTS
from . import make_score, register
from .selector import resolve, sample_root, validate_selector


def _skipped(spec, why):
    s = make_score(spec, None, None, why, detail={"skipped": True})
    s.advisory = True
    return s


def _validate(spec, case):
    validate_selector(spec.get("from"), roots=SAMPLE_ROOTS, what=f"check '{spec['id']}' from")
    tpl = spec.get("template")
    if not isinstance(tpl, str) or not tpl.strip():
        raise ConfigError(f"check '{spec['id']}': needs a string 'template'")
    w = spec.get("weight", "expected")
    if w not in VALID_RUBRIC_WEIGHTS:
        raise ConfigError(f"check '{spec['id']}': weight {w!r} not in {VALID_RUBRIC_WEIGHTS}")


class _Permissive(dict):
    """format_map helper: unknown template keys render as a visible placeholder
    instead of raising (data problems degrade, never raise)."""

    def __missing__(self, key):
        return f"<missing:{key}>"


def derive_items(spec, sample) -> tuple[list | None, str | None]:
    """Build rubric items from the sample's reference list. Returns
    (items, error) — error set on data problems."""
    res = resolve(sample_root(sample), spec["from"])
    if not res.ok:
        return None, res.error
    entries = res.value
    if not isinstance(entries, list):
        return None, f"{spec['from']}: expected a list, got {type(entries).__name__}"
    if not entries:
        return None, f"{spec['from']}: empty list — nothing to derive"
    weight = spec.get("weight", "expected")
    tpl = spec["template"]
    items = []
    for i, entry in enumerate(entries, 1):
        ctx = _Permissive(entry) if isinstance(entry, dict) else _Permissive(value=entry)
        text = string.Formatter().vformat(tpl, (), ctx)
        item_id = (entry.get("id") if isinstance(entry, dict) and entry.get("id")
                   else f"{spec['id']}_{i}")
        items.append({"id": str(item_id), "item": text, "weight": weight})
    return items, None


@register("gt.derived_rubric", judge=True, validator=_validate)
def gt_derived_rubric(spec, sample, run, ctx):
    items, err = derive_items(spec, sample)
    if items is None:
        return _skipped(spec, f"derived rubric skipped: {err}")
    if not ctx.judges:
        s = _skipped(spec, "judge skipped (--no-judge or no active judge backends)")
        s.detail["derived_items"] = items  # still auditable offline
        return s
    result = judge.run_rubric(run.output, items, ctx.judges,
                              ctx.case.case_id, transport=ctx.transport)
    if not result:
        return _skipped(spec, "derived rubric skipped (empty report)")
    if result.get("truncated"):
        ctx.truncations.append({"check_id": spec["id"], "sample_id": sample.id,
                                "report_chars": result.get("report_chars"),
                                "cap": judge.REPORT_CAP})
    s = make_score(spec, result.get("score"), result.get("essential_ok"),
                   f"derived rubric score {result.get('score')} — "
                   f"{sum(1 for it in result['items'] if it['pass'])}/{len(result['items'])} "
                   f"items passed"
                   + ("" if result.get("essential_ok") else " (essential item FAILED)"),
                   detail=result)
    s.advisory = True
    return s
