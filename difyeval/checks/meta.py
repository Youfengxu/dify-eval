"""Built-in meta.* checks — run-cost and run-time budgets.

Typically used ungated (advisory): a slow or expensive run is a signal, not
necessarily a failure. A case may still set ``gate: true`` — they are
deterministic, so gating is allowed (unlike judge checks).

meta.cost_max is per-currency with NO FX conversion: caps are declared per
currency and only compared against spend recorded in that same currency.
"""
from __future__ import annotations

from ..core import ConfigError
from . import degrade, is_marginal, make_score, register


def _validate_elapsed(spec, case):
    if not isinstance(spec.get("max"), (int, float)):
        raise ConfigError(f"check '{spec['id']}': needs a numeric 'max' (seconds)")


@register("meta.elapsed_max", validator=_validate_elapsed)
def meta_elapsed_max(spec, sample, run, ctx):
    elapsed = (run.meta or {}).get("elapsed")
    if not isinstance(elapsed, (int, float)):
        return degrade(spec, "meta.elapsed not present in run")
    cap = spec["max"]
    passed = elapsed <= cap
    return make_score(spec, float(elapsed), passed,
                      f"elapsed {elapsed}s (max {cap}s)",
                      marginal=is_marginal(elapsed, cap))


def _validate_cost(spec, case):
    caps = spec.get("max")
    if isinstance(caps, (int, float)):
        return
    if isinstance(caps, dict) and caps and all(
            isinstance(v, (int, float)) for v in caps.values()):
        return
    raise ConfigError(f"check '{spec['id']}': 'max' must be a number "
                      f"(with optional currency:, default USD) or a "
                      f"{{currency: amount}} mapping")


@register("meta.cost_max", validator=_validate_cost)
def meta_cost_max(spec, sample, run, ctx):
    caps = spec["max"]
    if isinstance(caps, (int, float)):
        caps = {spec.get("currency", "USD"): caps}
    cost = (run.usage or {}).get("cost")
    if not isinstance(cost, dict) or not cost:
        return degrade(spec, "no cost data in run.usage")
    lines, passed, marginal, worst_ratio = [], True, False, 0.0
    for cur in sorted(caps):
        cap = float(caps[cur])
        spent = float(cost.get(cur, 0.0))
        ok = spent <= cap
        passed = passed and ok
        marginal = marginal or is_marginal(spent, cap)
        if cap > 0:
            worst_ratio = max(worst_ratio, spent / cap)
        lines.append(f"{cur} {spent:g} {'<=' if ok else '>'} {cap:g}")
    uncapped = sorted(set(cost) - set(caps))
    if uncapped:
        lines.append(f"uncapped currencies present: {', '.join(uncapped)}")
    nwu = (run.usage or {}).get("nodes_without_usage")
    if nwu:
        lines.append(f"cost may be undercounted: {nwu} node(s) without usage")
    return make_score(spec, round(worst_ratio, 4), passed, "; ".join(lines), marginal=marginal)
