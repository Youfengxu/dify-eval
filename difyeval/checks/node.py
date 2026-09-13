"""Built-in node.* checks — assertions on journaled node outputs."""
from __future__ import annotations

import json
import re

from ..core import ConfigError, SAMPLE_ROOTS
from . import degrade, is_marginal, make_score, register
from .selector import maybe_json, resolve, sample_root, validate_selector


def _validate_node_exists(spec, case):
    if not isinstance(spec.get("node"), str) or not spec["node"]:
        raise ConfigError(f"check '{spec['id']}': needs a string 'node' (node id)")


@register("node.exists", validator=_validate_node_exists)
def node_exists(spec, sample, run, ctx):
    node_id = spec["node"]
    present = node_id in (run.nodes or {})
    if present:
        return make_score(spec, True, True, f"node '{node_id}' present")
    avail = sorted(run.nodes or {})
    ev = (f"node '{node_id}' absent (journaled: {', '.join(avail[:10])}"
          + (" ..." if len(avail) > 10 else "") + ")") if avail \
        else f"node '{node_id}' absent (no nodes journaled)"
    return make_score(spec, False, False, ev)


def _validate_json_count(spec, case):
    validate_selector(spec.get("selector"), what=f"check '{spec['id']}' selector")
    if "min" not in spec and "max" not in spec:
        raise ConfigError(f"check '{spec['id']}': needs min and/or max")
    for k in ("min", "max"):
        if k in spec and not isinstance(spec[k], (int, float)):
            raise ConfigError(f"check '{spec['id']}': {k} must be numeric")


@register("node.json_count", validator=_validate_json_count)
def node_json_count(spec, sample, run, ctx):
    res = resolve(run.to_dict(), spec["selector"])
    if not res.ok:
        return degrade(spec, res.error)
    ok, decoded, err = maybe_json(res.value)
    if not ok:
        return degrade(spec, f"{spec['selector']}: {err}")
    if not isinstance(decoded, (list, dict, str)):
        return degrade(spec, f"{spec['selector']}: {type(decoded).__name__} is not countable")
    count = len(decoded)
    lo, hi = spec.get("min"), spec.get("max")
    passed = (lo is None or count >= lo) and (hi is None or count <= hi)
    marginal = any(t is not None and is_marginal(count, t) for t in (lo, hi))
    bounds = " ".join(s for s in ((f"min={lo}" if lo is not None else ""),
                                  (f"max={hi}" if hi is not None else "")) if s)
    return make_score(spec, float(count), passed, f"count={count} ({bounds})", marginal=marginal)


def _validate_json_field(spec, case):
    validate_selector(spec.get("selector"), what=f"check '{spec['id']}' selector")
    if "equals_from" in spec:
        validate_selector(spec["equals_from"], roots=SAMPLE_ROOTS,
                          what=f"check '{spec['id']}' equals_from")
    if "field" in spec and not isinstance(spec["field"], str):
        raise ConfigError(f"check '{spec['id']}': field must be a string")


@register("node.json_field", validator=_validate_json_field)
def node_json_field(spec, sample, run, ctx):
    res = resolve(run.to_dict(), spec["selector"])
    if not res.ok:
        return degrade(spec, res.error)
    target = res.value
    label = spec["selector"]
    if "field" in spec:
        ok, decoded, err = maybe_json(target)
        if not ok or not isinstance(decoded, dict):
            return degrade(spec, f"{label}: not an object ({err or type(decoded).__name__})")
        field = spec["field"]
        if field not in decoded:
            ev = f"{label}: field '{field}' absent"
            return make_score(spec, False, False, ev)
        target = decoded[field]
        label = f"{label}.{field}"
    if "equals" in spec:
        expected = spec["equals"]
    elif "equals_from" in spec:
        eres = resolve(sample_root(sample), spec["equals_from"])
        if not eres.ok:
            return degrade(spec, eres.error)
        expected = eres.value
    else:
        # presence-only assertion: the selector (and field, if given) resolved
        return make_score(spec, True, True, f"{label} present "
                          f"(value: {_short(target)})")
    passed = target == expected
    ev = (f"{label} == {_short(expected)}" if passed
          else f"{label}: expected {_short(expected)}, got {_short(target)}")
    return make_score(spec, passed, passed, ev)


def _short(v, cap=60):
    s = json.dumps(v, ensure_ascii=False, sort_keys=True) if not isinstance(v, str) else repr(v)
    return s if len(s) <= cap else s[: cap - 3] + "..."


def _validate_node_regex(spec, case):
    validate_selector(spec.get("selector"), what=f"check '{spec['id']}' selector")
    pat = spec.get("pattern")
    if not isinstance(pat, str) or not pat:
        raise ConfigError(f"check '{spec['id']}': needs a string 'pattern'")
    try:
        re.compile(pat)
    except re.error as e:
        raise ConfigError(f"check '{spec['id']}': invalid regex {pat!r}: {e}")


@register("node.regex", validator=_validate_node_regex)
def node_regex(spec, sample, run, ctx):
    res = resolve(run.to_dict(), spec["selector"])
    if not res.ok:
        return degrade(spec, res.error)
    value = res.value
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    m = re.compile(spec["pattern"]).search(text)
    if m:
        return make_score(spec, True, True, f"matched: {m.group(0)[:80]!r}")
    return make_score(spec, False, False, f"no match for /{spec['pattern']}/ in {spec['selector']}")
