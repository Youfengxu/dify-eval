"""Built-in output.* checks — assertions on Run.output (the report text)."""
from __future__ import annotations

import json
import re

from ..core import ConfigError, SAMPLE_ROOTS
from . import degrade, make_score, register
from .selector import maybe_json, resolve, sample_root, validate_selector


# --------------------------------------------------------------------------
# output.nonempty
# --------------------------------------------------------------------------

@register("output.nonempty")
def output_nonempty(spec, sample, run, ctx):
    text = run.output or ""
    ok = bool(text.strip())
    ev = f"output has {len(text)} chars" if ok else "output is empty"
    if not ok and run.error:
        ev += f" (run error: {run.error})"
    return make_score(spec, ok, ok, ev)


# --------------------------------------------------------------------------
# output.contains / output.not_contains
# --------------------------------------------------------------------------

def _wanted_values(spec, sample):
    """Resolve the value list: literal `values`/`value`, or `values_from:
    reference.<path>`. Returns (values, error)."""
    if "values" in spec:
        vals = spec["values"]
        return ([str(v) for v in vals] if isinstance(vals, list) else [str(vals)]), None
    if "value" in spec:
        return [str(spec["value"])], None
    if "values_from" in spec:
        res = resolve(sample_root(sample), spec["values_from"])
        if not res.ok:
            return None, res.error
        v = res.value
        if isinstance(v, list):
            return [str(x) for x in v], None
        return [str(v)], None
    return None, "no values/value/values_from configured"


def _validate_contains(spec, case):
    if not any(k in spec for k in ("values", "value", "values_from")):
        raise ConfigError(f"check '{spec['id']}': needs one of values / value / values_from")
    if "values_from" in spec:
        validate_selector(spec["values_from"], roots=SAMPLE_ROOTS,
                          what=f"check '{spec['id']}' values_from")


@register("output.contains", validator=_validate_contains)
def output_contains(spec, sample, run, ctx):
    vals, err = _wanted_values(spec, sample)
    if vals is None:
        return degrade(spec, err)
    if not vals:
        return degrade(spec, "empty value list (nothing to assert)")
    text = run.output or ""
    missing = [v for v in vals if v not in text]
    passed = not missing
    value = round((len(vals) - len(missing)) / len(vals), 4)
    ev = (f"all {len(vals)} value(s) present" if passed
          else "missing: " + ", ".join(repr(m) for m in missing[:5])
          + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))
    return make_score(spec, value, passed, ev)


@register("output.not_contains", validator=_validate_contains)
def output_not_contains(spec, sample, run, ctx):
    vals, err = _wanted_values(spec, sample)
    if vals is None:
        return degrade(spec, err)
    text = run.output or ""
    found = [v for v in vals if v in text]
    passed = not found
    ev = ("none of the forbidden value(s) present" if passed
          else "found forbidden: " + ", ".join(repr(f) for f in found[:5])
          + (f" (+{len(found) - 5} more)" if len(found) > 5 else ""))
    return make_score(spec, not found, passed, ev)


# --------------------------------------------------------------------------
# output.regex
# --------------------------------------------------------------------------

def _validate_regex(spec, case):
    pat = spec.get("pattern")
    if not isinstance(pat, str) or not pat:
        raise ConfigError(f"check '{spec['id']}': needs a string 'pattern'")
    try:
        re.compile(pat)
    except re.error as e:
        raise ConfigError(f"check '{spec['id']}': invalid regex {pat!r}: {e}")


@register("output.regex", validator=_validate_regex)
def output_regex(spec, sample, run, ctx):
    try:
        pat = re.compile(spec["pattern"])
    except re.error as e:  # validated pre-run; belt and braces
        return degrade(spec, f"invalid regex: {e}")
    m = pat.search(run.output or "")
    if m:
        snippet = m.group(0)[:80]
        return make_score(spec, True, True, f"matched: {snippet!r}")
    return make_score(spec, False, False, f"no match for /{spec['pattern']}/")


# --------------------------------------------------------------------------
# output.json_schema — stdlib-only structural validation (dict-shape walker)
# --------------------------------------------------------------------------

_TYPE_MAP = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}

_MAX_SCHEMA_ERRORS = 10


def _type_ok(tname, value):
    types = _TYPE_MAP.get(tname)
    if types is None:
        return True  # unknown type name: no assertion (validated pre-run)
    if tname in ("number", "integer") and isinstance(value, bool):
        return False  # bool is not a number for our purposes
    return isinstance(value, types)


def _walk_schema(schema, value, path, errors):
    if len(errors) >= _MAX_SCHEMA_ERRORS:
        return
    if not isinstance(schema, dict):
        return
    tname = schema.get("type")
    if tname and not _type_ok(tname, value):
        errors.append(f"{path}: expected {tname}, got {type(value).__name__}")
        return
    if isinstance(value, dict):
        for req in schema.get("required", []) or []:
            if req not in value:
                errors.append(f"{path}: missing required key '{req}'")
        for key, sub in sorted((schema.get("properties") or {}).items()):
            if key in value:
                _walk_schema(sub, value[key], f"{path}.{key}", errors)
    if isinstance(value, list):
        items = schema.get("items")
        if items:
            for i, el in enumerate(value):
                if len(errors) >= _MAX_SCHEMA_ERRORS:
                    return
                _walk_schema(items, el, f"{path}[{i}]", errors)


def _validate_json_schema(spec, case):
    if not isinstance(spec.get("schema"), dict):
        raise ConfigError(f"check '{spec['id']}': needs a mapping 'schema'")
    bad = _collect_bad_types(spec["schema"])
    if bad:
        raise ConfigError(f"check '{spec['id']}': unknown schema type(s): {sorted(bad)} "
                          f"(known: {sorted(_TYPE_MAP)})")
    if "selector" in spec:
        validate_selector(spec["selector"], what=f"check '{spec['id']}' selector")


def _collect_bad_types(schema, bad=None):
    bad = bad if bad is not None else set()
    if isinstance(schema, dict):
        t = schema.get("type")
        if isinstance(t, str) and t not in _TYPE_MAP:
            bad.add(t)
        for sub in (schema.get("properties") or {}).values():
            _collect_bad_types(sub, bad)
        if schema.get("items"):
            _collect_bad_types(schema["items"], bad)
    return bad


@register("output.json_schema", validator=_validate_json_schema)
def output_json_schema(spec, sample, run, ctx):
    if "selector" in spec:
        res = resolve(run.to_dict(), spec["selector"])
        if not res.ok:
            return degrade(spec, res.error)
        target = res.value
    else:
        target = run.output or ""
    ok, decoded, err = maybe_json(target)
    if isinstance(decoded, str):
        return degrade(spec, f"target is not JSON: {err or 'plain string'}")
    errors: list[str] = []
    _walk_schema(spec["schema"], decoded, "$", errors)
    passed = not errors
    ev = ("schema satisfied" if passed
          else "; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""))
    return make_score(spec, passed, passed, ev)
