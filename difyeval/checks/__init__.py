"""Check registry + built-in check library.

A check is a function ``check(spec, sample, run, ctx) -> Score`` registered
under a dotted name::

    from difyeval.checks import register, make_score

    @register("csd.accounts_recall")
    def accounts_recall(spec, sample, run, ctx):
        ...

External repos ship packs (importable modules whose import registers their
checks); a case opts in via ``packs: [my_pkg.checks]``.

Registration may carry:

* ``judge=True``  — the check consults the LLM judge panel. Judge checks are
  advisory BY CONSTRUCTION: they can never gate, and ``gate: true`` on one is
  a loud config error (validated pre-run).
* ``validator=fn`` — optional config-time spec validation, called by
  ``difyeval.validate`` before any run (fail fast on bad params).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from ..core import ConfigError, Score


@dataclass
class CheckDef:
    name: str
    fn: Callable
    judge: bool = False
    validator: Callable | None = None


_REGISTRY: dict[str, CheckDef] = {}


def register(name: str, judge: bool = False, validator: Callable | None = None):
    """Decorator: register a check function under a dotted name."""

    def deco(fn):
        _REGISTRY[name] = CheckDef(name=name, fn=fn, judge=judge, validator=validator)
        return fn

    return deco


def get(name: str) -> CheckDef | None:
    return _REGISTRY.get(name)


def registered_types() -> list[str]:
    return sorted(_REGISTRY)


def register_pack(module_or_name):
    """Import a check-pack module (its import-time @register calls do the work)."""
    if isinstance(module_or_name, str):
        try:
            return importlib.import_module(module_or_name)
        except ImportError as e:
            raise ConfigError(f"cannot import check pack '{module_or_name}': {e}") from e
    return module_or_name


def load_packs(case) -> None:
    for pack in case.packs:
        register_pack(pack)


# --------------------------------------------------------------------------
# shared score helpers
# --------------------------------------------------------------------------

def make_score(spec, value, passed, evidence, marginal=False, detail=None) -> Score:
    """Build a Score honoring the gate/advisory convention: a check is
    advisory unless the case set ``gate: true`` (judge checks force advisory
    at the engine layer regardless)."""
    return Score(check_id=spec["id"], value=value, passed=passed, evidence=evidence,
                 marginal=marginal, advisory=not bool(spec.get("gate")), detail=detail or {})


def degrade(spec, evidence, detail=None) -> Score:
    """Data-problem Score: gated checks fail (court-grade: an unprovable gate
    is a failed gate); advisory checks record None."""
    passed = False if spec.get("gate") else None
    return make_score(spec, None, passed, evidence, detail=detail)


def is_marginal(value, threshold) -> bool:
    """Within 10% relative of a numeric threshold -> flagged (marginal)."""
    try:
        v, t = float(value), float(threshold)
    except (TypeError, ValueError):
        return False
    if t == 0:
        return v == 0
    return abs(v - t) <= 0.10 * abs(t)


# import built-in check modules so they self-register
from . import output as _output    # noqa: E402,F401
from . import node as _node        # noqa: E402,F401
from . import meta as _meta        # noqa: E402,F401
from . import judge_checks as _judge_checks  # noqa: E402,F401
from . import gt as _gt            # noqa: E402,F401
