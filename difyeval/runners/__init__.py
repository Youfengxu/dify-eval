"""Runner registry — every runner normalizes to the uniform Run contract.

A runner is ``fn(spec, sample, repeat, case) -> Run`` registered under the
``runner.type`` key from case.yml. Acquisition failures degrade to
``Run(error=..., meta={"status": "runner_error"})`` — gated checks then fail
visibly; the experiment never dies mid-flight on a data problem.
"""
from __future__ import annotations

from typing import Callable

from ..core import ConfigError

_RUNNERS: dict[str, Callable] = {}


def register_runner(name: str):
    def deco(fn):
        _RUNNERS[name] = fn
        return fn

    return deco


def get_runner(runner_spec: dict) -> Callable:
    rtype = runner_spec.get("type")
    fn = _RUNNERS.get(rtype)
    if fn is None:
        raise ConfigError(f"unknown runner type {rtype!r} — registered: {sorted(_RUNNERS)}")
    return fn


def registered_runners() -> list[str]:
    return sorted(_RUNNERS)


# import built-in runners so they self-register
from . import file as _file          # noqa: E402,F401
from . import difyctl as _difyctl    # noqa: E402,F401
from . import command as _command    # noqa: E402,F401
from . import dify_api as _dify_api  # noqa: E402,F401
