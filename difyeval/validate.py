"""Config-time validation — fail fast BEFORE any run is acquired or scored.

Everything here raises ConfigError (CLI exit code 2). Contrast with data
problems at score time, which degrade into Scores instead.
"""
from __future__ import annotations

from . import checks as registry
from .core import Case, ConfigError
from .runners import get_runner


def prepare_case(case: Case) -> None:
    """Load packs, then validate check types, judge/gate rules, per-check
    params, and the runner type. Call once, before any run."""
    registry.load_packs(case)
    validate_case(case)


def validate_case(case: Case) -> None:
    get_runner(case.runner)  # unknown runner type -> ConfigError
    for spec in case.checks:
        cd = registry.get(spec["type"])
        if cd is None:
            raise ConfigError(
                f"unknown check type {spec['type']!r} (check '{spec['id']}') — "
                f"registered types: {registry.registered_types()}")
        if cd.judge and spec.get("gate"):
            raise ConfigError(
                f"check '{spec['id']}' ({spec['type']}): the judge NEVER gates — "
                f"judge-derived checks are advisory by construction; remove 'gate: true'")
        if cd.validator is not None:
            cd.validator(spec, case)
