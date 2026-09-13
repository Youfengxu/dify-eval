"""Profile files — named system-under-test configurations.

A profile file is a small YAML mapping describing WHAT was evaluated (model
choices, workflow config, DSL version, runner endpoint) so an eval.json is
self-describing and two Experiments can be diffed profile-vs-profile with
``difyeval diff``.

Contract (config problems raise ConfigError — fail fast, exit 2)::

    profile: opus-4.6-baseline     # required — becomes the Experiment's profile label
    description: ...               # optional free text
    models: {orchestrator: ...}    # optional free-form mapping (recorded verbatim)
    config: {temperature: 0}       # optional free-form mapping (recorded verbatim)
    dsl: ../workflow.yml           # optional — auto --dsl, relative to this file
    runner: {base: https://...}    # optional — shallow-merged over case.runner

Unknown extra keys are allowed and preserved: the loaded mapping is embedded
VERBATIM into eval.json's ``manifest.profile_config`` so the audit record
carries the full profile, not a pointer to a file that may later change.
"""
from __future__ import annotations

import os

import yaml

from .core import Case, ConfigError


def load_profile(path: str) -> dict:
    """Load + validate a profile file. Returns the raw mapping (verbatim)."""
    if not os.path.isfile(path):
        raise ConfigError(f"profile file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in profile file {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"profile file {path} must be a YAML mapping")

    name = raw.get("profile")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"profile file {path}: 'profile' is required (non-empty string)")
    desc = raw.get("description")
    if desc is not None and not isinstance(desc, str):
        raise ConfigError(f"profile file {path}: 'description' must be a string")
    for key in ("models", "config"):
        if raw.get(key) is not None and not isinstance(raw[key], dict):
            raise ConfigError(f"profile file {path}: '{key}' must be a mapping")
    dsl = raw.get("dsl")
    if dsl is not None and (not isinstance(dsl, str) or not dsl.strip()):
        raise ConfigError(f"profile file {path}: 'dsl' must be a non-empty path string")
    runner = raw.get("runner")
    if runner is not None and not isinstance(runner, dict):
        raise ConfigError(f"profile file {path}: 'runner' must be a mapping of overrides")
    return raw


def resolve_profile_dsl(profile: dict, profile_path: str) -> str | None:
    """The profile's `dsl` path, resolved relative to the profile file.
    A configured-but-missing DSL file is a config problem — fail fast."""
    dsl = profile.get("dsl")
    if not dsl:
        return None
    if not os.path.isabs(dsl):
        dsl = os.path.join(os.path.dirname(os.path.abspath(profile_path)), dsl)
    if not os.path.isfile(dsl):
        raise ConfigError(f"profile dsl file not found: {dsl}")
    return dsl


def apply_runner_overrides(case: Case, profile: dict) -> None:
    """Shallow-merge the profile's `runner:` overrides (e.g. base, app,
    api_key_env) over case.runner. The merged runner is validated afterwards
    by prepare_case, so an unknown overridden type still fails fast."""
    overrides = profile.get("runner")
    if overrides:
        case.runner.update(overrides)
