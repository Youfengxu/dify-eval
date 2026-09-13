"""Selector semantics — dotted paths into the Run dict (or a Sample view).

``nodes.merge_s4.outputs.resolved_json`` walks dicts key by key (and lists by
integer index). When traversal reaches a *string* leaf but more path segments
remain, the string is auto-JSON-decoded and traversal continues — Dify node
outputs routinely carry JSON-in-a-string fields.

A missing path never raises: resolution returns ``Resolution(ok=False, error=
"selector not found: ...")`` and the calling check degrades (passed=False for
gated checks).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..core import ConfigError, RUN_ROOTS


@dataclass
class Resolution:
    ok: bool
    value: Any
    error: str | None = None


def split_selector(selector: str) -> list[str]:
    return [seg for seg in str(selector).split(".")]


def validate_selector(selector: Any, roots: tuple = RUN_ROOTS, what: str = "selector") -> None:
    """Config-time syntax validation. Raises ConfigError (fail fast)."""
    if not isinstance(selector, str) or not selector.strip():
        raise ConfigError(f"{what} must be a non-empty dotted path string, got {selector!r}")
    parts = split_selector(selector)
    if any(not p for p in parts):
        raise ConfigError(f"{what} {selector!r} has an empty path segment")
    if parts[0] not in roots:
        raise ConfigError(
            f"{what} {selector!r} must start with one of {list(roots)} (got {parts[0]!r})")


def resolve(root: Any, selector: str) -> Resolution:
    """Walk *root* along the dotted *selector*. Degrades, never raises."""
    parts = split_selector(selector)
    cur = root
    for i, part in enumerate(parts):
        at = ".".join(parts[:i]) or "(root)"
        if isinstance(cur, str):
            # further traversal requested into a string leaf -> auto-JSON-decode
            try:
                cur = json.loads(cur)
            except (json.JSONDecodeError, ValueError):
                return Resolution(False, None,
                                  f"selector not found: {selector} "
                                  f"(string at '{at}' is not JSON-decodable)")
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
                continue
            return Resolution(False, None,
                              f"selector not found: {selector} (missing key '{part}' at '{at}')")
        if isinstance(cur, list):
            try:
                idx = int(part)
                cur = cur[idx]
                continue
            except (ValueError, IndexError):
                return Resolution(False, None,
                                  f"selector not found: {selector} "
                                  f"(list at '{at}' has no index '{part}')")
        return Resolution(False, None,
                          f"selector not found: {selector} "
                          f"(cannot traverse {type(cur).__name__} at '{at}')")
    return Resolution(True, cur, None)


def maybe_json(value: Any) -> tuple[bool, Any, str | None]:
    """JSON-decode a string leaf when a check needs to count/inspect it.

    Non-strings pass through unchanged; an undecodable string is returned
    as-is with ok=True (the check decides whether raw-string is acceptable).
    """
    if isinstance(value, str):
        try:
            return True, json.loads(value), None
        except (json.JSONDecodeError, ValueError) as e:
            return False, value, f"not JSON-decodable: {e}"
    return True, value, None


def sample_root(sample) -> dict:
    """The view of a Sample that reference-side selectors resolve against
    (values_from / equals_from / from)."""
    return {
        "reference": sample.reference or {},
        "inputs": sample.inputs or {},
        "metadata": sample.metadata or {},
    }
