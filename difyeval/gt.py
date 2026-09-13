"""Ground-truth provenance convention — pure helpers, no I/O.

Any reference list in a case may hold dict items carrying optional
ground-truth metadata::

    reference:
      accounts:
        items:
          - handle: "@seed_account"
            provenance: human_verified            # optional — see tiers below
            discoverable: true                    # optional — see GATEABILITY
            validated: {by: analyst, date: 2026-07-01}
            source_run: "results/case__p__eval.json#s1:r1"
            notes: "the seeding account"
      must_reject:
        - {value: "@false_positive", reason: "news outlet, not a buzzer",
           source_run: "results/case__p__eval.json#s1:r1"}

Provenance tiers (exact strings):

* ``human_verified`` — an analyst confirmed the item (authoring research, or
  a review-loop ``confirm`` — carries ``validated: {by, date}``).
* ``human_supplied`` — provided by a human with the case but not
  independently re-verified. Items with NO ``provenance`` key default to
  this tier: a human hand-wrote them into the case file. Pipeline output
  must never be written into a reference list untagged — ``difyeval review
  apply`` always stamps the tier that matches the analyst's decision.
* ``pipeline_observed_unvalidated`` — surfaced by a run, accreted via a
  review-loop ``unvalidated`` decision; awaiting human confirmation.

TRUST vs GATEABILITY — the two orthogonal axes packs must honor:

* ``provenance`` governs TRUST: may the item back a gating (FAIL-capable)
  assertion? Only :data:`GATEABLE_TIERS` items may gate.
  ``pipeline_observed_unvalidated`` items are advisory-only until a human
  promotes them (the trust rule: pipeline-observed never gates). An
  UNKNOWN tier is treated as advisory too — never trust what you cannot
  classify — and :func:`validate_items` warns about it.
* ``discoverable: false`` governs GATEABILITY for recall-style checks: the
  item is real but the system under test cannot find it via its discovery
  surfaces (private account, deleted post, out-of-band tip). Recall packs
  must exclude ``discoverable: false`` items from gating DENOMINATORS —
  failing to find the unfindable is not a failure. Such items may still be
  counted in advisory "recall_all"-style metrics.

difyeval itself only ships the convention and these helpers; check packs
implement the actual exclusions (e.g. gate on
``[it for it in gateable(items) if it.get("discoverable") is not False]``).

``must_reject`` convention: a reference list (default path ``must_reject``)
of hard-negatives ``{value, reason, source_run}`` — things a run must NOT
claim. ``difyeval review apply`` appends one for every ``negative``
decision; :func:`reject_list` reads them back.
"""
from __future__ import annotations

import json

HUMAN_VERIFIED = "human_verified"
HUMAN_SUPPLIED = "human_supplied"
PIPELINE_OBSERVED_UNVALIDATED = "pipeline_observed_unvalidated"

#: All known provenance tiers, strongest first.
TIERS = (HUMAN_VERIFIED, HUMAN_SUPPLIED, PIPELINE_OBSERVED_UNVALIDATED)

#: Tiers whose items may back gating (FAIL-capable) assertions.
GATEABLE_TIERS = {HUMAN_VERIFIED, HUMAN_SUPPLIED}

#: Tier assumed for items without a ``provenance`` key (hand-authored).
DEFAULT_TIER = HUMAN_SUPPLIED

#: Identity fields checked (in order) by :func:`item_identity`.
IDENTITY_FIELDS = ("id", "handle", "value")


# --------------------------------------------------------------------------
# reference access
# --------------------------------------------------------------------------

def iter_items(reference, dotted_path: str) -> list:
    """The list at *dotted_path* inside a reference mapping; ``[]`` when the
    path is missing, unwalkable, or does not hold a list. Never raises."""
    cur = reference
    for part in str(dotted_path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return []
        cur = cur[part]
    return cur if isinstance(cur, list) else []


def iter_reference_lists(reference, _prefix: str = ""):
    """Yield ``(dotted_path, list)`` for every list value reachable through
    dict traversal of a reference mapping (lists are yielded, not descended
    into). Deterministic: dict keys in sorted order."""
    if not isinstance(reference, dict):
        return
    for key in sorted(reference, key=str):
        path = f"{_prefix}.{key}" if _prefix else str(key)
        v = reference[key]
        if isinstance(v, list):
            yield path, v
        elif isinstance(v, dict):
            yield from iter_reference_lists(v, path)


# --------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------

def item_tier(item, default: str = DEFAULT_TIER):
    """The item's provenance tier string. Non-dict items and dicts without a
    ``provenance`` key report *default* (hand-authored → ``human_supplied``).
    Unknown tier strings are returned verbatim (callers partition them as
    advisory; :func:`validate_items` warns)."""
    if isinstance(item, dict):
        p = item.get("provenance")
        if isinstance(p, str) and p.strip():
            return p.strip()
    return default


def partition_by_tier(items) -> tuple[list, list]:
    """Split items into ``(gateable, advisory)`` by provenance TRUST tier.

    ``gateable`` — tier in :data:`GATEABLE_TIERS` (missing provenance counts
    as :data:`DEFAULT_TIER`). ``advisory`` — everything else, including
    unknown tiers (never trust what you cannot classify). NOTE: this is the
    trust axis only — recall packs must additionally exclude
    ``discoverable: false`` items from gating denominators."""
    gate, adv = [], []
    for it in items or []:
        (gate if item_tier(it) in GATEABLE_TIERS else adv).append(it)
    return gate, adv


def gateable(items) -> list:
    """Just the trust-gateable partition of :func:`partition_by_tier`."""
    return partition_by_tier(items)[0]


def validate_items(items) -> list[str]:
    """Provenance-convention warnings for one reference list (never raises):
    unknown tier strings, and ``validated`` blocks missing ``by``/``date``.
    Non-dict items are fine (no provenance semantics attached)."""
    warnings = []
    for i, it in enumerate(items or []):
        if not isinstance(it, dict):
            continue
        label = item_identity(it)[1] or f"item[{i}]"
        tier = it.get("provenance")
        if tier is not None and tier not in TIERS:
            warnings.append(f"{label}: unknown provenance tier {tier!r} "
                            f"(known: {', '.join(TIERS)}) — treated as advisory")
        val = it.get("validated")
        if val is not None:
            if not isinstance(val, dict) or not val.get("by") or not val.get("date"):
                warnings.append(f"{label}: validated: must carry both 'by' and 'date'")
        disc = it.get("discoverable")
        if disc is not None and not isinstance(disc, bool):
            warnings.append(f"{label}: discoverable must be a boolean, got {disc!r}")
    return warnings


# --------------------------------------------------------------------------
# must_reject
# --------------------------------------------------------------------------

def reject_list(reference, path: str = "must_reject") -> list[dict]:
    """The hard-negative list at *path*: dict entries pass through, scalar
    entries are coerced to ``{"value": <scalar>}``. ``[]`` when absent."""
    out = []
    for entry in iter_items(reference, path):
        if isinstance(entry, dict):
            out.append(entry)
        elif entry is not None:
            out.append({"value": entry})
    return out


# --------------------------------------------------------------------------
# identity — the duplicate-detection key
# --------------------------------------------------------------------------

def _norm(v) -> str:
    return str(v).strip().lstrip("@").lower()


def item_identity(item) -> tuple[str, str]:
    """Stable identity key for duplicate detection across base + sidecar +
    review candidates.

    Rule (the shipped identity-key convention): the first of
    :data:`IDENTITY_FIELDS` (``id`` → ``handle`` → ``value``) present with a
    non-empty scalar wins; its value is compared normalized (whitespace
    stripped, leading ``@`` stripped, lowercased) so ``@User`` == ``user``.
    Dicts with none of those fields fall back to whole-dict equality
    (sorted-key JSON). Non-dict items compare by their normalized string.

    Returns ``(kind, normalized_value)`` where kind is the matched field
    name, ``"_dict"``, or ``"_scalar"``."""
    if isinstance(item, dict):
        for f in IDENTITY_FIELDS:
            v = item.get(f)
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (str, int, float)):
                s = _norm(v)
                if s:
                    return (f, s)
        return ("_dict", json.dumps(item, sort_keys=True, ensure_ascii=False,
                                    default=str))
    return ("_scalar", _norm(item))


def identity_value(item) -> str:
    """The normalized value half of :func:`item_identity` — what
    ``must_reject`` bans are matched against."""
    return item_identity(item)[1]
