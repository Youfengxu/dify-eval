"""Core data model: Sample, Run, Score, Case, Experiment + case.yml loading.

Two error regimes, deliberately distinct:

* Config problems (bad YAML, missing case_id, unknown check type, bad
  selector syntax) raise :class:`ConfigError` and fail fast BEFORE any run
  is acquired or scored (CLI exit code 2).
* Data problems (missing node, undecodable JSON, absent fixture) never
  raise — they degrade into a failing/None :class:`Score` or an error
  :class:`Run` so the experiment always completes and the failure is
  visible and attributable in the scorecard.
"""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml

from . import gt

VALID_MODES = ("reference", "live")
VALID_VERDICTS = ("PASS", "WARN", "FAIL")
VALID_RUBRIC_WEIGHTS = ("essential", "expected", "bonus")

#: Legal selector roots into a Run dict (see difyeval.checks.selector).
RUN_ROOTS = ("output", "nodes", "usage", "meta", "error")
#: Legal selector roots into a Sample (values_from / equals_from / from).
SAMPLE_ROOTS = ("reference", "inputs", "metadata")


class ConfigError(Exception):
    """Fatal configuration problem — fail fast, exit code 2."""


@dataclass
class Sample:
    """One dataset entry."""

    id: str
    inputs: dict = field(default_factory=dict)
    reference: dict | None = None
    metadata: dict = field(default_factory=dict)
    repeats: int = 1


@dataclass
class Run:
    """THE uniform contract every runner emits.

    ``nodes``  — {node_id: {outputs: dict, status?, elapsed?}}
    ``usage``  — {tokens?: int, cost?: {currency: amount}, nodes_without_usage?: int}
    ``meta``   — {status, elapsed, ...} (runner-specific extras allowed)
    ``error``  — set (with meta.status == "runner_error") when acquisition failed;
                 checks then run against the degraded Run and gated checks fail.
    """

    output: str = ""
    nodes: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "nodes": self.nodes,
            "usage": self.usage,
            "meta": self.meta,
            "error": self.error,
        }


@dataclass
class Score:
    """Result of one check against one Run.

    ``passed`` is None for advisory checks with no pass/fail notion (or when
    the check was skipped, e.g. judge checks under --no-judge).
    ``marginal`` — numeric-threshold checks within 10% relative of their
    threshold; rendered as ``(marginal)`` in the checks table.
    ``advisory`` — never contributes to FAIL. All judge-derived Scores are
    advisory by construction.
    ``detail`` — check-specific structured payload (rubric items + votes,
    judge per-dim medians, per-judge raw outputs) retained for the
    Experiment audit record.
    """

    check_id: str
    value: float | bool | None
    passed: bool | None
    evidence: str
    marginal: bool = False
    advisory: bool = False
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "value": self.value,
            "passed": self.passed,
            "evidence": self.evidence,
            "marginal": self.marginal,
            "advisory": self.advisory,
            "detail": self.detail,
        }


@dataclass
class Case:
    """A parsed + structurally-validated case.yml.

    ``raw`` echoes exactly what was authored in the base file; the optional
    accretions sidecar (see :func:`load_case`) merges into ``samples[..]
    .reference`` only. ``sidecar_path`` / ``sidecar_merged`` /
    ``load_warnings`` record what the sidecar contributed."""

    case_id: str
    mode: str
    packs: list
    runner: dict
    samples: list
    checks: list
    expect_verdict: str | None
    judge_notes: str
    path: str
    raw: dict
    sidecar_path: str | None = None
    #: [{sample, path, added, skipped}] — per merged reference list.
    sidecar_merged: list = field(default_factory=list)
    load_warnings: list = field(default_factory=list)

    @property
    def dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.path))

    def sha256(self) -> str | None:
        try:
            with open(self.path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return None


@dataclass
class SampleResult:
    """One sample x one repeat: the acquired Run and its Scores."""

    sample_id: str
    repeat: int  # 1-based
    run: Run
    scores: list = field(default_factory=list)


@dataclass
class Experiment:
    """One scored (dataset x profile) execution — the full audit record."""

    case: Case
    profile: str
    results: list = field(default_factory=list)  # list[SampleResult]
    verdict: str = "PASS"
    warn_reason: str | None = None
    scope: str = ""
    metrics: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)


@dataclass
class CheckContext:
    """Passed to every check: judge access + case-level config.

    ``judges``      — active judge backends ([] under --no-judge; judge checks
                      then emit a skipped advisory Score, no network touched).
    ``transport``   — injectable judge HTTP transport (tests stub this;
                      None means the real lazily-imported requests call).
    ``truncations`` — report-truncation events collected for the manifest and
                      the scorecard's judge block.
    """

    case: Case
    judges: list = field(default_factory=list)
    transport: Callable | None = None
    truncations: list = field(default_factory=list)


# --------------------------------------------------------------------------
# case.yml loading (structural validation only; check/runner-type validation
# lives in difyeval.validate, which needs the registries)
# --------------------------------------------------------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def load_case(path: str) -> Case:
    """Load and structurally validate a case.yml. Raises ConfigError."""
    _require(os.path.isfile(path), f"case file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    _require(isinstance(raw, dict), f"case file {path} must be a YAML mapping")

    case_id = raw.get("case_id")
    _require(isinstance(case_id, str) and case_id.strip(), "case_id is required (non-empty string)")

    mode = raw.get("mode", "reference")
    _require(mode in VALID_MODES, f"mode must be one of {VALID_MODES}, got {mode!r}")

    packs = raw.get("packs") or []
    _require(isinstance(packs, list) and all(isinstance(p, str) for p in packs),
             "packs must be a list of importable module names")

    runner = raw.get("runner")
    _require(isinstance(runner, dict) and isinstance(runner.get("type"), str),
             "runner must be a mapping with a string 'type'")

    dataset = raw.get("dataset")
    _require(isinstance(dataset, list) and dataset, "dataset must be a non-empty list")
    samples, seen_ids = [], set()
    for i, entry in enumerate(dataset):
        _require(isinstance(entry, dict), f"dataset[{i}] must be a mapping")
        sid = entry.get("id")
        _require(isinstance(sid, str) and sid.strip(), f"dataset[{i}] needs a non-empty string id")
        _require(sid not in seen_ids, f"duplicate sample id: {sid}")
        seen_ids.add(sid)
        repeats = entry.get("repeats", 1)
        _require(isinstance(repeats, int) and repeats >= 1,
                 f"dataset[{sid}].repeats must be an integer >= 1")
        inputs = entry.get("inputs") or {}
        _require(isinstance(inputs, dict), f"dataset[{sid}].inputs must be a mapping")
        reference = entry.get("reference")
        _require(reference is None or isinstance(reference, dict),
                 f"dataset[{sid}].reference must be a mapping or omitted")
        metadata = entry.get("metadata") or {}
        _require(isinstance(metadata, dict), f"dataset[{sid}].metadata must be a mapping")
        samples.append(Sample(id=sid, inputs=inputs, reference=reference,
                              metadata=metadata, repeats=repeats))

    checks = raw.get("checks")
    _require(isinstance(checks, list) and checks, "checks must be a non-empty list")
    seen_check_ids = set()
    for i, spec in enumerate(checks):
        _require(isinstance(spec, dict), f"checks[{i}] must be a mapping")
        cid = spec.get("id")
        _require(isinstance(cid, str) and cid.strip(), f"checks[{i}] needs a non-empty string id")
        _require(cid not in seen_check_ids, f"duplicate check id: {cid}")
        seen_check_ids.add(cid)
        _require(isinstance(spec.get("type"), str), f"check '{cid}' needs a string type")
        _require(isinstance(spec.get("gate", False), bool), f"check '{cid}': gate must be a boolean")

    expect = raw.get("expect_verdict")
    _require(expect is None or expect in VALID_VERDICTS,
             f"expect_verdict must be one of {VALID_VERDICTS}, got {expect!r}")

    judge_notes = raw.get("judge_notes") or ""
    _require(isinstance(judge_notes, str), "judge_notes must be a string")

    case = Case(case_id=case_id, mode=mode, packs=list(packs), runner=dict(runner),
                samples=samples, checks=list(checks), expect_verdict=expect,
                judge_notes=judge_notes, path=os.path.abspath(path), raw=raw)
    _merge_accretions(case)
    return case


# --------------------------------------------------------------------------
# accretions sidecar — append-only, run-accreted ground truth
#
# `<case minus .yml>.accretions.yml` (written by `difyeval review apply`,
# never by hand) merges into sample references at load time. The base case
# file is NEVER touched: comments survive, and the hand-authored /
# run-accreted boundary stays auditable. Sidecar shape:
#
#     samples:
#       _all:                      # applies to every sample, OR a sample id
#         reference:
#           accounts.items:        # dotted path to a LIST inside reference
#             - {handle: "@x", provenance: human_verified, ...}
#           must_reject:
#             - {value: "@y", reason: "...", source_run: "..."}
#
# Merge order is deterministic: base items first, then for each sample the
# `_all` block, then its own block — sidecar lists in file order. An item
# whose identity key (gt.item_identity) already exists in base+sidecar is
# skipped with a load warning, never an error. Structural problems in the
# sidecar (non-mapping, non-list path value, path colliding with a
# non-list) are config problems and raise ConfigError.
# --------------------------------------------------------------------------

def accretions_path(case_path: str) -> str:
    base, _ = os.path.splitext(case_path)
    return base + ".accretions.yml"


def load_accretions(case_path: str) -> dict:
    """The raw sidecar mapping for a case path ({} when absent).
    ConfigError on unreadable / unparseable / structurally alien files."""
    p = accretions_path(case_path)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"invalid accretions sidecar {p}: {e}") from e
    if raw is None:
        return {}
    _require(isinstance(raw, dict), f"accretions sidecar {p} must be a YAML mapping")
    _require(isinstance(raw.get("samples", {}), dict),
             f"accretions sidecar {p}: 'samples' must be a mapping")
    return raw


def _merge_accretions(case: Case) -> None:
    """Merge the case's sidecar (if any) into sample references, in place."""
    p = accretions_path(case.path)
    if not os.path.isfile(p):
        return
    case.sidecar_path = p
    blocks = load_accretions(case.path).get("samples") or {}
    sample_ids = {s.id for s in case.samples}
    for sid, block in blocks.items():
        if sid != "_all" and sid not in sample_ids:
            case.load_warnings.append(
                f"accretions sidecar: unknown sample id {sid!r} — block skipped")
            continue
        _require(isinstance(block, dict) and isinstance(block.get("reference", {}), dict),
                 f"accretions sidecar {p}: samples.{sid} must be a mapping with a "
                 f"'reference' mapping")
        for path, items in (block.get("reference") or {}).items():
            _require(isinstance(items, list),
                     f"accretions sidecar {p}: samples.{sid}.reference.{path} "
                     f"must be a list of items to append")
    merged: dict[tuple, dict] = {}
    for sample in case.samples:
        if any(isinstance(blocks.get(sid), dict) for sid in ("_all", sample.id)):
            # detach from case.raw before mutating — raw must keep echoing
            # exactly what was authored in the base file
            sample.reference = copy.deepcopy(sample.reference) \
                if sample.reference is not None else {}
        for sid in ("_all", sample.id):
            block = blocks.get(sid)
            if not isinstance(block, dict):
                continue
            for path, items in (block.get("reference") or {}).items():
                target = _reference_list(case, sample, str(path))
                seen = {gt.item_identity(x) for x in target}
                log = merged.setdefault((sample.id, str(path)),
                                        {"sample": sample.id, "path": str(path),
                                         "added": 0, "skipped": 0})
                for item in items:
                    key = gt.item_identity(item)
                    if key in seen:
                        log["skipped"] += 1
                        case.load_warnings.append(
                            f"accretions sidecar: duplicate item {key[1]!r} at "
                            f"{sample.id} · {path} — skipped")
                        continue
                    seen.add(key)
                    target.append(copy.deepcopy(item))
                    log["added"] += 1
    case.sidecar_merged = [merged[k] for k in sorted(merged)]


def _reference_list(case: Case, sample: Sample, dotted_path: str) -> list:
    """The (created-if-missing) list at *dotted_path* in the sample's
    reference. Intermediate mappings are created; a leaf that exists but is
    not a list is a config problem (ConfigError)."""
    if sample.reference is None:
        sample.reference = {}
    cur = sample.reference
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = cur[part] = {}
        _require(isinstance(nxt, dict),
                 f"accretions sidecar: reference path {dotted_path!r} in sample "
                 f"{sample.id!r} crosses a non-mapping at {part!r}")
        cur = nxt
    leaf = cur.get(parts[-1])
    if leaf is None:
        leaf = cur[parts[-1]] = []
    _require(isinstance(leaf, list),
             f"accretions sidecar: reference path {dotted_path!r} in sample "
             f"{sample.id!r} is not a list (found {type(leaf).__name__})")
    return leaf


def file_sha256(path: str) -> str | None:
    """sha256 of a file's bytes; graceful None when unreadable."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def git_sha(directory: str) -> str | None:
    """HEAD sha of the repo containing *directory*; graceful None."""
    import subprocess

    try:
        out = subprocess.run(["git", "-C", directory, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None
