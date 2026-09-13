"""``difyeval review`` — the ground-truth accretion loop (propose / apply).

For workflows whose truth grows from human review of live outputs: after a
scored run, an analyst MAY review what the pipeline found and fold it into
the case's ground truth — validated (gate-eligible) or unvalidated
(advisory-only). Skipping the loop entirely is always a no-op; reference
data only ever changes through an explicit ``apply``, and apply writes ONLY
the accretions sidecar (see difyeval.core) — never the base case file.

* ``propose`` loads a scored eval.json (runs are embedded in it), calls a
  pack-registered extractor per (sample, run) to surface candidate reference
  items, centrally drops candidates already present in the merged reference
  (base + sidecar, by gt.item_identity) or banned by ``must_reject``, and
  emits a decision sheet: every item ``decision: skip`` by default.
* ``apply`` validates the filled sheet and appends to the sidecar:
  ``confirm`` → ``provenance: human_verified`` + ``validated: {by, date}``;
  ``unvalidated`` → ``provenance: pipeline_observed_unvalidated``;
  ``negative`` → a ``{value, reason, source_run}`` entry under
  ``must_reject``; ``skip`` → nothing. Idempotent-safe: re-applying the
  same sheet finds every item already present and no-ops with a note.

Pack extractors register at import time (packs load via the case's
``packs:`` list)::

    from difyeval.review import register_review_extractor

    @register_review_extractor("csd")
    def extract(case, sample, run, experiment):
        # case: difyeval.core.Case (sidecar-merged view)
        # sample: difyeval.core.Sample — the run's dataset entry
        # run: difyeval.core.Run — one acquired run for that sample
        # experiment: the loaded eval.json mapping (scores, manifest, ...)
        return [{"path": "accounts.items",             # dotted reference path
                 "item": {"handle": "@discovered"},    # what would be appended
                 "evidence": {"urls": [...]}}]         # free-form, shown to the analyst

Tier-4 spot-validation: ``propose --sample-k K --seed S`` takes a seeded
deterministic random subsample of the candidates; ``apply`` on a sampled
sheet reports a precision estimate with a 95% Wilson score interval
(confirm vs negative decisions) — an ESTIMATE, never a census.
"""
from __future__ import annotations

import math
import os
import random
from datetime import datetime, timezone
from typing import Callable

import yaml

from . import checks as registry
from . import gt
from .core import ConfigError, Run, accretions_path, load_accretions, load_case
from .evalio import eval_case_id, eval_profile, load_eval

SHEET_SCHEMA = 1
DECISIONS = ("skip", "confirm", "unvalidated", "negative")

_EXTRACTORS: dict[str, Callable] = {}


def register_review_extractor(name: str):
    """Decorator: register a review extractor
    ``extract(case, sample, run, experiment) -> [candidate]`` under a name.
    Candidates are ``{path, item, evidence}``; do NOT dedupe against the
    reference inside the extractor — propose does that centrally."""

    def deco(fn):
        _EXTRACTORS[name] = fn
        return fn

    return deco


def registered_extractors() -> list[str]:
    return sorted(_EXTRACTORS)


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion (k successes of n).
    Ported verbatim from an earlier internal ground-truth review implementation."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return round(max(0.0, center - half), 3), round(min(1.0, center + half), 3)


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------

def _pick_extractor(name: str | None) -> tuple[str, Callable]:
    if name:
        fn = _EXTRACTORS.get(name)
        if fn is None:
            raise ConfigError(f"unknown review extractor {name!r} — registered: "
                              f"{registered_extractors() or '(none)'}")
        return name, fn
    if len(_EXTRACTORS) == 1:
        return next(iter(_EXTRACTORS.items()))
    names = ", ".join(registered_extractors()) \
        or "none — does the case list a pack that registers one?"
    raise ConfigError(f"pass --extractor: {len(_EXTRACTORS)} review "
                      f"extractor(s) registered ({names})")


def _run_from_dict(d: dict) -> Run:
    d = d if isinstance(d, dict) else {}
    return Run(output=d.get("output") or "", nodes=d.get("nodes") or {},
               usage=d.get("usage") or {}, meta=d.get("meta") or {},
               error=d.get("error"))


def _check_candidate(c, extractor: str):
    if not isinstance(c, dict) or not isinstance(c.get("path"), str) \
            or not c["path"].strip() or "item" not in c or c["item"] is None:
        raise ConfigError(f"extractor {extractor!r} emitted a malformed candidate "
                          f"(need {{path: str, item, evidence?}}): {c!r}")


def propose(eval_path: str, case_path: str, out_path: str, sample: str | None = None,
            extractor: str | None = None, sample_k: int | None = None,
            seed: int = 0) -> int:
    ev = load_eval(eval_path)
    if os.path.exists(out_path):
        raise ConfigError(f"refusing to overwrite existing sheet: {out_path}")
    if sample_k is not None and sample_k < 1:
        raise ConfigError("--sample-k must be >= 1")
    case = load_case(case_path)          # merged view: base + sidecar
    registry.load_packs(case)            # packs register checks AND extractors
    if eval_case_id(ev) not in ("?", case.case_id):
        raise ConfigError(f"eval.json is for case {eval_case_id(ev)!r}, not "
                          f"{case.case_id!r} — wrong --eval or --case?")
    ex_name, ex_fn = _pick_extractor(extractor)
    samples_by_id = {s.id: s for s in case.samples}

    # ---- extract candidates per (sample, run), eval.json stored order ----
    runs_seen, candidates, notes = [], [], []
    for r in ev.get("results") or []:
        if not isinstance(r, dict):
            continue
        sid, rep = r.get("sample_id"), r.get("repeat")
        if sample and sid != sample:
            continue
        s = samples_by_id.get(sid)
        if s is None:
            notes.append(f"eval result for unknown sample {sid!r} skipped")
            continue
        runs_seen.append(f"{sid}:r{rep}")
        for c in ex_fn(case, s, _run_from_dict(r.get("run")), ev) or []:
            _check_candidate(c, ex_name)
            candidates.append({"sample": sid, "repeat": rep,
                               "path": c["path"].strip(), "item": c["item"],
                               "evidence": c.get("evidence") or {}})
    if sample and sample not in samples_by_id:
        raise ConfigError(f"--sample {sample!r} is not a sample of {case.case_id!r}")

    # ---- central dedupe: merged reference + must_reject bans + in-sheet ----
    banned = {s.id: {gt.identity_value(e) for e in gt.reject_list(s.reference)}
              for s in case.samples}
    existing: dict[tuple, set] = {}
    kept, seen = [], set()
    for c in candidates:
        ident = gt.item_identity(c["item"])
        key = (c["sample"], c["path"], ident)
        ref = samples_by_id[c["sample"]].reference
        ex_key = (c["sample"], c["path"])
        if ex_key not in existing:
            existing[ex_key] = {gt.item_identity(x)
                                for x in gt.iter_items(ref, c["path"])}
        if key in seen or ident in existing[ex_key] \
                or ident[1] in banned[c["sample"]]:
            continue
        seen.add(key)
        kept.append(c)

    # ---- Tier-4 spot-validation: seeded deterministic subsample ----
    sampled = None
    if sample_k is not None and len(kept) > sample_k:
        idx = sorted(random.Random(seed).sample(range(len(kept)), sample_k))
        sampled = {"k": sample_k, "population": len(kept), "seed": seed,
                   "note": "spot-validation sample — apply reports a precision "
                           "ESTIMATE (Wilson 95% CI), not a census"}
        kept = [kept[i] for i in idx]

    sheet = {
        "review_sheet": SHEET_SCHEMA,
        "case_id": case.case_id,
        "case_file": os.path.abspath(case_path),
        "eval_path": os.path.abspath(eval_path),
        "profile": eval_profile(ev),
        "extractor": ex_name,
        "runs": runs_seen,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decisions": "skip | confirm | unvalidated | negative",
    }
    if sampled:
        sheet["sampled"] = sampled
    if notes:
        sheet["notes"] = notes
    sheet["items"] = [{"sample": c["sample"], "repeat": c["repeat"],
                       "path": c["path"], "item": c["item"],
                       "evidence": c["evidence"],
                       "decision": "skip", "reason": ""} for c in kept]

    header = (
        "# difyeval review sheet — ground-truth accretion (propose/apply loop)\n"
        "# Set `decision:` per item (default skip = drop):\n"
        "#   confirm     -> appended with provenance: human_verified + validated:{by,date}\n"
        "#   unvalidated -> appended with provenance: pipeline_observed_unvalidated (advisory)\n"
        "#   negative    -> appended to must_reject as {value, reason, source_run} (fill reason)\n"
        "#   skip        -> nothing\n"
        "# Then: difyeval review apply --sheet <this file> --case <case.yml> --analyst <you>\n")
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True,
                                        width=120))
    print(f"review sheet written: {out_path} — {len(kept)} candidate(s) "
          f"from {len(runs_seen)} run(s) [extractor: {ex_name}]"
          + (f" [SAMPLED {sampled['k']} of {sampled['population']}, "
             f"seed {sampled['seed']}]" if sampled else ""))
    print(f"next: set decisions, then `difyeval review apply --sheet {out_path} "
          f"--case {case_path} --analyst <name>`")
    return 0


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def _load_sheet(path: str) -> dict:
    if not os.path.isfile(path):
        raise ConfigError(f"sheet not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in sheet {path}: {e}") from e
    if not isinstance(raw, dict) or raw.get("review_sheet") != SHEET_SCHEMA:
        raise ConfigError(f"{path} is not a difyeval review sheet "
                          f"(expected review_sheet: {SHEET_SCHEMA})")
    if not isinstance(raw.get("items"), list):
        raise ConfigError(f"sheet {path}: 'items' must be a list")
    return raw


def _negative_value(item):
    """The raw designated-field value recorded as a must_reject `value`."""
    if isinstance(item, dict):
        for f in gt.IDENTITY_FIELDS:
            v = item.get(f)
            if isinstance(v, (str, int, float)) and not isinstance(v, bool) \
                    and str(v).strip():
                return v
        return gt.item_identity(item)[1]
    return item


def apply_sheet(sheet_path: str, case_path: str, analyst: str,
                date: str | None = None) -> int:
    if not isinstance(analyst, str) or not analyst.strip():
        raise ConfigError("--analyst must be a non-empty name")
    analyst = analyst.strip()
    date = date or datetime.now(timezone.utc).date().isoformat()
    sheet = _load_sheet(sheet_path)
    case = load_case(case_path)          # merged view for dedupe
    samples_by_id = {s.id: s for s in case.samples}

    # validate every decision BEFORE touching anything
    entries, bad = [], []
    for i, e in enumerate(sheet["items"]):
        if not isinstance(e, dict) or not isinstance(e.get("path"), str) \
                or "item" not in e or e.get("sample") not in samples_by_id:
            bad.append(f"items[{i}]: malformed (need sample/path/item of this case)")
            continue
        d = str(e.get("decision") or "skip").strip().lower()
        if d not in DECISIONS:
            bad.append(f"items[{i}] ({gt.identity_value(e['item'])}): "
                       f"decision={e.get('decision')!r}")
            continue
        entries.append((d, e))
    if bad:
        raise ConfigError("sheet has invalid entries — every decision must be one of "
                          f"{DECISIONS}: " + "; ".join(bad[:10])
                          + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""))

    banned = {s.id: {gt.identity_value(x) for x in gt.reject_list(s.reference)}
              for s in case.samples}
    existing: dict[tuple, set] = {}

    def known(sid, path, ident):
        key = (sid, path)
        if key not in existing:
            existing[key] = {gt.item_identity(x) for x in
                             gt.iter_items(samples_by_id[sid].reference, path)}
        return ident in existing[key]

    acc = load_accretions(case_path)
    added = {"confirm": 0, "unvalidated": 0, "negative": 0}
    skipped_known = 0

    def append(sid, path, item):
        acc.setdefault("samples", {}).setdefault(sid, {}) \
           .setdefault("reference", {}).setdefault(path, []).append(item)

    eval_src = sheet.get("eval_path") or "unknown-eval"
    for d, e in entries:
        if d == "skip":
            continue
        sid, path, item = e["sample"], e["path"].strip(), e["item"]
        src = f"{eval_src}#{sid}:r{e.get('repeat')}"
        ident = gt.item_identity(item)
        if d in ("confirm", "unvalidated"):
            if known(sid, path, ident):
                skipped_known += 1
                continue
            out = dict(item) if isinstance(item, dict) else {"value": item}
            out["provenance"] = (gt.HUMAN_VERIFIED if d == "confirm"
                                 else gt.PIPELINE_OBSERVED_UNVALIDATED)
            if d == "confirm":
                out["validated"] = {"by": analyst, "date": date}
            else:
                out.pop("validated", None)   # the tier comes from the DECISION
            out["source_run"] = src
            append(sid, path, out)
            existing[(sid, path)].add(ident)
            added[d] += 1
        else:  # negative
            if ident[1] in banned[sid]:
                skipped_known += 1
                continue
            append(sid, "must_reject", {
                "value": _negative_value(item),
                "reason": str(e.get("reason") or "").strip()
                          or "analyst-rejected in review",
                "source_run": src})
            banned[sid].add(ident[1])
            added["negative"] += 1

    total = sum(added.values())
    if not total:
        print("nothing to apply (all decisions skip / items already present) — "
              "sidecar unchanged")
        return 0

    header = ("# Run-accreted ground truth — written ONLY by `difyeval review apply`.\n"
              "# Merged into the case at load time (append-only; base file untouched).\n"
              "# Graduate items into the base case file by hand whenever you wish.\n")
    acc_path = accretions_path(case_path)
    with open(acc_path, "w", encoding="utf-8") as f:
        f.write(header + yaml.safe_dump(acc, sort_keys=False, allow_unicode=True,
                                        width=120))
    print(f"applied: {acc_path}")
    print(f"  confirmed (human_verified): {added['confirm']}  ·  "
          f"unvalidated (advisory): {added['unvalidated']}  ·  "
          f"negatives (must_reject): {added['negative']}"
          + (f"  ·  already present: {skipped_known}" if skipped_known else ""))

    # Tier-4 precision estimate on a sampled sheet (spot-validation)
    if sheet.get("sampled"):
        k_ok, k_bad = added["confirm"], added["negative"]
        n_dec = k_ok + k_bad
        if n_dec:
            ci = wilson_ci(k_ok, n_dec)
            print(f"  precision estimate (sampled spot-validation): "
                  f"{round(k_ok / n_dec, 3)} ({k_ok}/{n_dec} decided; "
                  f"95% Wilson CI {ci[0]}-{ci[1]}; sample "
                  f"{sheet['sampled'].get('k')} of "
                  f"{sheet['sampled'].get('population')})")

    reloaded = load_case(case_path)      # re-validate: the case must load cleanly
    n_items = sum(m["added"] for m in reloaded.sidecar_merged)
    print(f"  re-loaded {os.path.basename(case_path)}: OK — sidecar now carries "
          f"{n_items} merged item(s), {len(reloaded.load_warnings)} warning(s)")
    for w in reloaded.load_warnings:
        print(f"    warning: {w}")
    return 0
