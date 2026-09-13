"""``difyeval audit`` — the judge↔human validation loop.

Propose / apply / report idiom:

* ``propose`` emits a human-fillable YAML sheet from an eval.json: every
  rubric item (panel verdict, votes, evidence, per-judge votes) and every
  tradecraft dim (panel median, spread, per-judge scores), each with a
  ``human: null  # agree | disagree`` slot and an optional note.
* ``apply`` validates the filled sheet (every human slot must be exactly
  agree/disagree), stamps analyst + date + source sha256, and writes an
  immutable-ish ``<sheet>.applied.yml`` (re-apply refused).
* ``report`` aggregates applied sheets into per-judge-backend agreement:
  rubric items get % agreement AND Cohen's kappa (binary, textbook formula;
  degenerate marginals → "n/a (no variance)"; n < 10 → kappa refused with an
  "insufficient n" note — the honest-stats rule). Tradecraft dims are
  validated at PANEL level only (the human judged the panel median, not any
  single backend's score), so dims report % human-agree only.

Human semantics: ``agree`` = the panel's verdict / median assessment is
right; ``disagree`` = it is wrong. For rubric kappa, the human's implied
ground truth for an item is the panel verdict when agreeing, its negation
when disagreeing; each backend's vote is then compared against that truth.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import yaml

from .core import ConfigError, file_sha256
from .evalio import dim_entries, eval_case_id, eval_profile, load_eval, rubric_entries

SHEET_SCHEMA = 1
KAPPA_MIN_N = 10
_HUMAN_VALUES = ("agree", "disagree")


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------

def _y(v) -> str:
    """One-line YAML scalar/flow rendering (safe quoting via PyYAML)."""
    return yaml.safe_dump(v, default_flow_style=True, allow_unicode=True,
                          width=2 ** 20, sort_keys=True).splitlines()[0]


def render_sheet(ev: dict, eval_path: str) -> str:
    rubric = rubric_entries(ev)
    dims = dim_entries(ev)
    if not rubric and not dims:
        raise ConfigError(f"nothing to audit in {eval_path} — the eval contains no "
                          f"judge output (was it scored with --no-judge, or with no "
                          f"active judge backends?)")
    manifest = ev.get("manifest") or {}
    L = [
        "# difyeval audit sheet — judge↔human validation",
        "# Fill EVERY `human:` slot with agree or disagree:",
        "#   agree    = the panel's verdict / median assessment is right",
        "#   disagree = the panel got it wrong",
        "# Optionally add a short `note:`. Then:",
        "#   difyeval audit apply --sheet <this file> --analyst <your name>",
        f"audit_sheet: {SHEET_SCHEMA}",
        f"case_id: {_y(eval_case_id(ev))}",
        f"profile: {_y(eval_profile(ev))}",
        f"eval_path: {_y(os.path.abspath(eval_path))}",
        f"scored_at: {_y(manifest.get('scored_at'))}",
        f"judge_backends: {_y(manifest.get('judge_backends') or [])}",
    ]
    L.append("rubric:")
    if not rubric:
        L[-1] = "rubric: []"
    for e in rubric:
        L += [
            f"  - check: {_y(e['check'])}",
            f"    sample: {_y(e['sample'])}",
            f"    repeat: {_y(e['repeat'])}",
            f"    item: {_y(e['item'])}",
            f"    weight: {_y(e['weight'])}",
            f"    text: {_y(e['text'])}",
            f"    panel: {_y(e['panel'])}",
            f"    votes: {_y(e['votes'])}",
            f"    evidence: {_y(e['evidence'])}",
            f"    judge_votes: {_y(e['judge_votes'])}",
            "    human: null  # agree | disagree",
            "    note: \"\"",
        ]
    L.append("dims:")
    if not dims:
        L[-1] = "dims: []"
    for e in dims:
        L += [
            f"  - check: {_y(e['check'])}",
            f"    sample: {_y(e['sample'])}",
            f"    repeat: {_y(e['repeat'])}",
            f"    dim: {_y(e['dim'])}",
            f"    median: {_y(e['median'])}",
            f"    spread: {_y(e['spread'])}",
            f"    n: {_y(e['n'])}",
            f"    judge_scores: {_y(e['judge_scores'])}",
            "    human: null  # agree | disagree",
            "    note: \"\"",
        ]
    return "\n".join(L) + "\n"


def propose(eval_path: str, out_path: str) -> int:
    ev = load_eval(eval_path)
    if os.path.exists(out_path):
        raise ConfigError(f"refusing to overwrite existing sheet: {out_path}")
    sheet = render_sheet(ev, eval_path)
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(sheet)
    n_r = sheet.count("human: null")
    print(f"audit sheet written: {out_path} ({n_r} human slot(s) to fill)")
    print(f"next: fill every `human:` slot, then "
          f"`difyeval audit apply --sheet {out_path} --analyst <name>`")
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
    if not isinstance(raw, dict) or raw.get("audit_sheet") != SHEET_SCHEMA:
        raise ConfigError(f"{path} is not a difyeval audit sheet "
                          f"(expected audit_sheet: {SHEET_SCHEMA})")
    return raw


def applied_path_for(sheet_path: str) -> str:
    base, ext = os.path.splitext(sheet_path)
    if ext.lower() in (".yml", ".yaml"):
        return base + ".applied.yml"
    return sheet_path + ".applied.yml"


def apply_sheet(sheet_path: str, analyst: str, date: str | None = None) -> int:
    if not isinstance(analyst, str) or not analyst.strip():
        raise ConfigError("--analyst must be a non-empty name")
    if sheet_path.endswith(".applied.yml"):
        raise ConfigError(f"{sheet_path} is already an applied sheet — refusing to re-apply")
    raw = _load_sheet(sheet_path)
    if raw.get("applied"):
        raise ConfigError(f"sheet {sheet_path} already carries an 'applied' stamp — "
                          f"refusing to re-apply")

    bad = []
    for section in ("rubric", "dims"):
        entries = raw.get(section) or []
        if not isinstance(entries, list):
            raise ConfigError(f"sheet {sheet_path}: '{section}' must be a list")
        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                bad.append(f"{section}[{i}]: not a mapping")
                continue
            h = e.get("human")
            hv = h.strip().lower() if isinstance(h, str) else None
            if hv not in _HUMAN_VALUES:
                label = e.get("item") or e.get("dim") or i
                bad.append(f"{section}[{i}] ({label}): human={h!r}")
            else:
                e["human"] = hv  # normalize case/whitespace
    if bad:
        raise ConfigError("sheet has unfilled/invalid `human:` slots — every slot must "
                          "be agree or disagree: " + "; ".join(bad[:10])
                          + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""))

    out_path = applied_path_for(sheet_path)
    if os.path.exists(out_path):
        raise ConfigError(f"applied sheet already exists: {out_path} — refusing to "
                          f"re-apply (delete it deliberately if you must redo the audit)")
    raw["applied"] = {
        "analyst": analyst.strip(),
        "date": date or datetime.now(timezone.utc).date().isoformat(),
        "source_sheet": os.path.basename(sheet_path),
        "source_sha256": file_sha256(sheet_path),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    print(f"applied sheet written: {out_path} (analyst: {analyst.strip()})")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cohens_kappa(pairs: list) -> tuple[float | None, str | None]:
    """Textbook Cohen's kappa for two binary raters over paired (a, b) bools.
    Returns (kappa, None) or (None, reason) when undefined — degenerate
    marginals (pe == 1: at least one rater never varies) yield
    'n/a (no variance)'."""
    n = len(pairs)
    if n == 0:
        return None, "n=0"
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    d = sum(1 for x, y in pairs if not x and not y)
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    if pe >= 1.0:
        return None, "n/a (no variance)"
    return round((po - pe) / (1 - pe), 4), None


def _human_truth(entry: dict) -> bool:
    """The human's implied ground truth for a rubric item: the panel verdict
    when agreeing, its negation when disagreeing."""
    panel_pass = entry.get("panel") == "pass"
    return panel_pass if entry.get("human") == "agree" else not panel_pass


def render_report(sheets: list[tuple[str, dict]]) -> str:
    # rubric: per-backend (vote, human_truth) pairs; PANEL is a pseudo-backend
    pairs: dict[str, list] = {}
    skipped: dict[str, int] = {}
    n_rubric = 0
    dims_total, dims_agree = 0, 0
    per_dim: dict[str, list] = {}   # dim -> [human == agree bools]
    analysts = set()
    for _path, sheet in sheets:
        analysts.add(str((sheet.get("applied") or {}).get("analyst", "?")))
        for e in sheet.get("rubric") or []:
            if not isinstance(e, dict):
                continue
            n_rubric += 1
            truth = _human_truth(e)
            panel_pass = e.get("panel") == "pass"
            pairs.setdefault("PANEL (majority)", []).append((panel_pass, truth))
            for jn, vote in sorted((e.get("judge_votes") or {}).items()):
                if isinstance(vote, bool):
                    pairs.setdefault(jn, []).append((vote, truth))
                else:
                    skipped[jn] = skipped.get(jn, 0) + 1
        for e in sheet.get("dims") or []:
            if not isinstance(e, dict):
                continue
            dims_total += 1
            agree = e.get("human") == "agree"
            dims_agree += 1 if agree else 0
            per_dim.setdefault(str(e.get("dim")), []).append(agree)

    L = ["# difyeval audit report — judge↔human agreement", "",
         f"- sheets: {len(sheets)}",
         f"- analysts: {', '.join(sorted(analysts)) or '?'}",
         f"- honest-stats: Cohen's kappa needs n ≥ {KAPPA_MIN_N} and non-degenerate "
         f"marginals; below that only % agreement is reported. Tradecraft dims were "
         f"validated at panel level (the human judged the panel median), so dims get "
         f"% human-agree only — no kappa, no per-backend split.", ""]

    L += [f"## Rubric items — per judge backend (n items judged: {n_rubric})", ""]
    if not pairs:
        L.append("- no rubric items in the applied sheets")
    else:
        L += ["| backend | n | agreement | kappa |", "|---|---|---|---|"]
        order = (["PANEL (majority)"] if "PANEL (majority)" in pairs else []) \
            + sorted(k for k in pairs if k != "PANEL (majority)")
        for name in order:
            ps = pairs[name]
            n = len(ps)
            agree_pct = round(100.0 * sum(1 for a, b in ps if a == b) / n, 1)
            if n < KAPPA_MIN_N:
                kcell = f"n/a (insufficient n: {n} < {KAPPA_MIN_N})"
            else:
                k, reason = cohens_kappa(ps)
                kcell = reason if k is None else f"{k:g}"
            L.append(f"| {name} | {n} | {agree_pct}% | {kcell} |")
        for jn in sorted(skipped):
            L.append(f"- note: {jn} had {skipped[jn]} item(s) with no usable vote "
                     f"(judge error / missing) — excluded from its n")
    L.append("")

    L += [f"## Tradecraft dims — human agreement with the panel median (n: {dims_total})", ""]
    if not dims_total:
        L.append("- no tradecraft dims in the applied sheets")
    else:
        L += ["| dim | n | human agree |", "|---|---|---|"]
        for d in sorted(per_dim):
            xs = per_dim[d]
            L.append(f"| {d} | {len(xs)} | {round(100.0 * sum(xs) / len(xs), 1)}% |")
        L.append(f"| **all dims** | {dims_total} | "
                 f"{round(100.0 * dims_agree / dims_total, 1)}% |")
    return "\n".join(L) + "\n"


def report(sheets_glob: str, out: str | None = None) -> int:
    paths = sorted(glob.glob(sheets_glob))
    if not paths:
        raise ConfigError(f"--sheets {sheets_glob!r} matched no files")
    sheets = []
    for p in paths:
        sheet = _load_sheet(p)
        if not isinstance(sheet.get("applied"), dict):
            raise ConfigError(f"{p} is not an APPLIED sheet (no 'applied' stamp) — "
                              f"run `difyeval audit apply` on it first")
        sheets.append((p, sheet))
    md = render_report(sheets)
    print(md, end="")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"audit report written: {out}")
    return 0
