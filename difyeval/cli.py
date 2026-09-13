"""difyeval CLI.

Commands:
  run       acquire runs via the case's runner, then score
  score     score pre-acquired run files (--run / --runs-dir; falls back to
            the case's runner when it is offline-safe: file / difyctl-json)
  replay    alias of score — the fixture-replay path of the determinism contract
  validate  case.yml structural validation + check types + selector syntax
  diff      compare two eval.json Experiments (typically two profiles)
  audit     judge↔human validation loop: propose / apply / report
  review    ground-truth accretion loop: propose / apply (sidecar-only writes)
  drift     re-score anchor runs and compare against baseline eval.jsons
  new-case  scaffold a commented case.yml skeleton
  new-pack  scaffold a check-pack module with one worked example

Exit codes: 0 = verdict matches expect_verdict (default PASS) / clean drift;
            1 = FAIL / verdict mismatch / DRIFT / DETERMINISM-BREAK;
            2 = config error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from . import __version__
from . import gt
from .audit import apply_sheet, propose, report as audit_report
from .core import Case, ConfigError, SampleResult, load_case
from .diff import diff_command
from .drift import drift_command
from .engine import acquire_runs, build_context, score_results
from .profile import apply_runner_overrides, load_profile, resolve_profile_dsl
from .report import render_scorecard, write_eval_json
from .review import apply_sheet as review_apply, propose as review_propose
from .runners.file import assign_run_paths, load_run_file
from .validate import prepare_case

OFFLINE_RUNNERS = ("file", "difyctl-json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="difyeval",
                                 description="Universal Dify eval pipeline")
    ap.add_argument("--version", action="version", version=f"difyeval {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("case", help="path to case.yml")
    common.add_argument("--profile", default="default",
                        help="label for the system-under-test config (recorded, not interpreted)")
    common.add_argument("--profile-file", default=None,
                        help="profile YAML (profile name, description, models/config, "
                             "optional dsl + runner overrides); embedded verbatim into "
                             "manifest.profile_config")
    common.add_argument("--no-judge", action="store_true",
                        help="skip the LLM judge panel entirely (no network)")
    common.add_argument("--judges", default=None,
                        help="comma list of judge backend names (default: all active by env key)")
    common.add_argument("--out-dir", default=None,
                        help="where to write scorecard + eval.json (default: <case dir>/results)")
    common.add_argument("--dsl", default=None,
                        help="path to the workflow DSL to hash into the manifest/header")

    p_run = sub.add_parser("run", parents=[common],
                           help="acquire via the case's runner + score")
    p_run.add_argument("--save-fixture", action="store_true",
                       help="write acquired Run JSONs under <out-dir>/fixtures/<sample>/")

    for name, help_txt in (("score", "score pre-acquired run files"),
                           ("replay", "alias of score (fixture replay)")):
        p = sub.add_parser(name, parents=[common], help=help_txt)
        p.add_argument("--run", action="append", default=[], dest="run_paths",
                       help="run JSON file (repeatable; assigned to samples in dataset order)")
        p.add_argument("--runs-dir", default=None,
                       help="directory holding <sample_id>/run*.json or <sample_id>.json")

    sub.add_parser("validate", parents=[common], help="validate the case, exit")

    p_diff = sub.add_parser("diff", help="compare two eval.json Experiments")
    p_diff.add_argument("eval_a", help="baseline eval.json (A)")
    p_diff.add_argument("eval_b", help="candidate eval.json (B)")
    p_diff.add_argument("--out", default=None, help="also write the markdown diff here")
    p_diff.add_argument("--force", action="store_true",
                        help="diff even when the two case_ids differ")

    p_audit = sub.add_parser("audit", help="judge↔human validation loop")
    audit_sub = p_audit.add_subparsers(dest="audit_cmd", required=True)
    pa = audit_sub.add_parser("propose", help="emit a human-fillable sheet from an eval.json")
    pa.add_argument("--eval", required=True, dest="eval_path", help="scored eval.json")
    pa.add_argument("--out", required=True, help="where to write the sheet YAML")
    pb = audit_sub.add_parser("apply", help="validate + stamp a filled sheet")
    pb.add_argument("--sheet", required=True, help="the filled sheet YAML")
    pb.add_argument("--analyst", required=True, help="who filled the sheet")
    pb.add_argument("--date", default=None,
                    help="ISO date of the review (default: today, UTC)")
    pc = audit_sub.add_parser("report", help="per-judge-backend agreement over applied sheets")
    pc.add_argument("--sheets", required=True, help="glob of *.applied.yml sheets")
    pc.add_argument("--out", default=None, help="also write the markdown report here")

    p_review = sub.add_parser("review",
                              help="ground-truth accretion loop: propose / apply "
                                   "(writes only the case's accretions sidecar)")
    review_sub = p_review.add_subparsers(dest="review_cmd", required=True)
    rp = review_sub.add_parser("propose",
                               help="emit a decision sheet of run-discovered "
                                    "reference-item candidates")
    rp.add_argument("--eval", required=True, dest="eval_path",
                    help="scored eval.json (runs are embedded in it)")
    rp.add_argument("--case", required=True, help="the case.yml whose ground truth grows")
    rp.add_argument("--out", required=True, help="where to write the sheet YAML")
    rp.add_argument("--sample", default=None, help="restrict to this sample id")
    rp.add_argument("--extractor", default=None,
                    help="registered review extractor name (required when several "
                         "are registered)")
    rp.add_argument("--sample-k", type=int, default=None, dest="sample_k", metavar="K",
                    help="Tier-4 spot-validation: seeded-random subsample of K "
                         "candidates; apply then reports a Wilson-CI precision estimate")
    rp.add_argument("--seed", type=int, default=0,
                    help="seed for --sample-k (default 0; same seed = same sample)")
    ra = review_sub.add_parser("apply",
                               help="apply sheet decisions to the case's accretions "
                                    "sidecar (idempotent-safe)")
    ra.add_argument("--sheet", required=True, help="the filled sheet YAML")
    ra.add_argument("--case", required=True, help="the case.yml the sheet was proposed for")
    ra.add_argument("--analyst", required=True, help="who decided (stamped into validated:{by})")
    ra.add_argument("--date", default=None,
                    help="ISO date of the review (default: today, UTC)")

    p_drift = sub.add_parser("drift",
                             help="re-score anchor runs, compare against baselines "
                                  "(judged — the one networked command)")
    p_drift.add_argument("--anchors", required=True,
                         help="anchors.yml: [{case, runs: [...], baseline}]")
    p_drift.add_argument("--baseline-dir", default=None,
                         help="resolve relative baseline paths against this dir")
    p_drift.add_argument("--out", default=None, help="also write the markdown report here")
    p_drift.add_argument("--no-judge", action="store_true",
                         help="deterministic-only drift check (no network)")
    p_drift.add_argument("--judges", default=None,
                         help="comma list of judge backend names")

    p_new = sub.add_parser("new-case", help="scaffold a case.yml skeleton")
    p_new.add_argument("path", help="where to write the new case.yml")
    p_pack = sub.add_parser("new-pack", help="scaffold a check-pack module")
    p_pack.add_argument("path", help="where to write the new pack .py module")

    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    if args.cmd == "new-case":
        return _cmd_new_case(args.path)
    if args.cmd == "new-pack":
        return _cmd_new_pack(args.path)
    if args.cmd == "diff":
        return diff_command(args.eval_a, args.eval_b, out=args.out, force=args.force)
    if args.cmd == "audit":
        if args.audit_cmd == "propose":
            return propose(args.eval_path, args.out)
        if args.audit_cmd == "apply":
            return apply_sheet(args.sheet, args.analyst, date=args.date)
        return audit_report(args.sheets, out=args.out)
    if args.cmd == "review":
        if args.review_cmd == "propose":
            return review_propose(args.eval_path, args.case, args.out,
                                  sample=args.sample, extractor=args.extractor,
                                  sample_k=args.sample_k, seed=args.seed)
        return review_apply(args.sheet, args.case, args.analyst, date=args.date)
    if args.cmd == "drift":
        judges_only = args.judges.split(",") if args.judges else None
        return drift_command(args.anchors, baseline_dir=args.baseline_dir,
                             no_judge=args.no_judge, judges_only=judges_only,
                             out=args.out)

    case = load_case(args.case)
    profile_config = _apply_profile_file(args, case)
    prepare_case(case)  # packs + full validation — fail fast (exit 2)

    if args.cmd == "validate":
        print(f"OK: {case.case_id} — {len(case.samples)} sample(s), "
              f"{len(case.checks)} check(s), runner {case.runner.get('type')}")
        _report_ground_truth(case)
        return 0

    judges_only = args.judges.split(",") if args.judges else None
    ctx = build_context(case, no_judge=args.no_judge, judges_only=judges_only)
    if not args.no_judge and not ctx.judges:
        print("judge panel: none active (set a judge key env var, e.g. OPENROUTER_API_KEY) "
              "— judge checks will be skipped", file=sys.stderr)

    if args.cmd == "run":
        results = acquire_runs(case)
    else:  # score / replay
        results = _load_prescored_runs(case, args.run_paths, args.runs_dir)

    out_dir = args.out_dir or os.path.join(case.dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    if args.cmd == "run" and args.save_fixture:
        for r in results:
            fdir = os.path.join(out_dir, "fixtures", r.sample_id)
            os.makedirs(fdir, exist_ok=True)
            fpath = os.path.join(fdir, f"run-{r.repeat}.json")
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(r.run.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            print(f"fixture saved: {fpath}")

    exp = score_results(case, results, ctx, profile=args.profile, dsl_path=args.dsl,
                        profile_config=profile_config)

    scorecard = render_scorecard(exp)
    md_path = os.path.join(out_dir, f"{case.case_id}__{args.profile}__scorecard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(scorecard)
    json_path = os.path.join(out_dir, f"{case.case_id}__{args.profile}__eval.json")
    write_eval_json(exp, json_path)

    expected = case.expect_verdict or "PASS"
    match = exp.verdict == expected
    print(f"[{case.case_id}] verdict: {exp.verdict}"
          + (f" (expected {expected} — MISMATCH)" if not match else ""))
    if exp.warn_reason:
        print(f"[{case.case_id}] {exp.warn_reason}")
    print(f"[{case.case_id}] scorecard: {md_path}")
    print(f"[{case.case_id}] eval.json: {json_path}")
    return 0 if match else 1


# --------------------------------------------------------------------------
# validate: ground-truth surfacing (sidecar presence + provenance warnings)
# --------------------------------------------------------------------------

def _report_ground_truth(case: Case) -> None:
    if case.sidecar_path:
        total = sum(m["added"] for m in case.sidecar_merged)
        print(f"accretions sidecar: {case.sidecar_path} — {total} item(s) merged")
        for m in case.sidecar_merged:
            line = f"  {m['sample']} · {m['path']}: +{m['added']}"
            if m["skipped"]:
                line += f" ({m['skipped']} duplicate(s) skipped)"
            print(line)
    for w in case.load_warnings:
        print(f"warning: {w}")
    for sample in case.samples:
        for path, items in gt.iter_reference_lists(sample.reference):
            for w in gt.validate_items(items):
                print(f"warning: {sample.id} · reference.{path}: {w}")


# --------------------------------------------------------------------------
# profile file handling
# --------------------------------------------------------------------------

def _apply_profile_file(args, case: Case) -> dict | None:
    """Load --profile-file: its `profile` name becomes the Experiment label,
    its `dsl` auto-fills --dsl (explicit --dsl wins), its `runner:` overrides
    shallow-merge over case.runner. Returns the raw profile mapping for
    manifest.profile_config, or None without a profile file."""
    if not getattr(args, "profile_file", None):
        return None
    if args.profile != "default":
        raise ConfigError("pass either --profile or --profile-file, not both — "
                          "the profile file's 'profile' field is the label")
    prof = load_profile(args.profile_file)
    args.profile = prof["profile"]
    if args.dsl is None:
        args.dsl = resolve_profile_dsl(prof, args.profile_file)
    apply_runner_overrides(case, prof)
    return prof


# --------------------------------------------------------------------------
# score/replay run loading
# --------------------------------------------------------------------------

def _load_prescored_runs(case: Case, run_paths: list, runs_dir: str | None) -> list[SampleResult]:
    if run_paths:
        return assign_run_paths(case, run_paths)
    if runs_dir:
        return _scan_runs_dir(case, runs_dir)
    if case.runner.get("type") in OFFLINE_RUNNERS:
        return acquire_runs(case)  # the case's own runner is already offline
    raise ConfigError(
        f"score/replay needs --run or --runs-dir when the case runner is "
        f"{case.runner.get('type')!r} (only {list(OFFLINE_RUNNERS)} run offline)")


def _scan_runs_dir(case: Case, runs_dir: str) -> list[SampleResult]:
    results = []
    for sample in case.samples:
        candidates = (sorted(glob.glob(os.path.join(runs_dir, sample.id, "run-*.json")))
                      or sorted(glob.glob(os.path.join(runs_dir, sample.id, "run.json")))
                      or sorted(glob.glob(os.path.join(runs_dir, f"{sample.id}.json"))))
        if not candidates:
            raise ConfigError(f"--runs-dir {runs_dir}: no run file found for sample "
                              f"'{sample.id}' (looked for {sample.id}/run-*.json, "
                              f"{sample.id}/run.json, {sample.id}.json)")
        for i, p in enumerate(candidates, 1):
            results.append(SampleResult(sample_id=sample.id, repeat=i, run=load_run_file(p)))
    return results


# --------------------------------------------------------------------------
# scaffolds
# --------------------------------------------------------------------------

_CASE_SKELETON = '''\
# difyeval case — see the README for the full contract.
case_id: my_workflow_eval

# reference: checks compare against per-sample reference data.
# live: no reference — PASS means process-soundness only (scope line says so).
mode: reference

# Optional: importable modules whose import registers custom checks.
# packs: [my_pkg.checks]

runner:
  type: file                      # file | difyctl-json | dify-workflow | dify-chatflow | command
  path: fixtures/{sample_id}/run.json
  # -- dify-workflow / dify-chatflow --
  # base: https://dify.example.com   # or base_env: DIFY_BASE
  # api_key_env: DIFY_API_KEY
  # input_map: {caption: post_text}  # sample input key -> Dify variable
  # watch: [node_a, node_b]          # node ids to journal (omit = all)
  # report_key: report               # dify-workflow output key
  # -- command --
  # argv: ["python3", "sut.py", "--sample", "{sample_id}"]

dataset:
  - id: s1
    inputs:                        # -> Dify start-node variables, mechanically
      var: "..."
    reference:                     # optional; typed by the checks that use it
      expected: "..."
    # repeats: 2                   # re-run the same sample N times

checks:
  # deterministic, gate: true -> can FAIL the case
  - {id: has_out, type: output.nonempty, gate: true}
  # - {id: cites, type: node.json_count, selector: nodes.sources.outputs.sources_json, min: 3, gate: true}
  # - {id: origin, type: node.json_field, selector: nodes.merge.outputs.summary_json, field: earliest_url, equals_from: reference.expected}
  # judge checks are ADVISORY ALWAYS (gate: true on them is a config error);
  # a failing *essential* rubric item drives WARN.
  # - id: rubric
  #   type: judge.rubric
  #   items:
  #     - {id: a, weight: essential, item: "The report identifies X with a URL."}
  # - {id: craft, type: judge.tradecraft}

# Optional: anchor cases pin a non-PASS expectation (exit 0 iff verdict matches).
# expect_verdict: PASS

# Optional judge context, shown to every panel member.
# judge_notes: >
#   Anything the judges should know about this case.
'''

_PACK_SKELETON = '''\
"""Example difyeval check pack.

Register it in a case with::

    packs: [{module_name}]

(any importable dotted module path works — install your package or set
PYTHONPATH so the module resolves).
"""
from difyeval.checks import degrade, make_score, register
from difyeval.checks.selector import maybe_json, resolve


@register("{ns}.report_names_every_account")
def report_names_every_account(spec, sample, run, ctx):
    """Worked example: every account listed in a node's JSON output must be
    named in the report text. Params: `selector` (list of account dicts with
    a `username` key)."""
    res = resolve(run.to_dict(), spec.get("selector", ""))
    if not res.ok:
        return degrade(spec, res.error)          # data problem: degrade, never raise
    ok, decoded, err = maybe_json(res.value)     # node outputs are often JSON-in-a-string
    if not ok:
        return degrade(spec, err)
    accounts = decoded if isinstance(decoded, list) else []
    usernames = sorted({str(a.get("username", "")) for a in accounts
                        if isinstance(a, dict) and a.get("username")})
    if not usernames:
        return degrade(spec, "no usernames found at selector")
    missing = [u for u in usernames if u not in (run.output or "")]
    passed = not missing
    value = round((len(usernames) - len(missing)) / len(usernames), 4)
    ev = ("all %d account(s) named in report" % len(usernames) if passed
          else "missing from report: " + ", ".join(missing[:5]))
    return make_score(spec, value, passed, ev)
'''


def _cmd_new_case(path: str) -> int:
    if os.path.exists(path):
        raise ConfigError(f"refusing to overwrite existing file: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_CASE_SKELETON)
    print(f"scaffolded case: {path}")
    print("next: edit it, then `difyeval validate` it")
    return 0


def _cmd_new_pack(path: str) -> int:
    if os.path.exists(path):
        raise ConfigError(f"refusing to overwrite existing file: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    module_name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(_PACK_SKELETON.replace("{module_name}", module_name)
                .replace("{ns}", module_name))
    print(f"scaffolded check pack: {path}")
    print(f"register in a case with:  packs: [{module_name}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
