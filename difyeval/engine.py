"""Experiment engine: acquire runs, execute checks, assemble the Experiment."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable

from . import __version__
from . import checks as registry
from . import judge as judge_mod
from .core import Case, CheckContext, ConfigError, Experiment, Run, SampleResult, Score, \
    file_sha256, git_sha
from .runners import get_runner
from .validate import prepare_case
from .verdict import compute_metrics, compute_verdict, scope_statement


def build_context(case: Case, no_judge: bool = False, judges_only: list | None = None,
                  env: dict | None = None, transport: Callable | None = None) -> CheckContext:
    env = dict(os.environ) if env is None else env
    judges = [] if no_judge else judge_mod.active_judges(env, only=judges_only)
    return CheckContext(case=case, judges=judges, transport=transport)


def run_checks(case: Case, sample, run: Run, ctx: CheckContext) -> list[Score]:
    """Execute every check in case.yml order. A check that raises on messy
    data is caught here (degrade-never-raise safety net); judge checks are
    forced advisory regardless of what they return."""
    scores = []
    for spec in case.checks:
        cd = registry.get(spec["type"])
        try:
            score = cd.fn(spec, sample, run, ctx)
        except Exception as e:  # noqa: BLE001 — data problems degrade
            score = Score(check_id=spec["id"], value=None,
                          passed=False if (spec.get("gate") and not cd.judge) else None,
                          evidence=f"check error ({type(e).__name__}): {str(e)[:200]}",
                          advisory=not spec.get("gate") or cd.judge)
        if cd.judge:
            score.advisory = True  # the judge never gates — enforced
        scores.append(score)
    return scores


def acquire_runs(case: Case) -> list[SampleResult]:
    """Acquire via the case's runner: samples in dataset order x repeats."""
    runner = get_runner(case.runner)
    results = []
    for sample in case.samples:
        for repeat in range(1, sample.repeats + 1):
            run = runner(case.runner, sample, repeat, case)
            results.append(SampleResult(sample_id=sample.id, repeat=repeat, run=run))
    return results


def score_results(case: Case, results: list[SampleResult], ctx: CheckContext,
                  profile: str = "default", dsl_path: str | None = None,
                  profile_config: dict | None = None) -> Experiment:
    """Score pre-acquired runs and assemble the full Experiment."""
    samples_by_id = {s.id: s for s in case.samples}
    for r in results:
        sample = samples_by_id.get(r.sample_id)
        if sample is None:
            raise ConfigError(f"run refers to unknown sample id {r.sample_id!r}")
        r.scores = run_checks(case, sample, r.run, ctx)
    verdict, warn_reason = compute_verdict(case, results)
    exp = Experiment(
        case=case, profile=profile, results=results,
        verdict=verdict, warn_reason=warn_reason,
        scope=scope_statement(case),
        metrics=compute_metrics(case, results),
        manifest=build_manifest(case, ctx, profile, dsl_path, profile_config),
    )
    return exp


def build_manifest(case: Case, ctx: CheckContext, profile: str,
                   dsl_path: str | None = None,
                   profile_config: dict | None = None) -> dict:
    return {
        "difyeval_version": __version__,
        "git_sha": git_sha(case.dir),
        "case_sha256": case.sha256(),
        # accretions sidecar (run-accreted ground truth) merged at load, if any
        "accretions_sha256": file_sha256(case.sidecar_path) if case.sidecar_path else None,
        "dsl_sha256": file_sha256(dsl_path) if dsl_path else None,
        "judge_backends": judge_mod.redacted_backends(ctx.judges),
        "truncation": list(ctx.truncations),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        # the loaded --profile-file mapping, embedded VERBATIM (None without one)
        "profile_config": profile_config,
        "mode": case.mode,
    }


def run_experiment(case: Case, profile: str = "default", no_judge: bool = False,
                   judges_only: list | None = None, dsl_path: str | None = None,
                   transport: Callable | None = None,
                   results: list[SampleResult] | None = None,
                   profile_config: dict | None = None) -> Experiment:
    """One-call path: prepare (packs + validation, fail fast), acquire (unless
    pre-acquired results are passed), score, assemble."""
    prepare_case(case)
    ctx = build_context(case, no_judge=no_judge, judges_only=judges_only,
                        transport=transport)
    if results is None:
        results = acquire_runs(case)
    return score_results(case, results, ctx, profile=profile, dsl_path=dsl_path,
                         profile_config=profile_config)
