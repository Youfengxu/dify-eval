"""file runner — load a pre-acquired run from a JSON fixture.

Spec: ``{type: file, path: fixtures/{sample_id}/run.json}``. The path may be
per case (a literal path) or per sample via the ``{sample_id}`` / ``{repeat}``
placeholders; a sample may also override it entirely with
``metadata: {run_path: ...}``. Relative paths resolve against the case dir.

Accepts every shape normalize_run_payload knows (native Run, legacy
archive {report, nodes, elapsed, status}, difyctl JSON). A missing/broken
fixture is a data problem: it degrades to a runner_error Run.
"""
from __future__ import annotations

import json
import os

from ..core import ConfigError, Run, SampleResult
from . import register_runner
from .normalize import DEFAULT_REPORT_KEY, normalize_run_payload


def substitute_path(template: str, sample_id: str, repeat: int) -> str:
    # explicit replace, not str.format — stray braces in paths must not raise
    return template.replace("{sample_id}", sample_id).replace("{repeat}", str(repeat))


def resolve_fixture_path(spec: dict, sample, repeat: int, case) -> str | None:
    template = (sample.metadata or {}).get("run_path") or spec.get("path")
    if not template:
        return None
    path = substitute_path(str(template), sample.id, repeat)
    if not os.path.isabs(path):
        path = os.path.join(case.dir, path)
    return path


def load_run_file(path: str, report_key: str = DEFAULT_REPORT_KEY) -> Run:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except OSError as e:
        return Run(error=f"cannot read run file {path}: {e}", meta={"status": "runner_error"})
    except json.JSONDecodeError as e:
        return Run(error=f"run file {path} is not valid JSON: {e}", meta={"status": "runner_error"})
    return normalize_run_payload(obj, report_key)


def assign_run_paths(case, paths: list) -> list[SampleResult]:
    """Map explicit run-file paths onto the case's samples: with a single
    sample every path is one of its repeats; otherwise the counts must match
    1:1 in dataset order. Used by `score --run` and `drift`."""
    samples = case.samples
    results = []
    if len(samples) == 1:
        for i, p in enumerate(paths, 1):
            results.append(SampleResult(sample_id=samples[0].id, repeat=i,
                                        run=load_run_file(p)))
        return results
    if len(paths) == len(samples):
        for sample, p in zip(samples, paths):
            results.append(SampleResult(sample_id=sample.id, repeat=1,
                                        run=load_run_file(p)))
        return results
    raise ConfigError(
        f"run-file count ({len(paths)}) must equal the sample count ({len(samples)}), "
        f"or the dataset must have exactly one sample (paths become its repeats)")


@register_runner("file")
def run_file(spec: dict, sample, repeat: int, case) -> Run:
    path = resolve_fixture_path(spec, sample, repeat, case)
    if not path:
        return Run(error="file runner: no 'path' configured (runner.path or "
                         "sample metadata.run_path)", meta={"status": "runner_error"})
    return load_run_file(path, spec.get("report_key", DEFAULT_REPORT_KEY))
