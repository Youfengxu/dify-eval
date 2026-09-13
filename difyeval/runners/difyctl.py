"""difyctl-json runner — parse a saved ``difyctl run -o json`` output file.

Same path templating as the file runner; the payload shape is
``data.outputs.<report_key>`` (bare ``outputs`` tolerated, mirroring
the assembly step in a typical Dify CI pipeline). difyctl emits no node
trace, so Run.nodes is empty and node.* checks are unavailable; the full
outputs mapping is preserved at ``meta.outputs``.
"""
from __future__ import annotations

from ..core import Run
from . import register_runner
from .file import load_run_file, resolve_fixture_path
from .normalize import DEFAULT_REPORT_KEY


@register_runner("difyctl-json")
def run_difyctl_json(spec: dict, sample, repeat: int, case) -> Run:
    path = resolve_fixture_path(spec, sample, repeat, case)
    if not path:
        return Run(error="difyctl-json runner: no 'path' configured",
                   meta={"status": "runner_error"})
    return load_run_file(path, spec.get("report_key", DEFAULT_REPORT_KEY))
