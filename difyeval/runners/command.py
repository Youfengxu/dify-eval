"""command runner — shell out to an arbitrary system under test.

Spec::

    runner:
      type: command
      argv: ["python3", "run_my_thing.py", "--case", "{case_file}",
             "--sample", "{sample_id}", "--url", "{input.media_url}"]
      timeout: 600        # seconds, default 600

Placeholders: ``{case_file}`` (absolute case.yml path), ``{sample_id}``,
``{repeat}``, and ``{input.<k>}`` for any sample input. The sample's inputs
are ALSO written to the child's stdin as JSON (sorted keys), so commands can
read either. The command must print a Run-shaped JSON object on stdout (any
shape normalize_run_payload accepts).

Nonzero exit / timeout / unparseable stdout degrade to
``Run(error=..., meta.status="runner_error")`` — gated checks then fail.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess

from ..core import Run
from . import register_runner
from .normalize import DEFAULT_REPORT_KEY, normalize_run_payload

DEFAULT_TIMEOUT_S = 600
_INPUT_PH = re.compile(r"\{input\.([A-Za-z0-9_\-]+)\}")


def substitute_argv(argv: list[str], case, sample, repeat: int) -> list[str]:
    out = []
    for arg in argv:
        arg = arg.replace("{case_file}", case.path)
        arg = arg.replace("{sample_id}", sample.id)
        arg = arg.replace("{repeat}", str(repeat))
        arg = _INPUT_PH.sub(lambda m: str(sample.inputs.get(m.group(1), "")), arg)
        out.append(arg)
    return out


@register_runner("command")
def run_command(spec: dict, sample, repeat: int, case) -> Run:
    argv = spec.get("argv") or spec.get("command")
    if isinstance(argv, str):
        argv = shlex.split(argv)
    if not isinstance(argv, list) or not argv:
        return Run(error="command runner: needs 'argv' (list) or 'command' (string)",
                   meta={"status": "runner_error"})
    argv = substitute_argv([str(a) for a in argv], case, sample, repeat)
    timeout = spec.get("timeout", DEFAULT_TIMEOUT_S)
    stdin = json.dumps(sample.inputs, ensure_ascii=False, sort_keys=True)
    try:
        proc = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                              timeout=timeout, cwd=case.dir)
    except subprocess.TimeoutExpired:
        return Run(error=f"command timed out after {timeout}s: {argv[0]}",
                   meta={"status": "runner_error"})
    except OSError as e:
        return Run(error=f"command failed to start: {e}", meta={"status": "runner_error"})
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-500:]
        return Run(error=f"command exited {proc.returncode}"
                         + (f": {tail}" if tail else ""),
                   meta={"status": "runner_error"})
    try:
        obj = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return Run(error=f"command stdout is not valid JSON: {e}",
                   meta={"status": "runner_error"})
    return normalize_run_payload(obj, spec.get("report_key", DEFAULT_REPORT_KEY))
