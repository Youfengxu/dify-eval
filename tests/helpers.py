"""Shared test scaffolding — tiny factories, stub judge transports."""
from __future__ import annotations

import json
import os
import tempfile

from difyeval.core import Case, CheckContext, Run, Sample

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_CASE = os.path.join(REPO_ROOT, "examples", "text-source-discovery", "case.yml")


def make_sample(sid="s1", inputs=None, reference=None, metadata=None, repeats=1):
    return Sample(id=sid, inputs=inputs or {}, reference=reference,
                  metadata=metadata or {}, repeats=repeats)


def make_case(tmpdir=None, **over):
    kw = dict(case_id="t_case", mode="reference", packs=[],
              runner={"type": "file", "path": "run.json"},
              samples=[make_sample()], checks=[], expect_verdict=None,
              judge_notes="", path=os.path.join(tmpdir or tempfile.gettempdir(), "case.yml"),
              raw={})
    kw.update(over)
    return Case(**kw)


def make_run(output="hello world", nodes=None, usage=None, meta=None, error=None):
    return Run(output=output, nodes=nodes or {}, usage=usage or {},
               meta=meta if meta is not None else {"status": "succeeded", "elapsed": 10},
               error=error)


def make_ctx(case=None, judges=None, transport=None):
    return CheckContext(case=case or make_case(), judges=judges or [],
                        transport=transport)


def fake_judges(n=2, vendors=("a-vendor", "b-vendor")):
    out = []
    for i in range(n):
        out.append({"name": f"judge{i}", "vendor": vendors[i % len(vendors)],
                    "_key": "k", "_base": "http://stub", "_model": f"m{i}",
                    "params": {"temperature": 0}})
    return out


class ScriptedTransport:
    """Stub judge transport: returns queued (status, content) responses per
    call; records every (backend name, body) for assertions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, backend, body, timeout):
        self.calls.append((backend["name"], body))
        r = self.responses.pop(0) if self.responses else (200, "{}")
        if isinstance(r, Exception):
            raise r
        return r


def dim_reply(scores: dict) -> str:
    return json.dumps({d: {"score": s, "rationale": "r"} for d, s in scores.items()})


def rubric_reply(verdicts: dict) -> str:
    return json.dumps({i: {"pass": p, "evidence": f"ev-{i}"} for i, p in verdicts.items()})
