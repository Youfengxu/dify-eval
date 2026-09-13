import json
import os
import tempfile
import unittest

import yaml

from difyeval.core import ConfigError, load_case
from difyeval.drift import DIM_DRIFT_THRESHOLD, load_anchors, run_drift
from difyeval.engine import build_context, score_results
from difyeval.report import write_eval_json
from difyeval.runners.file import assign_run_paths
from difyeval.validate import prepare_case

from tests.helpers import ScriptedTransport, dim_reply, rubric_reply
from tests.test_cli import run_cli

ENV = {"OPENAI_API_KEY": "k"}  # activates exactly one judge backend, stubbed transport

CASE = {
    "case_id": "drift_anchor",
    "runner": {"type": "file", "path": "run.json"},
    "dataset": [{"id": "s1", "inputs": {}}],
    "checks": [
        {"id": "has_out", "type": "output.nonempty", "gate": True},
        {"id": "mentions", "type": "output.contains", "values": ["newswire"], "gate": True},
        {"id": "rubric", "type": "judge.rubric",
         "items": [{"id": "a", "weight": "expected", "item": "Names the source."}]},
        {"id": "craft", "type": "judge.tradecraft",
         "dims": [{"id": "d1", "criterion": "clarity of sourcing"}]},
    ],
}

BASE_REPLIES = [(200, rubric_reply({"a": True})), (200, dim_reply({"d1": 0.8}))]


def _bundle(td, judged=True):
    """Write case.yml + run.json + a baseline eval.json + anchors.yml into td."""
    case_path = os.path.join(td, "case.yml")
    with open(case_path, "w") as f:
        yaml.safe_dump(CASE, f)
    run_path = os.path.join(td, "run.json")
    with open(run_path, "w") as f:
        json.dump({"output": "report cites newswire.example"}, f)
    case = load_case(case_path)
    prepare_case(case)
    if judged:
        ctx = build_context(case, env=ENV,
                            transport=ScriptedTransport(list(BASE_REPLIES)))
    else:
        ctx = build_context(case, no_judge=True)
    exp = score_results(case, assign_run_paths(case, [run_path]), ctx)
    baseline_path = os.path.join(td, "baseline.json")
    write_eval_json(exp, baseline_path)
    anchors_path = os.path.join(td, "anchors.yml")
    with open(anchors_path, "w") as f:
        yaml.safe_dump({"anchors": [{"case": "case.yml", "runs": ["run.json"],
                                     "baseline": "baseline.json"}]}, f)
    return anchors_path


class TestDriftClean(unittest.TestCase):
    def test_identical_rescore_is_ok_exit_0(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td)
            md, rc = run_drift(anchors, env=ENV,
                               transport=ScriptedTransport(list(BASE_REPLIES)))
        self.assertEqual(rc, 0)
        self.assertIn("✅ OK", md)
        self.assertIn("identical (2 score(s))", md)  # both deterministic gates
        self.assertNotIn("DRIFT", md.replace("difyeval drift", ""))

    def test_dim_delta_at_threshold_is_not_drift(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td)
            replies = [(200, rubric_reply({"a": True})),
                       (200, dim_reply({"d1": round(0.8 - DIM_DRIFT_THRESHOLD, 4)}))]
            md, rc = run_drift(anchors, env=ENV, transport=ScriptedTransport(replies))
        self.assertEqual(rc, 0)  # strictly greater-than triggers, equal does not
        self.assertIn("✅ OK", md)


class TestJudgeDrift(unittest.TestCase):
    def test_dim_median_delta_over_threshold_is_drift_exit_1(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td)
            replies = [(200, rubric_reply({"a": True})), (200, dim_reply({"d1": 0.5}))]
            md, rc = run_drift(anchors, env=ENV, transport=ScriptedTransport(replies))
        self.assertEqual(rc, 1)
        self.assertIn("🔴 DRIFT", md)
        self.assertIn("d1: 0.8 → 0.5", md)
        self.assertIn("identical (2 score(s))", md)  # deterministic side still clean

    def test_rubric_item_flip_is_drift_exit_1(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td)
            replies = [(200, rubric_reply({"a": False})), (200, dim_reply({"d1": 0.8}))]
            md, rc = run_drift(anchors, env=ENV, transport=ScriptedTransport(replies))
        self.assertEqual(rc, 1)
        self.assertIn("🔴 DRIFT", md)
        self.assertIn("rubric flip: rubric/a (s1 r1): pass → fail", md)


class TestDeterminismBreak(unittest.TestCase):
    def test_changed_deterministic_score_is_break_exit_1(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td)
            # the frozen run "changes" (simulates code/config moving underfoot)
            with open(os.path.join(td, "run.json"), "w") as f:
                json.dump({"output": ""}, f)
            md, rc = run_drift(anchors, env=ENV,
                               transport=ScriptedTransport(list(BASE_REPLIES)))
        self.assertEqual(rc, 1)
        self.assertIn("❌ DETERMINISM-BREAK", md)
        self.assertIn("has_out (s1 r1): passed True → False", md)

    def test_break_is_distinct_from_drift(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td)
            with open(os.path.join(td, "run.json"), "w") as f:
                json.dump({"output": ""}, f)
            # judge ALSO drifts, but BREAK dominates the status
            replies = [(200, rubric_reply({"a": False})), (200, dim_reply({"d1": 0.2}))]
            md, rc = run_drift(anchors, env=ENV, transport=ScriptedTransport(replies))
        self.assertEqual(rc, 1)
        summary_row = next(ln for ln in md.split("\n") if ln.startswith("| drift_anchor"))
        self.assertIn("DETERMINISM-BREAK", summary_row)


class TestNoJudge(unittest.TestCase):
    def test_no_judge_cli_deterministic_only_exit_0(self):
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td, judged=False)
            out_path = os.path.join(td, "drift.md")
            rc, out, _ = run_cli(["drift", "--anchors", anchors, "--no-judge",
                                  "--out", out_path])
            self.assertEqual(rc, 0)
            self.assertIn("not assessed", out)
            self.assertIn("✅ OK", out)
            self.assertIn("deterministic-only drift check",
                          open(out_path, encoding="utf-8").read())

    def test_no_active_judges_noted_not_drift(self):
        # judged baseline, but the re-score finds no judge keys: judge side is
        # reported "not assessed" rather than flagging phantom flips
        with tempfile.TemporaryDirectory() as td:
            anchors = _bundle(td, judged=True)
            md, rc = run_drift(anchors, env={})  # no keys -> no active judges
        self.assertEqual(rc, 0)
        self.assertIn("not assessed", md)


class TestAnchorsConfig(unittest.TestCase):
    def test_missing_fields_fail_fast(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "anchors.yml")
            with open(p, "w") as f:
                yaml.safe_dump([{"runs": ["r.json"], "baseline": "b.json"}], f)
            with self.assertRaises(ConfigError) as cm:
                load_anchors(p)
            self.assertIn("'case' is required", str(cm.exception))

    def test_empty_anchors_fail_fast(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "anchors.yml")
            with open(p, "w") as f:
                yaml.safe_dump({"anchors": []}, f)
            with self.assertRaises(ConfigError):
                load_anchors(p)

    def test_paths_resolve_relative_to_anchors_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "anchors.yml")
            with open(p, "w") as f:
                yaml.safe_dump([{"case": "c.yml", "runs": ["r.json"],
                                 "baseline": "b.json"}], f)
            a = load_anchors(p)[0]
            self.assertEqual(a["case"], os.path.join(td, "c.yml"))
            self.assertEqual(a["baseline"], os.path.join(td, "b.json"))

    def test_baseline_dir_overrides_baseline_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "anchors.yml")
            with open(p, "w") as f:
                yaml.safe_dump([{"case": "c.yml", "runs": ["r.json"],
                                 "baseline": "b.json"}], f)
            a = load_anchors(p, baseline_dir="/base/line")[0]
            self.assertEqual(a["baseline"], "/base/line/b.json")
            self.assertEqual(a["case"], os.path.join(td, "c.yml"))  # unaffected

    def test_missing_anchors_file_exits_2_via_cli(self):
        rc, _, err = run_cli(["drift", "--anchors", "/no/such/anchors.yml"])
        self.assertEqual(rc, 2)
        self.assertIn("anchors file not found", err)


if __name__ == "__main__":
    unittest.main()
