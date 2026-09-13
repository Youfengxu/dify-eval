import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import yaml

from difyeval import cli

from tests.helpers import DEMO_CASE


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestCliSmoke(unittest.TestCase):
    def test_new_case_then_validate(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "case.yml")
            rc, out, _ = run_cli(["new-case", path])
            self.assertEqual(rc, 0)
            rc, out, _ = run_cli(["validate", path])
            self.assertEqual(rc, 0)
            self.assertIn("OK:", out)

    def test_new_case_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "case.yml")
            run_cli(["new-case", path])
            rc, _, err = run_cli(["new-case", path])
            self.assertEqual(rc, 2)
            self.assertIn("refusing", err)

    def test_run_demo_no_judge(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = run_cli(["run", DEMO_CASE, "--no-judge", "--out-dir", td])
            self.assertEqual(rc, 0)
            self.assertIn("verdict: PASS", out)
            self.assertTrue(os.path.exists(os.path.join(
                td, "text_source_discovery_demo__default__scorecard.md")))

    def test_score_with_explicit_run_files(self):
        demo_dir = os.path.dirname(DEMO_CASE)
        r1 = os.path.join(demo_dir, "fixtures", "recall_hoax", "run.json")
        r2 = os.path.join(demo_dir, "fixtures", "tariff_rumor", "run.json")
        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = run_cli(["score", DEMO_CASE, "--no-judge",
                                  "--run", r1, "--run", r2, "--out-dir", td])
            self.assertEqual(rc, 0)

    def test_unknown_check_type_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            case = {"case_id": "bad", "runner": {"type": "file", "path": "x.json"},
                    "dataset": [{"id": "s1", "inputs": {}}],
                    "checks": [{"id": "c", "type": "no.such_check"}]}
            path = os.path.join(td, "case.yml")
            yaml.safe_dump(case, open(path, "w"))
            rc, _, err = run_cli(["validate", path])
            self.assertEqual(rc, 2)
            self.assertIn("unknown check type", err)
            self.assertIn("output.nonempty", err)  # lists registered types

    def test_gate_on_judge_check_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            case = {"case_id": "bad", "runner": {"type": "file", "path": "x.json"},
                    "dataset": [{"id": "s1", "inputs": {}}],
                    "checks": [{"id": "r", "type": "judge.rubric", "gate": True,
                                "items": [{"id": "a", "item": "t"}]}]}
            path = os.path.join(td, "case.yml")
            yaml.safe_dump(case, open(path, "w"))
            rc, _, err = run_cli(["validate", path])
            self.assertEqual(rc, 2)
            self.assertIn("NEVER gates", err)

    def test_verdict_mismatch_exits_1(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            json.dump({"output": ""}, open(run_path, "w"))  # empty -> gate fails
            case = {"case_id": "anchor_fail", "runner": {"type": "file", "path": run_path},
                    "dataset": [{"id": "s1", "inputs": {}}],
                    "checks": [{"id": "c", "type": "output.nonempty", "gate": True}]}
            path = os.path.join(td, "case.yml")
            yaml.safe_dump(case, open(path, "w"))
            rc, out, _ = run_cli(["run", path, "--no-judge", "--out-dir", td])
            self.assertEqual(rc, 1)  # FAIL vs default expectation PASS
            self.assertIn("MISMATCH", out)

    def test_expect_verdict_fail_anchor_exits_0(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            json.dump({"output": ""}, open(run_path, "w"))
            case = {"case_id": "anchor", "runner": {"type": "file", "path": run_path},
                    "dataset": [{"id": "s1", "inputs": {}}],
                    "checks": [{"id": "c", "type": "output.nonempty", "gate": True}],
                    "expect_verdict": "FAIL"}
            path = os.path.join(td, "case.yml")
            yaml.safe_dump(case, open(path, "w"))
            rc, out, _ = run_cli(["run", path, "--no-judge", "--out-dir", td])
            self.assertEqual(rc, 0)  # anchor case: FAIL was expected

    def test_save_fixture_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = run_cli(["run", DEMO_CASE, "--no-judge", "--out-dir", td,
                                  "--save-fixture"])
            self.assertEqual(rc, 0)
            fx = os.path.join(td, "fixtures", "recall_hoax", "run-1.json")
            self.assertTrue(os.path.exists(fx))
            saved = json.load(open(fx))
            self.assertIn("output", saved)  # native Run shape
            # the saved fixture scores identically via --runs-dir
            rc2, out2, _ = run_cli(["score", DEMO_CASE, "--no-judge",
                                    "--runs-dir", os.path.join(td, "fixtures"),
                                    "--out-dir", os.path.join(td, "again")])
            self.assertEqual(rc2, 0)

    def test_new_pack_scaffold_registers_and_runs(self):
        with tempfile.TemporaryDirectory() as td:
            pack_path = os.path.join(td, "mypack.py")
            rc, _, _ = run_cli(["new-pack", pack_path])
            self.assertEqual(rc, 0)
            run_path = os.path.join(td, "run.json")
            json.dump({"output": "report names alice and bob",
                       "nodes": {"accounts": {"outputs": {
                           "accounts_json": json.dumps([{"username": "alice"},
                                                        {"username": "bob"}])}}}},
                      open(run_path, "w"))
            case = {"case_id": "packcase", "packs": ["mypack"],
                    "runner": {"type": "file", "path": run_path},
                    "dataset": [{"id": "s1", "inputs": {}}],
                    "checks": [{"id": "acc", "gate": True,
                                "type": "mypack.report_names_every_account",
                                "selector": "nodes.accounts.outputs.accounts_json"}]}
            case_path = os.path.join(td, "case.yml")
            yaml.safe_dump(case, open(case_path, "w"))
            sys.path.insert(0, td)
            try:
                rc, out, _ = run_cli(["run", case_path, "--no-judge", "--out-dir", td])
            finally:
                sys.path.remove(td)
                sys.modules.pop("mypack", None)
            self.assertEqual(rc, 0)
            self.assertIn("verdict: PASS", out)


if __name__ == "__main__":
    unittest.main()
