import json
import os
import tempfile
import unittest

from tests.test_cli import run_cli


def score(cid, value=None, passed=None, detail=None):
    return {"check_id": cid, "value": value, "passed": passed, "evidence": "",
            "marginal": False, "advisory": False, "detail": detail or {}}


def result(scores, sample="s1", repeat=1, usage=None):
    return {"sample_id": sample, "repeat": repeat,
            "run": {"output": "r", "nodes": {}, "usage": usage or {},
                    "meta": {"status": "succeeded"}, "error": None},
            "scores": scores}


def make_eval(case_id="c1", profile="p", verdict="PASS", checks=(), results=(),
              rate=None):
    return {"schema": "eval.json/1",
            "case": {"case_id": case_id, "checks": list(checks)},
            "case_path": "case.yml", "profile": profile, "verdict": verdict,
            "warn_reason": None, "scope": "",
            "results": list(results),
            "metrics": {"pass_rate": {"passed": 1, "total": 1, "rate": rate,
                                      "basis": "gated"}},
            "manifest": {"scored_at": "2026-01-01T00:00:00+00:00"}}


CHECKS = [{"id": "gate1", "type": "output.nonempty", "gate": True},
          {"id": "recall", "type": "node.json_count"},
          {"id": "rubric", "type": "judge.rubric"},
          {"id": "craft", "type": "judge.tradecraft"}]


def rubric_detail(a_pass):
    return {"items": [{"id": "a", "weight": "essential", "item": "Names the source.",
                       "pass": a_pass, "votes": "2/2" if a_pass else "0/2",
                       "evidence": "quoted"}],
            "per_judge": {}}


def craft_detail(d1):
    return {"median": {"d1": {"median": d1, "spread": 0.1, "n": 2}}, "per_judge": {}}


def _pair(td):
    ea = make_eval(verdict="PASS", checks=CHECKS, rate=1.0, results=[result(
        [score("gate1", value=True, passed=True),
         score("recall", value=0.5, passed=True),
         score("rubric", value=1.0, passed=True, detail=rubric_detail(True)),
         score("craft", value=0.8, detail=craft_detail(0.8))],
        usage={"tokens": 100, "cost": {"USD": 1.0}})])
    eb = make_eval(profile="q", verdict="FAIL", checks=CHECKS, rate=0.5,
                   results=[result(
        [score("gate1", value=False, passed=False),
         score("recall", value=0.8, passed=True),
         score("rubric", value=0.0, passed=False, detail=rubric_detail(False)),
         score("craft", value=0.5, detail=craft_detail(0.5))],
        usage={"tokens": 150, "cost": {"USD": 2.5}})])
    pa, pb = os.path.join(td, "a.json"), os.path.join(td, "b.json")
    json.dump(ea, open(pa, "w"))
    json.dump(eb, open(pb, "w"))
    return pa, pb


class TestDiff(unittest.TestCase):
    def test_full_diff_content(self):
        with tempfile.TemporaryDirectory() as td:
            pa, pb = _pair(td)
            rc, out, _ = run_cli(["diff", pa, pb])
        self.assertEqual(rc, 0)
        self.assertIn("## Verdict: PASS → FAIL", out)
        # pass flip on the gate is highlighted
        gate_row = next(ln for ln in out.split("\n") if ln.startswith("| gate1 "))
        self.assertIn("FLIP", gate_row)
        # numeric delta on the value column
        recall_row = next(ln for ln in out.split("\n") if ln.startswith("| recall "))
        self.assertIn("| 0.5 | 0.8 | +0.3 |", recall_row)
        # rubric item flip listed with the item text
        self.assertIn("| rubric | a | pass | fail |", out)
        self.assertIn("Names the source.", out)
        # judge dims median delta
        self.assertIn("| d1 | 0.8 | 0.5 | -0.3 |", out)
        # metrics + cost/tokens deltas
        self.assertIn("gated pass-rate: 1 → 0.5", out)
        self.assertIn("| tokens | 100 | 150 | +50 |", out)
        self.assertIn("| cost USD | 1 | 2.5 | +1.5 |", out)

    def test_out_writes_file(self):
        with tempfile.TemporaryDirectory() as td:
            pa, pb = _pair(td)
            out_path = os.path.join(td, "diff.md")
            rc, out, _ = run_cli(["diff", pa, pb, "--out", out_path])
            self.assertEqual(rc, 0)
            written = open(out_path, encoding="utf-8").read()
            self.assertIn("## Verdict: PASS → FAIL", written)

    def test_case_id_mismatch_refused_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            pa = os.path.join(td, "a.json")
            pb = os.path.join(td, "b.json")
            json.dump(make_eval(case_id="one"), open(pa, "w"))
            json.dump(make_eval(case_id="two"), open(pb, "w"))
            rc, _, err = run_cli(["diff", pa, pb])
            self.assertEqual(rc, 2)
            self.assertIn("case_id mismatch", err)
            rc, out, _ = run_cli(["diff", pa, pb, "--force"])
            self.assertEqual(rc, 0)
            self.assertIn("FORCED", out)

    def test_unreadable_eval_exit_2(self):
        rc, _, err = run_cli(["diff", "/no/such/a.json", "/no/such/b.json"])
        self.assertEqual(rc, 2)
        self.assertIn("cannot read", err)

    def test_not_an_eval_json_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.json")
            json.dump({"hello": 1}, open(p, "w"))
            rc, _, err = run_cli(["diff", p, p])
            self.assertEqual(rc, 2)
            self.assertIn("not a difyeval eval.json", err)

    def test_identical_evals_no_flips(self):
        with tempfile.TemporaryDirectory() as td:
            ev = make_eval(checks=CHECKS, rate=1.0, results=[result(
                [score("gate1", value=True, passed=True),
                 score("rubric", value=1.0, passed=True, detail=rubric_detail(True))])])
            pa, pb = os.path.join(td, "a.json"), os.path.join(td, "b.json")
            json.dump(ev, open(pa, "w"))
            json.dump(ev, open(pb, "w"))
            rc, out, _ = run_cli(["diff", pa, pb])
        self.assertEqual(rc, 0)
        self.assertNotIn("FLIP", out)
        self.assertIn("no flips (1 shared item(s) agree)", out)
        self.assertIn("## Verdict: PASS → PASS", out)

    def test_check_only_in_one_side_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            ea = make_eval(checks=CHECKS[:1], results=[result(
                [score("gate1", value=True, passed=True),
                 score("extra_a", passed=True)])])
            eb = make_eval(checks=CHECKS[:1], results=[result(
                [score("gate1", value=True, passed=True)])])
            pa, pb = os.path.join(td, "a.json"), os.path.join(td, "b.json")
            json.dump(ea, open(pa, "w"))
            json.dump(eb, open(pb, "w"))
            rc, out, _ = run_cli(["diff", pa, pb])
        self.assertEqual(rc, 0)
        row = next(ln for ln in out.split("\n") if ln.startswith("| extra_a "))
        self.assertIn("only in A", row)


if __name__ == "__main__":
    unittest.main()
