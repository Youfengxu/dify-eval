import json
import os
import tempfile
import unittest

import yaml

from difyeval.audit import applied_path_for, cohens_kappa

from tests.test_cli import run_cli
from tests.test_diff import make_eval, result, score

RUBRIC_DETAIL = {
    "items": [
        {"id": "a", "weight": "essential", "item": "Names the earliest source.",
         "pass": True, "votes": "2/2", "evidence": "\"first posted by X\""},
        {"id": "b", "weight": "expected", "item": "Cross-checks a second domain.",
         "pass": False, "votes": "1/2", "evidence": ""},
    ],
    "per_judge": {
        "j-openai": {"a": {"pass": True, "evidence": "x"}, "b": {"pass": True}},
        "j-kimi": {"a": {"pass": True}, "b": {"pass": False}},
    },
}

CRAFT_DETAIL = {
    "median": {"d1": {"median": 0.8, "spread": 0.2, "n": 2}},
    "per_judge": {"j-openai": {"d1": {"score": 0.9, "rationale": "r"}},
                  "j-kimi": {"d1": {"score": 0.7, "rationale": "r"}}},
}


def judged_eval():
    ev = make_eval(checks=[{"id": "rubric", "type": "judge.rubric"},
                           {"id": "craft", "type": "judge.tradecraft"}],
                   results=[result(
                       [score("rubric", value=0.6, passed=True, detail=RUBRIC_DETAIL),
                        score("craft", value=0.8, detail=CRAFT_DETAIL)])])
    ev["manifest"]["judge_backends"] = [
        {"name": "j-openai", "vendor": "openai", "model": "gpt-4o"},
        {"name": "j-kimi", "vendor": "moonshot", "model": "kimi-k2.6"}]
    return ev


def _write_eval(td, ev, name="eval.json"):
    p = os.path.join(td, name)
    with open(p, "w") as f:
        json.dump(ev, f)
    return p


class TestPropose(unittest.TestCase):
    def test_sheet_lists_every_item_and_dim_with_human_slots(self):
        with tempfile.TemporaryDirectory() as td:
            ev_path = _write_eval(td, judged_eval())
            sheet_path = os.path.join(td, "sheet.yml")
            rc, out, _ = run_cli(["audit", "propose", "--eval", ev_path,
                                  "--out", sheet_path])
            self.assertEqual(rc, 0)
            raw = open(sheet_path, encoding="utf-8").read()
            self.assertIn("# agree | disagree", raw)  # fill instructions inline
            sheet = yaml.safe_load(raw)
        self.assertEqual(sheet["audit_sheet"], 1)
        self.assertEqual(sheet["case_id"], "c1")
        self.assertEqual(len(sheet["rubric"]), 2)
        a = sheet["rubric"][0]
        self.assertEqual(a["item"], "a")
        self.assertEqual(a["panel"], "pass")
        self.assertEqual(a["votes"], "2/2")
        self.assertEqual(a["evidence"], '"first posted by X"')
        self.assertEqual(a["judge_votes"], {"j-kimi": True, "j-openai": True})
        self.assertIsNone(a["human"])
        b = sheet["rubric"][1]
        self.assertEqual(b["panel"], "fail")
        self.assertEqual(b["judge_votes"], {"j-kimi": False, "j-openai": True})
        self.assertEqual(len(sheet["dims"]), 1)
        d = sheet["dims"][0]
        self.assertEqual((d["dim"], d["median"], d["spread"], d["n"]),
                         ("d1", 0.8, 0.2, 2))
        self.assertEqual(d["judge_scores"], {"j-kimi": 0.7, "j-openai": 0.9})
        self.assertIsNone(d["human"])

    def test_propose_on_no_judge_eval_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            ev = make_eval(results=[result([score("gate1", passed=True)])])
            ev_path = _write_eval(td, ev)
            rc, _, err = run_cli(["audit", "propose", "--eval", ev_path,
                                  "--out", os.path.join(td, "s.yml")])
            self.assertEqual(rc, 2)
            self.assertIn("nothing to audit", err)

    def test_propose_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            ev_path = _write_eval(td, judged_eval())
            sheet_path = os.path.join(td, "sheet.yml")
            run_cli(["audit", "propose", "--eval", ev_path, "--out", sheet_path])
            rc, _, err = run_cli(["audit", "propose", "--eval", ev_path,
                                  "--out", sheet_path])
            self.assertEqual(rc, 2)
            self.assertIn("refusing", err)


def _proposed_sheet(td):
    ev_path = _write_eval(td, judged_eval())
    sheet_path = os.path.join(td, "sheet.yml")
    rc, _, _ = run_cli(["audit", "propose", "--eval", ev_path, "--out", sheet_path])
    assert rc == 0
    return sheet_path


def _fill(sheet_path, rubric=("agree", "disagree"), dims=("agree",)):
    sheet = yaml.safe_load(open(sheet_path, encoding="utf-8"))
    for e, h in zip(sheet["rubric"], rubric):
        e["human"] = h
    for e, h in zip(sheet["dims"], dims):
        e["human"] = h
    with open(sheet_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sheet, f, sort_keys=False)
    return sheet_path


class TestApply(unittest.TestCase):
    def test_apply_stamps_and_refuses_reapply(self):
        with tempfile.TemporaryDirectory() as td:
            sheet_path = _fill(_proposed_sheet(td))
            rc, out, _ = run_cli(["audit", "apply", "--sheet", sheet_path,
                                  "--analyst", "Ana Lyst", "--date", "2026-07-09"])
            self.assertEqual(rc, 0)
            applied = applied_path_for(sheet_path)
            self.assertTrue(applied.endswith("sheet.applied.yml"))
            data = yaml.safe_load(open(applied, encoding="utf-8"))
            self.assertEqual(data["applied"]["analyst"], "Ana Lyst")
            self.assertEqual(data["applied"]["date"], "2026-07-09")
            self.assertTrue(data["applied"]["source_sha256"])
            self.assertEqual(data["rubric"][0]["human"], "agree")
            # immutable-ish: a second apply is refused
            rc, _, err = run_cli(["audit", "apply", "--sheet", sheet_path,
                                  "--analyst", "Bob"])
            self.assertEqual(rc, 2)
            self.assertIn("refusing to re-apply", err)

    def test_apply_refuses_applied_file_as_input(self):
        with tempfile.TemporaryDirectory() as td:
            sheet_path = _fill(_proposed_sheet(td))
            run_cli(["audit", "apply", "--sheet", sheet_path, "--analyst", "Ana"])
            rc, _, err = run_cli(["audit", "apply",
                                  "--sheet", applied_path_for(sheet_path),
                                  "--analyst", "Ana"])
            self.assertEqual(rc, 2)
            self.assertIn("already an applied sheet", err)

    def test_apply_rejects_unfilled_or_invalid_human(self):
        with tempfile.TemporaryDirectory() as td:
            sheet_path = _proposed_sheet(td)  # humans still null
            rc, _, err = run_cli(["audit", "apply", "--sheet", sheet_path,
                                  "--analyst", "Ana"])
            self.assertEqual(rc, 2)
            self.assertIn("agree or disagree", err)
        with tempfile.TemporaryDirectory() as td:
            sheet_path = _fill(_proposed_sheet(td), rubric=("maybe", "agree"))
            rc, _, err = run_cli(["audit", "apply", "--sheet", sheet_path,
                                  "--analyst", "Ana"])
            self.assertEqual(rc, 2)
            self.assertIn("'maybe'", err)

    def test_apply_normalizes_case_and_whitespace(self):
        with tempfile.TemporaryDirectory() as td:
            sheet_path = _fill(_proposed_sheet(td), rubric=("Agree ", "DISAGREE"))
            rc, _, _ = run_cli(["audit", "apply", "--sheet", sheet_path,
                                "--analyst", "Ana"])
            self.assertEqual(rc, 0)
            data = yaml.safe_load(open(applied_path_for(sheet_path), encoding="utf-8"))
            self.assertEqual([e["human"] for e in data["rubric"]],
                             ["agree", "disagree"])


class TestKappa(unittest.TestCase):
    def test_hand_computed_example(self):
        # textbook 2x2: a=20 both-yes, b=5, c=10, d=15, n=50
        # po = 35/50 = 0.7; pe = (25*30 + 25*20)/2500 = 0.5; kappa = 0.4
        pairs = ([(True, True)] * 20 + [(True, False)] * 5
                 + [(False, True)] * 10 + [(False, False)] * 15)
        k, reason = cohens_kappa(pairs)
        self.assertIsNone(reason)
        self.assertEqual(k, 0.4)

    def test_degenerate_marginals_no_variance(self):
        k, reason = cohens_kappa([(True, True)] * 12)
        self.assertIsNone(k)
        self.assertEqual(reason, "n/a (no variance)")
        k, reason = cohens_kappa([(False, False)] * 12)
        self.assertEqual(reason, "n/a (no variance)")

    def test_empty(self):
        k, reason = cohens_kappa([])
        self.assertIsNone(k)

    def test_perfect_disagreement(self):
        # a=0, b=6, c=6, d=0 -> po=0, pe=0.5, kappa=-1
        k, reason = cohens_kappa([(True, False)] * 6 + [(False, True)] * 6)
        self.assertEqual(k, -1.0)


def _sheet_entry(item, panel, human, votes):
    return {"check": "rubric", "sample": "s1", "repeat": 1, "item": item,
            "weight": "expected", "text": "t", "panel": panel,
            "votes": "1/1", "evidence": "", "judge_votes": votes,
            "human": human, "note": ""}


def _dim_entry(dim, human):
    return {"check": "craft", "sample": "s1", "repeat": 1, "dim": dim,
            "median": 0.8, "spread": 0.1, "n": 2, "judge_scores": {},
            "human": human, "note": ""}


def _write_applied(td, name, rubric, dims, analyst="Ana"):
    sheet = {"audit_sheet": 1, "case_id": "c1", "profile": "p",
             "eval_path": "e.json", "scored_at": "T", "judge_backends": [],
             "rubric": rubric, "dims": dims,
             "applied": {"analyst": analyst, "date": "2026-07-09",
                         "source_sheet": name, "source_sha256": "x"}}
    p = os.path.join(td, name + ".applied.yml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(sheet, f, sort_keys=False)
    return p


class TestReport(unittest.TestCase):
    def _agreement_sheet(self, td):
        """12 rubric items engineered so backend jx forms the 2x2
        a=4, b=2, c=3, d=3 vs the human truth:
        po = 7/12; pe = (6*7 + 6*5)/144 = 0.5; kappa = 1/6 = 0.1667."""
        rubric = []
        # truth True x 7: panel pass + agree (5), panel fail + disagree (2)
        for i in range(5):
            rubric.append(_sheet_entry(f"t{i}", "pass", "agree",
                                       {"jx": i < 4}))       # 4x (T,T), 1x (F,T)
        for i in range(2):
            rubric.append(_sheet_entry(f"u{i}", "fail", "disagree",
                                       {"jx": False}))       # 2x (F,T)
        # truth False x 5: panel fail + agree (3), panel pass + disagree (2)
        for i in range(3):
            rubric.append(_sheet_entry(f"v{i}", "fail", "agree",
                                       {"jx": i < 2}))       # 2x (T,F), 1x (F,F)
        for i in range(2):
            rubric.append(_sheet_entry(f"w{i}", "pass", "disagree",
                                       {"jx": False}))       # 2x (F,F)
        dims = [_dim_entry("d1", "agree"), _dim_entry("d1", "agree"),
                _dim_entry("d2", "agree"), _dim_entry("d2", "disagree")]
        return _write_applied(td, "s1", rubric, dims)

    def test_agreement_and_kappa_per_backend(self):
        with tempfile.TemporaryDirectory() as td:
            self._agreement_sheet(td)
            rc, out, _ = run_cli(["audit", "report",
                                  "--sheets", os.path.join(td, "*.applied.yml")])
        self.assertEqual(rc, 0)
        # jx: agreement (4+3)/12 = 58.3%, kappa 0.1667 (n=12 >= 10)
        jx_row = next(ln for ln in out.split("\n") if ln.startswith("| jx "))
        self.assertIn("| 12 | 58.3% | 0.1667 |", jx_row)
        # the panel pseudo-backend agrees exactly when the human said agree: 8/12
        panel_row = next(ln for ln in out.split("\n")
                         if ln.startswith("| PANEL (majority) "))
        self.assertIn("| 12 | 66.7% |", panel_row)
        # dims: % only, per dim + overall, n everywhere
        self.assertIn("| d1 | 2 | 100.0% |", out)
        self.assertIn("| d2 | 2 | 50.0% |", out)
        self.assertIn("| **all dims** | 4 | 75.0% |", out)
        self.assertIn("n items judged: 12", out)

    def test_insufficient_n_refuses_kappa(self):
        with tempfile.TemporaryDirectory() as td:
            rubric = [_sheet_entry(f"i{i}", "pass", "agree", {"jy": True})
                      for i in range(4)]
            rubric.append(_sheet_entry("i4", "fail", "agree", {"jy": False}))
            _write_applied(td, "small", rubric, [])
            rc, out, _ = run_cli(["audit", "report",
                                  "--sheets", os.path.join(td, "*.applied.yml")])
        self.assertEqual(rc, 0)
        jy_row = next(ln for ln in out.split("\n") if ln.startswith("| jy "))
        self.assertIn("n/a (insufficient n: 5 < 10)", jy_row)
        self.assertIn("100.0%", jy_row)  # % agreement still reported

    def test_degenerate_marginals_reported(self):
        with tempfile.TemporaryDirectory() as td:
            rubric = [_sheet_entry(f"i{i}", "pass", "agree", {"jz": True})
                      for i in range(12)]
            _write_applied(td, "flat", rubric, [])
            rc, out, _ = run_cli(["audit", "report",
                                  "--sheets", os.path.join(td, "*.applied.yml")])
        self.assertEqual(rc, 0)
        jz_row = next(ln for ln in out.split("\n") if ln.startswith("| jz "))
        self.assertIn("n/a (no variance)", jz_row)

    def test_missing_votes_excluded_and_noted(self):
        with tempfile.TemporaryDirectory() as td:
            rubric = [_sheet_entry("i0", "pass", "agree", {"jx": True, "jy": None}),
                      _sheet_entry("i1", "pass", "agree", {"jx": True, "jy": True})]
            _write_applied(td, "gaps", rubric, [])
            rc, out, _ = run_cli(["audit", "report",
                                  "--sheets", os.path.join(td, "*.applied.yml")])
        self.assertEqual(rc, 0)
        jy_row = next(ln for ln in out.split("\n") if ln.startswith("| jy "))
        self.assertIn("| 1 |", jy_row)  # the None vote is excluded from n
        self.assertIn("jy had 1 item(s) with no usable vote", out)

    def test_report_refuses_unapplied_sheet(self):
        with tempfile.TemporaryDirectory() as td:
            sheet_path = _proposed_sheet(td)
            rc, _, err = run_cli(["audit", "report", "--sheets", sheet_path])
            self.assertEqual(rc, 2)
            self.assertIn("not an APPLIED sheet", err)

    def test_report_empty_glob_exits_2(self):
        rc, _, err = run_cli(["audit", "report", "--sheets", "/no/such/*.yml"])
        self.assertEqual(rc, 2)
        self.assertIn("matched no files", err)

    def test_report_out_writes_file(self):
        with tempfile.TemporaryDirectory() as td:
            self._agreement_sheet(td)
            out_path = os.path.join(td, "agreement.md")
            rc, _, _ = run_cli(["audit", "report",
                                "--sheets", os.path.join(td, "*.applied.yml"),
                                "--out", out_path])
            self.assertEqual(rc, 0)
            self.assertIn("judge↔human agreement",
                          open(out_path, encoding="utf-8").read())


class TestRoundTrip(unittest.TestCase):
    def test_propose_apply_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            sheet_path = _fill(_proposed_sheet(td),
                               rubric=("agree", "disagree"), dims=("agree",))
            rc, _, _ = run_cli(["audit", "apply", "--sheet", sheet_path,
                                "--analyst", "Ana", "--date", "2026-07-09"])
            self.assertEqual(rc, 0)
            rc, out, _ = run_cli(["audit", "report",
                                  "--sheets", os.path.join(td, "*.applied.yml")])
            self.assertEqual(rc, 0)
            self.assertIn("analysts: Ana", out)
            # item a: panel pass, human agree -> truth T; both judges voted T
            # item b: panel fail, human disagree -> truth T; kimi F, openai T
            kimi = next(ln for ln in out.split("\n") if ln.startswith("| j-kimi "))
            self.assertIn("| 2 | 50.0% |", kimi)
            openai = next(ln for ln in out.split("\n") if ln.startswith("| j-openai "))
            self.assertIn("| 2 | 100.0% |", openai)
            self.assertIn("insufficient n", kimi)  # n=2 < 10, honest-stats


if __name__ == "__main__":
    unittest.main()
