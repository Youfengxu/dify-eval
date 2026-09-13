import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import yaml

from difyeval import cli, review
from difyeval.core import accretions_path, load_case
from difyeval.review import register_review_extractor, wilson_ci


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


# ---- synthetic extractor: reads discovered accounts off a run node ----
@register_review_extractor("synth")
def synth_extractor(case, sample, run, experiment):
    out = []
    for a in (run.nodes.get("disc") or {}).get("outputs", {}).get("accounts", []):
        out.append({"path": "accounts.items",
                    "item": {"handle": a["handle"], "platform": a.get("platform")},
                    "evidence": {"n_findings": a.get("n", 1)}})
    return out


def write_bundle(td, handles=("@alpha", "@bravo", "@charlie"), base_items=None,
                 repeats=1):
    """A case.yml + a hand-built eval.json holding one sample with embedded runs."""
    run_path = os.path.join(td, "run.json")
    json.dump({"output": "hello"}, open(run_path, "w"))
    case = {"case_id": "rev_case",
            "runner": {"type": "file", "path": run_path},
            "dataset": [{"id": "s1", "inputs": {},
                         "reference": {"accounts": {"items": base_items or []}}}],
            "checks": [{"id": "c", "type": "output.nonempty", "gate": True}]}
    case_path = os.path.join(td, "case.yml")
    yaml.safe_dump(case, open(case_path, "w"))

    accounts = [{"handle": h, "platform": "tiktok", "n": i + 1}
                for i, h in enumerate(handles)]
    results = []
    for rep in range(1, repeats + 1):
        results.append({"sample_id": "s1", "repeat": rep,
                        "run": {"output": "report text",
                                "nodes": {"disc": {"outputs": {"accounts": accounts}}},
                                "usage": {}, "meta": {"status": "succeeded"},
                                "error": None},
                        "scores": []})
    ev = {"schema": "eval.json/1", "case": {"case_id": "rev_case"},
          "profile": "default", "results": results, "manifest": {}}
    eval_path = os.path.join(td, "eval.json")
    json.dump(ev, open(eval_path, "w"))
    return case_path, eval_path


def load_sheet(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


def set_decisions(sheet_path, decisions):
    """decisions: {normalized handle: (decision, reason)}"""
    sheet = load_sheet(sheet_path)
    for it in sheet["items"]:
        h = it["item"]["handle"].lstrip("@").lower()
        if h in decisions:
            d, reason = decisions[h]
            it["decision"] = d
            it["reason"] = reason
    yaml.safe_dump(sheet, open(sheet_path, "w"), sort_keys=False)


class TestWilsonCI(unittest.TestCase):
    def test_hand_computed_8_of_10(self):
        # p=0.8, z=1.96: center=0.716737..., half=0.226581... -> (0.490, 0.943)
        self.assertEqual(wilson_ci(8, 10), (0.49, 0.943))

    def test_zero_n(self):
        self.assertIsNone(wilson_ci(0, 0))

    def test_bounds_clamped(self):
        lo, hi = wilson_ci(10, 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


class TestPropose(unittest.TestCase):
    def test_sheet_contents(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            out = os.path.join(td, "sheet.yml")
            rc, msg, _ = run_cli(["review", "propose", "--eval", eval_path,
                                  "--case", case_path, "--out", out,
                                  "--extractor", "synth"])
            self.assertEqual(rc, 0)
            self.assertIn("3 candidate(s)", msg)
            sheet = load_sheet(out)
            self.assertEqual(sheet["review_sheet"], 1)
            self.assertEqual(sheet["case_id"], "rev_case")
            self.assertEqual(sheet["extractor"], "synth")
            self.assertEqual(sheet["runs"], ["s1:r1"])
            self.assertIn("skip | confirm | unvalidated | negative",
                          sheet["decisions"])
            self.assertEqual(len(sheet["items"]), 3)
            for it in sheet["items"]:
                self.assertEqual(it["decision"], "skip")
                self.assertEqual(it["sample"], "s1")
                self.assertEqual(it["path"], "accounts.items")
                self.assertIn("n_findings", it["evidence"])

    def test_candidates_already_in_reference_excluded_centrally(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(
                td, base_items=[{"handle": "@Alpha"}])  # identity-normalized dupe
            out = os.path.join(td, "sheet.yml")
            rc, _, _ = run_cli(["review", "propose", "--eval", eval_path,
                                "--case", case_path, "--out", out,
                                "--extractor", "synth"])
            self.assertEqual(rc, 0)
            handles = [it["item"]["handle"] for it in load_sheet(out)["items"]]
            self.assertEqual(handles, ["@bravo", "@charlie"])

    def test_repeat_runs_dedupe_in_sheet(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td, repeats=3)
            out = os.path.join(td, "sheet.yml")
            rc, _, _ = run_cli(["review", "propose", "--eval", eval_path,
                                "--case", case_path, "--out", out,
                                "--extractor", "synth"])
            self.assertEqual(rc, 0)
            sheet = load_sheet(out)
            self.assertEqual(sheet["runs"], ["s1:r1", "s1:r2", "s1:r3"])
            self.assertEqual(len(sheet["items"]), 3)  # not 9

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            out = os.path.join(td, "sheet.yml")
            open(out, "w").write("x")
            rc, _, err = run_cli(["review", "propose", "--eval", eval_path,
                                  "--case", case_path, "--out", out,
                                  "--extractor", "synth"])
            self.assertEqual(rc, 2)
            self.assertIn("refusing", err)

    def test_unknown_extractor_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            rc, _, err = run_cli(["review", "propose", "--eval", eval_path,
                                  "--case", case_path,
                                  "--out", os.path.join(td, "s.yml"),
                                  "--extractor", "no_such"])
            self.assertEqual(rc, 2)
            self.assertIn("unknown review extractor", err)
            self.assertIn("synth", err)  # lists registered

    def test_case_id_mismatch_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            ev = json.load(open(eval_path))
            ev["case"]["case_id"] = "other_case"
            json.dump(ev, open(eval_path, "w"))
            rc, _, err = run_cli(["review", "propose", "--eval", eval_path,
                                  "--case", case_path,
                                  "--out", os.path.join(td, "s.yml"),
                                  "--extractor", "synth"])
            self.assertEqual(rc, 2)
            self.assertIn("other_case", err)


class TestSampling(unittest.TestCase):
    HANDLES = tuple(f"@acct{i:02d}" for i in range(12))

    def test_seeded_subsample_is_deterministic(self):
        picks = []
        for i in range(2):
            with tempfile.TemporaryDirectory() as td:
                case_path, eval_path = write_bundle(td, handles=self.HANDLES)
                out = os.path.join(td, "sheet.yml")
                rc, _, _ = run_cli(["review", "propose", "--eval", eval_path,
                                    "--case", case_path, "--out", out,
                                    "--extractor", "synth",
                                    "--sample-k", "5", "--seed", "7"])
                self.assertEqual(rc, 0)
                sheet = load_sheet(out)
                self.assertEqual(sheet["sampled"]["k"], 5)
                self.assertEqual(sheet["sampled"]["population"], 12)
                self.assertEqual(sheet["sampled"]["seed"], 7)
                picks.append([it["item"]["handle"] for it in sheet["items"]])
        self.assertEqual(picks[0], picks[1])
        self.assertEqual(len(picks[0]), 5)
        # original candidate order preserved within the subsample
        self.assertEqual(picks[0], sorted(picks[0]))

    def test_different_seed_may_differ_and_k_ge_population_keeps_all(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)  # 3 candidates
            out = os.path.join(td, "sheet.yml")
            run_cli(["review", "propose", "--eval", eval_path, "--case", case_path,
                     "--out", out, "--extractor", "synth", "--sample-k", "5"])
            sheet = load_sheet(out)
            self.assertNotIn("sampled", sheet)   # k >= population: no sampling
            self.assertEqual(len(sheet["items"]), 3)


class TestApplyRoundTrip(unittest.TestCase):
    def _propose(self, td, case_path, eval_path, extra=()):
        out = os.path.join(td, "sheet.yml")
        rc, _, _ = run_cli(["review", "propose", "--eval", eval_path,
                            "--case", case_path, "--out", out,
                            "--extractor", "synth", *extra])
        self.assertEqual(rc, 0)
        return out

    def test_confirm_unvalidated_negative_land_with_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            sheet_path = self._propose(td, case_path, eval_path)
            set_decisions(sheet_path, {
                "alpha": ("confirm", ""),
                "bravo": ("unvalidated", ""),
                "charlie": ("negative", "state-media relay, not a buzzer")})
            rc, out, _ = run_cli(["review", "apply", "--sheet", sheet_path,
                                  "--case", case_path, "--analyst", "alice",
                                  "--date", "2026-07-23"])
            self.assertEqual(rc, 0)
            self.assertIn("confirmed (human_verified): 1", out)
            self.assertIn("unvalidated (advisory): 1", out)
            self.assertIn("negatives (must_reject): 1", out)

            acc = yaml.safe_load(open(accretions_path(case_path)))
            items = acc["samples"]["s1"]["reference"]["accounts.items"]
            by_handle = {i["handle"]: i for i in items}
            a = by_handle["@alpha"]
            self.assertEqual(a["provenance"], "human_verified")
            self.assertEqual(a["validated"], {"by": "alice", "date": "2026-07-23"})
            self.assertIn("#s1:r1", a["source_run"])
            b = by_handle["@bravo"]
            self.assertEqual(b["provenance"], "pipeline_observed_unvalidated")
            self.assertNotIn("validated", b)
            neg = acc["samples"]["s1"]["reference"]["must_reject"][0]
            self.assertEqual(neg["value"], "@charlie")
            self.assertEqual(neg["reason"], "state-media relay, not a buzzer")
            self.assertIn("#s1:r1", neg["source_run"])

            # the merged case now carries all three
            case = load_case(case_path)
            ref = case.samples[0].reference
            self.assertEqual(len(ref["accounts"]["items"]), 2)
            self.assertEqual(ref["must_reject"][0]["value"], "@charlie")
            # and the base file is untouched
            self.assertEqual(
                case.raw["dataset"][0]["reference"]["accounts"]["items"], [])

    def test_double_apply_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            sheet_path = self._propose(td, case_path, eval_path)
            set_decisions(sheet_path, {"alpha": ("confirm", ""),
                                       "charlie": ("negative", "r")})
            rc, _, _ = run_cli(["review", "apply", "--sheet", sheet_path,
                                "--case", case_path, "--analyst", "alice"])
            self.assertEqual(rc, 0)
            before = open(accretions_path(case_path), "rb").read()
            rc2, out2, _ = run_cli(["review", "apply", "--sheet", sheet_path,
                                    "--case", case_path, "--analyst", "alice"])
            self.assertEqual(rc2, 0)
            self.assertIn("nothing to apply", out2)
            self.assertEqual(open(accretions_path(case_path), "rb").read(), before)

    def test_propose_after_apply_excludes_accreted(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            sheet_path = self._propose(td, case_path, eval_path)
            set_decisions(sheet_path, {"alpha": ("confirm", ""),
                                       "charlie": ("negative", "r")})
            run_cli(["review", "apply", "--sheet", sheet_path,
                     "--case", case_path, "--analyst", "alice"])
            out2 = os.path.join(td, "sheet2.yml")
            rc, _, _ = run_cli(["review", "propose", "--eval", eval_path,
                                "--case", case_path, "--out", out2,
                                "--extractor", "synth"])
            self.assertEqual(rc, 0)
            handles = [it["item"]["handle"] for it in load_sheet(out2)["items"]]
            # @alpha accreted, @charlie banned via must_reject -> only @bravo left
            self.assertEqual(handles, ["@bravo"])

    def test_sampled_sheet_reports_wilson_precision(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(
                td, handles=tuple(f"@a{i:02d}" for i in range(10)))
            sheet_path = self._propose(td, case_path, eval_path,
                                       extra=("--sample-k", "4", "--seed", "1"))
            sheet = load_sheet(sheet_path)
            decisions = {}
            for i, it in enumerate(sheet["items"]):
                h = it["item"]["handle"].lstrip("@").lower()
                decisions[h] = ("confirm", "") if i < 3 else ("negative", "bad")
            set_decisions(sheet_path, decisions)
            rc, out, _ = run_cli(["review", "apply", "--sheet", sheet_path,
                                  "--case", case_path, "--analyst", "alice"])
            self.assertEqual(rc, 0)
            self.assertIn("precision estimate", out)
            self.assertIn("0.75 (3/4 decided", out)
            lo, hi = wilson_ci(3, 4)
            self.assertIn(f"95% Wilson CI {lo}-{hi}", out)

    def test_all_skip_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            sheet_path = self._propose(td, case_path, eval_path)
            rc, out, _ = run_cli(["review", "apply", "--sheet", sheet_path,
                                  "--case", case_path, "--analyst", "alice"])
            self.assertEqual(rc, 0)
            self.assertIn("nothing to apply", out)
            self.assertFalse(os.path.exists(accretions_path(case_path)))

    def test_invalid_decision_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            sheet_path = self._propose(td, case_path, eval_path)
            set_decisions(sheet_path, {"alpha": ("maybe", "")})
            rc, _, err = run_cli(["review", "apply", "--sheet", sheet_path,
                                  "--case", case_path, "--analyst", "alice"])
            self.assertEqual(rc, 2)
            self.assertIn("maybe", err)

    def test_missing_analyst_name_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, eval_path = write_bundle(td)
            sheet_path = self._propose(td, case_path, eval_path)
            rc, _, err = run_cli(["review", "apply", "--sheet", sheet_path,
                                  "--case", case_path, "--analyst", "  "])
            self.assertEqual(rc, 2)
            self.assertIn("analyst", err)

    def test_not_a_review_sheet_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            case_path, _ = write_bundle(td)
            bogus = os.path.join(td, "bogus.yml")
            yaml.safe_dump({"audit_sheet": 1}, open(bogus, "w"))
            rc, _, err = run_cli(["review", "apply", "--sheet", bogus,
                                  "--case", case_path, "--analyst", "a"])
            self.assertEqual(rc, 2)
            self.assertIn("review sheet", err)


class TestExtractorRegistry(unittest.TestCase):
    def test_signature_and_registration(self):
        self.assertIn("synth", review.registered_extractors())
        self.assertIs(review._EXTRACTORS["synth"], synth_extractor)

    def test_pack_module_registration_via_case_packs(self):
        # a pack listed in case packs registers its extractor at import time
        with tempfile.TemporaryDirectory() as td:
            pack = os.path.join(td, "revpack.py")
            open(pack, "w").write(
                "from difyeval.review import register_review_extractor\n"
                "@register_review_extractor('revpack')\n"
                "def extract(case, sample, run, experiment):\n"
                "    return [{'path': 'accounts.items',\n"
                "             'item': {'handle': '@from_pack'}, 'evidence': {}}]\n")
            case_path, eval_path = write_bundle(td)
            case = yaml.safe_load(open(case_path))
            case["packs"] = ["revpack"]
            yaml.safe_dump(case, open(case_path, "w"))
            import sys
            sys.path.insert(0, td)
            try:
                out = os.path.join(td, "sheet.yml")
                rc, _, _ = run_cli(["review", "propose", "--eval", eval_path,
                                    "--case", case_path, "--out", out,
                                    "--extractor", "revpack"])
            finally:
                sys.path.remove(td)
                sys.modules.pop("revpack", None)
                review._EXTRACTORS.pop("revpack", None)
            self.assertEqual(rc, 0)
            handles = [it["item"]["handle"] for it in load_sheet(out)["items"]]
            self.assertEqual(handles, ["@from_pack"])


if __name__ == "__main__":
    unittest.main()
