import unittest

from difyeval.core import SampleResult, Score
from difyeval.verdict import compute_metrics, compute_verdict, scope_statement

from tests.helpers import make_case, make_run, make_sample


def score(cid, passed, value=None, detail=None, advisory=False):
    return Score(check_id=cid, value=value, passed=passed, evidence="ev",
                 advisory=advisory, detail=detail or {})


def result(sid, repeat, scores):
    return SampleResult(sample_id=sid, repeat=repeat, run=make_run(), scores=scores)


CHECKS = [
    {"id": "g1", "type": "output.nonempty", "gate": True},
    {"id": "g2", "type": "node.json_count", "selector": "nodes.a.outputs.b",
     "min": 1, "gate": True},
    {"id": "rub", "type": "judge.rubric",
     "items": [{"id": "a", "item": "x", "weight": "essential"}]},
]


class TestVerdictTiers(unittest.TestCase):
    def _case(self, mode="reference"):
        return make_case(mode=mode, checks=CHECKS,
                         samples=[make_sample("s1"), make_sample("s2")])

    def test_fail_when_any_gate_fails(self):
        case = self._case()
        results = [result("s1", 1, [score("g1", True), score("g2", False),
                                    score("rub", True, detail={"essential_ok": True})])]
        v, _ = compute_verdict(case, results)
        self.assertEqual(v, "FAIL")

    def test_gated_none_counts_as_fail(self):
        case = self._case()
        results = [result("s1", 1, [score("g1", True), score("g2", None)])]
        v, _ = compute_verdict(case, results)
        self.assertEqual(v, "FAIL")

    def test_warn_on_essential_rubric_failure(self):
        case = self._case()
        results = [result("s1", 1, [score("g1", True), score("g2", True),
                                    score("rub", False, advisory=True,
                                          detail={"essential_ok": False})])]
        v, reason = compute_verdict(case, results)
        self.assertEqual(v, "WARN")
        self.assertIn("essential", reason)
        self.assertIn("rub", reason)

    def test_pass_when_gates_and_essentials_ok(self):
        case = self._case()
        results = [result("s1", 1, [score("g1", True), score("g2", True),
                                    score("rub", True, advisory=True,
                                          detail={"essential_ok": True})])]
        v, reason = compute_verdict(case, results)
        self.assertEqual(v, "PASS")
        self.assertIsNone(reason)

    def test_fail_outranks_warn(self):
        case = self._case()
        results = [result("s1", 1, [score("g1", False), score("g2", True),
                                    score("rub", False, advisory=True,
                                          detail={"essential_ok": False})])]
        v, _ = compute_verdict(case, results)
        self.assertEqual(v, "FAIL")

    def test_skipped_judge_never_warns(self):
        case = self._case()
        results = [result("s1", 1, [score("g1", True), score("g2", True),
                                    score("rub", None, advisory=True,
                                          detail={"skipped": True})])]
        v, _ = compute_verdict(case, results)
        self.assertEqual(v, "PASS")


class TestScope(unittest.TestCase):
    def test_reference_scope_lists_gates(self):
        s = scope_statement(make_case(mode="reference", checks=CHECKS))
        self.assertIn("reference mode", s)
        self.assertIn("g1, g2", s)

    def test_live_scope_disclaims_correctness(self):
        s = scope_statement(make_case(mode="live", checks=CHECKS))
        self.assertIn("process-soundness", s)
        self.assertIn("NOT established", s)

    def test_judge_check_never_in_gate_list(self):
        checks = list(CHECKS) + [{"id": "sneaky", "type": "judge.tradecraft"}]
        s = scope_statement(make_case(checks=checks))
        self.assertNotIn("sneaky", s)


class TestMetrics(unittest.TestCase):
    def _case(self):
        return make_case(checks=CHECKS,
                         samples=[make_sample("s1", repeats=2), make_sample("s2")])

    def test_pass_rate_over_runs(self):
        case = self._case()
        results = [
            result("s1", 1, [score("g1", True), score("g2", True, value=3.0)]),
            result("s1", 2, [score("g1", True), score("g2", False, value=1.0)]),
            result("s2", 1, [score("g1", True), score("g2", True, value=5.0)]),
        ]
        m = compute_metrics(case, results)
        self.assertEqual(m["pass_rate"]["passed"], 2)
        self.assertEqual(m["pass_rate"]["total"], 3)

    def test_honest_stats_n1_has_no_stderr(self):
        case = make_case(checks=CHECKS, samples=[make_sample("s1")])
        results = [result("s1", 1, [score("g2", True, value=4.0)])]
        m = compute_metrics(case, results)
        st = m["per_check"]["g2"]
        self.assertEqual(st["n"], 1)
        self.assertIn("value", st)
        self.assertNotIn("stderr", st)
        self.assertNotIn("mean", st)

    def test_stats_n2_mean_stderr(self):
        case = self._case()
        results = [
            result("s1", 1, [score("g2", True, value=3.0)]),
            result("s1", 2, [score("g2", True, value=5.0)]),
        ]
        m = compute_metrics(case, results)
        st = m["per_check"]["g2"]
        self.assertEqual(st["mean"], 4.0)
        self.assertEqual(st["n"], 2)
        self.assertIn("stderr", st)
        self.assertEqual(st["min"], 3.0)
        self.assertEqual(st["max"], 5.0)

    def test_repeat_consistency(self):
        case = self._case()
        results = [
            result("s1", 1, [score("g1", True), score("g2", True), score("rub", True)]),
            result("s1", 2, [score("g1", True), score("g2", False), score("rub", True)]),
            result("s2", 1, [score("g1", True)]),
        ]
        m = compute_metrics(case, results)
        self.assertEqual(m["repeat_consistency"]["s1"]["all_repeats_agree"],
                         round(2 / 3, 4))
        self.assertNotIn("s2", m["repeat_consistency"])  # single repeat: no claim

    def test_bool_values_excluded_from_numeric_stats(self):
        case = self._case()
        results = [result("s1", 1, [score("g1", True, value=True)])]
        m = compute_metrics(case, results)
        self.assertNotIn("g1", m["per_check"])


if __name__ == "__main__":
    unittest.main()
