import json
import unittest
from unittest.mock import patch

import difyeval.checks as registry
from difyeval import judge
from difyeval.core import ConfigError
from difyeval.engine import run_checks
from difyeval.validate import validate_case

from tests.helpers import (ScriptedTransport, dim_reply, fake_judges, make_case,
                           make_ctx, make_run, make_sample, rubric_reply)

DIMS = {"d1": "criterion one", "d2": "criterion two"}


class TestActiveJudges(unittest.TestCase):
    def test_activation_by_env_key_presence(self):
        js = judge.active_judges({"OPENAI_API_KEY": "sk-x"})
        self.assertEqual([j["name"] for j in js], ["openai"])
        self.assertEqual(js[0]["_model"], "gpt-4o")

    def test_only_filter(self):
        env = {"OPENAI_API_KEY": "k", "OPENROUTER_API_KEY": "k"}
        js = judge.active_judges(env, only=["openai"])
        self.assertEqual([j["name"] for j in js], ["openai"])

    def test_local_needs_base_and_model(self):
        self.assertEqual(judge.active_judges({"LOCAL_LLM_API_KEY": "k"}), [])
        js = judge.active_judges({"LOCAL_LLM_API_KEY": "k",
                                  "LOCAL_LLM_BASE_URL": "http://x",
                                  "LOCAL_LLM_MODEL": "m"})
        self.assertEqual([j["name"] for j in js], ["local"])

    def test_redacted_backends_carry_no_keys(self):
        js = judge.active_judges({"OPENAI_API_KEY": "sk-secret"})
        red = judge.redacted_backends(js)
        self.assertNotIn("sk-secret", json.dumps(red))
        self.assertEqual(red[0]["vendor"], "openai")


class TestPanel(unittest.TestCase):
    def test_median_spread_n_and_raw_retained(self):
        judges = fake_judges(3, vendors=("v1", "v2", "v3"))
        t = ScriptedTransport([
            (200, dim_reply({"d1": 0.2, "d2": 0.9})),
            (200, dim_reply({"d1": 0.4, "d2": 0.9})),
            (200, dim_reply({"d1": 0.6, "d2": None})),
        ])
        r = judge.run_panel("report body", DIMS, "", None, judges, transport=t)
        self.assertEqual(r["median"]["d1"], {"median": 0.4, "spread": 0.4, "n": 3})
        self.assertEqual(r["median"]["d2"]["n"], 2)
        self.assertTrue(r["independent_vendor"])
        self.assertIn("judge0", r["per_judge"])  # raw outputs retained
        self.assertEqual(r["per_judge"]["judge0"]["d1"]["rationale"], "r")

    def test_same_vendor_not_independent(self):
        judges = fake_judges(2, vendors=("v1", "v1"))
        t = ScriptedTransport([(200, dim_reply({"d1": 0.5})),
                               (200, dim_reply({"d1": 0.5}))])
        r = judge.run_panel("report", {"d1": "c"}, "", None, judges, transport=t)
        self.assertFalse(r["independent_vendor"])

    def test_empty_panel_or_report_returns_empty(self):
        self.assertEqual(judge.run_panel("r", DIMS, "", None, []), {})
        self.assertEqual(judge.run_panel("  ", DIMS, "", None, fake_judges(1)), {})

    @patch.object(judge, "RETRY_BACKOFF_S", 0)
    def test_retry_once_on_transport_error_then_success(self):
        t = ScriptedTransport([RuntimeError("conn reset"),
                               (200, dim_reply({"d1": 0.7}))])
        r = judge.run_panel("report", {"d1": "c"}, "", None, fake_judges(1), transport=t)
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(r["median"]["d1"]["median"], 0.7)

    @patch.object(judge, "RETRY_BACKOFF_S", 0)
    def test_retry_once_on_non_200_then_give_up(self):
        t = ScriptedTransport([(500, "err"), (503, "err")])
        r = judge.run_panel("report", {"d1": "c"}, "", None, fake_judges(1), transport=t)
        self.assertEqual(len(t.calls), 2)  # exactly one retry, then recorded as error
        self.assertEqual(r["per_judge"]["judge0"]["d1"]["score"], None)
        self.assertIn("HTTP 503", r["per_judge"]["judge0"]["d1"]["rationale"])

    def test_no_retry_on_json_parse_failure(self):
        t = ScriptedTransport([(200, "utter garbage, no json here")])
        r = judge.run_panel("report", {"d1": "c"}, "", None, fake_judges(1), transport=t)
        self.assertEqual(len(t.calls), 1)  # parse failure NEVER retried
        self.assertIn("unparseable", r["per_judge"]["judge0"]["d1"]["rationale"])

    def test_truncation_flagged_and_capped(self):
        big = "x" * (judge.REPORT_CAP + 500)
        t = ScriptedTransport([(200, dim_reply({"d1": 0.5}))])
        r = judge.run_panel(big, {"d1": "c"}, "", None, fake_judges(1), transport=t)
        self.assertTrue(r["truncated"])
        self.assertEqual(r["report_chars"], judge.REPORT_CAP + 500)
        body = t.calls[0][1]
        self.assertLess(len(body["messages"][1]["content"]), judge.REPORT_CAP + 2000)


class TestRubric(unittest.TestCase):
    ITEMS = [
        {"id": "a", "weight": "essential", "item": "Names the earliest source."},
        {"id": "b", "weight": "expected", "item": "Cross-checks a second domain."},
        {"id": "c", "weight": "bonus", "item": "States limitations."},
    ]

    def test_majority_vote_and_weighted_score(self):
        judges = fake_judges(3, vendors=("v1", "v2", "v3"))
        t = ScriptedTransport([
            (200, rubric_reply({"a": True, "b": True, "c": False})),
            (200, rubric_reply({"a": True, "b": False, "c": False})),
            (200, rubric_reply({"a": True, "b": True, "c": True})),
        ])
        r = judge.run_rubric("report", self.ITEMS, judges, "case_x", transport=t)
        by_id = {it["id"]: it for it in r["items"]}
        self.assertTrue(by_id["a"]["pass"])
        self.assertTrue(by_id["b"]["pass"])   # 2/3 majority
        self.assertFalse(by_id["c"]["pass"])  # 1/3
        self.assertEqual(r["score"], round(5 / 6, 3))  # (3+2)/(3+2+1)
        self.assertTrue(r["essential_ok"])
        self.assertEqual(by_id["a"]["votes"], "3/3")

    def test_essential_failure_flagged(self):
        t = ScriptedTransport([(200, rubric_reply({"a": False, "b": True, "c": True}))])
        r = judge.run_rubric("report", self.ITEMS, fake_judges(1), "case_x", transport=t)
        self.assertFalse(r["essential_ok"])

    def test_shuffle_is_deterministic_per_case_and_judge(self):
        t1 = ScriptedTransport([(200, rubric_reply({"a": True, "b": True, "c": True}))])
        t2 = ScriptedTransport([(200, rubric_reply({"a": True, "b": True, "c": True}))])
        judge.run_rubric("report", self.ITEMS, fake_judges(1), "case_x", transport=t1)
        judge.run_rubric("report", self.ITEMS, fake_judges(1), "case_x", transport=t2)
        self.assertEqual(t1.calls[0][1], t2.calls[0][1])  # identical prompt both times

    def test_shuffle_differs_across_judges_but_results_keyed_by_id(self):
        judges = fake_judges(2, vendors=("v1", "v2"))
        t = ScriptedTransport([(200, rubric_reply({"a": True, "b": True, "c": True}))] * 2)
        judge.run_rubric("report", self.ITEMS, judges, "case_x", transport=t)
        prompts = [c[1]["messages"][1]["content"] for c in t.calls]
        # both judges see all items; aggregation is order-independent
        for p in prompts:
            for iid in ("a", "b", "c"):
                self.assertIn(f"- {iid} [", p)

    def test_verdict_coercion_shapes(self):
        self.assertEqual(judge._rubric_verdict({"pass": "yes", "evidence": "e"}), (True, "e"))
        self.assertEqual(judge._rubric_verdict(True), (True, ""))
        self.assertEqual(judge._rubric_verdict("false"), (False, ""))
        self.assertEqual(judge._rubric_verdict(None), (False, ""))


class TestJudgeChecksIntegration(unittest.TestCase):
    def _case(self, checks):
        return make_case(checks=checks)

    def test_gate_true_on_judge_check_is_loud_config_error(self):
        case = self._case([{"id": "r", "type": "judge.rubric", "gate": True,
                            "items": [{"id": "a", "item": "x", "weight": "expected"}]}])
        with self.assertRaises(ConfigError) as cm:
            validate_case(case)
        self.assertIn("NEVER gates", str(cm.exception))

    def test_no_judge_skips_without_network(self):
        case = self._case([{"id": "r", "type": "judge.rubric",
                            "items": [{"id": "a", "item": "x", "weight": "essential"}]},
                           {"id": "t", "type": "judge.tradecraft"}])
        scores = run_checks(case, make_sample(), make_run("report"), make_ctx(case, judges=[]))
        for s in scores:
            self.assertIsNone(s.passed)
            self.assertTrue(s.advisory)
            self.assertTrue(s.detail.get("skipped"))
            self.assertIn("judge skipped", s.evidence)

    def test_tradecraft_default_dims_are_the_three(self):
        case = self._case([{"id": "t", "type": "judge.tradecraft"}])
        t = ScriptedTransport([(200, dim_reply({"uncertainty_expression": 0.8,
                                                "info_vs_judgment": 0.6,
                                                "alternatives_considered": 0.4}))])
        ctx = make_ctx(case, judges=fake_judges(1), transport=t)
        scores = run_checks(case, make_sample(), make_run("report"), ctx)
        s = scores[0]
        self.assertTrue(s.advisory)
        self.assertEqual(s.value, round((0.8 + 0.6 + 0.4) / 3, 3))
        sent = t.calls[0][1]["messages"][1]["content"]
        for d in ("uncertainty_expression", "info_vs_judgment", "alternatives_considered"):
            self.assertIn(d, sent)

    def test_tradecraft_unknown_dim_fails_fast(self):
        case = self._case([{"id": "t", "type": "judge.tradecraft",
                            "dims": ["sourcing_described"]}])
        with self.assertRaises(ConfigError):
            validate_case(case)

    def test_truncation_recorded_in_ctx(self):
        case = self._case([{"id": "r", "type": "judge.rubric",
                            "items": [{"id": "a", "item": "x", "weight": "expected"}]}])
        t = ScriptedTransport([(200, rubric_reply({"a": True}))])
        ctx = make_ctx(case, judges=fake_judges(1), transport=t)
        run_checks(case, make_sample(), make_run("y" * (judge.REPORT_CAP + 1)), ctx)
        self.assertEqual(len(ctx.truncations), 1)
        self.assertEqual(ctx.truncations[0]["check_id"], "r")

    def test_gt_derived_rubric_feeds_rubric_machinery(self):
        case = self._case([{"id": "gt", "type": "gt.derived_rubric",
                            "from": "reference.events",
                            "template": "Report covers: {what}", "weight": "essential"}])
        sample = make_sample(reference={"events": [{"what": "the first post"},
                                                   {"what": "the amplification"}]})
        t = ScriptedTransport([(200, rubric_reply({"gt_1": True, "gt_2": False}))])
        ctx = make_ctx(case, judges=fake_judges(1), transport=t)
        scores = run_checks(case, sample, make_run("report"), ctx)
        s = scores[0]
        self.assertTrue(s.advisory)
        self.assertFalse(s.detail["essential_ok"])  # gt_2 failed, weight essential
        self.assertIn("Report covers: the first post", t.calls[0][1]["messages"][1]["content"])

    def test_gt_derived_rubric_missing_reference_skips(self):
        case = self._case([{"id": "gt", "type": "gt.derived_rubric",
                            "from": "reference.events", "template": "{what}"}])
        ctx = make_ctx(case, judges=fake_judges(1),
                       transport=ScriptedTransport([]))
        scores = run_checks(case, make_sample(reference=None), make_run("report"), ctx)
        self.assertTrue(scores[0].detail.get("skipped"))
        self.assertIsNone(scores[0].passed)


if __name__ == "__main__":
    unittest.main()
