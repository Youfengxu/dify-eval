import json
import unittest

import difyeval.checks as registry
from difyeval.checks.gt import derive_items
from difyeval.core import ConfigError
from difyeval.engine import run_checks

from tests.helpers import make_case, make_ctx, make_run, make_sample


def run_one(check_type, spec_extra, run, sample=None, ctx=None, gate=False):
    spec = {"id": "c1", "type": check_type, **({"gate": True} if gate else {}), **spec_extra}
    cd = registry.get(check_type)
    return cd.fn(spec, sample or make_sample(), run, ctx or make_ctx())


class TestOutputChecks(unittest.TestCase):
    def test_nonempty_pass(self):
        s = run_one("output.nonempty", {}, make_run("report text"))
        self.assertTrue(s.passed)
        self.assertIn("chars", s.evidence)

    def test_nonempty_fail_mentions_run_error(self):
        s = run_one("output.nonempty", {}, make_run("", error="boom"), gate=True)
        self.assertFalse(s.passed)
        self.assertFalse(s.advisory)
        self.assertIn("boom", s.evidence)

    def test_contains_all_present(self):
        s = run_one("output.contains", {"values": ["foo", "bar"]}, make_run("foo bar baz"))
        self.assertTrue(s.passed)
        self.assertEqual(s.value, 1.0)

    def test_contains_missing_listed(self):
        s = run_one("output.contains", {"values": ["foo", "zap"]}, make_run("foo"), gate=True)
        self.assertFalse(s.passed)
        self.assertEqual(s.value, 0.5)
        self.assertIn("'zap'", s.evidence)

    def test_contains_values_from_reference(self):
        sample = make_sample(reference={"expected_domains": ["newswire.example"]})
        s = run_one("output.contains", {"values_from": "reference.expected_domains"},
                    make_run("see newswire.example today"), sample=sample)
        self.assertTrue(s.passed)

    def test_contains_values_from_missing_path_degrades(self):
        sample = make_sample(reference={})
        s = run_one("output.contains", {"values_from": "reference.nope"},
                    make_run("x"), sample=sample, gate=True)
        self.assertFalse(s.passed)  # gated degrade -> False
        self.assertIn("selector not found", s.evidence)

    def test_contains_degrade_advisory_is_none(self):
        sample = make_sample(reference={})
        s = run_one("output.contains", {"values_from": "reference.nope"},
                    make_run("x"), sample=sample, gate=False)
        self.assertIsNone(s.passed)

    def test_not_contains(self):
        ok = run_one("output.not_contains", {"values": ["secret"]}, make_run("clean"))
        self.assertTrue(ok.passed)
        bad = run_one("output.not_contains", {"values": ["secret"]}, make_run("a secret!"))
        self.assertFalse(bad.passed)
        self.assertIn("'secret'", bad.evidence)

    def test_regex(self):
        s = run_one("output.regex", {"pattern": r"\d{4}-\d{2}-\d{2}"},
                    make_run("published 2026-07-04 ok"))
        self.assertTrue(s.passed)
        s2 = run_one("output.regex", {"pattern": r"XYZ\d+"}, make_run("nope"))
        self.assertFalse(s2.passed)

    def test_json_schema_pass_and_fail(self):
        schema = {"type": "object", "required": ["a"],
                  "properties": {"a": {"type": "array", "items": {"type": "integer"}},
                                 "b": {"type": "string"}}}
        good = run_one("output.json_schema", {"schema": schema},
                       make_run(json.dumps({"a": [1, 2], "b": "x"})))
        self.assertTrue(good.passed)
        bad = run_one("output.json_schema", {"schema": schema},
                      make_run(json.dumps({"b": 5})), gate=True)
        self.assertFalse(bad.passed)
        self.assertIn("missing required key 'a'", bad.evidence)
        self.assertIn("expected string", bad.evidence)

    def test_json_schema_bool_is_not_integer(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        s = run_one("output.json_schema", {"schema": schema}, make_run('{"n": true}'))
        self.assertFalse(s.passed)

    def test_json_schema_non_json_output_degrades(self):
        s = run_one("output.json_schema", {"schema": {"type": "object"}},
                    make_run("plain prose"), gate=True)
        self.assertFalse(s.passed)
        self.assertIn("not JSON", s.evidence)

    def test_json_schema_via_selector(self):
        run = make_run("x", nodes={"n": {"outputs": {"j": '{"k": 1}'}}})
        s = run_one("output.json_schema",
                    {"schema": {"type": "object", "required": ["k"]},
                     "selector": "nodes.n.outputs.j"}, run)
        self.assertTrue(s.passed)


class TestNodeChecks(unittest.TestCase):
    RUN = make_run("text", nodes={
        "sources": {"status": "succeeded", "outputs": {
            "sources_json": json.dumps([{"url": "u1"}, {"url": "u2"}, {"url": "u3"}]),
            "summary_json": json.dumps({"earliest_url": "u1", "count": 3}),
        }},
    })

    def test_exists(self):
        self.assertTrue(run_one("node.exists", {"node": "sources"}, self.RUN).passed)
        miss = run_one("node.exists", {"node": "ghost"}, self.RUN)
        self.assertFalse(miss.passed)
        self.assertIn("sources", miss.evidence)

    def test_json_count_min_pass_marginal(self):
        s = run_one("node.json_count",
                    {"selector": "nodes.sources.outputs.sources_json", "min": 3}, self.RUN)
        self.assertTrue(s.passed)
        self.assertEqual(s.value, 3.0)
        self.assertTrue(s.marginal)  # exactly at threshold -> within 10%

    def test_json_count_not_marginal_when_clear(self):
        s = run_one("node.json_count",
                    {"selector": "nodes.sources.outputs.sources_json", "min": 1}, self.RUN)
        self.assertTrue(s.passed)
        self.assertFalse(s.marginal)

    def test_json_count_max_fail(self):
        s = run_one("node.json_count",
                    {"selector": "nodes.sources.outputs.sources_json", "max": 2},
                    self.RUN, gate=True)
        self.assertFalse(s.passed)

    def test_json_count_missing_selector_degrades(self):
        s = run_one("node.json_count", {"selector": "nodes.ghost.outputs.x", "min": 1},
                    self.RUN, gate=True)
        self.assertFalse(s.passed)
        self.assertIn("selector not found", s.evidence)

    def test_json_count_undecodable_degrades(self):
        run = make_run("x", nodes={"n": {"outputs": {"j": "not json"}}})
        s = run_one("node.json_count", {"selector": "nodes.n.outputs.j", "min": 1},
                    run, gate=True)
        self.assertFalse(s.passed)
        self.assertIn("not JSON-decodable", s.evidence)

    def test_json_field_presence(self):
        s = run_one("node.json_field",
                    {"selector": "nodes.sources.outputs.summary_json", "field": "earliest_url"},
                    self.RUN)
        self.assertTrue(s.passed)

    def test_json_field_absent(self):
        s = run_one("node.json_field",
                    {"selector": "nodes.sources.outputs.summary_json", "field": "nope"},
                    self.RUN)
        self.assertFalse(s.passed)

    def test_json_field_equals(self):
        s = run_one("node.json_field",
                    {"selector": "nodes.sources.outputs.summary_json", "field": "count",
                     "equals": 3}, self.RUN)
        self.assertTrue(s.passed)
        s2 = run_one("node.json_field",
                     {"selector": "nodes.sources.outputs.summary_json", "field": "count",
                      "equals": 4}, self.RUN)
        self.assertFalse(s2.passed)

    def test_json_field_equals_from(self):
        sample = make_sample(reference={"expected": "u1"})
        s = run_one("node.json_field",
                    {"selector": "nodes.sources.outputs.summary_json",
                     "field": "earliest_url", "equals_from": "reference.expected"},
                    self.RUN, sample=sample)
        self.assertTrue(s.passed)

    def test_node_regex(self):
        s = run_one("node.regex",
                    {"selector": "nodes.sources.outputs.sources_json", "pattern": r"u2"},
                    self.RUN)
        self.assertTrue(s.passed)


class TestMetaChecks(unittest.TestCase):
    def test_elapsed_pass_fail_marginal(self):
        run = make_run(meta={"status": "succeeded", "elapsed": 95})
        s = run_one("meta.elapsed_max", {"max": 100}, run)
        self.assertTrue(s.passed)
        self.assertTrue(s.marginal)  # 95 within 10% of 100
        s2 = run_one("meta.elapsed_max", {"max": 50}, run)
        self.assertFalse(s2.passed)

    def test_elapsed_missing_degrades(self):
        s = run_one("meta.elapsed_max", {"max": 10}, make_run(meta={}), gate=True)
        self.assertFalse(s.passed)

    def test_cost_dict_caps(self):
        run = make_run(usage={"cost": {"USD": 0.4, "EUR": 9.0}})
        s = run_one("meta.cost_max", {"max": {"USD": 0.5}}, run)
        self.assertTrue(s.passed)
        self.assertIn("uncapped currencies present: EUR", s.evidence)

    def test_cost_scalar_cap_default_usd_fail(self):
        run = make_run(usage={"cost": {"USD": 0.9}})
        s = run_one("meta.cost_max", {"max": 0.5}, run)
        self.assertFalse(s.passed)

    def test_cost_no_data_degrades(self):
        s = run_one("meta.cost_max", {"max": 1}, make_run(usage={}), gate=True)
        self.assertFalse(s.passed)
        self.assertIn("no cost data", s.evidence)

    def test_cost_undercount_note(self):
        run = make_run(usage={"cost": {"USD": 0.1}, "nodes_without_usage": 2})
        s = run_one("meta.cost_max", {"max": {"USD": 1.0}}, run)
        self.assertIn("undercounted", s.evidence)


class TestDerivedRubricItems(unittest.TestCase):
    def test_derive_from_reference_list(self):
        sample = make_sample(reference={"events": [
            {"date": "2026-07-04", "description": "first post"},
            {"id": "ev2", "date": "2026-07-05", "description": "amplified"},
        ]})
        spec = {"id": "gt1", "type": "gt.derived_rubric", "from": "reference.events",
                "template": "Report includes event: {date} — {description}",
                "weight": "expected"}
        items, err = derive_items(spec, sample)
        self.assertIsNone(err)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "gt1_1")
        self.assertEqual(items[1]["id"], "ev2")
        self.assertIn("2026-07-04", items[0]["item"])
        self.assertEqual(items[0]["weight"], "expected")

    def test_missing_template_key_renders_placeholder(self):
        sample = make_sample(reference={"events": [{"date": "d"}]})
        spec = {"id": "gt1", "from": "reference.events",
                "template": "{date} — {description}"}
        items, err = derive_items(spec, sample)
        self.assertIsNone(err)
        self.assertIn("<missing:description>", items[0]["item"])

    def test_non_list_reference_degrades(self):
        sample = make_sample(reference={"events": "oops"})
        spec = {"id": "gt1", "from": "reference.events", "template": "{value}"}
        items, err = derive_items(spec, sample)
        self.assertIsNone(items)
        self.assertIn("expected a list", err)


class TestValidators(unittest.TestCase):
    def _validate(self, spec):
        cd = registry.get(spec["type"])
        cd.validator(spec, make_case())

    def test_bad_regex_fails_fast(self):
        with self.assertRaises(ConfigError):
            self._validate({"id": "x", "type": "output.regex", "pattern": "("})

    def test_json_count_needs_bounds(self):
        with self.assertRaises(ConfigError):
            self._validate({"id": "x", "type": "node.json_count", "selector": "nodes.a"})

    def test_bad_selector_root_fails_fast(self):
        with self.assertRaises(ConfigError):
            self._validate({"id": "x", "type": "node.json_count",
                            "selector": "bogus.a", "min": 1})

    def test_unknown_schema_type_fails_fast(self):
        with self.assertRaises(ConfigError):
            self._validate({"id": "x", "type": "output.json_schema",
                            "schema": {"type": "objject"}})

    def test_rubric_weight_validated(self):
        with self.assertRaises(ConfigError):
            self._validate({"id": "x", "type": "judge.rubric",
                            "items": [{"id": "a", "item": "t", "weight": "critical"}]})


class TestDegradeNeverRaise(unittest.TestCase):
    def test_engine_catches_check_exceptions(self):
        @registry.register("test.explodes")
        def explodes(spec, sample, run, ctx):  # noqa: ARG001
            raise ValueError("kaboom")

        try:
            case = make_case(checks=[{"id": "boom", "type": "test.explodes", "gate": True}])
            scores = run_checks(case, make_sample(), make_run(), make_ctx(case))
            self.assertEqual(len(scores), 1)
            self.assertFalse(scores[0].passed)  # gated -> failed, not raised
            self.assertIn("kaboom", scores[0].evidence)
        finally:
            registry._REGISTRY.pop("test.explodes", None)


if __name__ == "__main__":
    unittest.main()
