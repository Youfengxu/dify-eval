import unittest

from difyeval.checks.selector import maybe_json, resolve, sample_root, validate_selector
from difyeval.core import ConfigError

from tests.helpers import make_sample


class TestResolve(unittest.TestCase):
    ROOT = {
        "output": "text",
        "nodes": {
            "merge_s4": {"outputs": {"resolved_json": '{"a": [1, 2, 3], "b": "x"}'}},
            "plain": {"outputs": {"lst": [{"k": "v0"}, {"k": "v1"}]}},
        },
        "meta": {"status": "succeeded"},
    }

    def test_plain_dict_path(self):
        r = resolve(self.ROOT, "meta.status")
        self.assertTrue(r.ok)
        self.assertEqual(r.value, "succeeded")

    def test_json_string_leaf_returned_raw_when_path_ends(self):
        r = resolve(self.ROOT, "nodes.merge_s4.outputs.resolved_json")
        self.assertTrue(r.ok)
        self.assertIsInstance(r.value, str)  # no further traversal requested

    def test_auto_json_decode_on_further_traversal(self):
        r = resolve(self.ROOT, "nodes.merge_s4.outputs.resolved_json.a")
        self.assertTrue(r.ok)
        self.assertEqual(r.value, [1, 2, 3])

    def test_list_index_traversal(self):
        r = resolve(self.ROOT, "nodes.plain.outputs.lst.1.k")
        self.assertTrue(r.ok)
        self.assertEqual(r.value, "v1")

    def test_missing_key_degrades(self):
        r = resolve(self.ROOT, "nodes.nope.outputs.x")
        self.assertFalse(r.ok)
        self.assertIn("selector not found", r.error)
        self.assertIn("'nope'", r.error)

    def test_non_json_string_leaf_degrades(self):
        r = resolve(self.ROOT, "output.deeper")
        self.assertFalse(r.ok)
        self.assertIn("not JSON-decodable", r.error)

    def test_bad_list_index_degrades(self):
        r = resolve(self.ROOT, "nodes.plain.outputs.lst.9.k")
        self.assertFalse(r.ok)
        self.assertIn("selector not found", r.error)

    def test_cannot_traverse_scalar(self):
        r = resolve(self.ROOT, "meta.status.deeper.yet")
        self.assertFalse(r.ok)


class TestMaybeJson(unittest.TestCase):
    def test_decodes_json_string(self):
        ok, v, err = maybe_json('[1, 2]')
        self.assertTrue(ok)
        self.assertEqual(v, [1, 2])

    def test_passthrough_non_string(self):
        ok, v, err = maybe_json({"a": 1})
        self.assertTrue(ok)
        self.assertEqual(v, {"a": 1})

    def test_undecodable_flags(self):
        ok, v, err = maybe_json("not json")
        self.assertFalse(ok)
        self.assertEqual(v, "not json")
        self.assertIn("not JSON-decodable", err)


class TestValidateSelector(unittest.TestCase):
    def test_valid(self):
        validate_selector("nodes.a.outputs.b")  # no raise

    def test_bad_root(self):
        with self.assertRaises(ConfigError):
            validate_selector("bogus.a.b")

    def test_empty_segment(self):
        with self.assertRaises(ConfigError):
            validate_selector("nodes..outputs")

    def test_non_string(self):
        with self.assertRaises(ConfigError):
            validate_selector(None)


class TestSampleRoot(unittest.TestCase):
    def test_reference_view(self):
        s = make_sample(reference={"events": [1]}, inputs={"q": "x"})
        root = sample_root(s)
        self.assertTrue(resolve(root, "reference.events").ok)
        self.assertEqual(resolve(root, "inputs.q").value, "x")

    def test_none_reference_is_empty(self):
        root = sample_root(make_sample(reference=None))
        r = resolve(root, "reference.events")
        self.assertFalse(r.ok)


if __name__ == "__main__":
    unittest.main()
