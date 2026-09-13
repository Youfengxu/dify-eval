import unittest

from difyeval import gt


class TestTierConstants(unittest.TestCase):
    def test_exact_tier_strings(self):
        self.assertEqual(gt.HUMAN_VERIFIED, "human_verified")
        self.assertEqual(gt.HUMAN_SUPPLIED, "human_supplied")
        self.assertEqual(gt.PIPELINE_OBSERVED_UNVALIDATED,
                         "pipeline_observed_unvalidated")
        self.assertEqual(gt.TIERS, ("human_verified", "human_supplied",
                                    "pipeline_observed_unvalidated"))

    def test_gateable_tiers(self):
        self.assertEqual(gt.GATEABLE_TIERS, {"human_verified", "human_supplied"})

    def test_semantics_documented_for_pack_authors(self):
        # the docstring is the pack-author contract: TRUST vs GATEABILITY
        doc = gt.__doc__
        self.assertIn("provenance", doc)
        self.assertIn("TRUST", doc)
        self.assertIn("discoverable", doc)
        self.assertIn("denominator", doc.lower())
        self.assertIn("must_reject", doc)
        self.assertIn("never gates", doc)


class TestIterItems(unittest.TestCase):
    REF = {"accounts": {"items": [{"handle": "@a"}, {"handle": "@b"}]},
           "flat": [1, 2, 3],
           "must_reject": [{"value": "@x", "reason": "r"}]}

    def test_nested_path(self):
        self.assertEqual(len(gt.iter_items(self.REF, "accounts.items")), 2)

    def test_flat_path(self):
        self.assertEqual(gt.iter_items(self.REF, "flat"), [1, 2, 3])

    def test_missing_path_is_empty(self):
        self.assertEqual(gt.iter_items(self.REF, "no.such.path"), [])
        self.assertEqual(gt.iter_items(None, "accounts.items"), [])
        self.assertEqual(gt.iter_items({}, "accounts.items"), [])

    def test_non_list_leaf_is_empty(self):
        self.assertEqual(gt.iter_items(self.REF, "accounts"), [])

    def test_iter_reference_lists_walks_sorted(self):
        found = list(gt.iter_reference_lists(self.REF))
        self.assertEqual([p for p, _ in found],
                         ["accounts.items", "flat", "must_reject"])
        # lists are yielded, never descended into
        self.assertEqual(found[1][1], [1, 2, 3])


class TestTiering(unittest.TestCase):
    ITEMS = [
        {"handle": "@v", "provenance": "human_verified"},
        {"handle": "@s", "provenance": "human_supplied"},
        {"handle": "@p", "provenance": "pipeline_observed_unvalidated"},
        {"handle": "@none"},                       # no provenance -> hand-authored
        {"handle": "@junk", "provenance": "made_up_tier"},
    ]

    def test_item_tier_default_is_human_supplied(self):
        self.assertEqual(gt.item_tier({"handle": "@x"}), "human_supplied")
        self.assertEqual(gt.item_tier("bare-string"), "human_supplied")
        self.assertEqual(gt.item_tier({"provenance": "  human_verified "}),
                         "human_verified")

    def test_partition(self):
        gate, adv = gt.partition_by_tier(self.ITEMS)
        self.assertEqual([i["handle"] for i in gate], ["@v", "@s", "@none"])
        self.assertEqual([i["handle"] for i in adv], ["@p", "@junk"])

    def test_pipeline_observed_never_gates(self):
        gate, _ = gt.partition_by_tier(
            [{"handle": "@p", "provenance": "pipeline_observed_unvalidated"}])
        self.assertEqual(gate, [])

    def test_unknown_tier_never_gates(self):
        gate, adv = gt.partition_by_tier([{"handle": "@x", "provenance": "wat"}])
        self.assertEqual(gate, [])
        self.assertEqual(len(adv), 1)

    def test_gateable_helper(self):
        self.assertEqual(gt.gateable(self.ITEMS), gt.partition_by_tier(self.ITEMS)[0])

    def test_empty_and_none(self):
        self.assertEqual(gt.partition_by_tier([]), ([], []))
        self.assertEqual(gt.partition_by_tier(None), ([], []))

    def test_discoverable_semantics_pack_side(self):
        # discoverable: false stays in the TRUST partition (it is a
        # GATEABILITY exclusion recall packs apply to denominators)
        items = [{"handle": "@private", "provenance": "human_verified",
                  "discoverable": False}]
        gate, _ = gt.partition_by_tier(items)
        self.assertEqual(len(gate), 1)
        denominator = [i for i in gt.gateable(items)
                       if i.get("discoverable") is not False]
        self.assertEqual(denominator, [])


class TestValidateItems(unittest.TestCase):
    def test_clean_items_no_warnings(self):
        items = [{"handle": "@a", "provenance": "human_verified",
                  "validated": {"by": "x", "date": "2026-07-01"},
                  "discoverable": True},
                 {"handle": "@b"}, "bare", 3]
        self.assertEqual(gt.validate_items(items), [])

    def test_unknown_tier_warns(self):
        w = gt.validate_items([{"handle": "@a", "provenance": "nope"}])
        self.assertEqual(len(w), 1)
        self.assertIn("unknown provenance tier", w[0])
        self.assertIn("a", w[0])

    def test_validated_without_by_or_date_warns(self):
        w = gt.validate_items([{"handle": "@a", "validated": {"by": "x"}},
                               {"handle": "@b", "validated": {"date": "2026-07-01"}},
                               {"handle": "@c", "validated": "yes"}])
        self.assertEqual(len(w), 3)
        for msg in w:
            self.assertIn("'by' and 'date'", msg)

    def test_non_bool_discoverable_warns(self):
        w = gt.validate_items([{"handle": "@a", "discoverable": "yes"}])
        self.assertEqual(len(w), 1)
        self.assertIn("discoverable", w[0])

    def test_empty(self):
        self.assertEqual(gt.validate_items([]), [])
        self.assertEqual(gt.validate_items(None), [])


class TestRejectList(unittest.TestCase):
    def test_default_path_and_coercion(self):
        ref = {"must_reject": [{"value": "@x", "reason": "r"}, "@bare", None]}
        out = gt.reject_list(ref)
        self.assertEqual(out, [{"value": "@x", "reason": "r"}, {"value": "@bare"}])

    def test_custom_path(self):
        ref = {"neg": {"list": ["@a"]}}
        self.assertEqual(gt.reject_list(ref, path="neg.list"), [{"value": "@a"}])

    def test_missing(self):
        self.assertEqual(gt.reject_list({}), [])
        self.assertEqual(gt.reject_list(None), [])


class TestItemIdentity(unittest.TestCase):
    def test_field_priority_id_first(self):
        self.assertEqual(gt.item_identity({"id": "K1", "handle": "@a"}),
                         ("id", "k1"))
        self.assertEqual(gt.item_identity({"handle": "@a", "value": "v"}),
                         ("handle", "a"))
        self.assertEqual(gt.item_identity({"value": "@V"}), ("value", "v"))

    def test_normalization_at_strip_and_case(self):
        self.assertEqual(gt.item_identity({"handle": " @UserOne "}),
                         gt.item_identity({"handle": "userone"}))

    def test_bool_and_empty_fields_skipped(self):
        self.assertEqual(gt.item_identity({"id": True, "handle": "@a"})[0], "handle")
        self.assertEqual(gt.item_identity({"id": "  ", "handle": "@a"})[0], "handle")

    def test_dict_fallback_is_sorted_json(self):
        a = gt.item_identity({"x": 1, "y": 2})
        b = gt.item_identity({"y": 2, "x": 1})
        self.assertEqual(a, b)
        self.assertEqual(a[0], "_dict")

    def test_scalar(self):
        self.assertEqual(gt.item_identity("@Handle"), ("_scalar", "handle"))
        self.assertEqual(gt.item_identity(42), ("_scalar", "42"))

    def test_identity_value(self):
        self.assertEqual(gt.identity_value({"handle": "@A"}), "a")


if __name__ == "__main__":
    unittest.main()
