import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import yaml

from difyeval import cli
from difyeval.core import ConfigError, accretions_path, load_accretions, load_case


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def write_case(td, dataset=None, name="case.yml"):
    run_path = os.path.join(td, "run.json")
    if not os.path.exists(run_path):
        json.dump({"output": "hello"}, open(run_path, "w"))
    case = {"case_id": "acc_case",
            "runner": {"type": "file", "path": run_path},
            "dataset": dataset or [{"id": "s1", "inputs": {}}],
            "checks": [{"id": "c", "type": "output.nonempty", "gate": True}]}
    path = os.path.join(td, name)
    yaml.safe_dump(case, open(path, "w"))
    return path


def write_sidecar(case_path, samples):
    p = accretions_path(case_path)
    yaml.safe_dump({"samples": samples}, open(p, "w"))
    return p


class TestNoSidecar(unittest.TestCase):
    def test_load_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {}, "reference": {"expected": "x"}}])
            case = load_case(path)
            self.assertIsNone(case.sidecar_path)
            self.assertEqual(case.sidecar_merged, [])
            self.assertEqual(case.load_warnings, [])
            self.assertEqual(case.samples[0].reference, {"expected": "x"})

    def test_load_accretions_absent_is_empty(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            self.assertEqual(load_accretions(path), {})


class TestSidecarMerge(unittest.TestCase):
    def test_append_after_base_in_file_order(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {},
                 "reference": {"accounts": {"items": [{"handle": "@base"}]}}}])
            write_sidecar(path, {"s1": {"reference": {"accounts.items": [
                {"handle": "@acc1", "provenance": "human_verified"},
                {"handle": "@acc2", "provenance": "pipeline_observed_unvalidated"},
            ]}}})
            case = load_case(path)
            items = case.samples[0].reference["accounts"]["items"]
            self.assertEqual([i["handle"] for i in items],
                             ["@base", "@acc1", "@acc2"])
            self.assertEqual(case.sidecar_merged,
                             [{"sample": "s1", "path": "accounts.items",
                               "added": 2, "skipped": 0}])
            self.assertEqual(case.load_warnings, [])

    def test_creates_missing_path_and_reference(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)  # sample has NO reference at all
            write_sidecar(path, {"s1": {"reference": {
                "accounts.items": [{"handle": "@new"}],
                "must_reject": [{"value": "@bad", "reason": "r"}]}}})
            case = load_case(path)
            ref = case.samples[0].reference
            self.assertEqual(ref["accounts"]["items"], [{"handle": "@new"}])
            self.assertEqual(ref["must_reject"], [{"value": "@bad", "reason": "r"}])

    def test_all_applies_to_every_sample_before_per_sample(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[{"id": "s1", "inputs": {}},
                                           {"id": "s2", "inputs": {}}])
            write_sidecar(path, {
                "_all": {"reference": {"accounts.items": [{"handle": "@everywhere"}]}},
                "s2": {"reference": {"accounts.items": [{"handle": "@only2"}]}}})
            case = load_case(path)
            s1 = case.samples[0].reference["accounts"]["items"]
            s2 = case.samples[1].reference["accounts"]["items"]
            self.assertEqual([i["handle"] for i in s1], ["@everywhere"])
            self.assertEqual([i["handle"] for i in s2], ["@everywhere", "@only2"])

    def test_all_items_are_independent_copies(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[{"id": "s1", "inputs": {}},
                                           {"id": "s2", "inputs": {}}])
            write_sidecar(path, {"_all": {"reference": {
                "accounts.items": [{"handle": "@shared", "notes": "n"}]}}})
            case = load_case(path)
            case.samples[0].reference["accounts"]["items"][0]["notes"] = "MUTATED"
            self.assertEqual(case.samples[1].reference["accounts"]["items"][0]["notes"], "n")

    def test_duplicate_of_base_item_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {},
                 "reference": {"accounts": {"items": [{"handle": "@UserOne"}]}}}])
            # same identity: @-strip + case-insensitive
            write_sidecar(path, {"s1": {"reference": {"accounts.items": [
                {"handle": "userone", "provenance": "human_verified"},
                {"handle": "@fresh"}]}}})
            case = load_case(path)
            items = case.samples[0].reference["accounts"]["items"]
            self.assertEqual([i["handle"] for i in items], ["@UserOne", "@fresh"])
            self.assertEqual(case.sidecar_merged[0]["added"], 1)
            self.assertEqual(case.sidecar_merged[0]["skipped"], 1)
            self.assertTrue(any("duplicate item 'userone'" in w
                                for w in case.load_warnings))

    def test_duplicate_within_sidecar_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            write_sidecar(path, {"s1": {"reference": {"accounts.items": [
                {"handle": "@a", "provenance": "human_verified"},
                {"handle": "@A"}]}}})
            case = load_case(path)
            items = case.samples[0].reference["accounts"]["items"]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["provenance"], "human_verified")

    def test_unknown_sample_id_warns_never_errors(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            write_sidecar(path, {"ghost": {"reference": {
                "accounts.items": [{"handle": "@x"}]}}})
            case = load_case(path)
            self.assertTrue(any("unknown sample id 'ghost'" in w
                                for w in case.load_warnings))
            self.assertIsNone(case.samples[0].reference)

    def test_base_file_never_mutated(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {},
                 "reference": {"accounts": {"items": [{"handle": "@base"}]}}}])
            write_sidecar(path, {"s1": {"reference": {
                "accounts.items": [{"handle": "@acc"}]}}})
            before = open(path, "rb").read()
            case = load_case(path)
            self.assertEqual(open(path, "rb").read(), before)
            # raw echoes the authored base only
            base_items = case.raw["dataset"][0]["reference"]["accounts"]["items"]
            self.assertEqual([i["handle"] for i in base_items], ["@base"])

    def test_deterministic_double_load(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {}, "reference": {"accounts": {"items": []}}}])
            write_sidecar(path, {"s1": {"reference": {"accounts.items": [
                {"handle": "@b"}, {"handle": "@a"}]}}})
            r1 = load_case(path).samples[0].reference
            r2 = load_case(path).samples[0].reference
            self.assertEqual(r1, r2)
            self.assertEqual([i["handle"] for i in r1["accounts"]["items"]],
                             ["@b", "@a"])  # sidecar file order, not sorted


class TestSidecarStructuralErrors(unittest.TestCase):
    def test_non_mapping_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            open(accretions_path(path), "w").write("- just\n- a list\n")
            with self.assertRaises(ConfigError):
                load_case(path)

    def test_path_value_not_a_list(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            write_sidecar(path, {"s1": {"reference": {"accounts.items": {"handle": "@x"}}}})
            with self.assertRaises(ConfigError):
                load_case(path)

    def test_path_collides_with_non_list(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {}, "reference": {"accounts": {"items": "text"}}}])
            write_sidecar(path, {"s1": {"reference": {
                "accounts.items": [{"handle": "@x"}]}}})
            with self.assertRaises(ConfigError) as cm:
                load_case(path)
            self.assertIn("not a list", str(cm.exception))

    def test_path_crosses_non_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {}, "reference": {"accounts": "text"}}])
            write_sidecar(path, {"s1": {"reference": {
                "accounts.items": [{"handle": "@x"}]}}})
            with self.assertRaises(ConfigError):
                load_case(path)


class TestValidateSurfacing(unittest.TestCase):
    def test_validate_reports_sidecar_counts_and_warnings(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {},
                 "reference": {"accounts": {"items": [{"handle": "@base"}]}}}])
            write_sidecar(path, {"s1": {"reference": {"accounts.items": [
                {"handle": "@base"},                       # duplicate -> warning
                {"handle": "@new", "provenance": "human_verified",
                 "validated": {"by": "a"}},                # missing date -> warning
                {"handle": "@odd", "provenance": "wat"}]}}})  # unknown tier -> warning
            rc, out, _ = run_cli(["validate", path])
            self.assertEqual(rc, 0)
            self.assertIn("OK:", out)
            self.assertIn("accretions sidecar:", out)
            self.assertIn("s1 · accounts.items: +2 (1 duplicate(s) skipped)", out)
            self.assertIn("duplicate item 'base'", out)
            self.assertIn("unknown provenance tier 'wat'", out)
            self.assertIn("'by' and 'date'", out)

    def test_validate_without_sidecar_is_quiet(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            rc, out, _ = run_cli(["validate", path])
            self.assertEqual(rc, 0)
            self.assertNotIn("accretions sidecar", out)
            self.assertNotIn("warning:", out)

    def test_validate_flags_hand_authored_provenance_problems(self):
        # provenance warnings surface even with NO sidecar (base-file lists)
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td, dataset=[
                {"id": "s1", "inputs": {},
                 "reference": {"accounts": {"items": [
                     {"handle": "@x", "provenance": "typo_tier"}]}}}])
            rc, out, _ = run_cli(["validate", path])
            self.assertEqual(rc, 0)
            self.assertIn("s1 · reference.accounts.items", out)
            self.assertIn("unknown provenance tier", out)


class TestManifest(unittest.TestCase):
    def test_accretions_sha_in_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            write_sidecar(path, {"s1": {"reference": {
                "accounts.items": [{"handle": "@x"}]}}})
            rc, _, _ = run_cli(["run", path, "--no-judge", "--out-dir", td])
            self.assertEqual(rc, 0)
            ev = json.load(open(os.path.join(td, "acc_case__default__eval.json")))
            self.assertTrue(ev["manifest"]["accretions_sha256"])

    def test_no_sidecar_manifest_null(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_case(td)
            rc, _, _ = run_cli(["run", path, "--no-judge", "--out-dir", td])
            self.assertEqual(rc, 0)
            ev = json.load(open(os.path.join(td, "acc_case__default__eval.json")))
            self.assertIsNone(ev["manifest"]["accretions_sha256"])


if __name__ == "__main__":
    unittest.main()
