import json
import os
import tempfile
import unittest

import yaml

from difyeval.core import ConfigError
from difyeval.profile import load_profile, resolve_profile_dsl

from tests.test_cli import run_cli

PROFILE = {
    "profile": "opus-baseline",
    "description": "orchestrator on opus 4.6",
    "models": {"orchestrator": "anthropic/claude-opus-4.6"},
    "config": {"temperature": 0, "reasoning_budget": 8401},
}


def _write_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)


def _mini_case(td, run_path, case_name="case.yml"):
    case = {"case_id": "profcase", "runner": {"type": "file", "path": run_path},
            "dataset": [{"id": "s1", "inputs": {}}],
            "checks": [{"id": "c", "type": "output.nonempty", "gate": True}]}
    path = os.path.join(td, case_name)
    _write_yaml(path, case)
    return path


class TestLoadProfile(unittest.TestCase):
    def _load(self, obj):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "prof.yml")
            _write_yaml(p, obj)
            return load_profile(p)

    def test_happy_path_returns_verbatim_mapping(self):
        prof = self._load(PROFILE)
        self.assertEqual(prof, PROFILE)

    def test_missing_file_fails_fast(self):
        with self.assertRaises(ConfigError):
            load_profile("/no/such/profile.yml")

    def test_bad_yaml_fails_fast(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "prof.yml")
            with open(p, "w") as f:
                f.write("profile: [unclosed")
            with self.assertRaises(ConfigError):
                load_profile(p)

    def test_non_mapping_fails_fast(self):
        with self.assertRaises(ConfigError):
            self._load(["not", "a", "mapping"])

    def test_profile_name_required(self):
        with self.assertRaises(ConfigError) as cm:
            self._load({"description": "no name"})
        self.assertIn("profile", str(cm.exception))

    def test_runner_must_be_mapping(self):
        with self.assertRaises(ConfigError):
            self._load({"profile": "p", "runner": "not-a-dict"})

    def test_models_config_must_be_mappings(self):
        with self.assertRaises(ConfigError):
            self._load({"profile": "p", "models": [1, 2]})
        with self.assertRaises(ConfigError):
            self._load({"profile": "p", "config": "x"})

    def test_unknown_extra_keys_preserved(self):
        prof = self._load({"profile": "p", "notes": "free-form extra"})
        self.assertEqual(prof["notes"], "free-form extra")

    def test_missing_dsl_file_fails_fast(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "prof.yml")
            _write_yaml(p, {"profile": "p", "dsl": "no-such.yml"})
            prof = load_profile(p)
            with self.assertRaises(ConfigError):
                resolve_profile_dsl(prof, p)


class TestProfileCli(unittest.TestCase):
    def test_profile_file_labels_and_embeds_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            json.dump({"output": "hello"}, open(run_path, "w"))
            case_path = _mini_case(td, run_path)
            prof_path = os.path.join(td, "prof.yml")
            _write_yaml(prof_path, PROFILE)
            rc, out, _ = run_cli(["run", case_path, "--no-judge",
                                  "--profile-file", prof_path, "--out-dir", td])
            self.assertEqual(rc, 0)
            # the profile name is the Experiment label -> file names + header
            md_path = os.path.join(td, "profcase__opus-baseline__scorecard.md")
            self.assertTrue(os.path.exists(md_path))
            md = open(md_path, encoding="utf-8").read()
            header_lines = [ln for ln in md.split("\n")
                            if ln.startswith("# difyeval scorecard")]
            self.assertEqual(len(header_lines), 1)  # single header line
            self.assertIn("opus-baseline", header_lines[0])
            # the profile mapping is embedded VERBATIM in the manifest
            ej = json.load(open(os.path.join(
                td, "profcase__opus-baseline__eval.json"), encoding="utf-8"))
            self.assertEqual(ej["manifest"]["profile_config"], PROFILE)
            self.assertEqual(ej["profile"], "opus-baseline")

    def test_no_profile_file_manifest_carries_none(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            json.dump({"output": "hello"}, open(run_path, "w"))
            case_path = _mini_case(td, run_path)
            rc, _, _ = run_cli(["run", case_path, "--no-judge", "--out-dir", td])
            self.assertEqual(rc, 0)
            ej = json.load(open(os.path.join(
                td, "profcase__default__eval.json"), encoding="utf-8"))
            self.assertIsNone(ej["manifest"]["profile_config"])

    def test_profile_dsl_auto_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            json.dump({"output": "hello"}, open(run_path, "w"))
            case_path = _mini_case(td, run_path)
            dsl_path = os.path.join(td, "workflow.yml")
            with open(dsl_path, "w") as f:
                f.write("app: {}\n")
            prof_path = os.path.join(td, "prof.yml")
            _write_yaml(prof_path, {**PROFILE, "dsl": "workflow.yml"})
            rc, _, _ = run_cli(["run", case_path, "--no-judge",
                                "--profile-file", prof_path, "--out-dir", td])
            self.assertEqual(rc, 0)
            ej = json.load(open(os.path.join(
                td, "profcase__opus-baseline__eval.json"), encoding="utf-8"))
            import hashlib
            expected = hashlib.sha256(open(dsl_path, "rb").read()).hexdigest()
            self.assertEqual(ej["manifest"]["dsl_sha256"], expected)

    def test_runner_overrides_applied(self):
        with tempfile.TemporaryDirectory() as td:
            for name, text in (("runA.json", "alpha"), ("runB.json", "beta")):
                json.dump({"output": text}, open(os.path.join(td, name), "w"))
            case_path = _mini_case(td, os.path.join(td, "runA.json"))
            prof_path = os.path.join(td, "prof.yml")
            _write_yaml(prof_path, {"profile": "b-variant",
                                    "runner": {"path": os.path.join(td, "runB.json")}})
            rc, _, _ = run_cli(["run", case_path, "--no-judge",
                                "--profile-file", prof_path, "--out-dir", td])
            self.assertEqual(rc, 0)
            ej = json.load(open(os.path.join(
                td, "profcase__b-variant__eval.json"), encoding="utf-8"))
            self.assertEqual(ej["results"][0]["run"]["output"], "beta")

    def test_profile_and_profile_file_together_is_config_error(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            json.dump({"output": "hello"}, open(run_path, "w"))
            case_path = _mini_case(td, run_path)
            prof_path = os.path.join(td, "prof.yml")
            _write_yaml(prof_path, PROFILE)
            rc, _, err = run_cli(["run", case_path, "--no-judge", "--profile", "x",
                                  "--profile-file", prof_path, "--out-dir", td])
            self.assertEqual(rc, 2)
            self.assertIn("not both", err)

    def test_plain_profile_label_keeps_working(self):
        with tempfile.TemporaryDirectory() as td:
            run_path = os.path.join(td, "run.json")
            json.dump({"output": "hello"}, open(run_path, "w"))
            case_path = _mini_case(td, run_path)
            rc, _, _ = run_cli(["run", case_path, "--no-judge", "--profile", "lbl",
                                "--out-dir", td])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(
                td, "profcase__lbl__scorecard.md")))


if __name__ == "__main__":
    unittest.main()
