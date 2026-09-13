"""The determinism contract: replaying the demo bundle twice with --no-judge
must produce byte-identical scorecards (minus the single generated line) and
semantically identical eval.json (minus manifest.scored_at)."""
import json
import os
import tempfile
import unittest

from difyeval import cli
from difyeval.report import strip_generated_line

from tests.helpers import DEMO_CASE


class TestDeterminism(unittest.TestCase):
    def _score_once(self, out_dir):
        rc = cli.main(["replay", DEMO_CASE, "--no-judge", "--out-dir", out_dir])
        self.assertEqual(rc, 0)
        with open(os.path.join(
                out_dir, "text_source_discovery_demo__default__scorecard.md"),
                encoding="utf-8") as f:
            md = f.read()
        with open(os.path.join(
                out_dir, "text_source_discovery_demo__default__eval.json"),
                encoding="utf-8") as f:
            ej = json.load(f)
        return md, ej

    def test_replay_twice_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            md1, ej1 = self._score_once(os.path.join(td, "a"))
            md2, ej2 = self._score_once(os.path.join(td, "b"))
        self.assertEqual(strip_generated_line(md1), strip_generated_line(md2))
        for ej in (ej1, ej2):
            ej["manifest"].pop("scored_at")
        self.assertEqual(json.dumps(ej1, sort_keys=True), json.dumps(ej2, sort_keys=True))

    def test_exactly_one_timestamp_line(self):
        with tempfile.TemporaryDirectory() as td:
            md, _ = self._score_once(td)
        gen_lines = [ln for ln in md.split("\n") if ln.startswith("- generated: ")]
        self.assertEqual(len(gen_lines), 1)

    def test_demo_verdict_and_scorecard_content(self):
        with tempfile.TemporaryDirectory() as td:
            md, ej = self._score_once(td)
        self.assertEqual(ej["verdict"], "PASS")
        self.assertIn("## Verdict: ✅ PASS", md)
        self.assertIn("(marginal)", md)          # 3 sources at min=3
        self.assertIn("Scope:", md)
        self.assertIn("judge skipped", md)        # --no-judge -> advisory skip visible
        self.assertEqual(ej["schema"], "eval.json/1")
        self.assertIsNone(ej["manifest"]["dsl_sha256"])
        self.assertTrue(ej["manifest"]["case_sha256"])


if __name__ == "__main__":
    unittest.main()
