import json
import os
import sys
import tempfile
import unittest

from difyeval.runners import get_runner, registered_runners
from difyeval.runners.command import run_command
from difyeval.runners.dify_api import consume_stream, run_chatflow, run_workflow
from difyeval.runners.file import run_file
from difyeval.runners.normalize import normalize_run_payload
from difyeval.runners.usage import aggregate_usage
from difyeval.core import ConfigError

from tests.helpers import make_case, make_sample


class TestNormalize(unittest.TestCase):
    def test_native_run_shape(self):
        r = normalize_run_payload({"output": "rep", "nodes": {"n": {"outputs": {}}},
                                   "usage": {"tokens": 5}, "meta": {"status": "succeeded"}})
        self.assertEqual(r.output, "rep")
        self.assertEqual(r.usage, {"tokens": 5})
        self.assertIsNone(r.error)

    def test_archive_shape_with_usage_aggregation(self):
        obj = {"report": "the report", "elapsed": 33, "status": "succeeded",
               "nodes": {"llm": {"status": "succeeded", "outputs": {
                   "usage": {"total_tokens": 100, "total_price": "0.01", "currency": "USD"}}},
                   "code": {"status": "succeeded", "outputs": {}}}}
        r = normalize_run_payload(obj)
        self.assertEqual(r.output, "the report")
        self.assertEqual(r.meta["status"], "succeeded")
        self.assertEqual(r.meta["elapsed"], 33)
        self.assertEqual(r.usage["tokens"], 100)
        self.assertEqual(r.usage["cost"], {"USD": 0.01})
        self.assertEqual(r.usage["nodes_without_usage"], 1)

    def test_difyctl_data_outputs_shape(self):
        obj = {"data": {"outputs": {"report": "difyctl report", "extra": 1},
                        "status": "succeeded", "elapsed_time": 12.5, "total_tokens": 900}}
        r = normalize_run_payload(obj)
        self.assertEqual(r.output, "difyctl report")
        self.assertEqual(r.nodes, {})  # no node trace — documented
        self.assertEqual(r.usage, {"tokens": 900})
        self.assertEqual(r.meta["outputs"]["extra"], 1)

    def test_difyctl_bare_outputs_tolerated(self):
        r = normalize_run_payload({"outputs": {"report": "bare"}})
        self.assertEqual(r.output, "bare")

    def test_difyctl_missing_report_key_dumps_outputs(self):
        r = normalize_run_payload({"outputs": {"other": [1, 2]}})
        self.assertEqual(json.loads(r.output), {"other": [1, 2]})

    def test_difyctl_custom_report_key(self):
        r = normalize_run_payload({"outputs": {"answer": "custom"}}, report_key="answer")
        self.assertEqual(r.output, "custom")

    def test_unrecognized_degrades(self):
        r = normalize_run_payload({"weird": 1})
        self.assertEqual(r.meta["status"], "runner_error")
        self.assertIn("unrecognized", r.error)

    def test_non_dict_degrades(self):
        r = normalize_run_payload([1, 2])
        self.assertEqual(r.meta["status"], "runner_error")


class TestFileRunner(unittest.TestCase):
    def test_path_template_and_both_shapes(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "s1"))
            with open(os.path.join(td, "s1", "run.json"), "w") as f:
                json.dump({"report": "arch", "nodes": {}, "elapsed": 1, "status": "succeeded"}, f)
            case = make_case(tmpdir=td)
            spec = {"type": "file", "path": os.path.join(td, "{sample_id}", "run.json")}
            r = run_file(spec, make_sample("s1"), 1, case)
            self.assertEqual(r.output, "arch")

    def test_sample_metadata_override(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "special.json")
            with open(p, "w") as f:
                json.dump({"output": "native"}, f)
            case = make_case(tmpdir=td)
            r = run_file({"type": "file", "path": "missing.json"},
                         make_sample("s1", metadata={"run_path": p}), 1, case)
            self.assertEqual(r.output, "native")

    def test_missing_file_degrades(self):
        case = make_case()
        r = run_file({"type": "file", "path": "/nonexistent/nope.json"},
                     make_sample("s1"), 1, case)
        self.assertEqual(r.meta["status"], "runner_error")
        self.assertIn("cannot read", r.error)

    def test_no_path_degrades(self):
        r = run_file({"type": "file"}, make_sample("s1"), 1, make_case())
        self.assertEqual(r.meta["status"], "runner_error")


class TestCommandRunner(unittest.TestCase):
    def _spec(self, code, **kw):
        return {"type": "command", "argv": [sys.executable, "-c", code], **kw}

    def test_stdin_inputs_and_run_json_stdout(self):
        code = ("import json,sys; inp=json.load(sys.stdin); "
                "print(json.dumps({'output': 'echo:' + inp['q'], "
                "'meta': {'status': 'succeeded'}}))")
        with tempfile.TemporaryDirectory() as td:
            r = run_command(self._spec(code), make_sample("s1", inputs={"q": "hi"}),
                            1, make_case(tmpdir=td))
        self.assertEqual(r.output, "echo:hi")

    def test_placeholder_substitution(self):
        code = "import sys; print('{\"output\": \"' + sys.argv[1] + '|' + sys.argv[2] + '\"}')"
        spec = {"type": "command",
                "argv": [sys.executable, "-c", code, "{sample_id}", "{input.url}"]}
        with tempfile.TemporaryDirectory() as td:
            r = run_command(spec, make_sample("sX", inputs={"url": "http://u"}),
                            1, make_case(tmpdir=td))
        self.assertEqual(r.output, "sX|http://u")

    def test_nonzero_exit_degrades(self):
        code = "import sys; sys.stderr.write('bad day'); sys.exit(3)"
        with tempfile.TemporaryDirectory() as td:
            r = run_command(self._spec(code), make_sample("s1"), 1, make_case(tmpdir=td))
        self.assertEqual(r.meta["status"], "runner_error")
        self.assertIn("exited 3", r.error)
        self.assertIn("bad day", r.error)

    def test_bad_stdout_degrades(self):
        with tempfile.TemporaryDirectory() as td:
            r = run_command(self._spec("print('not json')"), make_sample("s1"), 1,
                            make_case(tmpdir=td))
        self.assertEqual(r.meta["status"], "runner_error")
        self.assertIn("not valid JSON", r.error)

    def test_timeout_degrades(self):
        code = "import time; time.sleep(5)"
        with tempfile.TemporaryDirectory() as td:
            r = run_command(self._spec(code, timeout=1), make_sample("s1"), 1,
                            make_case(tmpdir=td))
        self.assertEqual(r.meta["status"], "runner_error")
        self.assertIn("timed out", r.error)


def sse(event: dict) -> str:
    return "data: " + json.dumps(event)


WORKFLOW_TRANSCRIPT = [
    sse({"event": "workflow_started", "data": {"id": "w1"}}),
    "",  # keep-alive blank line
    ": comment line",
    sse({"event": "node_finished", "data": {
        "node_id": "planner", "status": "succeeded", "elapsed_time": 2.0,
        "outputs": {"queries_json": "[\"q1\"]",
                    "usage": {"total_tokens": 50, "total_price": "0.001", "currency": "USD"}}}}),
    sse({"event": "node_finished", "data": {
        "node_id": "sources", "status": "succeeded", "elapsed_time": 9.0,
        "outputs": {"sources_json": "[{\"url\": \"u1\"}, {\"url\": \"u2\"}]"}}}),
    sse({"event": "node_finished", "data": {
        "node_id": "noise", "status": "succeeded", "elapsed_time": 0.1, "outputs": {}}}),
    "data: {malformed json",
    sse({"event": "workflow_finished", "data": {
        "status": "succeeded", "elapsed_time": 20.5, "total_tokens": 999,
        "outputs": {"report": "# The report", "aux": 1}}}),
]


class TestSSEConsumer(unittest.TestCase):
    def test_journals_watched_nodes_only(self):
        res = consume_stream(iter(WORKFLOW_TRANSCRIPT), watch={"sources", "planner"})
        self.assertEqual(sorted(res["nodes"]), ["planner", "sources"])
        self.assertEqual(res["status"], "succeeded")
        self.assertEqual(res["outputs"]["report"], "# The report")
        self.assertEqual(res["elapsed"], 20.5)

    def test_watch_none_journals_all(self):
        res = consume_stream(iter(WORKFLOW_TRANSCRIPT), watch=None)
        self.assertIn("noise", res["nodes"])

    def test_error_event(self):
        lines = [sse({"event": "error", "message": "boom", "data": {}})]
        res = consume_stream(iter(lines))
        self.assertEqual(res["status"], "error")
        self.assertIn("boom", res["error"])


class TestWorkflowRunner(unittest.TestCase):
    def _run(self, transcript, spec_extra=None, inputs=None):
        captured = {}

        def opener(url, headers, payload, read_timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["auth"] = headers.get("Authorization")
            return 200, iter(transcript), None

        spec = {"type": "dify-workflow", "base": "https://dify.example.com",
                "api_key_env": "TEST_DIFY_KEY", "watch": ["planner", "sources"],
                **(spec_extra or {})}
        os.environ["TEST_DIFY_KEY"] = "app-key"
        try:
            r = run_workflow(spec, make_sample("s1", inputs=inputs or {"caption": "hello"}),
                             1, make_case(), open_stream=opener)
        finally:
            del os.environ["TEST_DIFY_KEY"]
        return r, captured

    def test_recorded_transcript_end_to_end(self):
        r, cap = self._run(WORKFLOW_TRANSCRIPT)
        self.assertEqual(cap["url"], "https://dify.example.com/v1/workflows/run")
        self.assertEqual(cap["auth"], "Bearer app-key")
        self.assertEqual(r.output, "# The report")
        self.assertEqual(sorted(r.nodes), ["planner", "sources"])
        self.assertEqual(r.meta["status"], "succeeded")
        self.assertEqual(r.meta["elapsed"], 20.5)
        # node usage exists -> workflow_finished total NOT double-counted
        self.assertEqual(r.usage["tokens"], 50)
        self.assertEqual(r.usage["cost"], {"USD": 0.001})

    def test_input_map(self):
        _, cap = self._run(WORKFLOW_TRANSCRIPT,
                           spec_extra={"input_map": {"caption": "post_text"}},
                           inputs={"caption": "hello"})
        self.assertEqual(cap["payload"]["inputs"], {"post_text": "hello"})

    def test_workflow_totals_used_when_no_node_usage(self):
        transcript = [
            sse({"event": "workflow_finished", "data": {
                "status": "succeeded", "elapsed_time": 3, "total_tokens": 777,
                "outputs": {"report": "r"}}}),
        ]
        r, _ = self._run(transcript)
        self.assertEqual(r.usage["tokens"], 777)

    def test_http_error_degrades(self):
        def opener(url, headers, payload, read_timeout):
            return 404, None, "not found"

        spec = {"type": "dify-workflow", "base": "https://x", "api_key_env": "TEST_DIFY_KEY"}
        os.environ["TEST_DIFY_KEY"] = "k"
        try:
            r = run_workflow(spec, make_sample("s1"), 1, make_case(), open_stream=opener)
        finally:
            del os.environ["TEST_DIFY_KEY"]
        self.assertEqual(r.meta["status"], "http_404")
        self.assertIn("not found", r.error)

    def test_missing_key_env_degrades(self):
        spec = {"type": "dify-workflow", "base": "https://x", "api_key_env": "NOPE_KEY_VAR"}
        r = run_workflow(spec, make_sample("s1"), 1, make_case())
        self.assertEqual(r.meta["status"], "runner_error")
        self.assertIn("NOPE_KEY_VAR", r.error)


CHATFLOW_TRANSCRIPT = [
    sse({"event": "node_finished", "data": {
        "node_id": "union", "status": "succeeded", "elapsed_time": 5.0,
        "outputs": {"union_json": "[1, 2]"}}}),
    sse({"event": "message", "answer": "part one, "}),
    sse({"event": "message", "answer": "part two."}),
    sse({"event": "message_end", "data": {},
         "metadata": {"usage": {"total_tokens": 1234, "total_price": "0.02",
                                "currency": "USD"}}}),
    sse({"event": "workflow_finished", "data": {"status": "succeeded", "elapsed_time": 8.0,
                                                "outputs": {}}}),
]


class TestChatflowRunner(unittest.TestCase):
    def test_message_accumulation_and_usage(self):
        captured = {}

        def opener(url, headers, payload, read_timeout):
            captured["url"] = url
            captured["payload"] = payload
            return 200, iter(CHATFLOW_TRANSCRIPT), None

        spec = {"type": "dify-chatflow", "base": "https://dify.example.com",
                "api_key_env": "TEST_DIFY_KEY", "watch": ["union"]}
        os.environ["TEST_DIFY_KEY"] = "k"
        try:
            r = run_chatflow(spec, make_sample("s1", inputs={"query": "investigate X",
                                                             "extra": "e"}),
                             1, make_case(), open_stream=opener)
        finally:
            del os.environ["TEST_DIFY_KEY"]
        self.assertEqual(captured["url"], "https://dify.example.com/v1/chat-messages")
        self.assertEqual(captured["payload"]["query"], "investigate X")
        self.assertEqual(captured["payload"]["inputs"], {"extra": "e"})
        self.assertEqual(r.output, "part one, part two.")
        self.assertIn("union", r.nodes)
        # message_end usage IS added (the answer LLM is not a journaled node)
        self.assertEqual(r.usage["tokens"], 1234)
        self.assertEqual(r.usage["cost"], {"USD": 0.02})


class TestUsageAggregation(unittest.TestCase):
    def test_multi_currency_and_missing_nodes(self):
        nodes = {
            "a": {"outputs": {"usage": {"total_tokens": 10, "total_price": "0.5",
                                        "currency": "USD"}}},
            "b": {"outputs": {"usage": {"total_tokens": 5, "total_price": "2",
                                        "currency": "CNY"}}},
            "c": {"outputs": {}},
        }
        u = aggregate_usage(nodes)
        self.assertEqual(u["tokens"], 15)
        self.assertEqual(u["cost"], {"CNY": 2.0, "USD": 0.5})
        self.assertEqual(u["nodes_without_usage"], 1)

    def test_top_level_added(self):
        u = aggregate_usage({"a": {"outputs": {"usage": {"total_tokens": 10}}}},
                            top={"tokens": 7, "cost": {"USD": 0.1}})
        self.assertEqual(u["tokens"], 17)
        self.assertEqual(u["cost"], {"USD": 0.1})

    def test_empty(self):
        self.assertEqual(aggregate_usage({}), {})


class TestRegistry(unittest.TestCase):
    def test_all_five_runners_registered(self):
        self.assertEqual(registered_runners(),
                         ["command", "dify-chatflow", "dify-workflow",
                          "difyctl-json", "file"])

    def test_unknown_runner_is_config_error(self):
        with self.assertRaises(ConfigError):
            get_runner({"type": "bogus"})


if __name__ == "__main__":
    unittest.main()
