"""Offline orchestration tests; no Gemini credentials or network needed."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from spatial_data_agent.agent import GeminiDataAgent, GeminiBackend, declarations


class FakeBackend:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.histories = []

    def user(self, text):
        return {"user": text}

    def generate(self, history, system):
        self.histories.append(list(history))
        calls = next(self.turns)
        return {"model": calls, "thought_signature": "preserved"}, calls

    def tool_response(self, results):
        return {"tool_results": results}

    def close(self):
        pass


def call(name, **args):
    return (name, args, None)


def finish(status="answer", message="Done"):
    return [call("finish", status=status, message=message)]


class AgentTests(unittest.TestCase):
    def make_agent(self, root, turns, **kwargs):
        store = SimpleNamespace(root=Path(root) / "release")
        agent = GeminiDataAgent(store, exports_dir=Path(root) / "exports",
                                backend=FakeBackend(turns), **kwargs)
        agent.tools.call = Mock()
        return agent

    def test_discover_query_export_receipt(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [
                [call("describe", fields=["disease"])],
                [call("query", filters_json='{"disease":"MI"}')],
                [call("fetch", query_id="q1")], finish("ready")])
            agent.tools.call.side_effect = [{"fields": {}}, {"query_id": "q1", "n_obs": 20},
                                            {"status": "ready", "manifest_uri": "real/manifest.json"}]
            result = agent.run("Extract MI spots")
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["artifacts"][0]["manifest_uri"], "real/manifest.json")
            self.assertEqual(agent.tools.call.call_args_list[1].args,
                             ("query", {"filters": {"disease": "MI"}}))
            self.assertIn("thought_signature", agent.backend.histories[-1][1])

    def test_no_export_for_discovery_and_clarification(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [[call("describe", fields=["disease"])],
                                           finish("clarification_required", "Which disease?")])
            agent.tools.call.return_value = {"fields": {}}
            result = agent.run("Get cancer")
            self.assertEqual(result["status"], "clarification_required")
            self.assertEqual(result["artifacts"], [])
            self.assertEqual(agent.tools.call.call_count, 1)

    def test_unknown_query_empty_and_export_limit(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [], max_export_obs=100)
            for qid, n, code in [("none", None, "UNKNOWN_QUERY"),
                                  ("empty", 0, "EMPTY_QUERY"), ("large", 101, "EXPORT_LIMIT")]:
                if n is not None:
                    agent.queries[qid] = {"n_obs": n}
                with self.assertRaisesRegex(ValueError, ".+") as caught:
                    agent._dispatch("fetch", {"query_id": qid})
                self.assertEqual(caught.exception.code, code)
            agent.tools.call.assert_not_called()

    def test_cannot_fabricate_ready_or_use_unsupported_tool(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [finish("ready"), [call("shell", command="bad")], finish()])
            result = agent.run("Inspect")
            self.assertEqual(result["tool_trace"][0]["result"]["code"], "NO_ARTIFACT")
            self.assertEqual(result["tool_trace"][1]["result"]["code"], "INVALID_ARGUMENTS")
            agent.tools.call.assert_not_called()

    def test_tool_errors_and_malformed_json_return_to_model(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [[call("query", filters_json="oops")],
                [call("fetch", query_id="q")], finish("clarification_required")])
            agent.queries["q"] = {"n_obs": 10}
            agent.tools.call.return_value = {"status": "error", "code": "MISSING_GENE"}
            result = agent.run("Get a gene")
            self.assertEqual(result["tool_trace"][0]["result"]["code"], "INVALID_ARGUMENTS")
            self.assertEqual(result["artifacts"], [])

    def test_followup_uses_session_query(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [[call("query")], finish("selected"),
                                           [call("fetch", query_id="q")], finish("ready")])
            agent.tools.call.side_effect = [{"query_id": "q", "n_obs": 10},
                                            {"status": "ready", "manifest_uri": "x"}]
            self.assertEqual(agent.run("Find heart data")["status"], "selected")
            self.assertEqual(agent.run("Export those")["status"], "ready")

    def test_step_limit_and_empty_key(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [[]], max_steps=1)
            self.assertEqual(agent.run("Hi")["status"], "limit_reached")
            path = Path(root) / "key.txt"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                GeminiBackend(path, "unused")
            self.assertEqual({x["name"] for x in declarations()}, {"describe", "query", "fetch", "finish", "gene_systems", "resolve_genes"})


    def test_api_failure_does_not_expose_exception_payload(self):
        with TemporaryDirectory() as root:
            agent = self.make_agent(root, [])
            agent.backend.generate = Mock(side_effect=RuntimeError("private-key-in-sdk-error"))
            result = agent.run("Find available diseases")
            self.assertEqual(result["status"], "error")
            self.assertNotIn("private-key-in-sdk-error", str(result))

    def test_gemini_function_response_role_and_redacted_error(self):
        backend = GeminiBackend.__new__(GeminiBackend)
        backend.types = SimpleNamespace(Content=lambda **kw: kw, Part=lambda **kw: kw,
                                        FunctionResponse=lambda **kw: kw)
        message = backend.tool_response([("describe", "call1", {"fields": {}})])
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["parts"][0]["function_response"]["id"], "call1")
        backend._redact_key = "secret-key-value"
        error = SimpleNamespace(code=400, status="INVALID_ARGUMENT",
            message="Bad role: tool. key=secret-key-value; AIza12345_example")
        details = backend.error_details(error)
        self.assertEqual(details["code"], 400)
        self.assertIn("Bad role: tool", details["api_message"])
        self.assertNotIn("secret-key-value", str(details))
        self.assertNotIn("AIza12345_example", str(details))


if __name__ == "__main__":
    unittest.main()
