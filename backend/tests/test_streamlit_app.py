"""Streamlit-to-ASGI tests: real SSE transport/renderer, injected agent only."""

from pathlib import Path
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.app import create_app
from backend.tests.test_knowledge_agent_api import FakeAgent
from streamlit_app import api_base_url, iter_sse, scoped_stream_url


APP = Path(__file__).resolve().parents[2] / "streamlit_app.py"
BASE = "http://localhost:8005/v1/knowledge-agent"


class SseClientTest(unittest.TestCase):
    def test_urls_accept_legacy_scoped_and_base_endpoints(self):
        for suffix in ("", "/", "/stream", "/enterprise_rag/stream/"):
            with self.subTest(suffix=suffix):
                self.assertEqual(api_base_url(BASE + suffix), BASE)
                self.assertEqual(
                    scoped_stream_url(BASE + suffix, "maintenance"), BASE + "/maintenance/stream"
                )
        with self.assertRaises(ValueError):
            api_base_url("not a URL")

    def test_parser_ignores_sse_metadata_and_handles_multiline_data(self):
        events = list(iter_sse([
            ": heartbeat", "id: r:1", "event: answer.delta",
            'data: {"type":"answer.delta",', 'data: "data":{"text":"Xin chào"}}', "",
        ]))
        self.assertEqual(events, [{"type": "answer.delta", "data": {"text": "Xin chào"}}])


class StreamlitIntegrationTest(unittest.TestCase):
    def run_ui(self, api):
        # Exercise the real FastAPI routes without a socket or external services.
        stream_client = mock.MagicMock()
        stream_client.__enter__.return_value = stream_client
        stream_client.stream.side_effect = api.stream
        return (
            mock.patch("httpx.get", side_effect=lambda url, **kwargs: api.get(url)),
            mock.patch("httpx.Client", return_value=stream_client),
        )

    def test_enterprise_chat_and_switching_knowledge_bases(self):
        enterprise, maintenance = FakeAgent(), FakeAgent()
        with TestClient(create_app(knowledge_agents={
            "enterprise_rag": enterprise, "maintenance": maintenance,
        })) as api:
            get_patch, client_patch = self.run_ui(api)
            with get_patch, client_patch:
                ui = AppTest.from_file(str(APP)).run()
                self.assertFalse(ui.exception)
                self.assertEqual(ui.selectbox[0].value, "enterprise_rag")
                thread = ui.session_state.thread_id
                ui.chat_input[0].set_value("Who owns the project?").run()
                self.assertFalse(ui.exception)
                self.assertEqual(enterprise.calls, [("Who owns the project?", thread)])
                self.assertEqual(ui.session_state.messages[-1]["content"], "Hello")
                self.assertEqual(ui.status[0].state, "complete")
                ui.chat_input[0].set_value("Any follow-ups?").run()
                self.assertEqual(enterprise.calls[-1], ("Any follow-ups?", thread))
                ui.selectbox[0].select("maintenance").run()
                self.assertEqual(ui.session_state.messages, [])
                self.assertNotEqual(ui.session_state.thread_id, thread)
                ui.chat_input[0].set_value("sensor?").run()
                self.assertFalse(ui.exception)
                self.assertEqual(len(maintenance.calls), 1)
                self.assertEqual(len(enterprise.calls), 2)

    def test_backend_failure_is_shown_as_error(self):
        with TestClient(create_app(knowledge_agents={"enterprise_rag": FakeAgent(fail=True)})) as api:
            get_patch, client_patch = self.run_ui(api)
            with get_patch, client_patch:
                ui = AppTest.from_file(str(APP)).run()
                ui.chat_input[0].set_value("question").run()
                self.assertFalse(ui.exception)
                self.assertEqual(ui.status[0].state, "error")
                self.assertEqual(ui.session_state.messages[-1]["trace"][-1]["type"], "run.failed")

    def test_unavailable_api_disables_chat(self):
        with mock.patch("httpx.get", side_effect=httpx.ConnectError("offline")):
            ui = AppTest.from_file(str(APP)).run()
        self.assertFalse(ui.exception)
        self.assertTrue(ui.chat_input[0].disabled)
        self.assertTrue(ui.error)


if __name__ == "__main__":
    unittest.main()
