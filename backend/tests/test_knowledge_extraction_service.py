"""Unit tests for the structured knowledge extraction service."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.knowledge_bases import get_kb
from backend.knowledge_bases.maintenance.schema import MaintenanceExtraction
from backend.src.services.knowledge_extraction import (
    KnowledgeExtractionError,
    KnowledgeExtractionService,
    KnowledgeType,
)


MAINTENANCE = get_kb("maintenance")
PROMPT = MAINTENANCE.extract_prompt
BENCHMARK_CSV = MAINTENANCE.dataset.path


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.last_prompt = prompt
        return SimpleNamespace(content=self.response)


class FailingLLM:
    def invoke(self, prompt: str) -> SimpleNamespace:
        raise ConnectionError("endpoint unavailable")


class KnowledgeExtractionServiceTests(unittest.TestCase):
    def test_provenance_identifies_prompt_and_settings_without_secrets(self) -> None:
        client = SimpleNamespace(
            model_name="test-model", max_tokens=8192, api_key="secret",
            extra_body={"thinking": {"type": "disabled"}, "secret": "hidden"},
        )
        service = KnowledgeExtractionService(client, PROMPT)
        self.assertEqual(service.provenance, {
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "model": "test-model", "max_tokens": 8192, "thinking": "disabled",
        })
        changed = KnowledgeExtractionService(client, PROMPT + "\nNew rule")
        self.assertNotEqual(service.provenance["prompt_sha256"], changed.provenance["prompt_sha256"])

    @unittest.skipUnless(BENCHMARK_CSV.is_file(), "benchmark CSV is not available")
    def test_extracts_validated_items_from_benchmark_conversation(self) -> None:
        with BENCHMARK_CSV.open(newline="", encoding="utf-8-sig") as csv_file:
            conversation = next(csv.DictReader(csv_file))["raw_conversation"]

        client = FakeLLM(
            """```json
            {
              "knowledge_items": [{
                "subject": "rear photoelectric sensor",
                "claim": "Clean oil buildup from the lens before replacing the sensor.",
                "context": {"machine_id": "M01", "condition": "unstable signal"},
                "type": "rule",
                "valid_from": null,
                "valid_to": null
              }]
            }
            ```"""
        )

        result = KnowledgeExtractionService(client, PROMPT).extract(conversation)

        self.assertEqual(len(result.knowledge_items), 1)
        self.assertEqual(result.knowledge_items[0].type, KnowledgeType.RULE)
        self.assertEqual(result.knowledge_items[0].context["machine_id"], "M01")
        self.assertIn(conversation, client.last_prompt or "")
        self.assertNotIn("{RAW_CONVERSATION}", client.last_prompt or "")

    def test_uses_the_given_prompt_template(self) -> None:
        client = FakeLLM('{"knowledge_items": []}')

        KnowledgeExtractionService(client, "Other KB:\n{RAW_CONVERSATION}").extract(
            "  Expert: Check it.  "
        )

        self.assertEqual(client.last_prompt, "Other KB:\nExpert: Check it.")

    def test_compact_enterprise_output_uses_optional_field_defaults(self) -> None:
        client = FakeLLM(
            '{"knowledge_items":[{"subject":"uploads","claim":"Limit is 20 MB.","type":"rule"}]}'
        )
        item, = KnowledgeExtractionService(
            client, get_kb("enterprise_rag").extract_prompt
        ).extract("uploads\nLimit is 20 MB.").knowledge_items
        self.assertEqual(item.context, {})
        self.assertIsNone(item.valid_from)
        self.assertIsNone(item.valid_to)
        self.assertEqual(item.claim, "Limit is 20 MB.")

    def test_rejects_prompt_without_placeholder(self) -> None:
        with self.assertRaisesRegex(ValueError, "RAW_CONVERSATION"):
            KnowledgeExtractionService(FakeLLM("{}"), "no placeholder")

    def test_rejects_empty_conversation_before_calling_llm(self) -> None:
        client = FakeLLM('{"knowledge_items": []}')

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            KnowledgeExtractionService(client, PROMPT).extract("  ")

        self.assertIsNone(client.last_prompt)

    def test_rejects_invalid_llm_json(self) -> None:
        client = FakeLLM("not JSON")

        with self.assertRaisesRegex(KnowledgeExtractionError, "Expecting value at line 1, column 1"):
            KnowledgeExtractionService(client, PROMPT).extract("Expert: Check the sensor.")

    def test_reports_empty_response(self) -> None:
        with self.assertRaisesRegex(KnowledgeExtractionError, "empty extraction text"):
            KnowledgeExtractionService(FakeLLM("  "), PROMPT).extract("Expert: x")

    def test_reports_token_limit_including_reasoning_usage(self) -> None:
        response = SimpleNamespace(
            content="",
            response_metadata={"finish_reason": "length"},
            usage_metadata={
                "output_tokens": 8192,
                "output_token_details": {"reasoning": 8192},
            },
        )
        service = KnowledgeExtractionService(mock.Mock(invoke=mock.Mock(return_value=response)), PROMPT)
        with self.assertRaisesRegex(
            KnowledgeExtractionError,
            r"output token limit.*finish_reason=length, output_tokens=8192, reasoning_tokens=8192",
        ):
            service.extract("Expert: x")

    def test_does_not_accept_truncated_response_even_if_json_is_valid(self) -> None:
        response = SimpleNamespace(
            content='{"knowledge_items": []}',
            response_metadata={
                "finish_reason": "length",
                "token_usage": {"completion_tokens": 8192},
            },
        )
        service = KnowledgeExtractionService(mock.Mock(invoke=mock.Mock(return_value=response)), PROMPT)
        with self.assertRaisesRegex(KnowledgeExtractionError, "output_tokens=8192"):
            service.extract("Expert: x")

    def test_schema_error_names_field_without_logging_document(self) -> None:
        client = FakeLLM(
            '{"knowledge_items": [{"subject": "sensor", "claim": "private claim", '
            '"type": "private invalid value"}]}'
        )
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            with self.assertRaisesRegex(
                KnowledgeExtractionError, r"schema validation: knowledge_items.0.type: enum"
            ) as caught:
                KnowledgeExtractionService(client, PROMPT).extract("Expert: x")
        self.assertNotIn("private", str(caught.exception))
        self.assertEqual(output.getvalue(), "")

    def test_wraps_llm_client_errors(self) -> None:
        with self.assertRaisesRegex(
            KnowledgeExtractionError, "LLM request failed: endpoint unavailable"
        ):
            KnowledgeExtractionService(FailingLLM(), PROMPT).extract(
                "Expert: Check the sensor."
            )


CASE = {
    "subject": "WG-21", "valid_from": None, "valid_to": None,
    "context": {"machine_id": "M03", "machine_model": None, "condition": "cold start"},
    "case_card": {
        "context": "Cold start, line 2.", "observation": "Misfire on first welds.",
        "reasoning": "Loose cable connector.", "action": "Reseat the connector.",
        "outcome": None, "evidence": [{"turn_id": "T02", "quote": "It misfires cold."}],
    },
}


def lesson(**fields) -> dict:
    return {
        "subject": "weld gun", "claim": "Check the cable connector first on cold misfires.",
        "steps": None, "mandatory": False, "valid_from": None, "valid_to": None,
        "context": {"machine_model": "WG"}, "case_refs": [], **fields,
    }


def copy(value: dict) -> dict:
    return json.loads(json.dumps(value))


class MaintenanceExtractionTests(unittest.TestCase):
    def _extract(self, cases: list[dict], lessons: list[dict] = ()):
        client = FakeLLM(json.dumps({"cases": cases, "lessons": list(lessons)}))
        service = KnowledgeExtractionService(
            client, PROMPT, output_model=MaintenanceExtraction
        )
        return service.extract("Expert: x").knowledge_items

    def test_case_claim_is_rendered_from_its_card(self) -> None:
        item, = self._extract([CASE])
        self.assertEqual(item.type, KnowledgeType.CASE)
        self.assertEqual(item.claim, (
            "Cold start, line 2.\nObservation: Misfire on first welds.\n"
            "Reasoning: Loose cable connector.\nAction: Reseat the connector.\n"
            "Outcome: not stated"
        ))
        # A null scope key is dropped so the record stays unscoped for it.
        self.assertEqual(set(item.context), {"machine_id", "condition", "case_card"})
        card = item.context["case_card"]
        # Nested nulls stay: an explicit null outcome is part of the card.
        self.assertIsNone(card["outcome"])
        self.assertEqual(card["evidence"][0], {"turn_id": "T02", "quote": "It misfires cold."})
        self.assertEqual(item.supported_by, [])

    def test_lesson_type_is_derived_from_its_fields(self) -> None:
        steps = ["Isolate power.", "Reseat the connector."]
        items = self._extract([], [
            lesson(),
            lesson(steps=steps),
            lesson(mandatory=True),
            lesson(mandatory=True, steps=steps),
        ])
        self.assertEqual([item.type for item in items], [
            KnowledgeType.HEURISTIC, KnowledgeType.PROCEDURE,
            KnowledgeType.RULE, KnowledgeType.RULE,
        ])

    def test_lesson_steps_are_rendered_into_the_claim(self) -> None:
        item, = self._extract([], [lesson(steps=["Isolate power.", "Reseat the connector."])])
        self.assertEqual(item.claim, (
            "Check the cable connector first on cold misfires.\n"
            "1. Isolate power.\n2. Reseat the connector."
        ))
        self.assertEqual(item.context, {
            "machine_model": "WG", "steps": ["Isolate power.", "Reseat the connector."],
        })

    def test_case_refs_become_item_positions_after_cases(self) -> None:
        other = copy(CASE)
        other["subject"] = "WG-22"
        items = self._extract([CASE, other], [lesson(case_refs=[1, 0, 1])])
        self.assertEqual([item.type for item in items[:2]], [KnowledgeType.CASE] * 2)
        self.assertEqual(items[2].supported_by, [0, 1])

    def test_rejects_output_that_breaks_the_schema(self) -> None:
        def changed_card(**card) -> dict:
            case = copy(CASE)
            case["case_card"].update(card)
            return case

        no_card = copy(CASE)
        del no_card["case_card"]
        extra_scope = copy(CASE)
        extra_scope["context"]["shift"] = "night"
        written_claim = copy(CASE)
        written_claim["claim"] = "Restated card."
        rejected = {
            "cases.0.case_card: missing": ([no_card], []),
            "case_card.evidence": ([changed_card(evidence=[])], []),
            "case_card.severity": ([changed_card(severity="high")], []),
            "cases.0.context.shift": ([extra_scope], []),
            "cases.0.claim": ([written_claim], []),
            "lessons.0.case_refs points to no case": ([CASE], [lesson(case_refs=[1])]),
            "lessons.0.type": ([], [lesson(type="rule")]),
        }
        for location, (cases, lessons) in rejected.items():
            with self.subTest(location=location):
                with self.assertRaisesRegex(KnowledgeExtractionError, location):
                    self._extract(cases, lessons)

    def test_maintenance_kb_uses_case_lesson_schema(self) -> None:
        self.assertIs(MAINTENANCE.extraction_output_model, MaintenanceExtraction)
        self.assertTrue(MAINTENANCE.review_required)
        schema = KnowledgeExtractionService.output_schema(MaintenanceExtraction)
        self.assertEqual(list(schema["properties"]), ["cases", "lessons"])
        self.assertFalse(schema["additionalProperties"])
        case = schema["$defs"]["MaintenanceCase"]
        self.assertNotIn("claim", case["properties"])
        self.assertIn("case_card", case["required"])
        self.assertNotIn("type", schema["$defs"]["MaintenanceLesson"]["properties"])
        self.assertFalse(schema["$defs"]["MaintenanceScope"]["additionalProperties"])


class SupportLinkTests(unittest.TestCase):
    def test_rejects_self_and_missing_support_links(self) -> None:
        for refs in ([0], [2], [-1]):
            with self.subTest(refs=refs):
                payload = {"knowledge_items": [
                    {"subject": "a", "claim": "A.", "type": "rule", "supported_by": refs},
                    {"subject": "b", "claim": "B.", "type": "case"},
                ]}
                with self.assertRaisesRegex(
                    KnowledgeExtractionError,
                    "knowledge_items.0.supported_by points to no other item",
                ):
                    KnowledgeExtractionService(FakeLLM(json.dumps(payload)), PROMPT).extract("x")

    def test_accepts_links_to_other_items(self) -> None:
        payload = {"knowledge_items": [
            {"subject": "a", "claim": "A.", "type": "rule", "supported_by": [1]},
            {"subject": "b", "claim": "B.", "type": "case"},
        ]}
        items = KnowledgeExtractionService(FakeLLM(json.dumps(payload)), PROMPT).extract("x")
        self.assertEqual(items.knowledge_items[0].supported_by, [1])


class AsyncFakeLLM(FakeLLM):
    def __init__(self, response: str):
        super().__init__(response)
        self.async_calls = 0

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        self.async_calls += 1
        return self.invoke(prompt)


VALID_RESPONSE = (
    '{"knowledge_items": [{"subject": "sensor", "claim": "Clean the lens.", '
    '"context": {}, "type": "procedure"}]}'
)


class AsyncKnowledgeExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_aextract_uses_async_client(self) -> None:
        client = AsyncFakeLLM(VALID_RESPONSE)
        result = await KnowledgeExtractionService(client, PROMPT).aextract("Expert: x")
        self.assertEqual(client.async_calls, 1)
        self.assertEqual(result.knowledge_items[0].claim, "Clean the lens.")

    async def test_aextract_falls_back_to_sync_client(self) -> None:
        client = FakeLLM(VALID_RESPONSE)
        result = await KnowledgeExtractionService(client, PROMPT).aextract("Expert: x")
        self.assertEqual(len(result.knowledge_items), 1)
        self.assertIn("Expert: x", client.last_prompt)

    async def test_aextract_wraps_errors_and_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(KnowledgeExtractionError, "LLM request failed"):
            await KnowledgeExtractionService(FailingLLM(), PROMPT).aextract("Expert: x")
        with self.assertRaises(ValueError):
            await KnowledgeExtractionService(FakeLLM("{}"), PROMPT).aextract(" ")

    async def test_aextract_reports_json_error_details(self) -> None:
        with self.assertRaisesRegex(KnowledgeExtractionError, "line 1, column 1"):
            await KnowledgeExtractionService(AsyncFakeLLM("not JSON"), PROMPT).aextract("Expert: x")


class ExtractionClientConfigTests(unittest.TestCase):
    def test_json_mode_is_explicit_and_tracked_in_provenance(self) -> None:
        with mock.patch.dict(os.environ, self.SHARED, clear=True):
            service = KnowledgeExtractionService.from_env(PROMPT, json_mode=True)
            self.assertEqual(service._client._get_request_payload("Extract JSON.")["response_format"],
                             {"type": "json_object"})
            self.assertEqual(service.provenance["response_format"], {"type": "json_object"})

    def test_json_schema_requires_knowledge_item_type(self) -> None:
        with mock.patch.dict(os.environ, self.SHARED, clear=True):
            service = KnowledgeExtractionService.from_env(PROMPT, json_schema=True)
            response_format = service._client._get_request_payload("Extract JSON.")["response_format"]
            self.assertEqual(response_format["type"], "json_schema")
            schema = response_format["json_schema"]["schema"]
            self.assertIn("type", schema["$defs"]["KnowledgeItem"]["required"])
            self.assertEqual(service.provenance["response_format"], response_format)

    def test_json_schema_sends_output_model_schema(self) -> None:
        with mock.patch.dict(os.environ, self.SHARED, clear=True):
            service = KnowledgeExtractionService.from_env(
                PROMPT, json_schema=True, output_model=MaintenanceExtraction
            )
            response_format = service._client._get_request_payload("Extract JSON.")["response_format"]
            self.assertIn("CaseCard", response_format["json_schema"]["schema"]["$defs"])
            self.assertEqual(service.provenance["output_model"], "MaintenanceExtraction")

    SHARED = {
        "OPENROUTER_API_KEY": "shared-key",
        "OPENROUTER_MODEL": "shared-model",
        "OPENROUTER_BASE_URL": "https://example.test/api/coding/v3",
    }

    def _client(self, **env):
        with mock.patch.dict(os.environ, {**self.SHARED, **env}, clear=True):
            return KnowledgeExtractionService.from_env(PROMPT)._client

    def test_defaults_to_shared_llm_settings(self) -> None:
        client = self._client()
        self.assertEqual(client.model_name, "shared-model")
        self.assertEqual(client.openai_api_base, "https://example.test/api/coding/v3")
        self.assertIsNone(client.service_tier)
        self.assertIsNone(client.extra_body)

    def test_configures_provider_thinking_switch(self) -> None:
        for mode in ("disabled", "enabled"):
            with self.subTest(mode=mode):
                client = self._client(EXTRACTION_THINKING=mode)
                payload = client._get_request_payload("Extract a fact.")
                self.assertEqual(payload["extra_body"], {"thinking": {"type": mode}})

    def test_blank_thinking_keeps_provider_default(self) -> None:
        self.assertIsNone(self._client(EXTRACTION_THINKING=" ").extra_body)

    def test_rejects_invalid_thinking_switch(self) -> None:
        with self.assertRaisesRegex(ValueError, "EXTRACTION_THINKING must be"):
            self._client(EXTRACTION_THINKING="off")

    def test_extraction_overrides_for_modelark_bulk_endpoints(self) -> None:
        client = self._client(
            EXTRACTION_API_KEY="platform-key",
            EXTRACTION_MODEL="ep-bi-123",
            EXTRACTION_BASE_URL="https://example.test/api/v3/batch",
            EXTRACTION_SERVICE_TIER="flex",
            EXTRACTION_TIMEOUT_SECONDS="86400",
        )
        self.assertEqual(client.model_name, "ep-bi-123")
        self.assertEqual(client.openai_api_base, "https://example.test/api/v3/batch")
        self.assertEqual(client.openai_api_key.get_secret_value(), "platform-key")
        self.assertEqual(client.service_tier, "flex")
        self.assertEqual(client.request_timeout, 86400.0)

    def test_rejects_non_numeric_timeout(self) -> None:
        with self.assertRaisesRegex(ValueError, "EXTRACTION_TIMEOUT_SECONDS"):
            self._client(EXTRACTION_TIMEOUT_SECONDS="1h")


if __name__ == "__main__":
    unittest.main()
