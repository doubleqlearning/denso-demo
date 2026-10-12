"""Unit tests for embedding client configuration (no network)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from backend.knowledge_bases import get_kb
from backend.src.storage import KnowledgeStore
from backend.src.storage.embeddings import (
    LOCAL_API_KEY_PLACEHOLDER,
    QueryInstructionEmbeddings,
    build_embeddings_from_env,
)


class RecordingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0]


LOCAL_ENV = {
    "EMBEDDING_MODEL": "microsoft/harrier-oss-v1-0.6b",
    "EMBEDDING_BASE_URL": "http://localhost:8080/v1",
    "EMBEDDING_DIM": "1024",
}


class QueryInstructionEmbeddingsTest(unittest.IsolatedAsyncioTestCase):
    async def test_only_queries_get_the_instruction(self) -> None:
        inner = RecordingEmbeddings()
        embeddings = QueryInstructionEmbeddings(inner, "Retrieve maintenance knowledge")

        embeddings.embed_documents(["doc"])
        await embeddings.aembed_documents(["doc 2"])
        embeddings.embed_query("jig alignment")
        await embeddings.aembed_query("sensor")

        self.assertEqual(inner.documents, ["doc", "doc 2"])
        self.assertEqual(
            inner.queries,
            [
                "Instruct: Retrieve maintenance knowledge\nQuery: jig alignment",
                "Instruct: Retrieve maintenance knowledge\nQuery: sensor",
            ],
        )


class BuildEmbeddingsFromEnvTest(unittest.TestCase):
    def test_local_server_needs_no_api_key(self) -> None:
        with patch.dict("os.environ", LOCAL_ENV, clear=True):
            embeddings = build_embeddings_from_env()
        self.assertIsInstance(embeddings, OpenAIEmbeddings)
        self.assertEqual(
            embeddings.openai_api_key.get_secret_value(), LOCAL_API_KEY_PLACEHOLDER
        )
        self.assertEqual(embeddings.openai_api_base, "http://localhost:8080/v1")

    def test_hosted_provider_still_requires_api_key(self) -> None:
        env = {"EMBEDDING_MODEL": "text-embedding-3-small", "EMBEDDING_DIM": "1536"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaisesRegex(ValueError, "EMBEDDING_API_KEY"):
                build_embeddings_from_env()

    def test_batch_size_and_query_instruction(self) -> None:
        env = {
            **LOCAL_ENV,
            "EMBEDDING_BATCH_SIZE": "32",
        }
        with patch.dict("os.environ", env, clear=True):
            embeddings = build_embeddings_from_env(
                query_instruction="  Retrieve maintenance knowledge  "
            )
        self.assertIsInstance(embeddings, QueryInstructionEmbeddings)
        self.assertEqual(embeddings.instruction, "Retrieve maintenance knowledge")
        self.assertEqual(embeddings.embeddings.chunk_size, 32)

    def test_legacy_env_cannot_override_or_supply_kb_instruction(self) -> None:
        env = {**LOCAL_ENV, "EMBEDDING_QUERY_INSTRUCTION": "Old global instruction"}
        with patch.dict("os.environ", env, clear=True):
            explicit = build_embeddings_from_env(query_instruction="Company knowledge")
            default = build_embeddings_from_env()
            blank = build_embeddings_from_env(query_instruction="  ")
        self.assertEqual(explicit.instruction, "Company knowledge")
        self.assertIsInstance(default, OpenAIEmbeddings)
        self.assertIsInstance(blank, OpenAIEmbeddings)

    def test_rejects_invalid_batch_size(self) -> None:
        for value in ("0", "-1", "many"):
            env = {**LOCAL_ENV, "EMBEDDING_BATCH_SIZE": value}
            with self.subTest(value=value), patch.dict("os.environ", env, clear=True):
                with self.assertRaisesRegex(ValueError, "EMBEDDING_BATCH_SIZE"):
                    build_embeddings_from_env()


class KnowledgeBaseEmbeddingTest(unittest.IsolatedAsyncioTestCase):
    async def test_store_uses_each_kbs_instruction_without_touching_documents(self):
        inner = [RecordingEmbeddings(), RecordingEmbeddings()]
        env = {
            **LOCAL_ENV,
            "DATABASE_URL": "postgresql+psycopg://user:pass@localhost/test",
            "EMBEDDING_QUERY_INSTRUCTION": "Wrong global task",
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch("backend.src.storage.databases.ensure_database", new_callable=AsyncMock) as ensure,
            patch("backend.src.storage.embeddings.OpenAIEmbeddings", side_effect=inner),
            patch.object(KnowledgeStore, "from_url") as from_url,
        ):
            for name in ("maintenance", "enterprise_rag"):
                spec = get_kb(name)
                await KnowledgeStore.for_kb(spec, init_schema=False)
                ensure.assert_awaited_with(env["DATABASE_URL"], spec.database)
                embeddings = from_url.call_args.args[1]
                await embeddings.aembed_query("Who owns this?")
                await embeddings.aembed_documents(["Original document"])

        self.assertNotEqual(inner[0].queries, inner[1].queries)
        for client, name in zip(inner, ("maintenance", "enterprise_rag")):
            self.assertEqual(client.queries, [
                f"Instruct: {get_kb(name).embedding_query_instruction}\nQuery: Who owns this?"
            ])
            self.assertEqual(client.documents, ["Original document"])

    async def test_injected_embedding_client_keeps_its_own_format(self):
        supplied = RecordingEmbeddings()
        with (
            patch.dict("os.environ", {
                "DATABASE_URL": "postgresql+psycopg://user:pass@localhost/test",
                "EMBEDDING_DIM": "1",
            }, clear=True),
            patch("backend.src.storage.databases.ensure_database", new_callable=AsyncMock),
            patch("backend.src.storage.embeddings.build_embeddings_from_env") as build,
            patch.object(KnowledgeStore, "from_url") as from_url,
        ):
            await KnowledgeStore.for_kb(get_kb("enterprise_rag"), embeddings=supplied, init_schema=False)
        build.assert_not_called()
        self.assertIs(from_url.call_args.args[1], supplied)


if __name__ == "__main__":
    unittest.main()
