"""Official industrial QA pipeline: scoped retrieval, reranking, then answer."""
from pathlib import Path

from backend.knowledge_bases.industrial_qa.answering import IndustrialAnswerWriter
from backend.knowledge_bases.industrial_qa.retrieval import IndustrialRetriever, Ranking, RetrievalConfig
from backend.src.services import llm


def build_retriever(*, config=RetrievalConfig(), ranker=None):
    ranker = ranker if ranker is not None else llm.chat_client_from_env(
        "ANSWER", max_tokens=2048, response_format=llm.json_schema_format("source_ranking", Ranking))
    return IndustrialRetriever(ranker, config=config,
        prompt=Path(__file__).with_name("rerank.md").read_text().strip())


class IndustrialQaPipeline:
    def __init__(self, *, retriever=None, writer=None):
        self.retriever = retriever if retriever is not None else build_retriever()
        self.writer = writer if writer is not None else IndustrialAnswerWriter()

    async def retrieve(self, question, knowledge, documents, **kwargs):
        return await self.retriever.retrieve(question, knowledge, documents, **kwargs)

    async def write(self, question, retrieval, *, history=""):
        return await self.writer.write(question, {s.label: s.text for s in retrieval.sources}, history=history)
