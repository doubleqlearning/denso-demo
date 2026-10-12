"""LangGraph-backed ReAct agent that answers from the expert knowledge base."""

from __future__ import annotations

from collections.abc import Sequence

from backend.knowledge_bases.base import ContextFilter, Retrieval
from backend.src.agents.agent.state import KnowledgeAgentContext, KnowledgeAgentState
from backend.src.agents.agent.tools import (
    build_search_tool,
    get_document,
    get_evidence,
    get_knowledge,
    search_documents,
)


def build_tools(
    context_filters: Sequence[ContextFilter] = (),
    retrieval: Retrieval = Retrieval.KNOWLEDGE,
    *, filter_builder=None,
) -> list:
    if Retrieval(retrieval) is Retrieval.DOCUMENTS:
        return [search_documents, get_document]
    return [build_search_tool(context_filters, filter_builder=filter_builder), get_knowledge, get_evidence]


def build_graph(
    model,
    *,
    system_prompt: str,
    context_filters: Sequence[ContextFilter] = (),
    retrieval: Retrieval = Retrieval.KNOWLEDGE,
    checkpointer=None,
    filter_builder=None,
):
    """Build an agent graph around an injectable model, prompt, and checkpointer.

    The system prompt, search filters, and retrieval mode come from the
    knowledge base.
    """
    from langchain.agents import create_agent

    return create_agent(
        model=model,
        tools=build_tools(context_filters, retrieval, filter_builder=filter_builder),
        system_prompt=system_prompt,
        state_schema=KnowledgeAgentState,
        context_schema=KnowledgeAgentContext,
        checkpointer=checkpointer,
    )
