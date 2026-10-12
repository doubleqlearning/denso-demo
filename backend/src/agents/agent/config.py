"""Chat-model construction and run limits for the knowledge agent."""

from __future__ import annotations

import os

from backend.src.services.llm import DEFAULT_LLM_BASE_URL


MODEL_TIMEOUT_SECONDS = 60
MODEL_MAX_RETRIES = 2
# Reasoning models spend part of this budget thinking before the visible answer.
MODEL_MAX_TOKENS = 8000
RUN_TIMEOUT_SECONDS = 90
RECURSION_LIMIT = 20


def build_chat_model(
    *,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_LLM_BASE_URL,
    timeout_seconds: float = MODEL_TIMEOUT_SECONDS,
):
    """Build the OpenAI-compatible streaming chat client used by the agent."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        timeout=timeout_seconds,
        max_retries=MODEL_MAX_RETRIES,
        streaming=True,
        stream_usage=True,
        max_tokens=MODEL_MAX_TOKENS,
    )


def build_chat_model_from_env():
    """Build the chat client from the same OPENROUTER_* settings as extraction."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    missing = [
        name
        for name, value in (("OPENROUTER_API_KEY", api_key), ("OPENROUTER_MODEL", model))
        if not value
    ]
    if missing:
        raise ValueError("Missing required LLM configuration: " + ", ".join(missing))
    return build_chat_model(
        api_key=api_key,
        model=model,
        base_url=os.getenv("OPENROUTER_BASE_URL") or DEFAULT_LLM_BASE_URL,
    )
