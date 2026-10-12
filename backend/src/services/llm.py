"""OpenAI-compatible chat clients and JSON replies for the knowledge pipeline.

Extraction, verification, and consolidation each read their own
<PREFIX>_* settings, falling back to the shared OPENROUTER_* ones, so every
step can use another endpoint or model (e.g. a verifier from another family).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError


DEFAULT_LLM_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/coding/v3"


class LLMResponseError(RuntimeError):
    """Raised when an LLM reply cannot be obtained, parsed, or validated."""


def json_schema_format(
    name: str, schema: type[BaseModel] | dict[str, Any]
) -> dict[str, Any]:
    """response_format asking the endpoint for JSON matching a schema or model."""

    if not isinstance(schema, dict):
        schema = schema.model_json_schema()
    return {"type": "json_schema", "json_schema": {
        "name": name, "schema": schema,
        # Strict mode needs every key required and provider support;
        # the response is validated locally either way.
        "strict": False,
    }}


def chat_client_from_env(
    prefix: str,
    *,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    """Deterministic chat client configured by <prefix>_* or OPENROUTER_* settings.

    <prefix>_THINKING (enabled/disabled) is a ModelArk/DeepSeek request field;
    leave it unset for endpoints that do not accept it.
    """

    def setting(name: str) -> str | None:
        return os.getenv(f"{prefix}_{name}") or os.getenv(f"OPENROUTER_{name}")

    resolved_api_key = api_key or setting("API_KEY")
    resolved_model = model or setting("MODEL")
    missing = [
        f"OPENROUTER_{name}"
        for name, value in (("API_KEY", resolved_api_key), ("MODEL", resolved_model))
        if not value
    ]
    if missing:
        raise ValueError("Missing required LLM configuration: " + ", ".join(missing))

    options: dict[str, Any] = {}
    if response_format:
        options["model_kwargs"] = {"response_format": response_format}
    if thinking := os.getenv(f"{prefix}_THINKING", "").strip():
        if thinking not in {"enabled", "disabled"}:
            raise ValueError(f"{prefix}_THINKING must be enabled or disabled")
        options["extra_body"] = {"thinking": {"type": thinking}}
    if service_tier := os.getenv(f"{prefix}_SERVICE_TIER"):
        options["service_tier"] = service_tier
    if timeout := os.getenv(f"{prefix}_TIMEOUT_SECONDS"):
        try:
            options["timeout"] = float(timeout)
        except ValueError as exc:
            raise ValueError(f"{prefix}_TIMEOUT_SECONDS must be a number") from exc

    return ChatOpenAI(
        model=resolved_model,
        base_url=base_url or setting("BASE_URL") or DEFAULT_LLM_BASE_URL,
        api_key=resolved_api_key,
        temperature=0,
        streaming=False,
        max_tokens=max_tokens,
        **options,
    )


def client_provenance(client: Any) -> dict[str, Any]:
    """Non-secret settings of a client: model, token budget, thinking, format."""

    details: dict[str, Any] = {}
    for name, attribute in (("model", "model_name"), ("max_tokens", "max_tokens")):
        value = getattr(client, attribute, None)
        if value is not None:
            details[name] = value
    thinking = (getattr(client, "extra_body", None) or {}).get("thinking")
    if isinstance(thinking, dict) and "type" in thinking:
        details["thinking"] = thinking["type"]
    response_format = (getattr(client, "model_kwargs", None) or {}).get("response_format")
    if response_format:
        details["response_format"] = response_format
    return details


def invoke(client: Any, prompt: str, *, error: type[Exception] = LLMResponseError) -> Any:
    try:
        return client.invoke(prompt)
    except Exception as exc:
        raise error(f"LLM request failed: {exc}") from exc


async def ainvoke(client: Any, prompt: str, *, error: type[Exception] = LLMResponseError) -> Any:
    """Async call; clients without `ainvoke` run their blocking `invoke` in a thread."""

    method = getattr(client, "ainvoke", None)
    try:
        if method is not None:
            return await method(prompt)
        return await asyncio.to_thread(client.invoke, prompt)
    except Exception as exc:
        raise error(f"LLM request failed: {exc}") from exc


def response_json(
    response: Any,
    *,
    task: str,
    env_prefix: str,
    error: type[Exception] = LLMResponseError,
) -> Any:
    """Decoded JSON of a complete reply; `task` names the step in errors."""

    metadata = getattr(response, "response_metadata", None) or {}
    usage = getattr(response, "usage_metadata", None) or {}
    token_usage = metadata.get("token_usage") or {}
    finish_reason = metadata.get("finish_reason")
    output_tokens = usage.get("output_tokens", token_usage.get("completion_tokens"))
    reasoning_tokens = (usage.get("output_token_details") or {}).get(
        "reasoning", (token_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    )
    details = ", ".join(
        f"{key}={value}"
        for key, value in (
            ("finish_reason", finish_reason),
            ("output_tokens", output_tokens),
            ("reasoning_tokens", reasoning_tokens),
        )
        if value is not None
    )
    diagnostic = f" ({details})" if details else ""
    if finish_reason == "length":
        raise error(
            f"The LLM hit its output token limit before completing {task}"
            + diagnostic
            + ". Reasoning tokens also consume the output budget. For supporting "
            f"ModelArk/DeepSeek endpoints, set {env_prefix}_THINKING=disabled; "
            "otherwise reduce the input size or increase max_tokens."
        )
    content = response_text(response, error=error)
    if not content.strip():
        raise error(f"The LLM returned empty {task} text" + diagnostic)
    try:
        return json.loads(strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise error(
            f"The LLM returned invalid {task} JSON"
            + diagnostic
            + f": {exc.msg} at line {exc.lineno}, column {exc.colno}"
        ) from exc


def validated(
    output_model: type[BaseModel],
    payload: Any,
    *,
    task: str,
    error: type[Exception] = LLMResponseError,
) -> Any:
    """Validate a reply without quoting its contents in the error message."""

    try:
        return output_model.model_validate(payload)
    except ValidationError as exc:
        raise error(f"The LLM {task} JSON failed schema validation: {validation_summary(exc)}") from exc


def validation_summary(exc: ValidationError) -> str:
    errors = exc.errors(include_input=False, include_context=False, include_url=False)
    return "; ".join(_describe_error(error) for error in errors[:5])


def _describe_error(error: dict[str, Any]) -> str:
    location = ".".join(map(str, error["loc"])) or "<root>"
    if error["type"] == "value_error":
        # Raised by output schema validators, whose messages never quote input.
        return f"{location}: {error['msg']}"
    return f"{location}: {error['type']}"


def response_text(response: Any, *, error: type[Exception] = LLMResponseError) -> str:
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    # LangChain may represent multimodal message content as typed blocks.
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if text_parts:
            return "".join(text_parts)

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    raise error("The LLM response did not contain text")


def strip_json_fence(content: str) -> str:
    content = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    return fenced.group(1) if fenced else content
