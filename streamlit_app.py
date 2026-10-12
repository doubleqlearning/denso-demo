"""Small Streamlit client for the knowledge-agent SSE endpoint."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import httpx
import streamlit as st


DEFAULT_API_URL = os.getenv(
    "KNOWLEDGE_AGENT_API_URL",
    "http://localhost:8005/v1/knowledge-agent/stream",
)


def fetch_knowledge_bases(stream_url: str) -> list[str]:
    """Names served by the API, from the endpoint next to the stream URL."""

    list_url = api_base_url(stream_url) + "/knowledge-bases"
    response = httpx.get(list_url, timeout=5.0)
    response.raise_for_status()
    return [item["name"] for item in response.json()["knowledge_bases"]]


def api_base_url(endpoint: str) -> str:
    """Accept the API root, legacy stream URL, or a KB-specific stream URL."""
    url = urlsplit(endpoint.strip())
    path = re.sub(
        r"(/v1/knowledge-agent)(?:/[^/]+)?/stream$", r"\1", url.path.rstrip("/")
    )
    if url.scheme not in {"http", "https"} or not url.netloc or not path.endswith("/v1/knowledge-agent"):
        raise ValueError("URL phải trỏ tới /v1/knowledge-agent hoặc endpoint /stream của API")
    return urlunsplit((url.scheme, url.netloc, path, "", ""))


def scoped_stream_url(endpoint: str, kb: str) -> str:
    return f"{api_base_url(endpoint)}/{quote(kb, safe='')}/stream"


def iter_sse(lines: Iterable[str]) -> Iterator[dict]:
    """Parse JSON data fields from an SSE text stream."""

    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield json.loads("\n".join(data_lines))
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield json.loads("\n".join(data_lines))


def trace_label(event: dict) -> str:
    event_type = event.get("type", "unknown")
    data = event.get("data") or {}
    sequence = event.get("sequence", "?")
    if event_type == "tool.started":
        detail = json.dumps(data.get("input") or {}, ensure_ascii=False)
        return f"{sequence}. `{data.get('tool')}` started — {detail}"
    if event_type == "tool.completed":
        output = data.get("output") or {}
        count = output.get("count")
        suffix = f", {count} result(s)" if count is not None else ""
        return (
            f"{sequence}. `{data.get('tool')}` {data.get('status')}"
            f" in {data.get('duration_ms', '?')} ms{suffix}"
        )
    if event_type == "run.failed":
        return f"{sequence}. Failed — {data.get('message', 'Unknown error')}"
    return f"{sequence}. `{event_type}`"


def render_saved_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Citations"):
                st.json(message["citations"])
        if message.get("trace"):
            with st.expander("Trace"):
                st.json(message["trace"])


def main() -> None:
    st.set_page_config(page_title="Knowledge Agent Trace", page_icon="🔎", layout="wide")
    st.title("Knowledge Agent")
    st.caption("Stream câu trả lời, tool calls và kết quả retrieval đã được lọc an toàn.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("Knowledge Agent API", value=DEFAULT_API_URL)
        try:
            kb_names = fetch_knowledge_bases(api_url)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            kb_names = []
            st.error(f"Không lấy được danh sách knowledge base: {exc}")
        kb = st.selectbox(
            "Knowledge base", kb_names,
            index=kb_names.index("enterprise_rag") if "enterprise_rag" in kb_names else 0,
        )
        stream_url = scoped_stream_url(api_url, kb) if kb else None
        if stream_url:
            st.caption(f"SSE: {stream_url}")
        # Threads live in one knowledge base's agent; switching starts a new one.
        connection = (api_base_url(api_url), kb) if kb else None
        if st.session_state.get("connection") != connection:
            st.session_state.connection = connection
            st.session_state.kb = kb
            st.session_state.thread_id = uuid4().hex
            st.session_state.messages = []
        st.code(st.session_state.thread_id, language=None)
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = uuid4().hex
            st.session_state.messages = []
            st.rerun()

    for saved_message in st.session_state.messages:
        render_saved_message(saved_message)

    prompt = st.chat_input("Nhập câu hỏi về knowledge base...", disabled=not kb)
    if prompt:
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        render_saved_message(user_message)

        answer = ""
        citations: list[dict] = []
        trace: list[dict] = []
        failed = False
        completed = False

        with st.chat_message("assistant"):
            answer_slot = st.empty()
            run_status = st.status("Đang kết nối tới agent...", expanded=True)
            trace_slot = run_status.empty()

            try:
                timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
                with httpx.Client(timeout=timeout) as client:
                    with client.stream(
                        "POST",
                        stream_url,
                        json={
                            "query": prompt,
                            "thread_id": st.session_state.thread_id,
                        },
                        headers={"Accept": "text/event-stream"},
                    ) as response:
                        response.raise_for_status()
                        for event in iter_sse(response.iter_lines()):
                            event_type = event.get("type")
                            data = event.get("data") or {}
                            if event_type == "answer.delta":
                                answer += str(data.get("text") or "")
                                answer_slot.markdown(answer + "▌")
                            elif event_type == "answer.reset":
                                answer = ""
                                answer_slot.empty()
                            elif event_type == "answer.citations":
                                citations = list(data.get("citations") or [])
                                trace.append(event)
                            elif event_type == "run.failed":
                                failed = True
                                trace.append(event)
                                if not answer:
                                    answer = data.get("message", "Agent request failed")
                                    answer_slot.error(answer)
                                run_status.update(
                                    label=data.get("message", "Agent request failed"),
                                    state="error",
                                    expanded=True,
                                )
                            elif event_type == "run.completed":
                                completed = True
                                trace.append(event)
                            else:
                                trace.append(event)

                            if event_type != "answer.delta" and trace:
                                trace_slot.markdown(
                                    "\n\n".join(f"- {trace_label(item)}" for item in trace)
                                )

                answer_slot.markdown(answer or "_Agent không trả về nội dung._")
                if completed and not failed:
                    run_status.update(label="Hoàn tất", state="complete", expanded=False)
                elif not failed:
                    failed = True
                    run_status.update(
                        label="Agent stream kết thúc trước khi hoàn tất",
                        state="error",
                        expanded=True,
                    )
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                failed = True
                message = f"Không đọc được agent stream: {exc}"
                run_status.update(label=message, state="error", expanded=True)
                if not answer:
                    answer = message
                    answer_slot.error(message)

            if citations:
                with st.expander("Citations"):
                    st.json(citations)
            with st.expander("Raw trace"):
                st.json(trace)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer or "Agent không trả về nội dung.",
                "citations": citations,
                "trace": trace,
            }
        )


if __name__ == "__main__":
    main()
