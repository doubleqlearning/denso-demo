"""Async OpenAI-compatible client for ModelArk with an RPM limiter, retries and quota stop.

Env: MODELARK_BASEURL, MODELARK_APIKEY, MODELARK_MODELNAME.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

RPM = 50
CONCURRENCY = 20
TIMEOUT = 120
MAX_ATTEMPTS = 5
BACKOFF = [5, 15, 40, 90]  # seconds before attempts 2..5 (~2.5 min in total)
# ModelArk/DeepSeek reasoning otherwise spends the whole token budget thinking and returns no content.
EXTRA_BODY = {"thinking": {"type": "disabled"}}

QUOTA_MARKERS = ("quota", "usage limit", "exceeded your", "insufficient", "billing", "5-hour", "5 hour")


class QuotaExhausted(RuntimeError):
    """The provider refuses further calls (usage window exhausted). Re-run later to resume."""


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """Sliding 60 s window: at most `rpm` request starts per minute."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.starts: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.starts and now - self.starts[0] >= 60:
                    self.starts.popleft()
                if len(self.starts) < self.rpm:
                    self.starts.append(now)
                    return
                await asyncio.sleep(60 - (now - self.starts[0]) + 0.05)


def parse_json(text: str) -> Any:
    """Parse the first JSON object in a reply, tolerating ```json fences and <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    body = text[start : end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Manual notation such as Un\G50 arrives as an invalid JSON escape: keep the backslash literally.
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", body))


class LLMClient:
    def __init__(self) -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        base_url = os.getenv("MODELARK_BASEURL")
        api_key = os.getenv("MODELARK_APIKEY")
        self.model = os.getenv("MODELARK_MODELNAME")
        if not (base_url and api_key and self.model):
            raise SystemExit("Set MODELARK_BASEURL, MODELARK_APIKEY and MODELARK_MODELNAME in .env")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT, max_retries=0)
        self.limiter = RateLimiter(RPM)
        self.semaphore = asyncio.Semaphore(CONCURRENCY)
        self.quota_hit = False
        self.calls = 0
        self.tokens = 0

    async def chat(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 8192) -> str:
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            if self.quota_hit:
                raise QuotaExhausted("quota exhausted earlier in this run")
            if attempt:
                await asyncio.sleep(BACKOFF[attempt - 1] * (1 + random.random() * 0.3))
            async with self.semaphore:
                await self.limiter.acquire()
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                        extra_body=EXTRA_BODY,
                    )
                except APIStatusError as exc:
                    last_error = exc
                    message = str(exc).lower()
                    if exc.status_code in (402, 403, 429) and any(m in message for m in QUOTA_MARKERS):
                        self.quota_hit = True
                        raise QuotaExhausted(str(exc)) from exc
                    if exc.status_code == 429 or exc.status_code >= 500:
                        continue
                    raise LLMError(str(exc)) from exc
                except (APITimeoutError, APIConnectionError) as exc:
                    last_error = exc
                    continue
            self.calls += 1
            if response.usage:
                self.tokens += response.usage.total_tokens or 0
            content = response.choices[0].message.content or ""
            if content.strip():
                return content
            last_error = LLMError("empty reply")
        if isinstance(last_error, APIStatusError) and last_error.status_code == 429:
            # Throttled through the whole backoff schedule: treat as an exhausted usage window.
            self.quota_hit = True
            raise QuotaExhausted(str(last_error))
        raise LLMError(f"failed after {MAX_ATTEMPTS} attempts: {last_error}")

    async def chat_json(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 8192,
                        attempts: int = 3) -> Any:
        error: Exception | None = None
        for _ in range(attempts):
            reply = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
            try:
                return parse_json(reply)
            except (ValueError, json.JSONDecodeError) as exc:
                error = exc
        raise LLMError(f"invalid JSON after {attempts} attempts: {error}")


async def run_all(jobs: Iterable[Callable[[], Awaitable[Any]]], on_result: Callable[[Any], None],
                  label: str = "jobs") -> bool:
    """Run jobs concurrently (the client limits throughput), handing each result over as it completes.

    Returns False when the provider quota was exhausted; finished results are kept.
    """
    tasks = [asyncio.ensure_future(job()) for job in jobs]
    total, done, failed = len(tasks), 0, 0
    quota = False
    try:
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
            except QuotaExhausted:
                quota = True
                break
            except Exception as exc:  # one bad item must not stop the batch
                failed += 1
                print(f"[{label}] failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                continue
            done += 1
            on_result(result)
            if done % 10 == 0 or done == total:
                print(f"[{label}] {done}/{total} done, {failed} failed", flush=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if quota:
        print(f"[{label}] STOPPED: provider quota exhausted after {done} results. Re-run the same command "
              "after the usage window resets to resume.", flush=True)
    return not quota
