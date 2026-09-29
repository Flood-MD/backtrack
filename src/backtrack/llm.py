"""LLM connectors: OpenAI-compatible (OpenAI, OpenRouter, HuggingFace, local) and Anthropic."""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

import httpx

from .config import Config, ProviderConfig


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    name = "llm"
    model = ""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 4096, temperature: float = 0.2) -> str: ...

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> dict | list:
        """Ask for JSON; parse leniently; retry once with the parse error fed back."""
        system = system + "\n\nRespond with a single valid JSON value and nothing else."
        text = self.complete(system, user, max_tokens=max_tokens, temperature=0.1)
        try:
            return extract_json(text)
        except ValueError as e:
            retry = f"{user}\n\nYour previous reply was not valid JSON ({e}). Reply again with only valid JSON."
            return extract_json(self.complete(system, retry, max_tokens=max_tokens, temperature=0.0))


def extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found")
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        raise ValueError("unterminated JSON")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(str(e)) from e


def _post_with_retry(client: httpx.Client, url: str, *, headers: dict, payload: dict, tries: int = 4) -> dict:
    delay = 2.0
    last = ""
    for attempt in range(tries):
        try:
            r = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            last = f"network error: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise LLMError(last)
        if attempt < tries - 1:
            time.sleep(delay)
            delay *= 2
    raise LLMError(f"giving up after {tries} attempts; last error: {last}")


class OpenAICompatClient(LLMClient):
    def __init__(self, cfg: ProviderConfig, http: httpx.Client | None = None):
        self.cfg, self.name, self.model = cfg, cfg.name, cfg.model
        if not cfg.model:
            raise LLMError(f"Provider {cfg.name!r} has no model set")
        self.http = http or httpx.Client(timeout=300)

    def complete(self, system, user, max_tokens=4096, temperature=0.2):
        headers = {"Content-Type": "application/json", **self.cfg.extra_headers}
        if key := self.cfg.resolve_key():
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            self.cfg.token_param or "max_tokens": max_tokens,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        try:
            data = _post_with_retry(self.http, url, headers=headers, payload=payload)
        except LLMError as e:
            # Some reasoning models reject a custom temperature.
            if "temperature" in str(e) and "HTTP 400" in str(e):
                payload.pop("temperature")
                data = _post_with_retry(self.http, url, headers=headers, payload=payload)
            else:
                raise
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e


class AnthropicClient(LLMClient):
    API_VERSION = "2023-06-01"

    def __init__(self, cfg: ProviderConfig, http: httpx.Client | None = None):
        self.cfg, self.name, self.model = cfg, cfg.name, cfg.model
        if not cfg.model:
            raise LLMError(f"Provider {cfg.name!r} has no model set")
        self.http = http or httpx.Client(timeout=300)

    def complete(self, system, user, max_tokens=4096, temperature=0.2):
        headers = {
            "x-api-key": self.cfg.resolve_key(),
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
            **self.cfg.extra_headers,
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        url = self.cfg.base_url.rstrip("/") + "/v1/messages"
        data = _post_with_retry(self.http, url, headers=headers, payload=payload)
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e


def make_client(cfg: ProviderConfig, http: httpx.Client | None = None) -> LLMClient:
    if not cfg.resolve_key() and cfg.base_url.startswith("https://"):
        raise LLMError(
            f"No API key for provider {cfg.name!r}. Set ${cfg.api_key_env or 'YOUR_KEY_VAR'} "
            "or run `backtrack provider add`."
        )
    if cfg.kind == "anthropic":
        return AnthropicClient(cfg, http)
    if cfg.kind == "openai":
        return OpenAICompatClient(cfg, http)
    raise LLMError(f"Unknown provider kind {cfg.kind!r}")


def client_from_config(cfg: Config, name: str | None = None) -> LLMClient:
    return make_client(cfg.provider(name))
