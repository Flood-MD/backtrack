import json

import httpx
import pytest

from backtrack.config import new_provider
from backtrack.llm import AnthropicClient, LLMError, OpenAICompatClient, extract_json, make_client


def client(handler, kind, preset):
    p = new_provider(preset, model="m1", api_key="sk-test")
    cls = AnthropicClient if kind == "anthropic" else OpenAICompatClient
    return cls(p, httpx.Client(transport=httpx.MockTransport(handler)))


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.parametrize("preset,param", [("openai", "max_completion_tokens"), ("openrouter", "max_tokens"),
                                          ("huggingface", "max_tokens")])
def test_openai_compat_request_shape(preset, param):
    seen = {}

    def h(req):
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers["authorization"], json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    assert client(h, "openai", preset).complete("sys", "usr", max_tokens=50) == "hi"
    assert seen["url"].endswith("/chat/completions") and seen["auth"] == "Bearer sk-test"
    assert seen["body"][param] == 50 and seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_anthropic_request_shape():
    seen = {}

    def h(req):
        seen["url"], seen["headers"], seen["body"] = str(req.url), req.headers, json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}]})

    assert client(h, "anthropic", "anthropic").complete("sys", "usr") == "hello"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-test" and "anthropic-version" in seen["headers"]
    assert seen["body"]["system"] == "sys" and seen["body"]["messages"][0]["role"] == "user"


def test_retries_then_succeeds_and_hard_errors_do_not_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    n = {"c": 0}

    def flaky(req):
        n["c"] += 1
        return httpx.Response(429) if n["c"] < 3 else httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert client(flaky, "openai", "openai").complete("s", "u") == "ok" and n["c"] == 3
    n["c"] = 0

    def unauthorized(req):
        n["c"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(LLMError, match="401"):
        client(unauthorized, "openai", "openai").complete("s", "u")
    assert n["c"] == 1


def test_complete_json_retries_on_bad_json():
    replies = iter(["not json", '{"ok": true}'])

    def h(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    assert client(h, "openai", "openai").complete_json("s", "u") == {"ok": True}


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        make_client(new_provider("openai"))


def test_env_key_wins_over_stored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert new_provider("openai", api_key="stored").resolve_key() == "from-env"
