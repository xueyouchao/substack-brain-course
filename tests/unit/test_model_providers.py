"""Contract tests for the provider layer (docs/model-provider-config.md §3-§7).

The seams under test are `resolve_model` and `call_llm`: every assertion goes
through those public entry points, so the internal resolution helpers and client
singletons can be rewritten without touching this file. HTTP is intercepted with
respx, so the suite needs neither the network nor a real API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from kb.agents import providers
from kb.config import Settings

ALL_PROVIDERS = (
    "anthropic",
    "openai",
    "ollama",
    "ollama-anthropic",
    "opencode",
    "opencode-anthropic",
)


@pytest.fixture(autouse=True)
def _fresh_provider_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean singleton cache: clients are process-global by design."""
    monkeypatch.setattr(providers, "_anthropic_clients", {})
    monkeypatch.setattr(providers, "_openai_clients", {})
    monkeypatch.setattr(providers, "_warned_unknown_pricing", set())


def test_resolve_model_without_prefix_uses_default_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        providers, "settings", Settings.model_validate({"default_provider": "anthropic"})
    )

    resolved = providers.resolve_model("claude-haiku-5-5")

    assert (resolved.provider, resolved.model_id) == ("anthropic", "claude-haiku-5-5")
    assert resolved.spec.api_family == "anthropic_messages"


def test_resolve_model_splits_on_first_colon_only() -> None:
    """"ollama:gemma4:31b" must keep the Ollama tag: only the first colon splits."""
    resolved = providers.resolve_model("ollama:gemma4:31b")

    assert (resolved.provider, resolved.model_id) == ("ollama", "gemma4:31b")
    assert resolved.spec.api_family == "openai_chat"


@pytest.mark.parametrize(
    ("ref", "expected_family"),
    [
        ("anthropic:claude-haiku-5-5", "anthropic_messages"),
        ("openai:gpt-4o-mini", "openai_chat"),
        ("ollama:gemma4:31b", "openai_chat"),
        ("ollama-anthropic:claude-haiku-5-5", "anthropic_messages"),
        ("opencode:glm-5.3", "openai_chat"),
        ("opencode-anthropic:claude-haiku-5-5", "anthropic_messages"),
    ],
)
def test_resolve_model_routes_every_declared_provider(ref: str, expected_family: str) -> None:
    provider, _, model_id = ref.partition(":")

    resolved = providers.resolve_model(ref)

    assert resolved.provider == provider
    assert resolved.model_id == model_id
    assert resolved.spec.api_family == expected_family


def test_resolve_model_treats_unprefixed_tag_as_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare Ollama tag is a model name, not a provider prefix, so it keeps routing."""
    monkeypatch.setattr(
        providers, "settings", Settings.model_validate({"default_provider": "ollama"})
    )

    resolved = providers.resolve_model("gemma4:31b")

    assert (resolved.provider, resolved.model_id) == ("ollama", "gemma4:31b")


@pytest.mark.parametrize("ref", ["google:gemini-2.5-pro", "groq:llama-3.3-70b"])
def test_resolve_model_rejects_a_provider_name_outside_the_table(ref: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        providers.resolve_model(ref)

    message = str(excinfo.value)
    assert ref.partition(":")[0] in message
    for provider in ALL_PROVIDERS:
        assert provider in message


class Verdict(BaseModel):
    verdict: str
    confidence: int


class _FakeMessagesAPI:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeCompletionsAPI:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeChatAPI:
    def __init__(self, response: Any) -> None:
        self.completions = _FakeCompletionsAPI(response)


class _FakeAnthropicClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessagesAPI(
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )
        )


class _FakeOpenAIClient:
    def __init__(self, content: str | None) -> None:
        self.chat = _FakeChatAPI(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )
        )


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    """Point the module at a known key/base_url set, isolated from any ambient .env."""
    values: dict[str, str] = {
        "anthropic_api_key": "sk-ant-test",
        "openai_api_key": "sk-openai-test",
        "ollama_api_key": "ollama-test",
        "opencode_api_key": "opencode-test",
    }
    values.update(overrides)
    monkeypatch.setattr(providers, "settings", Settings.model_validate(values))


def _fake_anthropic(monkeypatch: pytest.MonkeyPatch, text: str) -> _FakeAnthropicClient:
    """Stand in for the SDK client, which speaks httpx2 and so escapes respx."""
    client = _FakeAnthropicClient(text)
    monkeypatch.setattr(providers, "get_anthropic_client", lambda provider="anthropic": client)
    return client


def _fake_openai(monkeypatch: pytest.MonkeyPatch, content: str | None) -> _FakeOpenAIClient:
    client = _FakeOpenAIClient(content)
    monkeypatch.setattr(providers, "get_openai_client", lambda provider: client)
    return client


VERDICT_JSON = json.dumps({"verdict": "yes", "confidence": 3})


async def test_anthropic_native_still_emits_output_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backwards-compatibility guard: a bare model name must keep the native schema path."""
    _settings(monkeypatch)
    client = _fake_anthropic(monkeypatch, VERDICT_JSON)

    result = await providers.call_llm(
        system_prompt="sys", user_input="hi", model="claude-haiku-5-5", output_schema=Verdict
    )

    sent = client.messages.calls[-1]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["format"]["schema"]["required"] == ["verdict", "confidence"]
    assert sent["system"] == "sys"
    assert sent["model"] == "claude-haiku-5-5"
    assert "JSON Schema" not in sent["system"]
    assert result["output"] == {"verdict": "yes", "confidence": 3}


async def test_openai_chat_native_sends_loose_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    client = _fake_openai(monkeypatch, VERDICT_JSON)

    await providers.call_llm(
        system_prompt="sys", user_input="hi", model="openai:gpt-4o-mini", output_schema=Verdict
    )

    sent = client.chat.completions.calls[-1]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is False
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert "JSON Schema" not in sent["messages"][0]["content"]


async def test_ollama_json_mode_sends_json_object_and_injects_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    client = _fake_openai(monkeypatch, VERDICT_JSON)

    await providers.call_llm(
        system_prompt="sys", user_input="hi", model="ollama:gemma4:31b", output_schema=Verdict
    )

    sent = client.chat.completions.calls[-1]
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["model"] == "gemma4:31b"
    assert sent["messages"][0]["content"].startswith("sys")
    assert "JSON Schema" in sent["messages"][0]["content"]
    assert '"verdict"' in sent["messages"][0]["content"]


async def test_opencode_prompt_only_omits_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    client = _fake_openai(monkeypatch, VERDICT_JSON)

    await providers.call_llm(
        system_prompt="sys", user_input="hi", model="opencode:glm-5.3", output_schema=Verdict
    )

    sent = client.chat.completions.calls[-1]
    assert "response_format" not in sent
    assert "JSON Schema" in sent["messages"][0]["content"]
    assert sent["model"] == "glm-5.3"


async def test_ollama_anthropic_entry_uses_prompt_only_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    client = _fake_anthropic(monkeypatch, VERDICT_JSON)

    await providers.call_llm(
        system_prompt="sys",
        user_input="hi",
        model="ollama-anthropic:claude-haiku-5-5",
        output_schema=Verdict,
    )

    sent = client.messages.calls[-1]
    assert "output_config" not in sent
    assert "JSON Schema" in sent["system"]


def test_ollama_structured_output_env_override_reaches_the_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch, ollama_structured_output="native")

    assert providers._structured_output_mode("ollama", providers.PROVIDERS["ollama"]) == "native"


def test_client_endpoints_come_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, ollama_base_url="http://localhost:11434/v1")

    ollama = providers.get_openai_client("ollama")

    assert str(ollama.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert ollama.auth_headers["Authorization"] == "Bearer ollama-test"


def test_empty_base_url_falls_back_to_the_provider_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty base_url must mean "unset", so the client still gets a real endpoint.

    Settings drops blank values, so the value is forced past validation here: the
    guard has to survive a Settings built elsewhere, a future field whose default is
    "", and any code that assigns the attribute directly.
    """
    _settings(monkeypatch)
    for field in (
        "ollama_base_url",
        "opencode_base_url",
        "ollama_anthropic_base_url",
        "opencode_anthropic_base_url",
    ):
        monkeypatch.setattr(providers.settings, field, "")

    assert str(providers.get_openai_client("ollama").base_url).rstrip("/") == "https://ollama.com/v1"
    assert (
        str(providers.get_openai_client("opencode").base_url).rstrip("/")
        == "https://opencode.ai/inference/openai/v1"
    )
    assert str(providers.get_anthropic_client("ollama-anthropic").base_url) == "https://ollama.com"
    assert (
        str(providers.get_anthropic_client("opencode-anthropic").base_url).rstrip("/")
        == "https://opencode.ai/inference/anthropic"
    )


def test_whitespace_only_base_url_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    monkeypatch.setattr(providers.settings, "ollama_base_url", "   ")

    assert str(providers.get_openai_client("ollama").base_url).rstrip("/") == "https://ollama.com/v1"


def test_empty_anthropic_base_url_keeps_the_sdk_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic's "" is the exception: pass no base_url and let the SDK choose."""
    _settings(monkeypatch)
    monkeypatch.setattr(providers.settings, "anthropic_base_url", "")

    assert str(providers.get_anthropic_client("anthropic").base_url) == "https://api.anthropic.com"


def test_anthropic_family_clients_do_not_share_an_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """One cached client for every anthropic_messages provider would aim Ollama at Anthropic."""
    _settings(monkeypatch)

    native = providers.get_anthropic_client("anthropic")
    ollama = providers.get_anthropic_client("ollama-anthropic")

    assert ollama is not native
    assert str(ollama.base_url) == "https://ollama.com"
    assert str(native.base_url) == "https://api.anthropic.com"


def test_clients_are_cached_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)

    assert providers.get_openai_client("ollama") is providers.get_openai_client("ollama")
    assert providers.get_openai_client("ollama") is not providers.get_openai_client("opencode")

async def test_subscription_cost_is_zero_and_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat-rate plan has no per-token price, so the row must say so instead of lying."""
    _settings(monkeypatch)
    _fake_openai(monkeypatch, VERDICT_JSON)

    result = await providers.call_llm(
        system_prompt="sys", user_input="hi", model="opencode:glm-5.3", output_schema=Verdict
    )

    assert result["usage"]["est_usd"] == 0.0
    assert result["usage"]["pricing_mode"] == "subscription"
    assert result["usage"]["provider"] == "opencode"


async def test_per_token_cost_uses_the_price_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    _fake_anthropic(monkeypatch, VERDICT_JSON)

    result = await providers.call_llm(
        system_prompt="sys", user_input="hi", model="claude-haiku-5-5", output_schema=Verdict
    )

    # 11 input tokens at $1/Mtok + 7 output tokens at $5/Mtok.
    assert result["usage"]["est_usd"] == 0.000046
    assert result["usage"]["pricing_mode"] == "per_token"


async def test_unknown_pricing_warns_once_per_provider_and_model(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A loop over 20 batches must not turn one config gap into 20 lines of noise."""
    _settings(monkeypatch)
    _fake_openai(monkeypatch, VERDICT_JSON)

    for _ in range(3):
        result = await providers.call_llm(
            system_prompt="sys", user_input="hi", model="ollama:gemma4:31b", output_schema=Verdict
        )

    assert result["usage"]["est_usd"] == 0.0
    assert result["usage"]["pricing_mode"] == "unknown"
    warnings = [r for r in caplog.records if "pricing" in r.getMessage().lower()]
    assert len(warnings) == 1
    assert "gemma4:31b" in warnings[0].getMessage()


async def test_usage_dict_keeps_the_cost_middleware_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kb/inngest_client.py does a subset check on these four keys, so they are frozen."""
    _settings(monkeypatch)
    _fake_openai(monkeypatch, "plain answer")

    result = await providers.call_llm(system_prompt="sys", user_input="hi", model="openai:gpt-4o")

    assert {"model", "input_tokens", "output_tokens", "est_usd"} <= result["usage"].keys()
    assert result["usage"]["input_tokens"] == 11
    assert result["usage"]["output_tokens"] == 7
    assert result["usage"]["model"] == "openai:gpt-4o"
    assert result["output"] == "plain answer"


async def test_malformed_json_raises_with_provider_and_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    fenced_garbage = "here you go:" + chr(10) + chr(96) * 3 + "json" + chr(10) + '{"verdict": '
    _fake_openai(monkeypatch, fenced_garbage)

    with pytest.raises(RuntimeError) as excinfo:
        await providers.call_llm(
            system_prompt="sys", user_input="hi", model="opencode:glm-5.3", output_schema=Verdict
        )

    message = str(excinfo.value)
    assert "opencode" in message
    assert "glm-5.3" in message
    assert "here you go" in message


async def test_schema_mismatch_raises_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    _fake_openai(monkeypatch, json.dumps({"verdict": "yes"}))

    with pytest.raises(RuntimeError) as excinfo:
        await providers.call_llm(
            system_prompt="sys", user_input="hi", model="opencode:glm-5.3", output_schema=Verdict
        )

    assert "confidence" in str(excinfo.value)


async def test_missing_api_key_names_the_provider_and_the_env_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch, ollama_api_key="")

    with pytest.raises(RuntimeError) as excinfo:
        await providers.call_llm(system_prompt="sys", user_input="hi", model="ollama:gemma4:31b")

    message = str(excinfo.value)
    assert "ollama" in message
    assert "OLLAMA_API_KEY" in message


async def test_missing_transform_schema_downgrades_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transform_schema is an SDK internal (docs §10): losing it must not lose the call."""
    _settings(monkeypatch)
    client = _fake_anthropic(monkeypatch, VERDICT_JSON)

    def _gone(schema: object) -> dict[str, Any]:
        raise AttributeError("transform_schema")

    monkeypatch.setattr(providers.anthropic, "transform_schema", _gone)

    result = await providers.call_llm(
        system_prompt="sys", user_input="hi", model="claude-haiku-5-5", output_schema=Verdict
    )

    sent = client.messages.calls[-1]
    assert "output_config" not in sent
    assert "JSON Schema" in sent["system"]
    assert result["output"] == {"verdict": "yes", "confidence": 3}
