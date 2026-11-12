"""Cost visibility and provider routing for POST /ask (ticket T3).

The seam is `ask_question`: `/ask` returns its dict untouched, so this is the one
place the public response contract can be observed. Retrieval and the snapshot
write are patched, because /ask is a read path with no Inngest step and needs
neither a database nor a network; the LLM itself is either a recorded fake or the
real `call_llm` over a fake SDK client.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from kb.agents import providers
from kb.config import Settings, settings
from kb.functions import ask as ask_module

QUESTION = "How does Inngest memoize completed steps?"
ANSWER_JSON_SAFE = "Inngest memoizes completed steps [1]."

PASSAGES: list[dict[str, Any]] = [
    {
        "passage_id": "passage-1",
        "title": "Durable steps",
        "author": "Ada",
        "url": "https://example.com/durable-steps",
        "text": "Inngest memoizes completed steps so a retry does not re-run them.",
        "published_at": "2026-01-02",
    },
    {
        "passage_id": "passage-2",
        "title": "Raft",
        "author": "Grace",
        "url": "https://example.com/raft",
        "text": "A Raft cluster elects a leader.",
        "published_at": "2026-02-03",
    },
]


class _FakeSession:
    """Only `commit` is reachable: the snapshot helper replaces the DB work."""

    async def commit(self) -> None:
        return None


class _FakeMessagesAPI:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _FakeCompletionsAPI:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


class _FakeOpenAIClient:
    """The Chat Completions stand-in, for every non-Anthropic provider."""

    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletionsAPI(content))


class _FakeAnthropicClient:
    """Stand-in for the SDK client, which speaks httpx2 and so escapes respx."""

    def __init__(self, text: str) -> None:
        self.messages = _FakeMessagesAPI(
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )
        )


def _patch_ask_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace retrieval, the snapshot write and the event emit for /ask."""

    async def fake_hybrid_search(
        session: Any, *, query: str, limit: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return PASSAGES, list(PASSAGES)

    async def fake_record_snapshot(session: Any, **kwargs: Any) -> str:
        return "snapshot-1"

    async def fake_emit_answer_produced(snapshot_id: str, retriever_variant: str) -> None:
        return None

    monkeypatch.setattr(ask_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(ask_module, "record_snapshot", fake_record_snapshot)
    monkeypatch.setattr(ask_module, "_emit_answer_produced", fake_emit_answer_produced)


@pytest.fixture
def call_log(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the kwargs of every `call_llm` invocation, in order."""
    log: list[dict[str, Any]] = []

    async def fake_call_llm(**kwargs: Any) -> dict[str, Any]:
        log.append(kwargs)
        return {
            "output": ANSWER_JSON_SAFE,
            "usage": {
                "model": kwargs["model"],
                "input_tokens": 120,
                "output_tokens": 8,
                "est_usd": 0.0002,
            },
        }

    _patch_ask_io(monkeypatch)
    monkeypatch.setattr(ask_module, "call_llm", fake_call_llm)
    return log


async def test_configured_synthesis_model_is_what_reaches_call_llm(
    monkeypatch: pytest.MonkeyPatch, call_log: list[dict[str, Any]]
) -> None:
    """SYNTHESIS_MODEL=ollama:gemma4:31b must reach the transport verbatim.

    The provider prefix is resolved inside call_llm, so /ask only has to stop
    hard-coding a direct Anthropic client and pass the setting through.
    """
    monkeypatch.setattr(settings, "synthesis_model", "ollama:gemma4:31b")

    await ask_module.ask_question(_FakeSession(), QUESTION)  # type: ignore[arg-type]
    await asyncio.sleep(0)  # let the fire-and-forget emit task finish

    assert len(call_log) == 1
    call = call_log[0]
    assert call["model"] == "ollama:gemma4:31b"
    assert call["system_prompt"] == ask_module.SYSTEM_PROMPT
    assert call["max_tokens"] == settings.synthesis_max_tokens
    # /ask answers in prose, so it relies on the default instead of passing None.
    assert "output_schema" not in call


async def test_synthesis_cost_is_persisted(call_log: list[dict[str, Any]]) -> None:
    """The costs row must be written even though /ask never enters an Inngest step."""
    await ask_module.ask_question(_FakeSession(), QUESTION)  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert call_log[0]["persist_cost"] is True


async def test_cost_row_reaches_the_writer_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real call_llm, fake SDK client: /ask's spend lands in the costs writer.

    Worked example (docs §7, per_token): 11 input tokens at 1.00 USD/Mtok plus
    7 output tokens at 5.00 USD/Mtok for claude-haiku-5-5.
    """
    _patch_ask_io(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_model", "anthropic:claude-haiku-5-5")
    monkeypatch.setattr(
        providers,
        "settings",
        Settings.model_validate(
            {"anthropic_api_key": "sk-ant-test", "default_provider": "anthropic"}
        ),
    )
    monkeypatch.setattr(
        providers,
        "get_anthropic_client",
        lambda provider="anthropic": _FakeAnthropicClient(ANSWER_JSON_SAFE),
    )
    written: list[dict[str, Any]] = []

    async def fake_record_llm_cost(**kwargs: Any) -> None:
        written.append(kwargs)

    monkeypatch.setattr(providers, "record_llm_cost", fake_record_llm_cost)

    await ask_module.ask_question(_FakeSession(), QUESTION)  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert written == [
        {
            "model": "anthropic:claude-haiku-5-5",
            "input_tokens": 11,
            "output_tokens": 7,
            "est_usd": 0.000046,
            "article_id": None,
        }
    ]


async def test_response_contract_is_unchanged(call_log: list[dict[str, Any]]) -> None:
    result = await ask_module.ask_question(_FakeSession(), QUESTION)  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert set(result) == {"answer", "citations", "as_of", "snapshot_id", "latency_ms"}
    assert result["answer"] == ANSWER_JSON_SAFE
    assert [c["passage_id"] for c in result["citations"]] == ["passage-1"]
    # as_of is the publication date of the newest CITED passage (passage-1).
    assert result["as_of"] == "2026-01-02T00:00:00"
    assert result["snapshot_id"] == "snapshot-1"
    assert isinstance(result["latency_ms"], int)


async def test_llm_failure_degrades_to_an_error_string(
    monkeypatch: pytest.MonkeyPatch,
    call_log: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Availability degradation, kept on purpose: a dead model must not 500 /ask."""

    async def failing_call_llm(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("provider 'ollama' requires OLLAMA_API_KEY")

    monkeypatch.setattr(ask_module, "call_llm", failing_call_llm)

    with caplog.at_level(logging.ERROR):
        result = await ask_module.ask_question(_FakeSession(), QUESTION)  # type: ignore[arg-type]
        await asyncio.sleep(0)

    assert result["answer"] == (
        "LLM synthesis failed (RuntimeError): provider 'ollama' requires OLLAMA_API_KEY"
    )
    assert result["snapshot_id"] == "snapshot-1"
    assert "LLM synthesis failed" in caplog.text


async def test_ask_answers_on_a_provider_with_no_anthropic_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behaviour, not module shape: /ask must run on a non-Anthropic provider alone.

    With ANTHROPIC_API_KEY unset and SYNTHESIS_MODEL=ollama:gemma4:31b, the real
    call_llm resolves the Chat Completions transport. A private Anthropic client
    left behind in kb.functions.ask would fail here instead of answering.
    """
    _patch_ask_io(monkeypatch)
    monkeypatch.setattr(settings, "synthesis_model", "ollama:gemma4:31b")
    monkeypatch.setattr(
        providers,
        "settings",
        Settings.model_validate(
            {
                "ollama_api_key": "ollama-test",
                "anthropic_api_key": "",
                "default_provider": "ollama",
            }
        ),
    )
    client = _FakeOpenAIClient(ANSWER_JSON_SAFE)
    monkeypatch.setattr(providers, "get_openai_client", lambda provider: client)

    async def fake_record_llm_cost(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(providers, "record_llm_cost", fake_record_llm_cost)

    result = await ask_module.ask_question(_FakeSession(), QUESTION)  # type: ignore[arg-type]
    await asyncio.sleep(0)

    assert result["answer"] == ANSWER_JSON_SAFE
    assert client.chat.completions.calls[-1]["model"] == "gemma4:31b"
