"""Contract tests for the provider-facing Settings surface (T1).

The seam under test is Settings construction: every assertion goes through the
public pydantic interface, so renaming an internal helper never breaks these.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb.config import Settings

# Values from the shared ticket contract: names and defaults are frozen.
PROVIDER_SURFACE_DEFAULTS: dict[str, str] = {
    "default_provider": "anthropic",
    "anthropic_base_url": "",
    "openai_base_url": "https://api.openai.com/v1",
    "openai_structured_output": "native",
    "openai_pricing_mode": "per_token",
    "ollama_base_url": "https://ollama.com/v1",
    "ollama_anthropic_base_url": "https://ollama.com",
    "ollama_api_key": "",
    "ollama_structured_output": "json_mode",
    "ollama_pricing_mode": "unknown",
    "opencode_base_url": "https://opencode.ai/inference/openai/v1",
    "opencode_anthropic_base_url": "https://opencode.ai/inference/anthropic",
    "opencode_api_key": "",
    "opencode_structured_output": "prompt_only",
    "opencode_pricing_mode": "subscription",
}

STRUCTURED_OUTPUT_FIELDS = (
    "openai_structured_output",
    "ollama_structured_output",
    "opencode_structured_output",
)
PRICING_MODE_FIELDS = ("openai_pricing_mode", "ollama_pricing_mode", "opencode_pricing_mode")


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ambient provider variables so an exported value cannot skew a case."""
    for name in (*PROVIDER_SURFACE_DEFAULTS, "digest_cron", "inngest_dev"):
        monkeypatch.delenv(name.upper(), raising=False)


def _from_env() -> Settings:
    """Build from the environment only, ignoring any local .env file."""
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(("name", "expected"), PROVIDER_SURFACE_DEFAULTS.items())
def test_provider_surface_default_matches_contract(name: str, expected: str) -> None:
    assert getattr(_from_env(), name) == expected


def test_existing_model_defaults_are_unchanged() -> None:
    """Zero-behaviour-change guard: T1 only adds fields, it never retunes them."""
    settings = _from_env()

    assert settings.synthesis_model == "claude-haiku-5-5"
    assert settings.extraction_model == "claude-haiku-5-5"
    assert settings.eval_judge_model == "claude-haiku-5-5"
    assert settings.research_model == "claude-haiku-5-5"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536


def test_env_overrides_provider_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama")
    monkeypatch.setenv("OLLAMA_STRUCTURED_OUTPUT", "native")
    monkeypatch.setenv("OPENCODE_PRICING_MODE", "per_token")

    settings = _from_env()

    assert settings.default_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.ollama_api_key == "ollama"
    assert settings.ollama_structured_output == "native"
    assert settings.opencode_pricing_mode == "per_token"


@pytest.mark.parametrize(("name", "expected"), PROVIDER_SURFACE_DEFAULTS.items())
def test_blank_env_value_falls_back_to_the_default(
    name: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cp .env.example .env` leaves blank assignments; blank means "not set"."""
    monkeypatch.setenv(name.upper(), "")

    assert getattr(_from_env(), name) == expected


def test_blank_literal_tier_does_not_break_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_PRICING_MODE", "")
    monkeypatch.setenv("OLLAMA_STRUCTURED_OUTPUT", "")

    settings = _from_env()

    assert settings.opencode_pricing_mode == "subscription"
    assert settings.ollama_structured_output == "json_mode"


def test_blank_values_in_a_dotenv_file_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real failure path: README tells users to copy .env.example verbatim."""
    (tmp_path / ".env").write_text(
        "OLLAMA_BASE_URL=\nOPENCODE_PRICING_MODE=\nOLLAMA_API_KEY=\nANTHROPIC_BASE_URL=\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.ollama_base_url == "https://ollama.com/v1"
    assert settings.opencode_pricing_mode == "subscription"
    assert settings.ollama_api_key == ""
    assert settings.anthropic_base_url == ""


def test_fields_defaulting_to_blank_still_read_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMGRAPH_USER", "")
    monkeypatch.setenv("INNGEST_EVENT_KEY", "")

    settings = _from_env()

    assert settings.memgraph_user == ""
    assert settings.inngest_event_key == ""


def test_blank_value_does_not_disable_the_existing_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIGEST_CRON", ' "0 8 * * 1" ')
    monkeypatch.setenv("INNGEST_DEV", "false")
    monkeypatch.setenv("INNGEST_BASE_URL", "")

    settings = _from_env()

    assert settings.digest_cron == "0 8 * * 1"
    assert settings.inngest_dev is False
    assert settings.inngest_base_url == "https://api.inngest.com"


@pytest.mark.parametrize("field", STRUCTURED_OUTPUT_FIELDS)
def test_structured_output_rejects_unknown_value(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: "garbage"})


@pytest.mark.parametrize("field", PRICING_MODE_FIELDS)
def test_pricing_mode_rejects_unknown_value(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: "per_request"})


def test_invalid_structured_output_from_env_is_rejected_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_STRUCTURED_OUTPUT", "garbage")

    with pytest.raises(ValidationError):
        _from_env()


@pytest.mark.parametrize("tier", ["native", "json_mode", "prompt_only"])
def test_every_structured_output_tier_is_accepted(tier: str) -> None:
    for field in STRUCTURED_OUTPUT_FIELDS:
        assert getattr(Settings.model_validate({field: tier}), field) == tier


@pytest.mark.parametrize("mode", ["per_token", "subscription", "unknown"])
def test_every_pricing_mode_is_accepted(mode: str) -> None:
    for field in PRICING_MODE_FIELDS:
        assert getattr(Settings.model_validate({field: mode}), field) == mode
