"""The single LLM entry point, routed by a "provider:" prefix on the model name.

call_llm keeps its signature, so the 17 call sites stay untouched while the provider
table decides the transport (Anthropic Messages or Chat Completions), the
structured-output tier and the pricing mode. Design: docs/model-provider-config.md.

Every call returns a `"usage"` dictionary shaped for `CostCaptureMiddleware`
(`kb/inngest_client.py`) and can also write directly to the `costs` table when
invoked outside an Inngest step middleware context.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

import anthropic
import openai
from pydantic import BaseModel

from kb.config import settings
from kb.db.queries import insert_cost
from kb.db.session import get_session

# Clients are cached per (provider, base_url): one process can talk to the
# Anthropic API and to an Ollama Messages endpoint at once, and a shared client
# would silently send the second provider's traffic to the first endpoint.
_anthropic_clients: dict[tuple[str, str], anthropic.AsyncAnthropic] = {}
_openai_clients: dict[tuple[str, str], openai.AsyncOpenAI] = {}

logger = logging.getLogger(__name__)

# Providers whose pricing is unknown warn once per (provider, model): inside a batch
# loop the same config gap would otherwise fill the log with identical lines.
_warned_unknown_pricing: set[tuple[str, str]] = set()

# A refused key (401/403), one class per SDK family. Catching these by name is what
# keeps the OpenCode hint below from swallowing unrelated failures.
_AUTH_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
)

# OpenCode's free tier is licensed for use inside the OpenCode apps only, so the
# same key and model id answer 401/403 when called from a pipeline.
_OPENCODE_FREE_TIER_HINT = "免费档模型只允许在 OpenCode 内部使用，请改用 Go 订阅或付费模型。"


@dataclass(frozen=True)
class ProviderSpec:
    """One row of the capability table (docs/model-provider-config.md §4).

    `base_url_setting` / `structured_setting` / `pricing_setting` name `Settings`
    attributes, so adding a provider is one table row plus env, never a branch in
    the call path. A `None` setting means the provider has no env knob of its own,
    and the row's `default_*` value from docs §4 applies instead.
    """

    api_family: str  # "anthropic_messages" | "openai_chat"
    base_url_setting: str
    key_setting: str  # empty string means "this provider needs no key"
    structured_setting: str | None
    pricing_setting: str | None
    # docs §4 tiers for this row, and the fallback whenever the setting above is
    # None: a new provider fills these in and never touches a lookup helper.
    default_structured: str  # "native" | "json_mode" | "prompt_only"
    default_pricing: str  # "per_token" | "subscription" | "unknown"
    # Canonical endpoint used when the env value is empty. An empty OLLAMA_BASE_URL
    # must mean "unset", not "call the empty host": `cp .env.example .env` ships
    # every base_url blank and pydantic keeps that "" instead of the default.
    default_base_url: str | None


# The two families exist because Ollama Cloud and OpenCode Console each serve both
# APIs from one key: Claude models go to Anthropic Messages, the rest to Chat
# Completions. Picking the family therefore means picking a provider name.
PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        api_family="anthropic_messages",
        base_url_setting="anthropic_base_url",
        key_setting="anthropic_api_key",
        structured_setting=None,
        pricing_setting=None,
        default_structured="native",
        default_pricing="per_token",
        default_base_url=None,  # None = let the Anthropic SDK pick its endpoint
    ),
    "openai": ProviderSpec(
        api_family="openai_chat",
        base_url_setting="openai_base_url",
        key_setting="openai_api_key",
        structured_setting="openai_structured_output",
        pricing_setting="openai_pricing_mode",
        default_structured="native",
        default_pricing="per_token",
        default_base_url="https://api.openai.com/v1",
    ),
    "ollama": ProviderSpec(
        api_family="openai_chat",
        base_url_setting="ollama_base_url",
        key_setting="ollama_api_key",
        structured_setting="ollama_structured_output",
        pricing_setting="ollama_pricing_mode",
        default_structured="json_mode",
        default_pricing="unknown",
        default_base_url="https://ollama.com/v1",
    ),
    "ollama-anthropic": ProviderSpec(
        api_family="anthropic_messages",
        base_url_setting="ollama_anthropic_base_url",
        key_setting="ollama_api_key",
        structured_setting=None,
        pricing_setting=None,
        default_structured="prompt_only",
        default_pricing="unknown",
        default_base_url="https://ollama.com",
    ),
    "opencode": ProviderSpec(
        api_family="openai_chat",
        base_url_setting="opencode_base_url",
        key_setting="opencode_api_key",
        structured_setting="opencode_structured_output",
        pricing_setting="opencode_pricing_mode",
        default_structured="prompt_only",
        default_pricing="subscription",
        default_base_url="https://opencode.ai/inference/openai/v1",
    ),
    "opencode-anthropic": ProviderSpec(
        api_family="anthropic_messages",
        base_url_setting="opencode_anthropic_base_url",
        key_setting="opencode_api_key",
        structured_setting="opencode_structured_output",
        pricing_setting="opencode_pricing_mode",
        default_structured="prompt_only",
        default_pricing="subscription",
        default_base_url="https://opencode.ai/inference/anthropic",
    ),
}

# Provider names users reach for that this layer deliberately does not serve. A
# match is a mistake worth a ValueError; anything else before the first colon is
# a model tag ("gemma4:31b") and stays part of the model id.
_UNSUPPORTED_PROVIDERS = frozenset(
    {
        "azure",
        "bedrock",
        "cohere",
        "deepseek",
        "fireworks",
        "gemini",
        "google",
        "groq",
        "mistral",
        "openrouter",
        "replicate",
        "together",
        "vertex",
        "xai",
    }
)


@dataclass(frozen=True)
class ResolvedModel:
    provider: str
    model_id: str
    spec: ProviderSpec


def resolve_model(ref: str) -> ResolvedModel:
    """Split `"provider:model_id"` on the first colon only and attach its spec.

    A prefix that is not a known provider is a model tag, not an error, which is
    what keeps `"gemma4:31b"` intact under `DEFAULT_PROVIDER=ollama`.
    """
    prefix, sep, remainder = ref.partition(":")
    if sep and prefix in PROVIDERS:
        return ResolvedModel(provider=prefix, model_id=remainder, spec=PROVIDERS[prefix])
    if sep and prefix.lower() in _UNSUPPORTED_PROVIDERS:
        raise ValueError(
            f"unknown provider {prefix!r} in model reference {ref!r}; "
            f"known providers: {', '.join(PROVIDERS)}. "
            "Any OpenAI-compatible gateway can be reached as 'openai:<model>' with "
            "OPENAI_BASE_URL set."
        )
    provider = settings.default_provider
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(
            f"DEFAULT_PROVIDER={provider!r} is not a known provider; "
            f"known providers: {', '.join(PROVIDERS)}"
        )
    return ResolvedModel(provider=provider, model_id=ref, spec=spec)


# Per-million-token pricing table (input_usd, output_usd)
MODEL_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-5-5": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def _base_url(spec: ProviderSpec) -> str:
    """Resolve the endpoint, treating an empty env value as "unset".

    An explicitly empty base_url is a valid pydantic value, so nothing downstream
    may assume the field default survived: a client built from "" would post to the
    empty host. Programmatic Settings construction still hits this path.
    """
    configured = str(getattr(settings, spec.base_url_setting) or "").strip()
    return configured or (spec.default_base_url or "")


def _api_key(provider: str, spec: ProviderSpec) -> str:
    if not spec.key_setting:
        return ""
    key = str(getattr(settings, spec.key_setting) or "")
    if not key:
        raise RuntimeError(
            f"provider {provider!r} requires {spec.key_setting.upper()} "
            f"(Settings.{spec.key_setting}); set it in .env to invoke live models."
        )
    return key


def _setting_or_default(setting_name: str | None, default: str) -> str:
    """One lookup for both tiers: the env knob when the row has one, else its default."""
    return str(getattr(settings, setting_name)) if setting_name is not None else default


def _structured_output_mode(resolved: ResolvedModel) -> str:
    spec = resolved.spec
    return _setting_or_default(spec.structured_setting, spec.default_structured)


def _pricing_mode(resolved: ResolvedModel) -> str:
    spec = resolved.spec
    return _setting_or_default(spec.pricing_setting, spec.default_pricing)


def get_anthropic_client(provider: str) -> anthropic.AsyncAnthropic:
    """Return the cached `AsyncAnthropic` client for a provider and its base_url."""
    spec = PROVIDERS[provider]
    base_url = _base_url(spec)
    cache_key = (provider, base_url)
    client = _anthropic_clients.get(cache_key)
    if client is None:
        if base_url:
            client = anthropic.AsyncAnthropic(api_key=_api_key(provider, spec), base_url=base_url)
        else:
            client = anthropic.AsyncAnthropic(api_key=_api_key(provider, spec))
        _anthropic_clients[cache_key] = client
    return client


def get_openai_client(provider: str) -> openai.AsyncOpenAI:
    """Return the cached `AsyncOpenAI` client for a provider and its base_url."""
    spec = PROVIDERS[provider]
    base_url = _base_url(spec)
    cache_key = (provider, base_url)
    client = _openai_clients.get(cache_key)
    if client is None:
        if base_url:
            client = openai.AsyncOpenAI(api_key=_api_key(provider, spec), base_url=base_url)
        else:
            client = openai.AsyncOpenAI(api_key=_api_key(provider, spec))
        _openai_clients[cache_key] = client
    return client


def _warn_unknown_pricing_once(resolved: ResolvedModel) -> None:
    """Say once that a model's spend is untracked, so a reviewer sees it in the log."""
    key = (resolved.provider, resolved.model_id)
    if key in _warned_unknown_pricing:
        return
    _warned_unknown_pricing.add(key)
    logger.warning(
        "pricing_mode=unknown for provider=%r model=%r: est_usd is reported as 0.0, so USD "
        "budgets cannot bind. Set the provider's pricing mode to per_token to track spend.",
        resolved.provider,
        resolved.model_id,
    )


def estimate_llm_cost_usd(
    *,
    resolved: ResolvedModel,
    pricing_mode: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD for one call. Flat-rate plans report 0.0 (docs §7)."""
    if pricing_mode != "per_token":
        if pricing_mode == "unknown":
            _warn_unknown_pricing_once(resolved)
        return 0.0
    in_price, out_price = MODEL_PRICING_PER_MTOK.get(
        resolved.model_id,
        (
            settings.anthropic_input_price_per_million_tokens,
            settings.anthropic_output_price_per_million_tokens,
        ),
    )
    usd = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return round(usd, 6)


async def record_llm_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    est_usd: float,
    article_id: uuid.UUID | None = None,
) -> None:
    """Persist a single LLM invocation cost row to the `costs` table."""
    async with get_session() as session:
        await insert_cost(
            session,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            est_usd=est_usd,
            article_id=article_id,
        )
        await session.commit()


def _extract_json_object(raw_text: str) -> str:
    """Strip optional markdown code fences around JSON output."""
    stripped = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _schema_instruction(system_prompt: str, schema: type[BaseModel]) -> str:
    """Append the prompt-side schema contract used by the non-native tiers."""
    return (
        f"{system_prompt}\n\n"
        "只输出符合以下 JSON Schema 的单个 JSON 对象，不要包含解释或 markdown 围栏：\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    )


def _anthropic_json_schema(
    schema: type[BaseModel], *, provider: str, model_id: str
) -> dict[str, Any] | None:
    """Native JSON schema for Anthropic, or None when the SDK cannot build one.

    `transform_schema` is an Anthropic SDK internal (docs §10): when it is gone the
    request must still go out, so the caller downgrades to prompt_only instead of
    losing a whole call to an AttributeError.
    """
    try:
        return anthropic.transform_schema(schema)
    except (AttributeError, ImportError):
        logger.warning(
            "anthropic.transform_schema is unavailable; downgrading provider=%r model=%r "
            "structured output from native to prompt_only",
            provider,
            model_id,
        )
        return None


def _anthropic_response_output(response: Any) -> Any:
    """Return the first tool_use payload, else the first text block."""
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use" and hasattr(block, "input"):
            return block.input
        if block_type == "text" and hasattr(block, "text"):
            return str(block.text)
    return None


async def _call_anthropic_messages(
    resolved: ResolvedModel,
    *,
    system_prompt: str,
    user_input: str,
    max_tokens: int,
    output_schema: type[BaseModel] | None,
) -> tuple[Any, int, int]:
    client = get_anthropic_client(resolved.provider)
    mode = _structured_output_mode(resolved)
    kwargs: dict[str, Any] = {
        "model": resolved.model_id,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_input}],
    }
    if output_schema is not None:
        schema = (
            _anthropic_json_schema(
                output_schema, provider=resolved.provider, model_id=resolved.model_id
            )
            if mode == "native"
            else None
        )
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        else:
            kwargs["system"] = _schema_instruction(system_prompt, output_schema)

    response = await client.messages.create(**kwargs)
    usage_obj = getattr(response, "usage", None)
    input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
    return _anthropic_response_output(response), input_tokens, output_tokens


async def _call_openai_chat(
    resolved: ResolvedModel,
    *,
    system_prompt: str,
    user_input: str,
    max_tokens: int,
    output_schema: type[BaseModel] | None,
) -> tuple[Any, int, int]:
    client = get_openai_client(resolved.provider)
    mode = _structured_output_mode(resolved)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    kwargs: dict[str, Any] = {
        "model": resolved.model_id,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if output_schema is not None:
        if mode == "native":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                    # Loose on purpose: Ollama and OpenCode do not all implement
                    # strict json_schema, and a 400 is worse than a retry-worthy body.
                    "strict": False,
                },
            }
        else:
            if mode == "json_mode":
                kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"] = [
                {
                    "role": "system",
                    "content": _schema_instruction(system_prompt, output_schema),
                },
                {"role": "user", "content": user_input},
            ]

    response = await client.chat.completions.create(**kwargs)
    usage_obj = getattr(response, "usage", None)
    input_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    content = getattr(response.choices[0].message, "content", None) if response.choices else None
    return content, input_tokens, output_tokens


# docs §6: the error message quotes the head of the offending output, and truncating
# there is what keeps one bad response from flooding the log. Named rather than
# inline because kb/config.py's rule is "Never a magic number inline".
_ERROR_SNIPPET_CHARS = 200


def _parse_json_payload(raw: Any) -> Any:
    """Reach the object behind JSON text, however many layers a gateway added.

    The chain of docs §6 holds for all three tiers, so text is unwrapped before it
    is validated: a Chat Completions body arrives as a str, and a gateway that
    re-encodes the structure wraps the same object one layer deeper. Unwrapping
    terminates because a JSON string literal is always longer than its payload.
    """
    data = raw
    while isinstance(data, str):
        data = json.loads(_extract_json_object(data))
    return data


def _validate_structured_output(
    raw: Any, schema: type[BaseModel], *, resolved: ResolvedModel
) -> BaseModel:
    """Fallback chain for every tier: strip fences, parse JSON, validate.

    A bare JSONDecodeError says nothing about which model misbehaved, so re-raise
    with the provider, the model and the head of the offending text.
    """
    try:
        return schema.model_validate(_parse_json_payload(raw))
    except ValueError as exc:  # json.JSONDecodeError and pydantic.ValidationError
        snippet = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        raise RuntimeError(
            f"provider={resolved.provider!r} model={resolved.model_id!r} did not return valid "
            f"{schema.__name__} JSON ({exc}); raw output starts with: "
            f"{snippet[:_ERROR_SNIPPET_CHARS]!r}"
        ) from exc


async def call_llm(
    *,
    system_prompt: str,
    user_input: str,
    model: str,
    max_tokens: int = 2048,
    output_schema: type[BaseModel] | None = None,
    article_id: uuid.UUID | None = None,
    persist_cost: bool = False,
) -> dict[str, Any]:
    """Invoke `model` through its provider transport and return
    `{"output": ..., "usage": {...}}`.

    `model` may carry a `provider:` prefix (docs §3); a bare model name keeps
    routing to `DEFAULT_PROVIDER`, so existing call sites need no change.
    """
    resolved = resolve_model(model)
    pricing_mode = _pricing_mode(resolved)
    transport = (
        _call_anthropic_messages
        if resolved.spec.api_family == "anthropic_messages"
        else _call_openai_chat
    )
    try:
        raw_output, input_tokens, output_tokens = await transport(
            resolved,
            system_prompt=system_prompt,
            user_input=user_input,
            max_tokens=max_tokens,
            output_schema=output_schema,
        )
    except _AUTH_ERRORS as exc:
        # Only the transport is wrapped: every other failure keeps its own type.
        if not resolved.provider.startswith("opencode"):
            raise
        raise RuntimeError(
            f"provider={resolved.provider!r} model={resolved.model_id!r} rejected the "
            f"credentials (401/403: {exc}). {_OPENCODE_FREE_TIER_HINT}"
        ) from exc

    est_usd = estimate_llm_cost_usd(
        resolved=resolved,
        pricing_mode=pricing_mode,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    usage: dict[str, Any] = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "est_usd": est_usd,
        "article_id": str(article_id) if article_id is not None else None,
        "provider": resolved.provider,
        "pricing_mode": pricing_mode,
    }

    if persist_cost:
        await record_llm_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            est_usd=est_usd,
            article_id=article_id,
        )

    if output_schema is not None:
        validated = _validate_structured_output(raw_output, output_schema, resolved=resolved)
        return {"output": validated.model_dump(mode="json"), "usage": usage}

    return {"output": str(raw_output or ""), "usage": usage}
