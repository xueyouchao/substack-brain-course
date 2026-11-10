"""Every tunable number in the pipeline lives here. Never a magic number inline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_blank(value: object) -> bool:
    return isinstance(value, str) and not value.strip()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- External services ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    database_url: str = (
        "postgresql+asyncpg://substack_brain:substack_brain_pw@localhost:5433/substack_brain"
    )
    inngest_dev: bool = True
    inngest_base_url: str = "http://localhost:8288"
    inngest_event_key: str = ""
    inngest_signing_key: str = ""
    lab_port: int = 8000

    # --- Fetching ---
    user_agent: str = "substack-brain/0.1 (+https://github.com/neural-maze/substack-brain-course)"
    request_timeout_s: float = 15.0
    max_article_bytes: int = 5_000_000

    # --- Chunking ---
    chunk_size: int = 1_000
    chunk_overlap: int = 150

    # --- Embeddings ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 96
    # Verified 2026-09-19 against OpenAI's pricing page; no output-token cost for embeddings.
    embedding_price_per_million_tokens: float = 0.02

    # --- Discovery & backfill ---
    discovery_limit: int = 5
    publications_file: str = "publications.yaml"
    # backfill-publication sends kb/article.discovered events in batches of this size.
    backfill_batch_size: int = 25

    # --- Flow control for ingest-article (week 2) ---
    # How many articles from the SAME publication can be in flight at once.
    concurrency_per_publication: int = 3
    # How many ingest-article runs may START per minute, across all publications.
    # embed_texts sends a whole article in one request (up to 96 chunks), so this
    # is roughly "embedding requests per minute". Keep it under your OpenAI quota:
    # POST /ask also embeds every question.
    ingest_runs_per_minute: int = 50
    # Inngest priority is a time shift in SECONDS (allowed range -600..600), not a
    # rank: a manual run is scheduled as if it had arrived 600s earlier, and a
    # backfill run as if it had arrived 600s later.
    priority_manual_s: int = 600
    priority_backfill_s: int = -600

    # --- Retrieval & answers (week 2) ---
    rrf_k: int = 60
    retrieval_limit: int = 5
    synthesis_model: str = "claude-haiku-5-5"
    synthesis_max_tokens: int = 1024

    # --- Claim extraction & Memgraph graph (week 4) ---
    memgraph_uri: str = "bolt://localhost:7687"
    memgraph_user: str = ""
    memgraph_password: str = ""
    extraction_model: str = "claude-haiku-5-5"
    extraction_max_tokens: int = 4096
    graph_retrieval_limit: int = 15
    concurrency_extract_claims: int = 2
    # Anthropic Claude pricing per million tokens (USD) for CostCaptureMiddleware
    anthropic_input_price_per_million_tokens: float = 1.00
    anthropic_output_price_per_million_tokens: float = 5.00

    # --- Evaluation suite & binary LLM judges (week 5) ---
    eval_judge_model: str = "claude-haiku-5-5"
    eval_judge_max_tokens: int = 1024
    concurrency_eval_runs: int = 2
    eval_taxonomy_file: str = "evals/taxonomy.yaml"
    eval_labeled_traces_file: str = "evals/labeled_traces.jsonl"
    eval_benchmark_file: str = "evals/golden_benchmark.yaml"
    eval_min_tpr: float = 0.90
    eval_min_tnr: float = 0.90

    # --- Bounded research agent & weekly digest (week 6) ---
    research_model: str = "claude-haiku-5-5"
    research_max_tokens: int = 2048
    research_max_iterations: int = 3
    research_max_subqueries_per_hop: int = 3
    research_passages_per_hop: int = 5
    research_max_usd: float = 0.25
    concurrency_research_runs: int = 2
    digest_cron: str = "0 8 * * 1"
    digest_lookback_days: int = 7
    digest_max_tokens: int = 2048

    # --- Model provider routing (docs/model-provider-config.md) ---
    # Any *_MODEL above may carry an explicit "provider:" prefix. A bare model
    # name falls back to this provider, so an .env that only sets
    # ANTHROPIC_API_KEY keeps working with no edit.
    default_provider: str = "anthropic"
    # Empty means "let the Anthropic SDK choose its own endpoint"; set it to
    # aim the anthropic_messages transport at any compatible gateway.
    anthropic_base_url: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    # Ollama Cloud serves both API families from one key, so the two families
    # need separate hosts instead of sharing a single base_url.
    # For a local Ollama, point ollama_base_url at http://localhost:11434/v1
    # and give ollama_api_key any non-empty value: it is ignored there.
    ollama_base_url: str = "https://ollama.com/v1"
    ollama_anthropic_base_url: str = "https://ollama.com"
    ollama_api_key: str = ""
    # Literal rather than str: a typo must fail at startup, not silently
    # downgrade structured output to free-form prose per request.
    ollama_structured_output: Literal["native", "json_mode", "prompt_only"] = "json_mode"
    ollama_pricing_mode: Literal["per_token", "subscription", "unknown"] = "unknown"
    # Same two-family split as Ollama: OpenCode Console routes Claude models to
    # Anthropic Messages and the rest to Chat Completions, and both families
    # share OPENCODE_API_KEY, so the user only ever configures one key.
    opencode_base_url: str = "https://opencode.ai/inference/openai/v1"
    opencode_anthropic_base_url: str = "https://opencode.ai/inference/anthropic"
    opencode_api_key: str = ""
    opencode_structured_output: Literal["native", "json_mode", "prompt_only"] = "prompt_only"
    opencode_pricing_mode: Literal["per_token", "subscription", "unknown"] = "subscription"

    @model_validator(mode="before")
    @classmethod
    def _treat_blank_as_unset(cls, data: Any) -> Any:
        # `cp .env.example .env` leaves lines like `OLLAMA_BASE_URL=` behind, and a
        # blank assignment is nobody's intent: keep it, and it silently wipes a
        # non-empty default (or, for a Literal field, refuses to start at all
        # because "" is not a valid tier). Dropping the key here lets the field
        # default apply, in one place, for env vars and .env files alike.
        if not isinstance(data, dict):
            return data
        return {key: value for key, value in data.items() if not _is_blank(value)}

    @field_validator("digest_cron", mode="before")
    @classmethod
    def _strip_cron_quotes(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().strip("\"'")
        return v

    @model_validator(mode="after")
    def _resolve_inngest_cloud_base_url(self) -> Settings:
        if not self.inngest_dev and "localhost:8288" in self.inngest_base_url:
            self.inngest_base_url = "https://api.inngest.com"
        return self


settings = Settings()

