"""
LLM provider factory.

Configured via environment variables:
  LLM_PROVIDER=grok    (default) — xAI Grok, requires XAI_API_KEY
  LLM_PROVIDER=ollama  — local Ollama, no API key required

Additional overrides:
  GROK_MODEL     (default: grok-beta)
  OLLAMA_MODEL   (default: llama3.2)
  OLLAMA_BASE_URL (default: http://localhost:11434/v1)
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a configured ChatOpenAI-compatible LLM instance."""
    provider = os.getenv("LLM_PROVIDER", "grok").lower()

    if provider == "grok":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "XAI_API_KEY environment variable is not set. "
                "Required for LLM_PROVIDER=grok."
            )
        return ChatOpenAI(
            model=os.getenv("GROK_MODEL", "grok-3"),
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            temperature=0,
            max_retries=1,
        )

    if provider == "ollama":
        return ChatOpenAI(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",   # required by OpenAI client but unused by Ollama
            temperature=0,
            max_retries=1,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        "Valid options: 'grok', 'ollama'."
    )


def get_structured_llm(schema):
    """
    Return an LLM runnable bound to `schema` for structured output.

    - Grok: uses function/tool calling (schema enforced server-side).
    - Ollama: uses json_mode (schema included in prompt; LLM returns JSON).
    """
    llm = get_llm()
    provider = os.getenv("LLM_PROVIDER", "grok").lower()

    if provider == "ollama":
        return llm.with_structured_output(schema, method="json_mode")
    return llm.with_structured_output(schema)
