"""
Extraction Critic Agent — reviews LLM extraction quality.

Returns {"valid": bool, "correction_instruction": str}.
Increments retry_count when extraction is rejected.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.llm_provider import get_structured_llm
from schemas.state import InvoiceGraphState

logger = logging.getLogger("invoice-agent")


# ── Critic output schema ──────────────────────────────────────────────────────

class CriticOutput(BaseModel):
    valid: bool
    correction_instruction: str = ""


# ── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an invoice extraction quality reviewer. You will be shown the original \
invoice text and the structured data extracted from it.

Your job is to verify the extraction is correct and complete.

CHECK FOR:
1. invoice_number — must be present and match what appears in the document
2. vendor         — must be present and match the document (full company name)
3. items          — list must not be empty; each item needs a name and positive quantity
4. quantities     — must be positive integers only (never 0, never negative)
5. due_date       — if present in the document, must be captured (any format is OK)
6. No hallucinated data (values not in the original text)
7. Item names must be correctly normalized (e.g., "Widget A" → "WidgetA")

Return:
- valid: true if the extraction is acceptable
- correction_instruction: if valid is false, a specific, actionable instruction \
  describing exactly what needs to be fixed. Leave empty string if valid is true.
"""


# ── Node function ─────────────────────────────────────────────────────────────

def run_critic(state: InvoiceGraphState) -> dict:
    """LangGraph node: review extraction quality and decide retry vs. proceed."""
    invoice_text = state["invoice_text"]
    invoice_data = state.get("invoice_data")
    error = state.get("error")
    retry_count = state.get("retry_count", 0)

    extraction_method = state.get("extraction_method", "")

    # Deterministic paths (JSON/CSV): skip LLM critique entirely.
    # Data quality issues (negative qty, schema errors) are handled by validation_agent.
    if extraction_method in ("deterministic_json", "deterministic_csv"):
        if invoice_data is None:
            logger.warning(f"[CRITIC] deterministic schema error — routing to validation")
        else:
            logger.info(f"[CRITIC] deterministic extraction — accepted")
        return {
            "critic_result": {"valid": True, "correction_instruction": ""},
            "correction_instruction": None,
        }

    # Fast-path: ingestion returned no data (network error / hard exception)
    if invoice_data is None:
        correction = error or "Extraction returned no data. Re-extract all fields."
        logger.warning(f"[CRITIC] extraction produced no data, requesting retry")
        return {
            "critic_result": {"valid": False, "correction_instruction": correction},
            "correction_instruction": correction,
            "retry_count": retry_count + 1,
        }

    from langchain_core.messages import HumanMessage, SystemMessage
    user_content = (
        f"ORIGINAL INVOICE TEXT:\n{invoice_text}\n\n"
        f"EXTRACTED DATA:\n{invoice_data}"
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        structured_llm = get_structured_llm(CriticOutput)
        result: CriticOutput = structured_llm.invoke(messages)

        if result.valid:
            logger.info("[CRITIC] extraction accepted")
            return {
                "critic_result": result.model_dump(),
                "correction_instruction": None,
            }
        else:
            logger.warning(
                f"[CRITIC] extraction rejected (retry {retry_count + 1}): "
                f"{result.correction_instruction}"
            )
            return {
                "critic_result": result.model_dump(),
                "correction_instruction": result.correction_instruction,
                "retry_count": retry_count + 1,
            }

    except Exception as exc:
        # If the critic itself fails, be conservative: treat as invalid
        msg = f"Critic LLM failed: {exc}"
        logger.error(f"[CRITIC] {msg}")
        return {
            "critic_result": {"valid": False, "correction_instruction": msg},
            "correction_instruction": msg,
            "retry_count": retry_count + 1,
        }
