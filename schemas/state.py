"""
Shared LangGraph state definition.
All agents read from and write partial updates to this TypedDict.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class InvoiceGraphState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    invoice_path: str            # original file path (for logging / audit)
    invoice_text: str            # normalized text from file_loader

    # ── LLM extraction ───────────────────────────────────────────────────────
    invoice_data: Optional[dict]         # serialized InvoiceData (or None on failure)

    # ── Critic ───────────────────────────────────────────────────────────────
    critic_result: Optional[dict]        # {"valid": bool, "correction_instruction": str}
    correction_instruction: Optional[str]  # forwarded to ingestion on retry

    # ── Validation ───────────────────────────────────────────────────────────
    validation_result: Optional[dict]    # {"status": "valid"|"invalid", "issues": [...]}

    # ── Approval ─────────────────────────────────────────────────────────────
    approval_result: Optional[dict]      # {"decision": "approve|reject|escalate", "reason": str}

    # ── Control ──────────────────────────────────────────────────────────────
    retry_count: int             # incremented each time critic rejects; max = 2
    error: Optional[str]         # last error message (cleared on successful extraction)
    extraction_method: Optional[str]  # "llm_extraction" | "deterministic_json" | "deterministic_csv"

    # ── Audit accumulator ────────────────────────────────────────────────────
    audit: dict                  # partial audit data collected across nodes
