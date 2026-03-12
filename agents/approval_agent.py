"""
Approval Agent — 100% deterministic Python. No LLM calls, no prompts.

Decision hierarchy (in strict order):
  1. risk_category == "fraud_suspected"  → escalate  (overrides invalid)
  2. validation_status != "valid"        → reject
  3. total_amount > ESCALATION_THRESHOLD → escalate  (default $10,000; env: INVOICE_ESCALATION_THRESHOLD)
  4. math_discrepancy issue present      → escalate  (human must review arithmetic errors before payment)
  5. otherwise                           → approve
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.state import InvoiceGraphState

logger = logging.getLogger("invoice-agent")

_ESCALATION_THRESHOLD = float(os.getenv("INVOICE_ESCALATION_THRESHOLD", "10000"))


def run_approval(state: InvoiceGraphState) -> dict:
    """LangGraph node: apply business rules and return approve/reject/escalate."""
    validation   = state.get("validation_result") or {}
    invoice_data = state.get("invoice_data") or {}

    val_status    = validation.get("status", "invalid")
    issues: list[dict] = validation.get("issues", [])
    risk_category = validation.get("risk_category", "low_risk")

    # ── Rule 1: Fraud suspected — escalate (overrides invalid) ──────────────────
    if risk_category == "fraud_suspected":
        fraud_issues = "; ".join(i.get("message", "") for i in issues)
        reason = f"Potential fraudulent invoice — {fraud_issues}".rstrip(" —")
        logger.info(f"[APPROVAL] escalated — fraud_suspected (risk_score={validation.get('risk_score')})")
        return {"approval_result": {"decision": "escalate", "reason": reason}}

    # ── Rule 2: Hard validation failure → reject ──────────────────────────────
    # Guard against None/missing status — anything that is not explicitly "valid" is rejected.
    if val_status != "valid":
        reason = "Validation failed: " + "; ".join(i.get("message", "") for i in issues)
        logger.info(f"[APPROVAL] rejected — {reason}")
        return {"approval_result": {"decision": "reject", "reason": reason.strip(": ") or "Validation failed"}}

    # ── Rule 3: Escalate on high-value invoice ────────────────────────────────
    total = invoice_data.get("total_amount") or 0.0
    if total > _ESCALATION_THRESHOLD:
        reason = f"Invoice amount ${total:,.2f} exceeds ${_ESCALATION_THRESHOLD:,.0f} threshold"
        logger.info(f"[APPROVAL] escalated — {reason}")
        return {"approval_result": {"decision": "escalate", "reason": reason}}

    # ── Rule 4: Escalate on math discrepancy ─────────────────────────────────
    # An invoice with known arithmetic errors must not be paid without human review.
    if any(i.get("type") == "math_discrepancy" for i in issues):
        msg = next(i["message"] for i in issues if i.get("type") == "math_discrepancy")
        reason = f"Amount discrepancy requires manual review: {msg}"
        logger.info(f"[APPROVAL] escalated — math_discrepancy")
        return {"approval_result": {"decision": "escalate", "reason": reason}}

    # ── Rule 5: Approve ───────────────────────────────────────────────────────
    logger.info("[APPROVAL] approved")
    return {"approval_result": {"decision": "approve", "reason": "All validation checks passed"}}
