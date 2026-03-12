"""
Validation Agent — deterministic checks using Python tools only.
LLMs are never called here. No SQLite access either (tools handle that).

Checks:
  0. Vendor:    vendor field is non-empty
  1. Inventory: item exists, quantity <= stock
  2. Date:      due_date is a valid ISO date (if present); overdue dates raise risk_score +0.3
  3. Math:      tax-aware total check (subtotal + tax + shipping + handling + fees)

Produces:
  - validation_status: "valid" | "invalid"
  - risk_score  (0.0 – 1.0)
  - risk_category: "low_risk" | "high_risk" | "fraud_suspected"
"""

from __future__ import annotations

import datetime
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.models import InvoiceData
from schemas.state import InvoiceGraphState
from tools.inventory_tools import get_inventory_item, normalize_item_name

logger = logging.getLogger("invoice-agent")

# Tolerance for total-amount comparison after adding charges ($0.50)
_MATH_TOLERANCE = 0.50

# Risk thresholds
_FRAUD_THRESHOLD      = 0.7
_HIGH_RISK_THRESHOLD  = 0.4


def _extract_charges(text: str) -> dict:
    """
    Regex-scan invoice text for tax, shipping, handling, and miscellaneous fees.
    Returns a dict with float values; missing items default to 0.0.
    Uses multiple patterns per charge type for maximum coverage.
    """
    charges: dict[str, float] = {"tax": 0.0, "shipping": 0.0, "handling": 0.0, "fees": 0.0}

    def _find_amount(patterns: list[str]) -> float:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    pass
        return 0.0

    # Tax: explicit "Sales Tax" pattern first, then generic tax/vat/gst/hst
    charges["tax"] = _find_amount([
        r"sales\s+tax\s*(?:\([^)]*\))?\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]*)",
        r"(?:tax|vat|gst|hst)(?:\s+amount)?\s*(?:\([^)]*\))?\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]*)",
    ])

    charges["shipping"] = _find_amount([
        r"(?:shipping|freight|delivery)\s*(?:\([^)]*\))?\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]*)",
    ])

    charges["handling"] = _find_amount([
        r"handling\s*(?:\([^)]*\))?\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]*)",
    ])

    charges["fees"] = _find_amount([
        r"(?:fees?|surcharge|misc(?:ellaneous)?)\s*(?:\([^)]*\))?\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]*)",
    ])

    return charges


def run_validation(state: InvoiceGraphState) -> dict:
    """LangGraph node: validate extracted invoice data against inventory and vendor DB."""
    raw = state.get("invoice_data")
    invoice_text: str = state.get("invoice_text") or ""

    if not raw:
        error_msg = state.get("error") or "No invoice data to validate"
        return {
            "validation_result": {
                "status": "invalid",
                "issues": [{"type": "extraction_error", "message": error_msg}],
                "risk_score": 0.0,
                "risk_category": "low_risk",
            }
        }

    try:
        invoice = InvoiceData(**raw)
    except Exception as exc:
        return {
            "validation_result": {
                "status": "invalid",
                "issues": [{"type": "schema_error", "message": f"InvoiceData schema error: {exc}"}],
                "risk_score": 0.0,
                "risk_category": "low_risk",
            }
        }

    issues: list[dict] = []
    status = "valid"
    risk_score = 0.0
    # Audit clarity tracking
    normalized_items: list[str] = []
    normalization_applied = False

    # ── 0. Vendor presence check ──────────────────────────────────────────────
    if not invoice.vendor or not invoice.vendor.strip():
        issues.append({"type": "vendor_missing", "message": "Vendor missing from invoice"})
        status = "invalid"

    # ── 1. Inventory validation (aggregated) ─────────────────────────────────
    # Group line items by normalized name so that split lines for the same item
    # are summed before comparing against stock.
    item_groups: dict[str, list] = {}
    for item in invoice.items:
        key = normalize_item_name(item.name)
        if key not in item_groups:
            item_groups[key] = []
        item_groups[key].append(item)

    for _norm_key, group in item_groups.items():
        total_qty  = sum(i.quantity for i in group)
        rep_item   = group[0]  # representative item for name/DB lookup

        # Reject invoices with invalid (zero or negative) quantities
        if any(i.quantity <= 0 for i in group):
            msg = f"Item '{rep_item.name}' has invalid quantity {rep_item.quantity} (must be > 0)"
            logger.warning(f"[VALIDATION] {msg}")
            issues.append({"type": "invalid_quantity", "message": msg})
            status = "invalid"
            risk_score += 0.3
            continue

        try:
            result = get_inventory_item(rep_item.name)
        except Exception as exc:
            issues.append({"type": "inventory_error", "message": f"DB error for '{rep_item.name}': {exc}"})
            status = "invalid"
            continue

        if result["stock"] is None:
            msg = f"Item '{rep_item.name}' not found in inventory"
            logger.warning(f"[VALIDATION] {msg}")
            issues.append({"type": "inventory_not_found", "message": msg})
            status = "invalid"
            risk_score += 0.7  # directly reaches fraud_suspected threshold

        elif result["stock"] == 0:
            msg = f"Item '{rep_item.name}' has zero inventory (stock = 0)"
            logger.warning(f"[VALIDATION] {msg}")
            issues.append({"type": "inventory_zero_stock", "message": msg})
            status = "invalid"
            risk_score += 0.7  # directly reaches fraud_suspected threshold

        elif total_qty > result["stock"]:
            display_name = rep_item.name
            if len(group) > 1:
                display_name = f"{rep_item.name} (aggregated {len(group)} lines)"
            msg = (
                f"Insufficient stock for '{display_name}': "
                f"requested {total_qty}, available {result['stock']}"
            )
            logger.warning(f"[VALIDATION] stock mismatch — {msg}")
            issues.append({"type": "stock_insufficient", "message": msg})
            status = "invalid"

        else:
            # Item matched successfully — record canonical DB name
            canonical = result["item"]
            normalized_items.append(canonical)
            if rep_item.name != canonical:
                normalization_applied = True
                logger.info(f"[VALIDATION] normalized '{rep_item.name}' → '{canonical}'")

    # ── 2. Date validation ────────────────────────────────────────────────────
    # Flag if due_date is present but not a valid ISO date (e.g. raw "yesterday")
    if invoice.due_date is not None:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', invoice.due_date):
            issues.append({
                "type": "invalid_due_date",
                "message": f"Due date '{invoice.due_date}' is not a recognized date format",
            })
        else:
            # Valid ISO date — check if it's already past due.
            try:
                due = datetime.date.fromisoformat(invoice.due_date)
                today = datetime.date.today()
                if due < today:
                    days_overdue = (today - due).days
                    msg = f"Invoice is overdue by {days_overdue} day(s) (due {invoice.due_date})"
                    logger.warning(f"[VALIDATION] {msg}")
                    issues.append({"type": "overdue_invoice", "message": msg})
                    risk_score += 0.3
            except ValueError:
                pass  # malformed date already caught above

    elif re.search(
        r"\b(?:due|net\s*\d+|pay(?:able|ment)?\s+(?:by|date)|expires?)"
        r"|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
        invoice_text,
        re.IGNORECASE,
    ):
        issues.append({
            "type": "invalid_due_date",
            "message": "Due date mentioned in invoice but could not be parsed",
        })

    # ── 3. Tax-aware math validation ──────────────────────────────────────────
    subtotal: float | None = None
    charges: dict | None = None
    expected_total: float | None = None

    if invoice.total_amount is not None:
        priced_items = [i for i in invoice.items if i.unit_price is not None and i.quantity > 0]
        if priced_items:
            subtotal = sum(i.quantity * i.unit_price for i in priced_items)
            charges = _extract_charges(invoice_text)

            # Tax priority: tax_rate from invoice > tax_amount from invoice > text regex
            # Only use invoice fields if they are explicitly non-zero (LLM may return 0.0)
            if invoice.tax_rate:  # truthy: skips None and 0.0
                rate = invoice.tax_rate
                if rate > 1.0:  # percentage (e.g., 6.0) → decimal (0.06)
                    rate = rate / 100.0
                charges["tax"] = round(subtotal * rate, 2)
            elif invoice.tax_amount:  # truthy: skips None and 0.0
                charges["tax"] = invoice.tax_amount

            expected_total = subtotal + sum(charges.values())

            delta = abs(expected_total - invoice.total_amount)
            if delta > _MATH_TOLERANCE:
                msg = (
                    f"Total amount {invoice.total_amount} does not match "
                    f"expected {expected_total:.2f} "
                    f"(subtotal {subtotal:.2f}"
                    + (f" + tax {charges['tax']:.2f}" if charges["tax"] else "")
                    + (f" + shipping {charges['shipping']:.2f}" if charges["shipping"] else "")
                    + (f" + handling {charges['handling']:.2f}" if charges["handling"] else "")
                    + (f" + fees {charges['fees']:.2f}" if charges["fees"] else "")
                    + f", delta {delta:.2f})"
                )
                logger.warning(f"[VALIDATION] {msg}")
                issues.append({"type": "math_discrepancy", "message": msg})
                risk_score += 0.3

    # ── Risk categorization ───────────────────────────────────────────────────
    risk_score = round(min(risk_score, 1.0), 2)
    if risk_score >= _FRAUD_THRESHOLD:
        risk_category = "fraud_suspected"
    elif risk_score >= _HIGH_RISK_THRESHOLD:
        risk_category = "high_risk"
    else:
        risk_category = "low_risk"

    if not issues:
        logger.info("[VALIDATION] all checks passed")
    else:
        logger.info(
            f"[VALIDATION] {len(issues)} issue(s) — status={status} "
            f"risk={risk_score} ({risk_category})"
        )

    validation_result: dict = {
        "status":               status,
        "issues":               issues,
        "risk_score":           risk_score,
        "risk_category":        risk_category,
        # Audit clarity fields
        "normalized_items":     normalized_items,
        "normalization_applied": normalization_applied,
    }

    # Attach math breakdown when available (surfaced in audit report)
    if subtotal is not None and charges is not None and expected_total is not None:
        validation_result["subtotal"]       = round(subtotal, 2)
        validation_result["tax"]            = charges["tax"]
        validation_result["shipping"]       = charges["shipping"]
        validation_result["handling"]       = charges["handling"]
        validation_result["fees"]           = charges["fees"]
        validation_result["expected_total"] = round(expected_total, 2)

    return {"validation_result": validation_result}
