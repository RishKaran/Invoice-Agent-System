"""
Payment Tool — executes mock payment only when approval_decision == "approve".

Safety rule: Any caller must pass the approval_decision explicitly.
If the decision is not "approve", this function raises immediately — no payment occurs.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

logger = logging.getLogger("invoice-agent")


def _find_existing_payment(invoice_number: str) -> str | None:
    """
    Scan outputs/receipt_*.json files for a completed payment matching invoice_number.
    Returns the existing transaction_id, or None if no prior payment is found.
    """
    if not invoice_number or not _OUTPUTS_DIR.exists():
        return None
    for receipt_file in sorted(_OUTPUTS_DIR.glob("receipt_*.json"), reverse=True):
        try:
            data = json.loads(receipt_file.read_text(encoding="utf-8"))
            if (data.get("invoice_number") == invoice_number
                    and data.get("status") == "completed"):
                return data.get("transaction_id")
        except Exception:
            continue
    return None


def mock_payment(vendor: str, amount: float, invoice_number: str = "") -> dict:
    """
    Simulate an external payment API call.
    Matches the interface specified in the case brief:
        def mock_payment(vendor, amount):
            print(f"Paid {amount} to {vendor}")
            return {"status": "success"}
    Extended to return a finance-grade receipt for auditability.
    """
    print(f"Paid {amount} to {vendor}")
    now = datetime.datetime.utcnow()
    return {
        "transaction_id": f"TXN-{now.strftime('%Y%m%d%H%M%S')}-{now.microsecond // 1000:03d}",
        "invoice_number": invoice_number,
        "vendor": vendor,
        "amount": amount,
        "currency": "USD",
        "payment_method": "ACH",
        "payment_timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "completed",
    }


def execute_payment(
    vendor: str,
    amount: float,
    approval_decision: str,
    invoice_number: str = "",
) -> dict:
    """
    Execute a payment only when approval_decision is exactly "approve".

    Args:
        vendor:             Vendor name from the validated InvoiceData.
        amount:             Total amount to pay.
        approval_decision:  Must be "approve"; any other value raises RuntimeError.
        invoice_number:     Invoice identifier included in the receipt for traceability.

    Returns:
        Payment receipt dict from mock_payment().

    Raises:
        RuntimeError: If approval_decision != "approve".
    """
    if approval_decision != "approve":
        raise RuntimeError(
            f"Payment blocked: invoice not approved. "
            f"approval_decision='{approval_decision}' (expected 'approve')."
        )

    # Guard against duplicate payments (e.g., when ALLOW_DUPLICATES_FOR_TESTING is set)
    existing_txn = _find_existing_payment(invoice_number)
    if existing_txn:
        logger.warning(
            f"[PAYMENT] duplicate payment prevented for {invoice_number!r} "
            f"(existing txn={existing_txn})"
        )
        return {
            "transaction_id":    existing_txn,
            "invoice_number":    invoice_number,
            "vendor":            vendor,
            "amount":            amount,
            "currency":          "USD",
            "payment_method":    "ACH",
            "payment_timestamp": "",
            "status":            "duplicate_payment_prevented",
        }

    receipt = mock_payment(vendor, amount, invoice_number)

    # Persist receipt to outputs/ for auditability.
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = _OUTPUTS_DIR / f"receipt_{receipt['transaction_id']}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    return receipt
