"""
Smoke test — verifies schemas, tools, and DB without running any agents.
Run from project root: python _smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = " PASS"
FAIL = " FAIL"
results: list[tuple[str, bool, str]] = []


def check(label: str, fn):
    try:
        fn()
        results.append((label, True, ""))
    except Exception as exc:
        results.append((label, False, str(exc)))


# ── 1. Pydantic schemas ──────────────────────────────────────────────────────

from schemas.models import InvoiceData, InvoiceItem

def test_invoice_valid():
    data = InvoiceData(
        invoice_number="INV-1011",
        vendor="Summit Manufacturing Co.",
        total_amount=3000.0,
        items=[
            InvoiceItem(name="WidgetA", quantity=6, unit_price=250.0),
            InvoiceItem(name="WidgetB", quantity=3, unit_price=500.0),
        ],
        due_date="20/02/2026",
    )
    assert data.due_date == "2026-02-20", f"Due date normalization failed: {data.due_date}"

def test_invoice_empty_items():
    try:
        InvoiceData(invoice_number="X", vendor="Y", items=[])
        raise AssertionError("Should have raised ValidationError")
    except Exception as exc:
        assert "items" in str(exc).lower() or "min_length" in str(exc).lower() or "list should have at least" in str(exc).lower()

def test_invoice_zero_qty_passes_schema():
    """quantity=0 (and negative) are allowed by schema; validation_agent rejects them explicitly."""
    item = InvoiceItem(name="Bad", quantity=0)
    assert item.quantity == 0  # schema allows it; validation_agent flags invalid_quantity

check("Schema: valid invoice + date normalization", test_invoice_valid)
check("Schema: empty items list raises error", test_invoice_empty_items)
check("Schema: quantity=0 passes schema (caught by validation_agent)", test_invoice_zero_qty_passes_schema)


# ── 2. File Loader ───────────────────────────────────────────────────────────

from tools.file_loader import extract_invoice_content

TXT_CONTENT = textwrap.dedent("""\
    Invoice Number: INV-1011
    Vendor: Summit Manufacturing Co.
    Items:
    WidgetA quantity 6 unit_price 250
    WidgetB quantity 3 unit_price 500
    Due Date: 2026-02-20
    Total Amount: 3000
""")

JSON_CONTENT = json.dumps({
    "invoice_number": "INV-2020",
    "vendor": "Acme Industrial Supplies",
    "total_amount": 1500,
    "due_date": "2026-03-15",
    "items": [
        {"name": "GadgetX", "quantity": 5, "unit_price": 300},
    ],
})

def test_loader_txt():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write(TXT_CONTENT)
        name = f.name
    text = extract_invoice_content(name)
    assert "INV-1011" in text
    assert "WidgetA" in text

def test_loader_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
        f.write(JSON_CONTENT)
        name = f.name
    text = extract_invoice_content(name)
    assert "Acme" in text
    assert "GadgetX" in text

def test_loader_missing_file():
    try:
        extract_invoice_content("/nonexistent/path/inv.txt")
        raise AssertionError("Should raise FileNotFoundError")
    except FileNotFoundError:
        pass

def test_loader_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        name = f.name
    try:
        extract_invoice_content(name)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass

def test_loader_filename_fallback():
    with tempfile.NamedTemporaryFile(
        prefix="invoice_9999_", suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write("Vendor: Summit Manufacturing Co.\nItems:\nWidgetA quantity 2\n")
        name = f.name
    text = extract_invoice_content(name)
    assert "INV-9999" in text, f"Fallback invoice number not injected: {text[:200]}"

check("Loader: parse TXT", test_loader_txt)
check("Loader: parse JSON", test_loader_json)
check("Loader: missing file raises FileNotFoundError", test_loader_missing_file)
check("Loader: unsupported format raises ValueError", test_loader_unsupported_format)
check("Loader: filename-based invoice number fallback", test_loader_filename_fallback)


# ── 3. Inventory Tool ────────────────────────────────────────────────────────

from tools.inventory_tools import get_inventory_item

def test_inventory_found():
    result = get_inventory_item("WidgetA")
    assert result["stock"] == 15, result

def test_inventory_normalized_lookup():
    result = get_inventory_item("widget-a")  # hyphen + mixed case
    # SQLite lookup strips hyphens; WidgetA has no hyphen so this won't match —
    # but the normalization should not crash.
    assert "stock" in result

def test_inventory_not_found():
    result = get_inventory_item("NoSuchItem")
    assert result["stock"] is None

check("Inventory: WidgetA stock=15", test_inventory_found)
check("Inventory: hyphenated name does not crash", test_inventory_normalized_lookup)
check("Inventory: unknown item returns stock=None", test_inventory_not_found)


# ── 4. Payment Tool ──────────────────────────────────────────────────────────

from tools.payment_tool import execute_payment

def test_payment_approved():
    receipt = execute_payment("Summit Manufacturing Co.", 3000.0, approval_decision="approve")
    assert receipt["status"] == "completed"
    assert receipt["amount"] == 3000.0

def test_payment_blocked_reject():
    try:
        execute_payment("Summit Manufacturing Co.", 3000.0, approval_decision="reject")
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as exc:
        assert "blocked" in str(exc).lower()

def test_payment_blocked_escalate():
    try:
        execute_payment("Summit Manufacturing Co.", 3000.0, approval_decision="escalate")
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as exc:
        assert "blocked" in str(exc).lower()

check("Payment: approved -> paid receipt", test_payment_approved)
check("Payment: reject -> RuntimeError", test_payment_blocked_reject)
check("Payment: escalate -> RuntimeError", test_payment_blocked_escalate)


# ── Report ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  SMOKE TEST RESULTS")
print("=" * 55)
passed = 0
for label, ok, err in results:
    tag = PASS if ok else FAIL
    print(f"{tag}  {label}")
    if not ok:
        print(f"       -> {err}")
    passed += ok

print("=" * 55)
print(f"  {passed}/{len(results)} tests passed")
print("=" * 55 + "\n")

if __name__ == "__main__" and passed < len(results):
    sys.exit(1)
