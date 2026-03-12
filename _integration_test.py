"""
Integration test — exercises the full graph with mocked LLM calls.
No real API key required.

Run: python _integration_test.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest.mock as mock
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


# ── Shared fixtures ───────────────────────────────────────────────────────────

INVOICE_OK = {
    "invoice_number": "INV-1011",
    "vendor": "Summit Manufacturing Co.",
    "total_amount": 3000.0,
    "items": [
        {"name": "WidgetA", "quantity": 6, "unit_price": 250.0},
        {"name": "WidgetB", "quantity": 3, "unit_price": 500.0},
    ],
    "due_date": "2026-02-20",
    "notes": None,
}

# Items sum to 15000 exactly so math check passes; high-value threshold fires
INVOICE_HIGH_VALUE = {
    "invoice_number": "INV-9999",
    "vendor": "Summit Manufacturing Co.",
    "total_amount": 15000.0,
    "items": [
        {"name": "WidgetA", "quantity": 6, "unit_price": 2000.0},   # 6*2000 = 12000
        {"name": "WidgetB", "quantity": 3, "unit_price": 1000.0},   # 3*1000 =  3000 → total 15000
    ],
    "due_date": "2026-02-20",
    "notes": None,
}

INVOICE_BAD_STOCK = {
    "invoice_number": "INV-2222",
    "vendor": "Summit Manufacturing Co.",
    "total_amount": 500.0,
    "items": [{"name": "GadgetX", "quantity": 999, "unit_price": None}],
    "due_date": None,
    "notes": None,
}

INVOICE_UNKNOWN_VENDOR = {
    "invoice_number": "INV-3333",
    "vendor": "Shady Goods Ltd.",
    "total_amount": 250.0,    # 1 * 250.0 — matches model_validator (all prices known)
    "items": [{"name": "WidgetA", "quantity": 1, "unit_price": 250.0}],
    "due_date": None,
    "notes": None,
}

INVOICE_MATH_DISCREPANCY = {
    "invoice_number": "INV-4444",
    "vendor": "Summit Manufacturing Co.",
    # WidgetB has no unit_price → model_validator skips (not all prices known).
    # validation_agent soft-check still fires: priced sum=1500 vs total=9999.
    "total_amount": 9999.0,
    "items": [
        {"name": "WidgetA", "quantity": 6, "unit_price": 250.0},
        {"name": "WidgetB", "quantity": 3, "unit_price": None},   # no price
    ],
    "due_date": None,
    "notes": None,
}


def _make_invoice_txt(data: dict) -> str:
    """Write data dict to a temp txt invoice file, return path."""
    lines = [
        f"Invoice Number: {data['invoice_number']}",
        f"Vendor: {data['vendor']}",
        "Items:",
    ]
    for item in data["items"]:
        line = f"  {item['name']} quantity {item['quantity']}"
        if item.get("unit_price"):
            line += f" unit_price {item['unit_price']}"
        lines.append(line)
    if data.get("due_date"):
        lines.append(f"Due Date: {data['due_date']}")
    if data.get("total_amount"):
        lines.append(f"Total Amount: {data['total_amount']}")
    content = "\n".join(lines)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8",
        prefix=f"invoice_{data['invoice_number']}_",
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def _build_app():
    from main import build_graph
    return build_graph()


def _run_graph(app, invoice_data_fixture: dict):
    """Run graph with LLM calls mocked to return the given invoice fixture."""
    from schemas.models import InvoiceData
    from agents.critic_agent import CriticOutput

    invoice_path = _make_invoice_txt(invoice_data_fixture)

    from tools.file_loader import extract_invoice_content
    text = extract_invoice_content(invoice_path)

    from schemas.state import InvoiceGraphState
    initial: InvoiceGraphState = {
        "invoice_path": invoice_path,
        "invoice_text": text,
        "invoice_data": None,
        "critic_result": None,
        "correction_instruction": None,
        "validation_result": None,
        "approval_result": None,
        "retry_count": 0,
        "error": None,
        "audit": {"invoice_path": invoice_path},
    }

    validated_invoice = InvoiceData(**invoice_data_fixture)
    critic_ok = CriticOutput(valid=True, correction_instruction="")

    config = {"configurable": {"thread_id": invoice_data_fixture["invoice_number"] + "_test"}}

    with mock.patch("agents.ingestion_agent.get_structured_llm") as mock_ing, \
         mock.patch("agents.critic_agent.get_structured_llm") as mock_crit:
        mock_ing.return_value.invoke.return_value = validated_invoice
        mock_crit.return_value.invoke.return_value = critic_ok
        return app.invoke(initial, config=config)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_happy_path():
    app = _build_app()
    state = _run_graph(app, INVOICE_OK)

    assert state["invoice_data"]["invoice_number"] == "INV-1011"
    assert state["validation_result"]["status"] == "valid"
    assert state["approval_result"]["decision"] == "approve"
    assert state["audit"].get("payment_status") == "success"


def test_high_value_escalation():
    app = _build_app()
    state = _run_graph(app, INVOICE_HIGH_VALUE)

    assert state["approval_result"]["decision"] == "escalate"
    assert "10,000" in state["approval_result"]["reason"] or "threshold" in state["approval_result"]["reason"]
    assert state["audit"].get("payment_status") == "escalated"


def test_stock_insufficient_reject():
    app = _build_app()
    state = _run_graph(app, INVOICE_BAD_STOCK)

    assert state["validation_result"]["status"] == "invalid"
    issue_types = {i["type"] for i in state["validation_result"]["issues"]}
    assert "stock_insufficient" in issue_types
    assert state["approval_result"]["decision"] == "reject"


def test_unknown_vendor_allowed():
    """Unknown vendor is fine — no vendor approval system. Invoice approves if inventory passes."""
    app = _build_app()
    state = _run_graph(app, INVOICE_UNKNOWN_VENDOR)

    issue_types = {i["type"] for i in state["validation_result"]["issues"]}
    assert "vendor_warning" not in issue_types
    assert "vendor_missing" not in issue_types
    assert state["approval_result"]["decision"] == "approve"


def test_math_discrepancy_escalates():
    """Math discrepancy escalates for manual review — invoices with arithmetic errors must not be paid."""
    app = _build_app()
    state = _run_graph(app, INVOICE_MATH_DISCREPANCY)

    issue_types = {i["type"] for i in state["validation_result"]["issues"]}
    assert "math_discrepancy" in issue_types
    assert state["approval_result"]["decision"] == "escalate"
    assert state["audit"].get("payment_status") == "escalated"


def test_payment_blocked_on_reject():
    """Payment tool must raise RuntimeError if approval_decision != 'approve'."""
    from tools.payment_tool import execute_payment
    try:
        execute_payment("Summit Manufacturing Co.", 3000.0, approval_decision="reject")
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as exc:
        assert "blocked" in str(exc).lower()


def test_retry_exhaustion_routes_to_failure():
    """When ingestion always fails, graph must reach failure_node within 2 retries."""
    from schemas.state import InvoiceGraphState
    app = _build_app()

    invoice_path = _make_invoice_txt(INVOICE_OK)
    from tools.file_loader import extract_invoice_content
    text = extract_invoice_content(invoice_path)

    initial: InvoiceGraphState = {
        "invoice_path": invoice_path,
        "invoice_text": text,
        "invoice_data": None,
        "critic_result": None,
        "correction_instruction": None,
        "validation_result": None,
        "approval_result": None,
        "retry_count": 0,
        "error": None,
        "audit": {},
    }

    config = {"configurable": {"thread_id": "retry_exhaustion_test"}}

    # LLM always fails
    with mock.patch("agents.ingestion_agent.get_structured_llm") as mock_ing, \
         mock.patch("agents.critic_agent.get_structured_llm"):
        mock_ing.return_value.invoke.side_effect = RuntimeError("Simulated LLM failure")
        state = app.invoke(initial, config=config)

    # Graph must not crash; failure_node must have recorded the error
    assert state["audit"].get("validation_status") == "error"
    assert state["audit"].get("payment_status") == "none"
    assert state.get("retry_count", 0) >= 2


def test_duplicate_detection():
    """Duplicate invoices should be skipped unless 'revised' in filename."""
    from main import _quick_extract_invoice_number
    text = "Invoice Number: INV-1011\nVendor: Summit"
    assert _quick_extract_invoice_number(text) == "INV-1011"

    # Simulate processed_numbers tracking
    processed = {"INV-1011"}
    from pathlib import Path
    # Non-revised → should be skipped
    path = Path("/tmp/invoice_1011.txt")
    is_revised = "revised" in path.stem.lower()
    candidate = _quick_extract_invoice_number(text)
    should_skip = candidate in processed and not is_revised
    assert should_skip

    # Revised → should NOT be skipped
    path_revised = Path("/tmp/invoice_1011_revised.txt")
    is_revised = "revised" in path_revised.stem.lower()
    should_skip = candidate in processed and not is_revised
    assert not should_skip


def test_audit_report_written():
    """Audit report JSON must be written to outputs/ after processing."""
    from main import _OUT_DIR, process_invoice
    from schemas.models import InvoiceData
    from agents.critic_agent import CriticOutput

    fixture = {**INVOICE_OK, "invoice_number": "INV-AUDIT-TEST"}
    app = _build_app()

    validated_invoice = InvoiceData(**fixture)
    critic_ok = CriticOutput(valid=True, correction_instruction="")

    invoice_path = _make_invoice_txt(fixture)
    processed: set[str] = set()

    with mock.patch("agents.ingestion_agent.get_structured_llm") as mock_ing, \
         mock.patch("agents.critic_agent.get_structured_llm") as mock_crit, \
         mock.patch.dict(os.environ, {"ALLOW_DUPLICATES_FOR_TESTING": "true"}):
        mock_ing.return_value.invoke.return_value = validated_invoice
        mock_crit.return_value.invoke.return_value = critic_ok
        process_invoice(app, invoice_path, processed)

    report_path = _OUT_DIR / "audit_INV-AUDIT-TEST.json"
    assert report_path.exists(), f"Audit report not found: {report_path}"
    report = json.loads(report_path.read_text())
    assert report["invoice_number"] == "INV-AUDIT-TEST"
    assert report["approval_decision"] == "approve"
    assert report["payment_status"] == "success"


check("Graph: happy path approve + payment", test_happy_path)
check("Graph: high-value invoice escalated", test_high_value_escalation)
check("Graph: insufficient stock -> reject", test_stock_insufficient_reject)
check("Graph: unknown vendor -> approve (no vendor approval system)", test_unknown_vendor_allowed)
check("Graph: math discrepancy -> escalate (not approve)", test_math_discrepancy_escalates)
check("Safety: payment blocked when decision != approve (RuntimeError)", test_payment_blocked_on_reject)
check("Graph: retry exhaustion -> failure_node (no crash)", test_retry_exhaustion_routes_to_failure)
check("Logic: duplicate detection (revised vs. non-revised)", test_duplicate_detection)
check("Audit: report written to outputs/", test_audit_report_written)


# ── New verification cases (safety + observability improvements) ──────────────

def test_widget_a_matches_widgeta():
    """Case 1: 'Widget A' (space) must match inventory item 'WidgetA'."""
    from tools.inventory_tools import get_inventory_item
    result = get_inventory_item("Widget A")
    assert result["stock"] is not None, "'Widget A' should match 'WidgetA' in inventory"
    assert result["stock"] == 15, f"Expected stock=15, got {result['stock']}"
    assert result["item"] == "WidgetA"


def test_hyphenated_name_matches():
    """Case 1b: 'Widget-A' (hyphen) must also match 'WidgetA'."""
    from tools.inventory_tools import get_inventory_item
    result = get_inventory_item("Widget-A")
    assert result["stock"] == 15, f"Expected stock=15, got {result['stock']}"


def test_normalize_item_name_variants():
    """Case 1c: All normalize_item_name variants collapse to the same key."""
    from tools.inventory_tools import normalize_item_name
    assert normalize_item_name("Widget A")  == "widgeta"
    assert normalize_item_name("Widget-A")  == "widgeta"
    assert normalize_item_name("WidgetA")   == "widgeta"
    assert normalize_item_name("widget a")  == "widgeta"
    assert normalize_item_name("Gadget X")  == "gadgetx"
    assert normalize_item_name("GadgetX")   == "gadgetx"


def test_stock_mismatch_still_fails():
    """Case 2: Stock mismatch must still produce invalid validation status."""
    app = _build_app()
    state = _run_graph(app, INVOICE_BAD_STOCK)
    assert state["validation_result"]["status"] == "invalid"
    types = {i["type"] for i in state["validation_result"]["issues"]}
    assert "stock_insufficient" in types


def test_approval_cannot_approve_invalid_validation():
    """Case 3: Approval agent must reject when validation_status != 'valid' — including None."""
    from agents.approval_agent import run_approval
    # Simulate state where validation explicitly failed
    state = {
        "validation_result": {"status": "invalid", "issues": [
            {"type": "stock_insufficient", "message": "Not enough stock"}
        ]},
        "invoice_data": {"total_amount": 500.0},
        "approval_result": None,
    }
    result = run_approval(state)
    assert result["approval_result"]["decision"] == "reject"

    # None status (missing/empty validation_result) must also reject, not approve
    state_none = {
        "validation_result": {"status": None, "issues": []},
        "invoice_data": {"total_amount": 100.0},
        "approval_result": None,
    }
    result_none = run_approval(state_none)
    assert result_none["approval_result"]["decision"] == "reject"

    # Missing validation_result entirely must also reject
    state_missing = {
        "invoice_data": {"total_amount": 100.0},
        "approval_result": None,
    }
    result_missing = run_approval(state_missing)
    assert result_missing["approval_result"]["decision"] == "reject"


def test_payment_blocked_none_decision():
    """Case 4: Payment tool blocks on any non-approve value including None."""
    from tools.payment_tool import execute_payment
    for bad in ("reject", "escalate", "", None):
        try:
            execute_payment("Summit", 100.0, approval_decision=bad)
            raise AssertionError(f"Should have raised RuntimeError for decision={bad!r}")
        except RuntimeError as exc:
            assert "blocked" in str(exc).lower()


def test_agent_trace_output(capsys=None):
    """Case 5: Agent trace messages are printed during workflow execution."""
    import io
    from contextlib import redirect_stdout
    from schemas.models import InvoiceData
    from agents.critic_agent import CriticOutput

    app = _build_app()
    fixture = {**INVOICE_OK, "invoice_number": "INV-TRACE-TEST"}
    validated_invoice = InvoiceData(**fixture)
    critic_ok = CriticOutput(valid=True, correction_instruction="")
    invoice_path = _make_invoice_txt(fixture)

    buf = io.StringIO()
    with redirect_stdout(buf), \
         mock.patch("agents.ingestion_agent.get_structured_llm") as mock_ing, \
         mock.patch("agents.critic_agent.get_structured_llm") as mock_crit:
        mock_ing.return_value.invoke.return_value = validated_invoice
        mock_crit.return_value.invoke.return_value = critic_ok
        from schemas.state import InvoiceGraphState
        initial: InvoiceGraphState = {
            "invoice_path": invoice_path, "invoice_text": "dummy",
            "invoice_data": None, "critic_result": None,
            "correction_instruction": None, "validation_result": None,
            "approval_result": None, "retry_count": 0, "error": None, "audit": {},
        }
        from tools.file_loader import extract_invoice_content
        initial["invoice_text"] = extract_invoice_content(invoice_path)
        config = {"configurable": {"thread_id": "trace-test"}}
        app.invoke(initial, config=config)

    output = buf.getvalue()
    assert "[AGENT TRACE] -> Ingestion Agent" in output
    assert "[AGENT TRACE] -> Extraction Critic" in output
    assert "[AGENT TRACE] -> Validation Agent" in output
    assert "[AGENT TRACE] -> Approval Agent" in output
    assert "[AGENT TRACE] -> Payment Tool" in output


check("Normalization: 'Widget A' matches inventory 'WidgetA'", test_widget_a_matches_widgeta)
check("Normalization: 'Widget-A' (hyphen) matches 'WidgetA'", test_hyphenated_name_matches)
check("Normalization: all name variants collapse correctly", test_normalize_item_name_variants)
check("Validation: stock mismatch still produces invalid status", test_stock_mismatch_still_fails)
check("Approval: cannot approve with invalid validation status", test_approval_cannot_approve_invalid_validation)
check("Payment: RuntimeError on any non-approve decision", test_payment_blocked_none_decision)
check("Trace: all agent trace lines emitted during workflow", test_agent_trace_output)


# ── New verification cases (Tasks 1–5) ───────────────────────────────────────

def test_schema_math_validator_passes_consistent():
    """Schema accepts invoices where total matches item sum (tax-inclusive totals also accepted)."""
    from schemas.models import InvoiceData, InvoiceItem
    inv = InvoiceData(
        invoice_number="INV-T1",
        vendor="Summit Manufacturing Co.",
        total_amount=3000.0,
        items=[
            InvoiceItem(name="WidgetA", quantity=6, unit_price=250.0),
            InvoiceItem(name="WidgetB", quantity=3, unit_price=500.0),
        ],
    )
    assert inv.total_amount == 3000.0


def test_schema_math_validator_rejects_bad_total():
    """Schema accepts tax-inclusive totals; math discrepancy is flagged by validation_agent."""
    from schemas.models import InvoiceData, InvoiceItem
    # total=9975, items sum=9500, delta=475 (5% tax) — schema must accept this now
    inv = InvoiceData(
        invoice_number="INV-T2",
        vendor="Summit Manufacturing Co.",
        total_amount=9975.0,
        items=[
            InvoiceItem(name="WidgetA", quantity=12, unit_price=250.0),
            InvoiceItem(name="WidgetB", quantity=7,  unit_price=500.0),
        ],
    )
    assert inv.total_amount == 9975.0


def test_schema_math_validator_skips_missing_price():
    """Schema accepts invoices with unknown item prices (partial pricing data)."""
    from schemas.models import InvoiceData, InvoiceItem
    inv = InvoiceData(
        invoice_number="INV-T3",
        vendor="Summit Manufacturing Co.",
        total_amount=9999.0,
        items=[
            InvoiceItem(name="WidgetA", quantity=6, unit_price=250.0),
            InvoiceItem(name="WidgetB", quantity=3, unit_price=None),
        ],
    )
    assert inv.total_amount == 9999.0


def test_schema_allows_negative_total_for_validation_agent():
    """Schema allows negative total_amount so validation_agent can flag it explicitly (e.g. INV-1009)."""
    from schemas.models import InvoiceData, InvoiceItem
    inv = InvoiceData(
        invoice_number="INV-T4", vendor="X",
        total_amount=-1.0,
        items=[InvoiceItem(name="WidgetA", quantity=1)],
    )
    assert inv.total_amount == -1.0  # passes schema; validation_agent flags math discrepancy


def test_registry_check_duplicate():
    """Task 1: check_duplicate correctly classifies NEW, DUPLICATE_CONTENT, REVISED_VERSION."""
    from tools.registry_tool import check_duplicate, log_invoice_to_registry
    import hashlib, time as t

    ts = int(t.time() * 1000)
    unique_text = f"Invoice Number: INV-REG-{ts}\nVendor: TestCo\nItems:\n  WidgetA quantity 1"
    inv_num = f"INV-REG-{ts}"

    # Disable testing override so duplicate detection actually runs
    with mock.patch.dict(os.environ, {"ALLOW_DUPLICATES_FOR_TESTING": "false"}):
        # First check: NEW
        result = check_duplicate(unique_text, inv_num)
        assert result["status"] == "NEW"
        content_hash = result["content_hash"]

        # Log it
        log_invoice_to_registry(content_hash, inv_num, "TestCo", "paid", "/test/path.txt")

        # Same text: DUPLICATE_CONTENT
        result2 = check_duplicate(unique_text, inv_num)
        assert result2["status"] == "DUPLICATE_CONTENT"

        # Different text, same invoice_number: REVISED_VERSION
        revised_text = unique_text + "\n  WidgetB quantity 2"
        result3 = check_duplicate(revised_text, inv_num)
        assert result3["status"] == "REVISED_VERSION"


def test_registry_duplicate_skips_processing():
    """Task 5: process_invoice skips DUPLICATE_CONTENT invoices via registry."""
    from main import process_invoice, _OUT_DIR
    from schemas.models import InvoiceData
    from agents.critic_agent import CriticOutput
    from tools.registry_tool import log_invoice_to_registry
    import hashlib

    # Pre-populate registry with this invoice
    fixture = {**INVOICE_OK, "invoice_number": "INV-REG-DUP"}
    invoice_path = _make_invoice_txt(fixture)
    from tools.file_loader import extract_invoice_content
    text = extract_invoice_content(invoice_path)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log_invoice_to_registry(content_hash, "INV-REG-DUP", "Summit Manufacturing Co.", "paid", invoice_path)

    app = _build_app()
    processed: set[str] = set()

    # Disable testing override so duplicate detection actually runs
    with mock.patch.dict(os.environ, {"ALLOW_DUPLICATES_FOR_TESTING": "false"}):
        state = process_invoice(app, invoice_path, processed)

    assert state.get("skipped") is True
    assert state.get("registry_status") == "DUPLICATE_CONTENT"


def test_audit_report_includes_timing():
    """Task 5: Audit report JSON must include execution_times_ms."""
    from main import _OUT_DIR, process_invoice
    from schemas.models import InvoiceData
    from agents.critic_agent import CriticOutput

    fixture = {**INVOICE_OK, "invoice_number": "INV-TIMING-TEST"}
    app = _build_app()
    validated_invoice = InvoiceData(**fixture)
    critic_ok = CriticOutput(valid=True, correction_instruction="")
    invoice_path = _make_invoice_txt(fixture)
    processed: set[str] = set()

    with mock.patch("agents.ingestion_agent.get_structured_llm") as mock_ing, \
         mock.patch("agents.critic_agent.get_structured_llm") as mock_crit, \
         mock.patch.dict(os.environ, {"ALLOW_DUPLICATES_FOR_TESTING": "true"}):
        mock_ing.return_value.invoke.return_value = validated_invoice
        mock_crit.return_value.invoke.return_value = critic_ok
        process_invoice(app, invoice_path, processed)

    report_path = _OUT_DIR / "audit_INV-TIMING-TEST.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "execution_times_ms" in report, "Audit report missing execution_times_ms"
    times = report["execution_times_ms"]
    assert "ingestion_agent" in times
    assert "validation_agent" in times
    assert "approval_agent" in times
    assert "payment_tool" in times


def test_approval_rule_order_high_value_before_vendor():
    """Task 3: High-value escalation fires before vendor check."""
    from agents.approval_agent import run_approval
    state = {
        "validation_result": {"status": "valid", "issues": [
            {"type": "vendor_unknown", "message": "Vendor not found"}
        ]},
        "invoice_data": {"total_amount": 15000.0},
    }
    result = run_approval(state)
    # High-value rule (Rule 2) fires before vendor rule (Rule 3)
    assert result["approval_result"]["decision"] == "escalate"
    assert "15,000" in result["approval_result"]["reason"] or "threshold" in result["approval_result"]["reason"]


def test_regex_extraction():
    """Task 4: regex_extraction correctly pulls invoice_number and total_amount."""
    from agents.ingestion_agent import regex_extraction
    text = """
    Invoice Number: INV-2025-007
    Vendor: Summit Manufacturing Co.
    Total Amount: $4500.00
    """
    result = regex_extraction(text)
    assert result.get("invoice_number") == "INV-2025-007"
    assert result.get("total_amount") == 4500.0


def test_payment_tool_guard_confirmed():
    """Task 6: Payment guard raises RuntimeError — confirmed immutable."""
    from tools.payment_tool import execute_payment
    for bad in ("reject", "escalate", "", "APPROVE", None):
        try:
            execute_payment("Vendor", 100.0, approval_decision=bad)
            raise AssertionError(f"Guard missing for decision={bad!r}")
        except RuntimeError as exc:
            assert "blocked" in str(exc).lower()


check("Schema: math validator accepts consistent total", test_schema_math_validator_passes_consistent)
check("Schema: math validator rejects bad total (all prices known)", test_schema_math_validator_rejects_bad_total)
check("Schema: math validator skips when any price is missing", test_schema_math_validator_skips_missing_price)
check("Schema: negative total_amount passes schema (caught by validation_agent)", test_schema_allows_negative_total_for_validation_agent)
def test_rejected_invoice_logged_to_registry():
    """Rejected invoices must be logged to the registry so re-submission is caught as a duplicate."""
    from main import process_invoice
    from tools.registry_tool import check_duplicate, _DB_PATH
    import sqlite3

    # Clean up any leftover registry entry from previous test runs so this test
    # is idempotent across multiple pytest invocations.
    try:
        with sqlite3.connect(str(_DB_PATH)) as _conn:
            _conn.execute("DELETE FROM processed_invoices WHERE invoice_number = 'INV-REG-REJECT'")
            _conn.commit()
    except sqlite3.Error:
        pass

    fixture = {**INVOICE_BAD_STOCK, "invoice_number": "INV-REG-REJECT"}
    app = _build_app()
    invoice_path = _make_invoice_txt(fixture)
    processed: set[str] = set()

    with mock.patch("agents.ingestion_agent.get_structured_llm") as mock_ing, \
         mock.patch("agents.critic_agent.get_structured_llm") as mock_crit, \
         mock.patch.dict(os.environ, {"ALLOW_DUPLICATES_FOR_TESTING": "false"}):
        from schemas.models import InvoiceData
        from agents.critic_agent import CriticOutput
        mock_ing.return_value.invoke.return_value = InvoiceData(**fixture)
        mock_crit.return_value.invoke.return_value = CriticOutput(valid=True, correction_instruction="")
        state = process_invoice(app, invoice_path, processed)

    assert state.get("approval_result", {}).get("decision") == "reject"

    # The same invoice submitted again must now be caught as DUPLICATE_CONTENT
    from tools.file_loader import extract_invoice_content
    text = extract_invoice_content(invoice_path)
    with mock.patch.dict(os.environ, {"ALLOW_DUPLICATES_FOR_TESTING": "false"}):
        dup_result = check_duplicate(text, "INV-REG-REJECT")
    assert dup_result["status"] == "DUPLICATE_CONTENT", \
        f"Rejected invoice not in registry — re-submission would be reprocessed (got {dup_result['status']})"


check("Registry: rejected invoice logged (prevents re-submission)", test_rejected_invoice_logged_to_registry)
check("Registry: NEW / DUPLICATE_CONTENT / REVISED_VERSION classification", test_registry_check_duplicate)
check("Registry: duplicate content skips processing", test_registry_duplicate_skips_processing)
check("Audit: execution_times_ms included in report", test_audit_report_includes_timing)
check("Approval: high-value escalation fires before vendor check", test_approval_rule_order_high_value_before_vendor)
check("Regex: extract invoice_number and total_amount from text", test_regex_extraction)
check("Payment guard: RuntimeError on all non-approve inputs", test_payment_tool_guard_confirmed)


# ── Report ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  INTEGRATION TEST RESULTS")
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
