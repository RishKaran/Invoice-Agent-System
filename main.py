"""
Invoice Agent System — CLI entry point and LangGraph orchestration.

Usage:
  python main.py --invoice_path=data/invoices/invoice1.txt
  python main.py --invoice_folder=data/invoices/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Directory setup ───────────────────────────────────────────────────────────
_LOG_DIR = ROOT / "logs"
_OUT_DIR = ROOT / "outputs"
_DB_DIR  = ROOT / "database"
_LOG_DIR.mkdir(exist_ok=True)
_OUT_DIR.mkdir(exist_ok=True)
_DB_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "system.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("invoice-agent")

# ── LangGraph imports ─────────────────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.approval_agent import run_approval
from agents.critic_agent import run_critic
from agents.ingestion_agent import run_ingestion
from agents.validation_agent import run_validation
from schemas.state import InvoiceGraphState
from tools.file_loader import extract_invoice_content
from tools.payment_tool import execute_payment
from tools.registry_tool import check_duplicate, log_invoice_to_registry

_SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".json", ".csv", ".xml", ".xlsx", ".xls"}


# ─────────────────────────────────────────────────────────────────────────────
# Agent trace
# ─────────────────────────────────────────────────────────────────────────────

def _trace(message: str) -> None:
    """Print a live agent trace line to stdout (unbuffered)."""
    print(f"[AGENT TRACE] -> {message}", flush=True)


def _timed_node(label: str, key: str, fn, state: InvoiceGraphState) -> dict:
    """Run a node function, measure wall-clock time, and inject ms into audit."""
    _trace(label)
    start = time.time()
    result = fn(state)
    elapsed = round((time.time() - start) * 1000, 2)

    audit = dict(state.get("audit") or {})
    audit.setdefault("execution_times_ms", {})[key] = elapsed
    result["audit"] = audit
    return result


def _traced_ingestion(state: InvoiceGraphState) -> dict:
    return _timed_node("Ingestion Agent", "ingestion_agent", run_ingestion, state)


def _traced_critic(state: InvoiceGraphState) -> dict:
    return _timed_node("Extraction Critic", "critic_agent", run_critic, state)


def _traced_validation(state: InvoiceGraphState) -> dict:
    result = _timed_node("Validation Agent", "validation_agent", run_validation, state)
    if (result.get("validation_result") or {}).get("status") == "invalid":
        _trace("Validation failed — routing to approval decision")
    return result


def _traced_approval(state: InvoiceGraphState) -> dict:
    return _timed_node("Approval Agent", "approval_agent", run_approval, state)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal nodes (payment, manual review, failure)
# ─────────────────────────────────────────────────────────────────────────────

def payment_node(state: InvoiceGraphState) -> dict:
    """Execute payment, record receipt in audit, log to registry."""
    _trace("Payment Tool")
    start = time.time()
    invoice_data = state.get("invoice_data") or {}
    approval = state.get("approval_result") or {}
    audit = dict(state.get("audit") or {})

    try:
        receipt = execute_payment(
            vendor=invoice_data.get("vendor", ""),
            amount=float(invoice_data.get("total_amount") or 0),
            approval_decision=approval.get("decision", ""),
            invoice_number=invoice_data.get("invoice_number", ""),
        )
        logger.info(f"[PAYMENT] executed {receipt['transaction_id']}")
        audit["payment_status"] = "success"
        audit["transaction_id"] = receipt["transaction_id"]
    except Exception as exc:
        logger.error(f"[PAYMENT] failed: {exc}")
        audit["payment_status"] = "failed"
        audit["payment_error"] = str(exc)

    audit.setdefault("execution_times_ms", {})["payment_tool"] = round((time.time() - start) * 1000, 2)

    log_invoice_to_registry(
        content_hash=audit.get("content_hash", ""),
        invoice_number=invoice_data.get("invoice_number", ""),
        vendor=invoice_data.get("vendor", ""),
        status=audit["payment_status"],
        file_path=audit.get("invoice_path", ""),
    )
    return {"audit": audit}


def reject_node(state: InvoiceGraphState) -> dict:
    """Log rejection, record in audit, log to registry."""
    invoice_data = state.get("invoice_data") or {}
    approval = state.get("approval_result") or {}
    audit = dict(state.get("audit") or {})

    inv_num = invoice_data.get("invoice_number", "UNKNOWN")
    reason = approval.get("reason", "validation failed")
    logger.info(f"[APPROVAL] {inv_num} rejected: {reason}")

    audit["payment_status"] = "rejected"

    log_invoice_to_registry(
        content_hash=audit.get("content_hash", ""),
        invoice_number=invoice_data.get("invoice_number", ""),
        vendor=invoice_data.get("vendor", ""),
        status="rejected",
        file_path=audit.get("invoice_path", ""),
    )
    return {"audit": audit}


def manual_review_node(state: InvoiceGraphState) -> dict:
    """Log escalation, record in audit, log to registry."""
    invoice_data = state.get("invoice_data") or {}
    approval = state.get("approval_result") or {}
    audit = dict(state.get("audit") or {})

    inv_num = invoice_data.get("invoice_number", "UNKNOWN")
    reason = approval.get("reason", "unknown reason")
    logger.info(f"[APPROVAL] {inv_num} escalated for manual review: {reason}")

    audit["payment_status"] = "escalated"

    log_invoice_to_registry(
        content_hash=audit.get("content_hash", ""),
        invoice_number=invoice_data.get("invoice_number", ""),
        vendor=invoice_data.get("vendor", ""),
        status="escalated",
        file_path=audit.get("invoice_path", ""),
    )
    return {"audit": audit}


def failure_node(state: InvoiceGraphState) -> dict:
    """Log failure, write partial audit report, log to registry, terminate safely."""
    error = (state.get("error")
             or state.get("correction_instruction")
             or "Unknown failure")
    retry_count = state.get("retry_count", 0)
    audit = dict(state.get("audit") or {})

    logger.error(f"[FAILURE] agent retries exhausted ({retry_count}): {error}")

    audit["validation_status"] = "error"
    audit["payment_status"] = "none"
    audit["issues"] = [error]

    _write_audit_report(state, audit, override_payment_status="none")

    log_invoice_to_registry(
        content_hash=audit.get("content_hash", ""),
        invoice_number=(state.get("invoice_data") or {}).get("invoice_number", ""),
        vendor=(state.get("invoice_data") or {}).get("vendor", ""),
        status="failed",
        file_path=audit.get("invoice_path", ""),
    )
    return {"audit": audit}


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_critic(state: InvoiceGraphState) -> str:
    critic = state.get("critic_result") or {}
    if critic.get("valid"):
        return "validation_agent"
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        return "failure_node"
    _trace("Critic requested correction — retrying ingestion")
    return "ingestion_agent"


def _route_after_approval(state: InvoiceGraphState) -> str:
    approval = state.get("approval_result") or {}
    decision = approval.get("decision", "reject")
    if decision == "approve":
        return "payment_node"
    if decision == "escalate":
        _trace("Escalated for manual review")
        return "manual_review_node"
    return "reject_node"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Compile the LangGraph invoice processing workflow."""
    workflow = StateGraph(InvoiceGraphState)

    # ── Nodes (traced wrappers used for ingestion/critic/validation/approval) ──
    workflow.add_node("ingestion_agent",   _traced_ingestion)
    workflow.add_node("critic_agent",      _traced_critic)
    workflow.add_node("validation_agent",  _traced_validation)
    workflow.add_node("approval_agent",    _traced_approval)
    workflow.add_node("payment_node",       payment_node)
    workflow.add_node("reject_node",        reject_node)
    workflow.add_node("manual_review_node", manual_review_node)
    workflow.add_node("failure_node",       failure_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.set_entry_point("ingestion_agent")
    workflow.add_edge("ingestion_agent", "critic_agent")

    workflow.add_conditional_edges(
        "critic_agent",
        _route_after_critic,
        {
            "ingestion_agent":  "ingestion_agent",   # retry
            "validation_agent": "validation_agent",   # proceed
            "failure_node":     "failure_node",        # give up
        },
    )

    workflow.add_edge("validation_agent", "approval_agent")

    workflow.add_conditional_edges(
        "approval_agent",
        _route_after_approval,
        {
            "payment_node":       "payment_node",
            "reject_node":        "reject_node",
            "manual_review_node": "manual_review_node",
        },
    )

    workflow.add_edge("payment_node",       END)
    workflow.add_edge("reject_node",        END)
    workflow.add_edge("manual_review_node", END)
    workflow.add_edge("failure_node",       END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# Audit report
# ─────────────────────────────────────────────────────────────────────────────

def _write_audit_report(state: dict, audit: dict, override_payment_status: str | None = None) -> None:
    invoice_data = state.get("invoice_data") or {}
    validation = state.get("validation_result") or {}
    approval = state.get("approval_result") or {}

    report = {
        "invoice_number":    invoice_data.get("invoice_number") or audit.get("invoice_number"),
        "vendor":            invoice_data.get("vendor") or audit.get("vendor"),
        "validation_status": validation.get("status") or audit.get("validation_status", "unknown"),
        "approval_decision": (approval.get("decision") or audit.get("approval_decision")),
        "payment_status":    override_payment_status or audit.get("payment_status", "none"),
        "issues":            [i.get("message", str(i)) for i in validation.get("issues", [])]
                             or audit.get("issues", []),
    }
    report["invoice_hash"] = audit.get("content_hash")
    if "transaction_id" in audit:
        report["transaction_id"] = audit["transaction_id"]
    if "execution_times_ms" in audit:
        report["execution_times_ms"] = audit["execution_times_ms"]
    # Extraction method (deterministic_json / deterministic_csv / llm_extraction)
    if state.get("extraction_method"):
        report["extraction_method"] = state["extraction_method"]
    # Risk, math breakdown, and audit clarity fields from validation
    for _key in ("risk_score", "risk_category", "subtotal", "tax", "shipping",
                 "handling", "fees", "expected_total",
                 "normalized_items", "normalization_applied"):
        if _key in validation:
            report[_key] = validation[_key]

    inv_num = (report["invoice_number"]
               or audit.get("candidate_number")
               or f"unknown-{audit.get('content_hash', 'x')[:8]}")
    report_path = _OUT_DIR / f"audit_{inv_num}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"[AUDIT] report saved: {report_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Invoice number quick-scan (for duplicate detection before graph runs)
# ─────────────────────────────────────────────────────────────────────────────

def _quick_extract_invoice_number(text: str) -> str | None:
    """Regex scan for an invoice number in normalized text."""
    match = re.search(
        r"invoice\s*(?:(?:number|no\.?|#)\s*[:\-]?\s*|[:\-#]\s*)([A-Za-z0-9][A-Za-z0-9\-]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


# ─────────────────────────────────────────────────────────────────────────────
# Core processor
# ─────────────────────────────────────────────────────────────────────────────

def process_invoice(
    app,
    invoice_path: str,
    processed_numbers: set[str],
) -> dict:
    """
    Load, run graph, handle duplicates, save audit.
    Returns the final LangGraph state dict.
    """
    path = Path(invoice_path)
    is_revised = "revised" in path.stem.lower()

    # ── Load and normalize ────────────────────────────────────────────────────
    try:
        text = extract_invoice_content(invoice_path)
    except Exception as exc:
        logger.error(f"[LOADER] failed to load '{path.name}': {exc}")
        return {"error": str(exc), "_file": path.name}

    # ── Registry + in-memory duplicate detection ──────────────────────────────
    candidate_number = _quick_extract_invoice_number(text)

    registry_result = check_duplicate(text, candidate_number or "UNKNOWN")
    content_hash = registry_result["content_hash"]
    registry_status = registry_result["status"]

    if registry_status == "DUPLICATE_CONTENT":
        logger.info(f"[REGISTRY] Duplicate invoice detected – skipping processing")
        dup_report = {
            "invoice_number": candidate_number or "unknown",
            "vendor": None,
            "status": "duplicate_skipped",
            "invoice_hash": content_hash,
            "original_invoice_hash": registry_result.get("original_invoice_hash"),
            "original_transaction_id": registry_result.get("original_transaction_id"),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        inv_label = candidate_number or "unknown"
        dup_path = _OUT_DIR / f"audit_{inv_label}_duplicate.json"
        dup_path.write_text(json.dumps(dup_report, indent=2), encoding="utf-8")
        logger.info(f"[AUDIT] duplicate report saved: {dup_path.name}")
        return {"skipped": True, "invoice_number": candidate_number, "registry_status": "DUPLICATE_CONTENT"}

    if registry_status == "REVISED_VERSION":
        logger.info(f"[REGISTRY] Revised invoice detected – processing new version")

    # Fallback: in-memory set check (catches within-session duplicates pre-registry)
    if candidate_number and candidate_number in processed_numbers:
        if not is_revised and registry_status == "NEW":
            logger.info(f"[DUPLICATE] skipping {candidate_number} (in-memory, not revised)")
            return {"skipped": True, "invoice_number": candidate_number}
        logger.info(f"[DUPLICATE] processing revised version of {candidate_number}")

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state: InvoiceGraphState = {
        "invoice_path":          invoice_path,
        "invoice_text":          text,
        "invoice_data":          None,
        "critic_result":         None,
        "correction_instruction": None,
        "validation_result":     None,
        "approval_result":       None,
        "retry_count":           0,
        "error":                 None,
        "extraction_method":     None,
        "audit":                 {
            "invoice_path":    str(path),
            "content_hash":    content_hash,
            "registry_status": registry_status,
            "candidate_number": candidate_number,  # pre-graph regex scan; filename fallback
        },
    }

    config = {"configurable": {"thread_id": path.stem}}

    # ── Run graph ─────────────────────────────────────────────────────────────
    try:
        final_state = app.invoke(initial_state, config=config)
    except Exception as exc:
        logger.error(f"[GRAPH] unexpected error for '{path.name}': {exc}")
        return {"error": str(exc), "_file": path.name}

    # ── Post-flight duplicate check (using extracted invoice_number) ──────────
    invoice_data = final_state.get("invoice_data") or {}
    invoice_number = invoice_data.get("invoice_number")

    if invoice_number:
        if invoice_number in processed_numbers and not is_revised:
            # Graph ran but invoice was a duplicate — still return result
            logger.warning(f"[DUPLICATE] {invoice_number} was processed again (post-graph detection)")
        processed_numbers.add(invoice_number)

    # ── Save audit report ─────────────────────────────────────────────────────
    audit = final_state.get("audit") or {}
    _write_audit_report(final_state, audit)

    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# CLI output
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(state: dict) -> None:
    """Print the human-readable summary to stdout."""
    sep = "=" * 48

    if state.get("skipped"):
        print(f"\n{sep}")
        print(f"  Duplicate Skipped")
        print(sep)
        print(f"  Invoice Number : {state.get('invoice_number', 'N/A')}")
        print(f"{sep}\n")
        return

    if state.get("error") and not state.get("invoice_data"):
        print(f"\n{sep}")
        print(f"  Processing Failed")
        print(sep)
        print(f"  Error : {state['error']}")
        print(f"{sep}\n")
        return

    invoice_data = state.get("invoice_data") or {}
    validation   = state.get("validation_result") or {}
    approval     = state.get("approval_result") or {}
    audit        = state.get("audit") or {}

    val_label      = "PASSED" if validation.get("status") == "valid" else "FAILED"
    approval_label = (approval.get("decision") or "none").upper()
    payment_label  = (audit.get("payment_status") or "none").upper()

    print(f"\n{sep}")
    print(f"  Invoice Processed")
    print(sep)
    print(f"  Invoice Number : {invoice_data.get('invoice_number', 'N/A')}")
    print(f"  Vendor         : {invoice_data.get('vendor', 'N/A')}")
    print(f"  Validation     : {val_label}")
    print(f"  Approval       : {approval_label}")
    print(f"  Payment        : {payment_label}")

    issues = validation.get("issues", [])
    if issues:
        print(f"  Issues         :")
        for issue in issues:
            print(f"    - {issue.get('message', issue)}")

    approval_reason = approval.get("reason")
    if approval_reason and approval_label != "APPROVE":
        print(f"  Reason         : {approval_reason}")

    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="invoice-agent",
        description="AI-powered invoice processing system",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--invoice_path",
        metavar="FILE",
        help="Path to a single invoice file (txt, pdf, json, csv, xml, xlsx)",
    )
    group.add_argument(
        "--invoice_folder",
        metavar="DIR",
        help="Path to a folder of invoice files (processed sequentially)",
    )
    args = parser.parse_args()

    logger.info("[SYSTEM] building invoice processing graph")
    app = build_graph()
    processed_numbers: set[str] = set()

    if args.invoice_path:
        state = process_invoice(app, args.invoice_path, processed_numbers)
        _print_result(state)

    elif args.invoice_folder:
        folder = Path(args.invoice_folder)
        if not folder.is_dir():
            print(f"Error: '{folder}' is not a directory.")
            sys.exit(1)

        invoice_files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        if not invoice_files:
            print(f"No supported invoice files found in '{folder}'.")
            sys.exit(0)

        logger.info(f"[SYSTEM] processing {len(invoice_files)} invoice(s) from '{folder}'")

        tally: dict[str, int] = {"approve": 0, "escalate": 0, "reject": 0, "skipped": 0, "error": 0}
        batch_rows: list[dict] = []

        for invoice_file in invoice_files:
            logger.info(f"[SYSTEM] --- {invoice_file.name} ---")
            state = process_invoice(app, str(invoice_file), processed_numbers)
            _print_result(state)

            invoice_data = state.get("invoice_data") or {}
            validation   = state.get("validation_result") or {}
            approval     = state.get("approval_result") or {}
            audit        = state.get("audit") or {}

            if state.get("skipped"):
                tally["skipped"] += 1
                batch_rows.append({
                    "invoice_number":    state.get("invoice_number", ""),
                    "vendor":            "",
                    "file":              invoice_file.name,
                    "validation_status": "skipped",
                    "approval_decision": "skipped",
                    "payment_status":    "skipped",
                    "total_amount":      "",
                    "transaction_id":    "",
                    "risk_score":        "",
                    "risk_category":     "",
                    "reason":            "duplicate skipped",
                })
            elif state.get("error") and not state.get("invoice_data"):
                tally["error"] += 1
                batch_rows.append({
                    "invoice_number":    "",
                    "vendor":            "",
                    "file":              invoice_file.name,
                    "validation_status": "error",
                    "approval_decision": "error",
                    "payment_status":    "error",
                    "total_amount":      "",
                    "transaction_id":    "",
                    "risk_score":        "",
                    "risk_category":     "",
                    "reason":            state.get("error", ""),
                })
            else:
                decision = (approval.get("decision") or "error")
                tally[decision] = tally.get(decision, 0) + 1
                batch_rows.append({
                    "invoice_number":    invoice_data.get("invoice_number", ""),
                    "vendor":            invoice_data.get("vendor", ""),
                    "file":              invoice_file.name,
                    "validation_status": validation.get("status", ""),
                    "approval_decision": decision,
                    "payment_status":    audit.get("payment_status", ""),
                    "total_amount":      invoice_data.get("total_amount", ""),
                    "transaction_id":    audit.get("transaction_id", ""),
                    "risk_score":        validation.get("risk_score", ""),
                    "risk_category":     validation.get("risk_category", ""),
                    "reason":            approval.get("reason", ""),
                })

        # ── CSV audit export ──────────────────────────────────────────────────
        # Deduplicate: per invoice_number, keep the last non-skipped row
        # (revised version supersedes original). Skipped rows are included only
        # when no real record exists for that invoice. Rows without an invoice
        # number (load errors) are always kept.
        final_rows: dict[str, dict] = {}   # invoice_number -> winning row
        no_number_rows: list[dict] = []
        for row in batch_rows:
            inv_num = row["invoice_number"]
            if not inv_num:
                no_number_rows.append(row)
                continue
            existing = final_rows.get(inv_num)
            if existing is None:
                final_rows[inv_num] = row
            elif row["approval_decision"] != "skipped":
                final_rows[inv_num] = row  # revised/reprocessed version wins
            # else: existing non-skipped row wins over incoming skipped row

        deduped_rows = list(final_rows.values()) + no_number_rows

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = _OUT_DIR / f"batch_summary_{ts}.csv"
        _CSV_FIELDS = [
            "invoice_number", "vendor", "file", "validation_status",
            "approval_decision", "payment_status", "total_amount",
            "transaction_id", "risk_score", "risk_category", "reason",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(deduped_rows)
        logger.info(f"[AUDIT] batch CSV saved: {csv_path.name}")

        sep = "=" * 48
        print(f"\n{sep}")
        print(f"  Batch Summary — {len(invoice_files)} invoices")
        print(sep)
        print(f"  Approved   : {tally['approve']}")
        print(f"  Escalated  : {tally['escalate']}")
        print(f"  Rejected   : {tally['reject']}")
        if tally["skipped"]:
            print(f"  Skipped    : {tally['skipped']}")
        if tally["error"]:
            print(f"  Errors     : {tally['error']}")
        print(f"  Report     : {csv_path.name}")
        print(f"{sep}\n")


if __name__ == "__main__":
    main()
