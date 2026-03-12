"""
Invoice Registry Tool — persistent deduplication and audit trail.

Uses the processed_invoices table in inventory.db.
Provides two functions consumed by main.py orchestration:

  check_duplicate()         — pre-flight content/version check
  log_invoice_to_registry() — post-processing persistence
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger("invoice-agent")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _PROJECT_ROOT / "database" / "inventory.db"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"


def _init_registry_db() -> None:
    """
    Create the database directory and processed_invoices table if they don't exist.
    Called automatically at module import. Safe to call multiple times.
    """
    _DB_PATH.parent.mkdir(exist_ok=True)
    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_invoices (
                    content_hash   TEXT PRIMARY KEY,
                    invoice_number TEXT,
                    vendor         TEXT,
                    status         TEXT,
                    file_path      TEXT
                )
            """)
            conn.commit()
    except sqlite3.Error:
        pass  # Non-fatal; check_duplicate and log_invoice_to_registry handle missing table.


# Auto-initialize on import so the registry is always ready.
_init_registry_db()


def _find_transaction_id_for_invoice(invoice_number: str) -> str | None:
    """
    Scan outputs/receipt_*.json files for a receipt matching invoice_number.
    Returns the transaction_id string, or None if not found.
    """
    if not invoice_number or not _OUTPUTS_DIR.exists():
        return None
    for receipt_file in sorted(_OUTPUTS_DIR.glob("receipt_*.json"), reverse=True):
        try:
            data = json.loads(receipt_file.read_text(encoding="utf-8"))
            if data.get("invoice_number") == invoice_number:
                return data.get("transaction_id")
        except Exception:
            continue
    return None


def check_duplicate(text: str, invoice_number: str) -> dict:
    """
    Determine whether this invoice has been seen before.

    Logic (in order):
      1. Compute SHA-256 of the normalized invoice text.
      2. If the hash already exists in processed_invoices → DUPLICATE_CONTENT.
      3. Else if the same invoice_number exists with a different hash → REVISED_VERSION.
      4. Otherwise → NEW.

    Returns:
        {
            "status": "NEW" | "DUPLICATE_CONTENT" | "REVISED_VERSION",
            "content_hash": str
        }
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    """
    Development override: allows repeated testing locally.
    In production environments this variable should NOT be set.
    """
    
    if os.getenv("ALLOW_DUPLICATES_FOR_TESTING", "false").lower() == "true":
        logger.info("[REGISTRY] ALLOW_DUPLICATES_FOR_TESTING=true — skipping duplicate check")
        return {"status": "NEW", "content_hash": content_hash}

    if not _DB_PATH.exists():
        return {"status": "NEW", "content_hash": content_hash}

    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            # Check 1: exact content match
            row = conn.execute(
                "SELECT invoice_number FROM processed_invoices WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if row is not None:
                original_inv_num = row[0]
                original_txn_id = _find_transaction_id_for_invoice(original_inv_num)
                logger.info(
                    f"[REGISTRY] content_hash match — DUPLICATE_CONTENT"
                    f" (original_transaction_id={original_txn_id})"
                )
                return {
                    "status": "DUPLICATE_CONTENT",
                    "content_hash": content_hash,
                    "original_invoice_hash": content_hash,
                    "original_transaction_id": original_txn_id,
                }

            # Check 2: same invoice number, different content (revised)
            if invoice_number and invoice_number != "UNKNOWN":
                row2 = conn.execute(
                    "SELECT 1 FROM processed_invoices WHERE invoice_number = ? LIMIT 1",
                    (invoice_number,),
                ).fetchone()
                if row2 is not None:
                    logger.info(f"[REGISTRY] invoice_number match, new hash — REVISED_VERSION")
                    return {"status": "REVISED_VERSION", "content_hash": content_hash}

    except sqlite3.OperationalError:
        # processed_invoices table not yet created — treat as NEW
        pass

    logger.info(f"[REGISTRY] no prior record found — NEW")
    return {"status": "NEW", "content_hash": content_hash}


def log_invoice_to_registry(
    content_hash: str,
    invoice_number: str,
    vendor: str,
    status: str,
    file_path: str,
) -> None:
    """
    Persist a processed invoice record.

    Uses INSERT OR REPLACE so re-processing a revised invoice updates the row.
    """
    if not _DB_PATH.exists():
        return  # DB not initialized; skip silently

    try:
        with sqlite3.connect(str(_DB_PATH)) as conn:
            # FIX 1: mark any prior entry for the same invoice_number as superseded
            if invoice_number:
                conn.execute(
                    "UPDATE processed_invoices SET status='superseded' "
                    "WHERE invoice_number=? AND content_hash!=?",
                    (invoice_number, content_hash),
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_invoices
                    (content_hash, invoice_number, vendor, status, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content_hash, invoice_number or "", vendor or "", status, file_path),
            )
            conn.commit()
    except sqlite3.OperationalError:
        pass  # Table not created yet; non-fatal
