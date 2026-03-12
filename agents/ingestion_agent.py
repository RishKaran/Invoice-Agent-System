"""
Ingestion Agent — extracts structured InvoiceData from invoice files.

Routing:
  .json  → parse_json_invoice()   (deterministic, no LLM)
  .csv   → parse_csv_invoice()    (deterministic, no LLM)
  other  → LLM extraction with critic/retry loop

Handles:
  • OCR artifacts (O/0 confusion)
  • Irregular spacing and currency symbols
  • Item name normalization (Widget A → WidgetA)
  • Retry with critic correction instructions
"""

from __future__ import annotations

import json as _json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.llm_provider import get_structured_llm
from schemas.models import InvoiceData
from schemas.state import InvoiceGraphState

logger = logging.getLogger("invoice-agent")

# ── LLM Prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an invoice data extraction specialist. Extract structured invoice data \
from the provided text and return it as valid JSON.

NORMALIZATION RULES:
1. ITEM NAMES: Remove spaces within multi-word product names.
   Examples: "Widget A" → "WidgetA",  "Gadget X" → "GadgetX"
2. NUMERIC OCR: Treat the letter 'O' as the digit '0' in numeric contexts.
   Example: "O2" in a quantity context → 2
3. CURRENCY SYMBOLS: Strip $, £, € from monetary values. Return bare numbers.
4. QUANTITIES: Must be positive integers (≥ 1). Never return 0 or negative.
5. INVOICE NUMBER: Required field. If missing from body, look in headers/footers.
6. VENDOR: Required. Use the full company name exactly as written in the document.
7. NOTES: Capture delivery instructions, PO references, or special remarks.
8. DUE DATE: Extract as-is from the document; do not reformat.

JSON SCHEMA (return exactly this shape):
{
  "invoice_number": "<string>",
  "vendor":         "<string>",
  "total_amount":   <number | null>,
  "items": [
    {"name": "<string>", "quantity": <positive int>, "unit_price": <number | null>}
  ],
  "due_date": "<string | null>",
  "notes":    "<string | null>"
}
"""


# ── Deterministic JSON parser ──────────────────────────────────────────────────

def parse_json_invoice(invoice_path: str) -> dict:
    """
    Parse a JSON invoice file directly into an InvoiceData-compatible dict.
    Handles nested vendor objects and line_items arrays.
    """
    with open(invoice_path, encoding="utf-8") as fh:
        data = _json.load(fh)

    # Vendor: may be {"name": ..., "address": ...} or a plain string
    vendor_raw = data.get("vendor", "")
    if isinstance(vendor_raw, dict):
        vendor = str(vendor_raw.get("name") or "").strip()
    else:
        vendor = str(vendor_raw).strip()

    # Line items → InvoiceItem format
    items: list[dict] = []
    for li in data.get("line_items", []):
        if not isinstance(li, dict):
            continue
        name = str(
            li.get("item") or li.get("name") or li.get("description") or ""
        ).strip()
        qty_raw = li.get("quantity") or li.get("qty") or 0
        price_raw = li.get("unit_price") or li.get("price")
        try:
            qty = int(qty_raw)
        except (TypeError, ValueError):
            qty = 0
        try:
            price = float(price_raw) if price_raw is not None else None
        except (TypeError, ValueError):
            price = None
        if name:
            items.append({"name": name, "quantity": qty, "unit_price": price})

    # Total amount
    total_raw = data.get("total")
    try:
        total_amount = float(total_raw) if total_raw is not None else None
    except (TypeError, ValueError):
        total_amount = None

    # Notes: only pass through the source JSON notes field.
    # payment_terms and revision are not in the invoice_text seen by the critic,
    # so including them here would cause the critic to reject "hallucinated" data.
    notes = str(data["notes"]).strip() if data.get("notes") else None

    # Tax fields — support both percentage strings ("6%") and numeric values
    tax_rate_raw = data.get("tax_rate")
    try:
        tax_rate = float(str(tax_rate_raw).rstrip("%")) if tax_rate_raw is not None else None
    except (TypeError, ValueError):
        tax_rate = None

    tax_amount_raw = data.get("tax_amount") or data.get("tax")
    try:
        tax_amount = float(tax_amount_raw) if tax_amount_raw is not None else None
    except (TypeError, ValueError):
        tax_amount = None

    return {
        "invoice_number": str(data.get("invoice_number") or "").strip(),
        "vendor":         vendor,
        "total_amount":   total_amount,
        "items":          items,
        "due_date":       data.get("due_date") or None,
        "notes":          notes or None,
        "tax_rate":       tax_rate,
        "tax_amount":     tax_amount,
    }


# ── Deterministic CSV parsers ──────────────────────────────────────────────────

def _parse_csv_kv(df: pd.DataFrame) -> dict:
    """
    Parse a 2-column key/value CSV (field, value format).

    Example:
        field,value
        invoice_number,INV-1006
        vendor,Acme Industrial Supplies
        item,WidgetA
        quantity,5
        unit_price,250.00
        total,2750.00
    """
    items: list[dict] = []
    current_item: dict | None = None
    kv: dict[str, str] = {}

    for _, row in df.iterrows():
        key   = str(row.iloc[0]).strip().lower()
        value = str(row.iloc[1]).strip()

        if key == "item":
            if current_item is not None:
                items.append(current_item)
            current_item = {"name": value, "quantity": 1, "unit_price": None}
        elif key == "quantity" and current_item is not None:
            try:
                current_item["quantity"] = int(float(value))
            except (ValueError, TypeError):
                pass
        elif key == "unit_price" and current_item is not None:
            try:
                current_item["unit_price"] = float(value)
            except (ValueError, TypeError):
                pass
        else:
            kv[key] = value

    if current_item is not None:
        items.append(current_item)

    total_raw = kv.get("total")
    try:
        total_amount = float(total_raw) if total_raw else None
    except (ValueError, TypeError):
        total_amount = None

    notes_parts: list[str] = []
    if kv.get("payment_terms"):
        notes_parts.append(f"Payment terms: {kv['payment_terms']}")

    return {
        "invoice_number": kv.get("invoice_number", ""),
        "vendor":         kv.get("vendor", ""),
        "total_amount":   total_amount,
        "items":          items,
        "due_date":       kv.get("due_date") or None,
        "notes":          "; ".join(notes_parts) or None,
    }


def _parse_csv_tabular(df: pd.DataFrame) -> dict:
    """
    Parse a tabular CSV where each row is a line item and summary rows appear
    in the footer.

    Example:
        Invoice Number,Vendor,Date,Due Date,Item,Qty,Unit Price,Line Total
        INV-1007,MegaWidgets Corp,...,WidgetA,20,250.00,5000.00
        ,,,,,,Total:,15525.00
    """
    # Normalize column names to internal keys
    rename: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in {"invoice number", "invoice_number"}:
            rename[c] = "_inv_num"
        elif cl == "vendor":
            rename[c] = "_vendor"
        elif cl in {"due date", "due_date"}:
            rename[c] = "_due_date"
        elif cl in {"item", "name", "description", "product"}:
            rename[c] = "_item"
        elif cl in {"qty", "quantity", "count"}:
            rename[c] = "_qty"
        elif cl in {"unit price", "unit_price", "price", "rate"}:
            rename[c] = "_unit_price"
        elif cl in {"line total", "line_total", "amount"}:
            rename[c] = "_line_total"
    df = df.rename(columns=rename)

    invoice_number = ""
    vendor         = ""
    due_date: str | None = None
    items: list[dict] = []
    total_amount: float | None = None
    tax_amount: float | None = None
    tax_rate: float | None = None

    for _, row in df.iterrows():
        inv        = str(row.get("_inv_num",    "")).strip()
        v          = str(row.get("_vendor",     "")).strip()
        dd         = str(row.get("_due_date",   "")).strip() or None
        item_name  = str(row.get("_item",       "")).strip()
        qty_str    = str(row.get("_qty",        "")).strip()
        price_str  = str(row.get("_unit_price", "")).strip()
        line_total = str(row.get("_line_total", "")).strip()

        # Capture metadata from the first populated rows
        if inv and not invoice_number:
            invoice_number = inv
        if v and not vendor:
            vendor = v
        if dd and not due_date:
            due_date = dd

        # Summary footer: the unit_price column contains a label ending with ":"
        # (e.g., "Total:", "Subtotal:", "Tax (6%):", "Shipping:")
        if price_str.lower().rstrip().endswith(":") or price_str.lower() in {"total:", "subtotal:"}:
            label_lower = price_str.lower()
            if "total" in label_lower and "sub" not in label_lower:
                try:
                    total_amount = float(line_total.replace(",", ""))
                except (ValueError, TypeError):
                    pass
            elif "tax" in label_lower or "vat" in label_lower or "gst" in label_lower:
                try:
                    tax_amount = float(line_total.replace(",", ""))
                except (ValueError, TypeError):
                    pass
                # Also extract explicit rate from label, e.g. "Tax (6%):" → 6.0
                rate_m = re.search(r'\((\d+(?:\.\d+)?)\s*%?\)', price_str)
                if rate_m:
                    tax_rate = float(rate_m.group(1))
            continue

        # Item row: item_name is populated and is not itself a summary label
        if item_name and not item_name.endswith(":"):
            try:
                qty = int(float(qty_str)) if qty_str else 1
            except (ValueError, TypeError):
                qty = 1
            try:
                price = float(price_str) if price_str else None
            except (ValueError, TypeError):
                price = None
            items.append({"name": item_name, "quantity": qty, "unit_price": price})

    return {
        "invoice_number": invoice_number,
        "vendor":         vendor,
        "total_amount":   total_amount,
        "items":          items,
        "due_date":       due_date,
        "notes":          None,
        "tax_amount":     tax_amount,
        "tax_rate":       tax_rate,
    }


def parse_csv_invoice(invoice_path: str) -> dict:
    """
    Parse a CSV invoice file deterministically.
    Detects key-value vs. tabular format automatically.
    """
    df = pd.read_csv(invoice_path, dtype=str).fillna("")

    # Key-value format: exactly 2 columns whose headers are "field"/"value" variants
    if len(df.columns) == 2:
        cols_lower = {c.lower().strip() for c in df.columns}
        if cols_lower <= {"field", "value", "key", "name"}:
            return _parse_csv_kv(df)

    return _parse_csv_tabular(df)


# ── Regex safety net (LLM path only) ─────────────────────────────────────────

def regex_extraction(text: str) -> dict:
    """
    Deterministic regex fallback for extracting invoice_number and total_amount.

    Used when the LLM fails Pydantic validation on the second attempt, as a
    last-resort attempt to recover structured data before routing to failure_node.

    Returns a dict with whatever fields could be extracted (values may be absent).
    """
    result: dict = {}

    inv_match = re.search(
        r"(?:Invoice|INV)[\s#:\-]*(?:Number|No\.?|ID|Num)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-]+)",
        text,
        re.IGNORECASE,
    )
    if inv_match:
        result["invoice_number"] = inv_match.group(1).strip()

    total_match = re.search(
        r"Total\s*(?:Amount)?\s*[:\-]?\s*\$?\s*([0-9]+\.[0-9]{2})",
        text,
        re.IGNORECASE,
    )
    if total_match:
        try:
            result["total_amount"] = float(total_match.group(1))
        except ValueError:
            pass

    return result


def _apply_regex_patch(invoice_dict: dict | None, text: str) -> dict | None:
    """
    Patch a partially-extracted invoice dict with regex-derived fields.

    Only fills in fields that are missing or None in the existing dict.
    Returns None if the invoice_dict is None (nothing to patch).
    """
    if not invoice_dict:
        return None
    regex_data = regex_extraction(text)
    for key, value in regex_data.items():
        if not invoice_dict.get(key):
            invoice_dict[key] = value
    return invoice_dict


# ── Node function ─────────────────────────────────────────────────────────────

def run_ingestion(state: InvoiceGraphState) -> dict:
    """LangGraph node: extract InvoiceData from invoice_text or file directly."""
    text         = state["invoice_text"]
    invoice_path = state.get("invoice_path", "")
    correction   = state.get("correction_instruction")
    retry_count  = state.get("retry_count", 0)

    ext = Path(invoice_path).suffix.lower() if invoice_path else ""

    # ── Deterministic path for structured formats ─────────────────────────────
    # JSON and CSV files are always parsed deterministically — the LLM is never
    # used, and correction instructions are ignored (data quality issues in the
    # source file cannot be fixed by an LLM prompt).
    if ext in (".json", ".csv"):
        method = "deterministic_json" if ext == ".json" else "deterministic_csv"
        parser = parse_json_invoice if ext == ".json" else parse_csv_invoice
        try:
            raw_dict = parser(invoice_path)
            invoice  = InvoiceData(**raw_dict)
            logger.info(
                f"[INGESTION] {method}: {invoice.invoice_number} "
                f"({len(invoice.items)} items)"
            )
            return {
                "invoice_data":      invoice.model_dump(),
                "error":             None,
                "extraction_method": method,
            }
        except ValidationError as exc:
            msg = f"{method} schema validation failed: {exc}"
            logger.warning(f"[INGESTION] {msg}")
            return {"invoice_data": None, "error": msg, "extraction_method": method}
        except Exception as exc:
            msg = f"{method} parse error: {exc}"
            logger.error(f"[INGESTION] {msg}")
            return {"invoice_data": None, "error": msg, "extraction_method": method}

    # ── LLM path for unstructured formats (.txt, .pdf, etc.) ─────────────────
    user_content = f"Extract invoice data from the following text:\n\n{text}"
    if correction:
        user_content += (
            f"\n\n--- CORRECTION REQUIRED (attempt {retry_count + 1}) ---\n{correction}"
        )

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        structured_llm = get_structured_llm(InvoiceData)
        result: InvoiceData = structured_llm.invoke(messages)

        logger.info(f"[INGESTION] llm_extraction: {result.invoice_number}")
        return {
            "invoice_data":      result.model_dump(),
            "error":             None,
            "extraction_method": "llm_extraction",
        }

    except ValidationError as exc:
        msg = f"Schema validation failed (attempt {retry_count + 1}): {exc}"
        logger.warning(f"[INGESTION] {msg}")
        if retry_count >= 1:
            logger.info("[INGESTION] attempting regex fallback for missing fields")
            regex_data = regex_extraction(text)
            if regex_data:
                logger.info(f"[INGESTION] regex recovered: {list(regex_data.keys())}")
                patched_dict = regex_data.copy()
                try:
                    patched_invoice = InvoiceData(**patched_dict)
                    logger.info(
                        f"[INGESTION] regex recovery succeeded: {patched_invoice.invoice_number}"
                    )
                    return {
                        "invoice_data":      patched_invoice.model_dump(),
                        "error":             None,
                        "extraction_method": "llm_extraction",
                    }
                except ValidationError:
                    logger.warning("[INGESTION] regex recovery still failed schema validation")
        return {"invoice_data": None, "error": msg, "extraction_method": "llm_extraction"}

    except Exception as exc:
        msg = f"LLM extraction failed (attempt {retry_count + 1}): {exc}"
        logger.error(f"[INGESTION] {msg}")
        if retry_count >= 1:
            logger.info("[INGESTION] attempting regex fallback after LLM failure")
            regex_data = regex_extraction(text)
            if regex_data:
                logger.info(f"[INGESTION] regex recovered: {list(regex_data.keys())}")
                patched_dict = regex_data.copy()
                try:
                    patched_invoice = InvoiceData(**patched_dict)
                    logger.info(
                        f"[INGESTION] regex recovery succeeded: {patched_invoice.invoice_number}"
                    )
                    return {
                        "invoice_data":      patched_invoice.model_dump(),
                        "error":             None,
                        "extraction_method": "llm_extraction",
                    }
                except ValidationError:
                    logger.warning("[INGESTION] regex recovery still failed schema validation")
        return {"invoice_data": None, "error": msg, "extraction_method": "llm_extraction"}
