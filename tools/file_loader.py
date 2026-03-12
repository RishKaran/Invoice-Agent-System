"""
Universal Invoice Loader — normalizes all supported formats into plain text
before handing off to the ingestion agent.

Supported: txt, pdf, json, csv, xml, xlsx
"""

from __future__ import annotations

import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_invoice_content(path: str) -> str:
    """Parse an invoice file and return normalized human-readable text."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Invoice file not found: {path}")

    ext = p.suffix.lower().lstrip(".")
    parsers = {
        "txt":  _parse_txt,
        "pdf":  _parse_pdf,
        "json": _parse_json,
        "csv":  _parse_csv,
        "xml":  _parse_xml,
        "xlsx": _parse_xlsx,
        "xls":  _parse_xlsx,
    }

    if ext not in parsers:
        raise ValueError(f"Unsupported file format: '.{ext}'. Supported: {sorted(parsers)}")

    try:
        text = parsers[ext](p)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse '{p.name}': {exc}") from exc

    if not isinstance(text, str):
        raise TypeError(f"Parser for '{ext}' must return str, got {type(text)}")

    # Inject a fallback invoice number derived from filename when absent.
    if "invoice number" not in text.lower():
        fallback = _invoice_number_from_filename(p.stem)
        if fallback:
            text = f"Invoice Number: {fallback}\n" + text

    return text.strip()


# ---------------------------------------------------------------------------
# Format parsers
# ---------------------------------------------------------------------------

def _parse_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _parse_pdf(p: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required for PDF support: pip install pdfplumber") from exc

    pages: list[str] = []
    with pdfplumber.open(str(p)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

    if not pages:
        raise ValueError("PDF contained no extractable text.")
    return "\n".join(pages)


def _parse_json(p: Path) -> str:
    data: Any = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        # Treat list as a collection of line items
        return _normalize_dict({"items": data})
    if isinstance(data, dict):
        return _normalize_dict(data)
    raise ValueError("JSON invoice must be an object or array.")


def _parse_csv(p: Path) -> str:
    rows: list[dict] = []
    with p.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            rows = list(reader)

    # Key-value format: two columns (key, value) with no obvious header
    if rows and len(rows[0]) == 2:
        keys = [k.lower().strip() for k in rows[0].keys()]
        if set(keys) <= {"key", "value", "field", "name"}:
            kv = {r[list(r.keys())[0]]: r[list(r.keys())[1]] for r in rows}
            return _normalize_dict(kv)

    # Standard tabular format: each row is a line item
    header = list(rows[0].keys()) if rows else []
    lines = [f"Items:"]
    meta: dict[str, str] = {}
    item_rows: list[str] = []
    summary_rows: list[str] = []

    ITEM_KEYS = {"item", "name", "description", "product"}
    QTY_KEYS  = {"quantity", "qty", "count"}
    PRICE_KEYS= {"unit_price", "price", "unit price", "rate"}
    META_KEYS = {"invoice_number", "invoice number", "vendor", "total_amount",
                 "total amount", "due_date", "due date"}
    _CHARGE_KEYWORDS = ("tax", "vat", "gst", "hst", "shipping", "freight",
                        "delivery", "handling", "fee", "surcharge")

    for row in rows:
        row_lower = {k.lower().strip(): v for k, v in row.items()}
        item_name = _first(row_lower, ITEM_KEYS)
        qty       = _first(row_lower, QTY_KEYS)
        price     = _first(row_lower, PRICE_KEYS)
        for key in META_KEYS:
            if key in row_lower and row_lower[key]:
                meta[key] = row_lower[key]
        if item_name:
            parts = [str(item_name)]
            if qty:
                parts.append(f"quantity {qty}")
            if price:
                parts.append(f"unit_price {price}")
            item_rows.append("  " + " ".join(parts))
        else:
            # Footer summary row: look for a charge label + amount pair
            vals = [v.strip() for v in row.values() if v.strip()]
            if len(vals) >= 2:
                label, amount = vals[-2], vals[-1]
                if label.rstrip().endswith(":") and any(kw in label.lower() for kw in _CHARGE_KEYWORDS):
                    summary_rows.append(f"{label.rstrip(': ').strip()}: {amount}")

    out_parts: list[str] = []
    if "invoice_number" in meta or "invoice number" in meta:
        out_parts.append(f"Invoice Number: {meta.get('invoice_number') or meta.get('invoice number')}")
    if "vendor" in meta:
        out_parts.append(f"Vendor: {meta['vendor']}")
    if item_rows:
        out_parts.append("Items:")
        out_parts.extend(item_rows)
    out_parts.extend(summary_rows)
    if "total_amount" in meta or "total amount" in meta:
        out_parts.append(f"Total Amount: {meta.get('total_amount') or meta.get('total amount')}")
    if "due_date" in meta or "due date" in meta:
        out_parts.append(f"Due Date: {meta.get('due_date') or meta.get('due date')}")

    return "\n".join(out_parts) if out_parts else "\n".join(
        [",".join(header)] + [",".join(r.values()) for r in rows]
    )


def _parse_xml(p: Path) -> str:
    tree = ET.parse(str(p))
    root = tree.getroot()

    def _find(tag: str) -> str | None:
        el = root.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else None

    lines: list[str] = []

    inv_num = _find("invoice_number") or _find("invoiceNumber") or _find("InvoiceNumber")
    if inv_num:
        lines.append(f"Invoice Number: {inv_num}")

    vendor = _find("vendor") or _find("vendor_name") or _find("Vendor")
    if vendor:
        lines.append(f"Vendor: {vendor}")

    # Items — look for <item>, <lineItem>, <line_item> containers
    item_containers = root.findall(".//item") or root.findall(".//lineItem") or root.findall(".//line_item")
    if item_containers:
        lines.append("Items:")
        for item in item_containers:
            name  = (item.findtext("name") or item.findtext("description") or "").strip()
            qty   = (item.findtext("quantity") or item.findtext("qty") or "").strip()
            price = (item.findtext("unit_price") or item.findtext("price") or "").strip()
            parts = [name]
            if qty:
                parts.append(f"quantity {qty}")
            if price:
                parts.append(f"unit_price {price}")
            lines.append("  " + " ".join(parts))

    due = _find("due_date") or _find("dueDate")
    if due:
        lines.append(f"Due Date: {due}")

    # Include tax so the validation agent can verify math
    tax_amount = _find("tax_amount") or _find("taxAmount")
    if tax_amount:
        lines.append(f"Tax: {tax_amount}")
    else:
        tax_rate = _find("tax_rate") or _find("taxRate")
        if tax_rate:
            lines.append(f"Tax Rate: {tax_rate}")

    total = _find("total_amount") or _find("totalAmount") or _find("total")
    if total:
        lines.append(f"Total Amount: {total}")

    notes = _find("notes") or _find("note")
    if notes:
        lines.append(f"Notes: {notes}")

    return "\n".join(lines) if lines else ET.tostring(root, encoding="unicode")


def _parse_xlsx(p: Path) -> str:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and openpyxl are required for XLSX support: pip install pandas openpyxl") from exc

    df = pd.read_excel(str(p), dtype=str)
    df = df.fillna("")

    cols_lower = {c.lower().strip(): c for c in df.columns}
    item_col  = _first(cols_lower, {"item", "name", "description", "product"})
    qty_col   = _first(cols_lower, {"quantity", "qty", "count"})
    price_col = _first(cols_lower, {"unit_price", "price", "unit price", "rate"})

    lines: list[str] = ["Items:"]
    for _, row in df.iterrows():
        name  = str(row[item_col]).strip()  if item_col  else ""
        qty   = str(row[qty_col]).strip()   if qty_col   else ""
        price = str(row[price_col]).strip() if price_col else ""
        if not name:
            continue
        parts = [name]
        if qty:
            parts.append(f"quantity {qty}")
        if price:
            parts.append(f"unit_price {price}")
        lines.append("  " + " ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_dict(data: dict) -> str:
    """Convert a parsed JSON/dict invoice to normalized text."""
    FIELD_MAP = {
        "invoice_number": "Invoice Number",
        "invoicenumber":  "Invoice Number",
        "vendor":         "Vendor",
        "vendor_name":    "Vendor",
        "total_amount":   "Total Amount",
        "totalamount":    "Total Amount",
        "total":          "Total Amount",
        "due_date":       "Due Date",
        "duedate":        "Due Date",
        "notes":          "Notes",
        "note":           "Notes",
        "subtotal":       "Subtotal",
        "tax_amount":     "Tax Amount",
        "taxamount":      "Tax Amount",
        "tax_rate":       "Tax Rate",
        "taxrate":        "Tax Rate",
    }
    ITEM_KEYS = {"items", "lineitems", "line_items", "products"}

    lines: list[str] = []
    items_data = None

    for raw_key, value in data.items():
        key = raw_key.lower().replace(" ", "").replace("-", "")
        if key in ITEM_KEYS:
            items_data = value
            continue
        label = FIELD_MAP.get(key)
        if label and value:
            lines.append(f"{label}: {value}")

    if items_data and isinstance(items_data, list):
        lines.append("Items:")
        for item in items_data:
            if isinstance(item, dict):
                item_lower = {k.lower().replace(" ", "_"): v for k, v in item.items()}
                name  = item_lower.get("name") or item_lower.get("description") or item_lower.get("item") or ""
                qty   = item_lower.get("quantity") or item_lower.get("qty") or ""
                price = item_lower.get("unit_price") or item_lower.get("price") or ""
                parts = [str(name)]
                if qty:
                    parts.append(f"quantity {qty}")
                if price:
                    parts.append(f"unit_price {price}")
                lines.append("  " + " ".join(parts))
            else:
                lines.append(f"  {item}")

    return "\n".join(lines)


def _first(mapping: dict, keys: set) -> Any:
    """Return the value of the first matching key from a set of candidates."""
    for k in keys:
        if k in mapping:
            return mapping[k]
    return None


def _invoice_number_from_filename(stem: str) -> str | None:
    """Extract a numeric ID from filename and return as INV-XXXX."""
    match = re.search(r"(\d+)", stem)
    if match:
        return f"INV-{match.group(1)}"
    return None
