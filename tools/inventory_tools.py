"""
Inventory Tool — read-only lookup against the inventory catalog.

Source priority:
  1. database/inventory.db  (SQLite — authoritative production source)
  2. data/inventory.json    (JSON fallback for development / offline use)

SQLite is queried first on every call. The JSON fallback is only used when
the database file is absent or the inventory table cannot be read.

LLMs must never call SQLite directly; they use get_inventory_item() as a tool.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

# Resolve paths relative to project root so this works regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH      = _PROJECT_ROOT / "database" / "inventory.db"
_CATALOG_PATH = _PROJECT_ROOT / "data" / "inventory.json"


def _load_json_catalog() -> dict[str, tuple[str, int]] | None:
    """
    Load data/inventory.json if it exists.
    Returns {normalized_name: (canonical_name, stock)}, or None if absent/invalid.
    Used only when the SQLite database is unavailable.
    """
    if not _CATALOG_PATH.exists():
        return None
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        catalog: dict[str, tuple[str, int]] = {}
        for entry in data.get("items", []):
            canonical = entry["item"]
            catalog[normalize_item_name(canonical)] = (canonical, int(entry["stock"]))
        return catalog
    except Exception:
        return None


# Module-level JSON cache — loaded at most once per process, only on fallback.
_JSON_CATALOG: dict[str, tuple[str, int]] | None = None
_CATALOG_LOADED = False


def normalize_item_name(name: str) -> str:
    """
    Canonical item name normalization for inventory matching.

    Rules (applied in order):
      1. Strip leading/trailing whitespace
      2. Remove parenthetical modifiers: "(rush order)", "(expedited)", "(5%)"
      3. Remove known order modifier words: rush order, rush, expedited, priority,
         urgent, sample, replacement
      4. Remove all remaining non-alphanumeric characters (spaces, hyphens, etc.)
      5. Convert to lowercase

    Examples:
      "Widget A"              -> "widgeta"
      "Widget-A"              -> "widgeta"
      "WidgetA"               -> "widgeta"
      "widget a"              -> "widgeta"
      "Gadget X"              -> "gadgetx"
      "Widget A (rush order)" -> "widgeta"
      "WidgetA - rush"        -> "widgeta"
      "Widget A rush order"   -> "widgeta"
      "Widget A (expedited)"  -> "widgeta"
    """
    name = name.strip()
    # Expand camelCase so "WidgetARushOrder" → "Widget A Rush Order"
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', name)
    # Remove parenthetical content: "(rush order)", "(expedited)", "(5%)", etc.
    name = re.sub(r"\([^)]*\)", "", name)
    # Remove known modifier words (whole-word match; longer phrases before shorter)
    name = re.sub(
        r"\b(rush\s+order|rush|expedited|priority|urgent|sample|replacement)\b",
        "",
        name,
        flags=re.IGNORECASE,
    )
    # Strip everything non-alphanumeric (spaces, hyphens, stray punctuation)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def get_inventory_item(item_name: str) -> dict:
    """
    Look up an item in the inventory using normalized name matching.

    Source priority:
      1. database/inventory.db  (SQLite — always queried first)
      2. data/inventory.json    (fallback when SQLite is absent or unreadable)

    Compares normalized names so that formatting differences ("Widget A" vs
    "WidgetA" vs "widget-a") and order modifiers ("Widget A (rush order)")
    never produce a false "not found" result.

    Returns:
        {"item": str, "stock": int | None}
        stock is None when no normalized match exists in either source.
    """
    global _JSON_CATALOG, _CATALOG_LOADED

    normalized_input = normalize_item_name(item_name)

    # ── SQLite primary ────────────────────────────────────────────────────────
    if _DB_PATH.exists():
        try:
            with sqlite3.connect(str(_DB_PATH)) as conn:
                rows = conn.execute("SELECT item, stock FROM inventory").fetchall()
            for row in rows:
                if normalize_item_name(row[0]) == normalized_input:
                    return {"item": row[0], "stock": row[1]}
            # Database reachable but item not in inventory table.
            return {"item": item_name, "stock": None}
        except sqlite3.Error:
            pass  # Table missing or unreadable — fall through to JSON fallback.

    # ── JSON fallback (development / offline) ─────────────────────────────────
    if not _CATALOG_LOADED:
        _JSON_CATALOG = _load_json_catalog()
        _CATALOG_LOADED = True

    if _JSON_CATALOG is not None:
        if normalized_input in _JSON_CATALOG:
            canonical, stock = _JSON_CATALOG[normalized_input]
            return {"item": canonical, "stock": stock}

    return {"item": item_name, "stock": None}
