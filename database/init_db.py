"""
Database initializer — run once to create and seed inventory.db.
Usage: python database/init_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent / "inventory.db"


def init_database() -> None:
    with sqlite3.connect(str(_DB_PATH)) as conn:
        # ── inventory ────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                item  TEXT PRIMARY KEY,
                stock INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.executemany(
            "INSERT OR REPLACE INTO inventory (item, stock) VALUES (?, ?)",
            [
                ("WidgetA", 15),
                ("WidgetB", 10),
                ("GadgetX", 5),
                ("FakeItem", 0),
            ],
        )

        # ── processed_invoices ────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_invoices (
                content_hash TEXT PRIMARY KEY,
                invoice_number TEXT,
                vendor TEXT,
                status TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT
            )
        """)

        conn.commit()

    print(f"Database initialized: {_DB_PATH}")


if __name__ == "__main__":
    init_database()
