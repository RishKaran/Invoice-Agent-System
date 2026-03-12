from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Tolerance for schema-level math check ($0.05 — tighter than validation agent's $0.10)
_MATH_TOLERANCE = 0.05

DATE_FORMATS = [
    "%Y-%m-%d",    # 2026-02-25
    "%d/%m/%Y",    # 25/02/2026
    "%m/%d/%Y",    # 02/25/2026
    "%d-%b-%Y",    # 25-Feb-2026
    "%d-%B-%Y",    # 25-February-2026
    "%b %d %Y",    # Feb 25 2026
    "%B %d %Y",    # February 25 2026
    "%d-%m-%Y",    # 25-02-2026
    "%b %d, %Y",   # Feb 25, 2026
    "%B %d, %Y",   # February 25, 2026
    "%d %b %Y",    # 25 Feb 2026
    "%d %B %Y",    # 25 February 2026
]


def parse_date(date_str: str) -> str:
    """
    Parse a date string using DATE_FORMATS and return an ISO-format date string.
    Raises ValueError with a helpful message if no format matches.
    """
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse due_date: '{date_str}'. Supported formats: {DATE_FORMATS}"
    )


class InvoiceItem(BaseModel):
    name: str
    quantity: int  # no gt=0 constraint — validation_agent checks for invalid quantities
    unit_price: Optional[float] = None


class InvoiceData(BaseModel):
    invoice_number: str
    vendor: str
    total_amount: Optional[float] = None  # no gt=0 — validation_agent checks sign
    items: list[InvoiceItem] = Field(min_length=1)
    due_date: Optional[str] = None
    notes: Optional[str] = None
    # Tax fields — populated by deterministic parsers; used for math validation
    tax_rate: Optional[float] = Field(default=None, ge=0)    # percentage or decimal (6.0 or 0.06)
    tax_amount: Optional[float] = Field(default=None, ge=0)  # explicit dollar amount

    @field_validator("due_date", mode="before")
    @classmethod
    def normalize_due_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            return parse_date(v)
        except ValueError:
            return v  # keep raw string; validation_agent flags as invalid_due_date
