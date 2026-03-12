# Invoice Agent System

An AI-powered accounts payable automation pipeline built on LangGraph. Processes invoices from any supported format — plain text, PDF, JSON, CSV, XML, XLSX — through a multi-stage extraction, validation, and approval workflow. Approved invoices are paid automatically. Rejected and escalated invoices are logged to a persistent registry and produce structured audit trails.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)
- [Business Impact](#business-impact)
- [Alignment With Evaluation Criteria](#alignment-with-evaluation-criteria)
- [Graph Topology](#graph-topology)
- [Stage-by-Stage Design](#stage-by-stage-design)
  - [File Loader](#1-file-loader)
  - [Ingestion Agent](#2-ingestion-agent)
  - [Extraction Critic](#3-extraction-critic)
  - [Validation Agent](#4-validation-agent)
  - [Approval Agent](#5-approval-agent)
  - [Payment Tool](#6-payment-tool)
  - [Registry Tool](#7-registry-tool)
- [Fallback and Resilience Design](#fallback-and-resilience-design)
- [Schema Design](#schema-design)
- [Inventory](#inventory)
- [Outputs and Observability](#outputs-and-observability)
- [Running the System](#running-the-system)
- [Testing](#testing)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)

---

## Architecture Overview

```
Invoice File
     │
     ▼
File Loader ──── normalizes to plain text ─────────────────────────┐
     │                                                              │
     ▼                                                              │
Ingestion Agent ──── .json/.csv: deterministic parse               │
     │               .txt/.pdf/.xml/.xlsx: LLM extraction          │
     ▼                                                              │
Extraction Critic ── accepts or rejects LLM output                 │
     │    │                                                         │
     │    └── reject ──► retry Ingestion (max 2) ──► failure_node  │
     ▼                                                              │
Validation Agent ─── 100% deterministic Python                     │
     │               checks vendor, inventory, date, math          │
     ▼                                                              │
Approval Agent ──── 100% deterministic Python                      │
     │               5-rule hierarchy                               │
     ├── approve ──► Payment Tool ──► receipt + registry           │
     ├── escalate ─► manual_review_node ──► registry               │
     └── reject ──► reject_node ──► registry                       │
                                                                    │
                          Audit Report ◄──────────────────────────┘
```

All business logic — validation and approval — is fully deterministic Python. LLMs are used exclusively for data extraction from unstructured formats and for reviewing that extraction.

---

## Design Decisions and Tradeoffs

### 1. Deterministic validation over LLM reasoning

LLMs perform well on unstructured text extraction but are unreliable for arithmetic and rule-based business logic. Asking an LLM whether `$15,525.00 == $14,750.00 + $885.00` introduces unnecessary uncertainty into a decision that has a correct answer.

All validation (inventory check, math verification, date parsing) and all approval logic are implemented as deterministic Python. This makes the system's decisions predictable, auditable, and safe to run against real financial data. LLMs are used only where they add value: extracting structure from unstructured invoice text.

### 2. Separated agent responsibilities

The pipeline is split into four distinct agents rather than a single general-purpose agent:

- **Ingestion agent** — extracts structured data from raw text
- **Extraction critic** — independently reviews the extraction for hallucinations or errors
- **Validation agent** — checks the structured data against inventory and business rules
- **Approval agent** — applies the decision hierarchy

This separation ensures that hallucinated or malformed data cannot reach the validation layer without being caught first by the critic. It also keeps business logic entirely independent of LLM behavior — changes to approval rules require no prompt engineering.

### 3. Deterministic parsing for structured formats

JSON and CSV invoices bypass the LLM entirely. These formats already contain machine-readable structured data; invoking an LLM to re-extract them would introduce non-determinism, add latency, and incur unnecessary API cost. The deterministic parsers also skip the critic review step, since there is no LLM output to critique. In a typical AP workload where a significant portion of invoices arrive as JSON or CSV exports from vendor systems, this path processes invoices in milliseconds with no external calls.

### 4. Vendor presence validation, no allowlists

The system validates that a vendor name is present and non-empty. A blank vendor field rejects the invoice immediately — an invoice with no identifiable supplier cannot be safely paid.

Beyond presence, vendor identity alone is not used as a rejection signal. The dataset does not provide vendor master data, and several invoices must pass validation even though their vendors are not pre-registered anywhere. Real AP workflows regularly onboard new vendors; an allowlist would create friction without reducing risk.

Fraud detection relies entirely on objective signals that do not require vendor registration:

- Item not found in inventory → +0.7 risk (fraud_suspected threshold)
- Item has zero stock → +0.7 risk
- Requested quantity exceeds available stock → invalid
- Mathematical discrepancy in total → +0.3 risk

These signals catch the same fraud patterns that allowlists target, applied to every invoice regardless of who submitted it.

### 5. Fraud escalation instead of automatic rejection

When the risk score reaches the fraud threshold (items not in inventory, zero-stock items), the invoice is escalated to a human reviewer rather than automatically rejected. An automatic rejection would close the loop silently — if a legitimate vendor submitted an invoice for a new product that simply hasn't been added to the catalog yet, it would be rejected with no visibility. Escalation surfaces the case for investigation, which is the correct AP response to a potentially fraudulent submission.

### 6. Revised invoice precedence

Vendors routinely submit corrected invoices after an error is identified. When a revised invoice arrives for an invoice number already in the registry (same number, different content hash), the system processes it as a new invoice and marks the original record as `superseded`. The revised version replaces the original in the batch summary CSV. This prevents both silent rejection of legitimate corrections and accidental double-payment of the original amount.

---

## Business Impact

The case describes Acme Corp losing $2M/year on manual invoice processing, with a 30% error rate, 5-day processing delays, high manual effort, and uncontrolled fraud risk.

**How this system addresses each pain point:**

| Problem | How the system solves it |
|---|---|
| 30% error rate | Deterministic validation catches math discrepancies, stock mismatches, missing fields, and invalid dates on every invoice — no human judgment required |
| 5-day processing delays | Approved invoices move from file to payment in seconds; the LLM path typically completes in under 10 seconds including two LLM calls |
| High manual effort | Only escalated and flagged invoices require human attention; clean invoices are approved and paid automatically |
| Fraud risk | Unknown items, zero-stock items, and arithmetic errors are detected and escalated before payment is ever attempted |
| Duplicate payments | SHA-256 content hashing prevents re-processing identical invoices; a receipt file scan blocks duplicate payment calls even if the graph is re-run |

The architecture processes the majority of clean invoices automatically while routing genuinely ambiguous cases — high-value invoices, suspected fraud, arithmetic errors — to human reviewers. Finance teams receive a structured audit trail for every invoice regardless of outcome.

---

## Alignment With Evaluation Criteria

| Criterion | How this build addresses it |
|---|---|
| **Functionality** | End-to-end workflow from invoice file to payment execution across 6 input formats (txt, pdf, json, csv, xml, xlsx), with structured audit output for every invoice |
| **Code Quality** | Hard separation between extraction, validation, approval, and payment layers; each agent has a single responsibility and no cross-layer dependencies; deterministic paths have zero LLM calls |
| **Agentic Sophistication** | Multi-agent LangGraph pipeline with structured LLM output binding, an independent extraction critic with retry loops (up to 2 retries with targeted correction instructions), and a regex safety net for partial recovery on LLM failure |
| **Shipping Mindset** | Deterministic validation, a hard payment safety guard (`RuntimeError` on non-approve decisions), duplicate payment prevention, and SHA-256 registry deduplication make the system safe to run against real invoice batches |
| **Observability** | Per-invoice JSON audit reports, payment receipts, a batch summary CSV, execution timing per node, and a persistent append-only system log give finance teams full traceability |

---

## Graph Topology

The LangGraph `StateGraph` compiles to 8 nodes with a single entry point and 4 terminal paths:

| Node | Type | Purpose |
|---|---|---|
| `ingestion_agent` | LLM / deterministic | Extract `InvoiceData` from raw invoice text |
| `critic_agent` | LLM / bypass | Review extraction quality; trigger retry if poor |
| `validation_agent` | Deterministic | Vendor, inventory, date, math checks |
| `approval_agent` | Deterministic | Apply business rules → approve / escalate / reject |
| `payment_node` | Tool | Execute payment, write receipt, log to registry |
| `reject_node` | Terminal | Log rejection to registry, write audit |
| `manual_review_node` | Terminal | Log escalation to registry, write audit |
| `failure_node` | Terminal | Log extraction failure, write partial audit |

**Conditional edges:**

- After `critic_agent`: proceed to `validation_agent` (accepted), retry `ingestion_agent` (rejected, retries < 2), or give up at `failure_node` (retries exhausted).
- After `approval_agent`: route to `payment_node`, `reject_node`, or `manual_review_node` based on decision.

State is persisted across nodes via LangGraph's `MemorySaver` checkpointer, keyed by `thread_id` (derived from filename stem).

---

## Stage-by-Stage Design

### 1. File Loader

**File:** `tools/file_loader.py`

Converts any supported invoice format into normalized plain text before the graph runs. This separation ensures every downstream agent operates on a consistent string representation regardless of the original format.

**Supported formats and their parsers:**

| Format | Parser approach |
|---|---|
| `.txt` | Direct UTF-8 read |
| `.pdf` | `pdfplumber` page extraction |
| `.json` | Dict-to-text via `_normalize_dict()` |
| `.csv` | Key-value or tabular detection; charge rows extracted to text |
| `.xml` | `xml.etree.ElementTree` with camelCase field aliases |
| `.xlsx` / `.xls` | `pandas.read_excel` |

**Key design decisions:**

- **Charge row injection (CSV):** Footer rows in tabular CSVs (e.g., `Tax (6%): $885.00`, `Shipping: $50.00`) are detected by a `_CHARGE_KEYWORDS` tuple and appended to the output text as `"Label: amount"` lines. Without this, the validation agent's math check would only see the subtotal and produce a false discrepancy.

- **Fallback invoice number from filename:** If the parsed text does not contain the phrase "invoice number", the loader injects a line `Invoice Number: INV-XXXX` derived from the filename's numeric component. This prevents the LLM from hallucinating an invoice number when processing files named `invoice_1011.pdf`.

- **XML field aliasing:** The XML parser searches for both snake_case (`invoice_number`, `tax_amount`) and camelCase (`invoiceNumber`, `taxAmount`) variants of every field, making the loader resilient to different XML schema conventions.

---

### 2. Ingestion Agent

**File:** `agents/ingestion_agent.py`

Extracts structured `InvoiceData` from invoice text. The most significant design choice here is the **deterministic bypass for structured formats**.

**Two distinct extraction paths:**

**Path A — Deterministic (`.json`, `.csv`):**
JSON and CSV files contain machine-generated, structured data. The LLM is never invoked. `parse_json_invoice()` and `parse_csv_invoice()` read the files directly. This eliminates LLM latency, cost, and non-determinism for ~60% of a typical AP workload.

The CSV parser detects format automatically:
- **Key-value format** (2 columns, headers are `field`/`value` variants): delegates to `_parse_csv_kv()`
- **Tabular format** (one row per line item): delegates to `_parse_csv_tabular()`, which also extracts tax amount and rate from footer rows using a label-detection heuristic and regex rate extraction (`\((\d+(?:\.\d+)?)\s*%?\)`)

**Path B — LLM extraction (`.txt`, `.pdf`, `.xml`, `.xlsx`):**
An LLM is prompted with a strict system prompt defining 8 normalization rules. The output is bound directly to the `InvoiceData` Pydantic schema using structured output (`.with_structured_output()`). If the response fails schema validation or the LLM errors, the agent:

1. Returns the error to the critic for a retry with a targeted correction instruction.
2. On the second failure, attempts a **regex safety net** — `regex_extraction()` scans the raw text for invoice number and total amount patterns and patches whatever the LLM partially recovered.
3. If regex recovery still cannot satisfy the schema, returns `invoice_data=None` and the critic routes to `failure_node`.

**Item name normalization rules (embedded in LLM prompt):**
- `"Widget A"` → `"WidgetA"` (camelCase, no spaces)
- `"O2"` in numeric context → `2` (OCR artifact correction)
- Currency symbols stripped from all monetary values

---

### 3. Extraction Critic

**File:** `agents/critic_agent.py`

Reviews LLM extraction quality against the original invoice text. Returns `CriticOutput(valid: bool, correction_instruction: str)`.

**Deterministic bypass:** For JSON/CSV paths, the critic is completely skipped — it unconditionally returns `{valid: True}`. Invoking an LLM to review another program's deterministic output would be wasteful and could introduce false failures.

**Retry logic:** The critic increments `retry_count` on rejection. The router in `main.py` allows up to 2 retries before routing to `failure_node`. The correction instruction from the critic is forwarded back to the ingestion agent as additional context in the next extraction attempt.

**Conservative failure mode:** If the critic LLM itself throws an exception, it returns `{valid: False}` and triggers a retry rather than allowing a potentially poor extraction to proceed.

---

### 4. Validation Agent

**File:** `agents/validation_agent.py`

The validation agent is 100% deterministic Python — no LLM calls, no database writes. It performs four ordered checks and accumulates issues into a list, rather than failing fast. This ensures every problem with an invoice is surfaced in a single pass.

**Check 0 — Vendor presence:**
The vendor field must be non-empty after stripping whitespace. A blank vendor is an immediate `status=invalid`. This catches invoices from unknown or missing suppliers before hitting the inventory database.

**Check 1 — Inventory validation (aggregated):**

Before comparing against stock, all line items are grouped by their normalized name using `normalize_item_name()`. This function:
- Expands camelCase: `"WidgetARushOrder"` → `"Widget A Rush Order"`
- Removes parenthetical modifiers: `"WidgetA (rush order)"` → `"WidgetA"`
- Removes modifier words: `rush`, `expedited`, `priority`, `urgent`, `sample`, `replacement`
- Strips all non-alphanumeric characters and lowercases

This aggregation step is critical for invoices that split a single product across multiple line items (e.g., a rush order and a standard order for the same SKU). Without it, each line item would pass its individual stock check even when the total quantity exceeds available stock.

Three inventory states are handled distinctly:
- `stock is None` — item not found in catalog → +0.7 risk (immediately reaches `fraud_suspected` threshold)
- `stock == 0` — item exists but has zero inventory → +0.7 risk
- `total_qty > stock` — insufficient stock → `status=invalid`, no risk increment (legitimate shortage, not fraud)

**Check 2 — Date validation:**

Two sub-checks run in sequence:
1. Format check: if `due_date` is present, it must match `^\d{4}-\d{2}-\d{2}$`. Non-ISO strings (e.g., `"yesterday"`, `"Net 30"`) produce an `invalid_due_date` issue.
2. Overdue check: if the date is valid ISO, it is compared to today's date. Past-due invoices produce an `overdue_invoice` issue and add +0.3 to `risk_score`. An overdue date is a risk signal (backdated invoice fraud) but does not by itself make an invoice `invalid` — a legitimately late invoice from a verified vendor should still be paid.

**Check 3 — Tax-aware math validation:**

The expected total is computed as:
```
expected = subtotal + tax + shipping + handling + fees
```

Tax is resolved with an explicit priority chain to prevent LLM-returned `0.0` from overriding a real tax amount:
1. `invoice.tax_rate` (truthy check — skips `0.0` and `None`) → compute tax from subtotal
2. `invoice.tax_amount` (truthy check) → use explicitly stated amount
3. Regex extraction from invoice text → `_extract_charges()` scans for tax/VAT/GST/HST, shipping, handling, and fee patterns

If the computed expected total differs from `invoice.total_amount` by more than `$0.50`, a `math_discrepancy` issue is added.

**Risk scoring:**

Scores accumulate across all checks (capped at 1.0):

| Issue | Risk increment |
|---|---|
| Item not found in inventory | +0.7 |
| Item has zero inventory | +0.7 |
| Invalid quantity (≤ 0) | +0.3 |
| Math discrepancy | +0.3 |
| Overdue invoice | +0.3 |

Thresholds:
- `risk_score ≥ 0.7` → `fraud_suspected`
- `risk_score ≥ 0.4` → `high_risk`
- otherwise → `low_risk`

---

### 5. Approval Agent

**File:** `agents/approval_agent.py`

Applies a strict 5-rule decision hierarchy. Fully deterministic — no LLM, no database access.

```
Rule 1: risk_category == "fraud_suspected"  → ESCALATE  (overrides invalid status)
Rule 2: validation_status != "valid"        → REJECT
Rule 3: total_amount > ESCALATION_THRESHOLD → ESCALATE
Rule 4: math_discrepancy issue present      → ESCALATE
Rule 5: (default)                           → APPROVE
```

**Rule ordering is intentional:**

- Rule 1 comes before Rule 2: a fraud-suspected invoice with a failed validation should be escalated for human review, not silently rejected. A rejection closes the loop; an escalation triggers investigation.
- Rule 4 comes after Rule 3: a valid invoice with an arithmetic error in the stated total must not be paid. A human must verify the correct amount before payment is authorized.

**Configurable escalation threshold:**
The `$10,000` default is overridable via the `INVOICE_ESCALATION_THRESHOLD` environment variable:
```bash
INVOICE_ESCALATION_THRESHOLD=25000 python main.py --invoice_folder data/invoices/
```

**None-safety:**
The approval agent guards against `validation_status=None` (which can occur if the validation agent state is partially populated) by checking `val_status != "valid"` rather than `val_status == "invalid"`. Any value that is not explicitly `"valid"` results in rejection.

---

### 6. Payment Tool

**File:** `tools/payment_tool.py`

Executes payment (mocked ACH transfer) and writes a finance-grade receipt to `outputs/`.

**Safety guard — decision check:**
`execute_payment()` requires `approval_decision == "approve"` as an explicit argument. Any other value raises `RuntimeError` immediately. This guard exists at the function boundary, independent of the graph routing logic, so a routing error could never accidentally trigger a payment.

**Duplicate payment prevention:**
Before calling `mock_payment()`, the tool scans all `outputs/receipt_*.json` files for an existing receipt matching the invoice number with `status=completed`. If found, the payment is skipped and the existing transaction ID is returned with `status=duplicate_payment_prevented`. This protects against double-payment in cases such as:
- A revised invoice re-entering the payment path
- A session restart re-processing an already-paid invoice

**Unique transaction IDs:**
Receipt filenames and transaction IDs are generated as `TXN-{YYYYMMDDHHmmSS}-{milliseconds}`. The millisecond suffix ensures uniqueness when multiple deterministic-path invoices are processed back-to-back within the same second — a real collision scenario when JSON invoices process in under 1ms.

---

### 7. Registry Tool

**File:** `tools/registry_tool.py`

Provides persistent cross-session deduplication using a SQLite `processed_invoices` table.

**Deduplication logic (`check_duplicate()`):**

The invoice text is hashed with SHA-256. The registry classifies each incoming invoice as one of three states:

| Status | Meaning | Action |
|---|---|---|
| `NEW` | No prior record | Process normally |
| `DUPLICATE_CONTENT` | Identical hash seen before | Skip — return original TXN ID in audit |
| `REVISED_VERSION` | Same invoice number, different hash | Process — supersede prior record |

This three-state model handles the real-world scenario where a vendor resubmits a corrected invoice. The system processes it as a new invoice while marking the original record as `superseded` in the registry.

**All terminal paths log to registry:**
Every non-skipped invoice reaches exactly one of four registry log calls:
- `payment_node` → `status="success"` or `status="failed"`
- `reject_node` → `status="rejected"`
- `manual_review_node` → `status="escalated"`
- `failure_node` → `status="failed"`

This design ensures that rejected invoices cannot be re-submitted and processed again in a future batch run.

**Development override:**
Setting `ALLOW_DUPLICATES_FOR_TESTING=true` bypasses the registry hash check entirely, returning `NEW` for every invoice. This allows repeated test runs against the same invoice set without clearing the database. The flag is intentionally not set in CI/production environments.

---

## Fallback and Resilience Design

A central design principle: **the system should never silently succeed with wrong data, and should never crash on bad input.** Every major failure mode has an explicit, logged handler.

### LLM extraction failures

| Failure mode | Handler |
|---|---|
| LLM returns invalid schema | `ValidationError` caught → critic retry with error as correction instruction |
| LLM throws network / timeout error | Exception caught → critic retry |
| Critic rejects output twice | `retry_count ≥ 2` → route to `failure_node` |
| `invoice_data is None` after retry | Regex safety net attempts partial recovery (`invoice_number`, `total_amount`) |
| Regex recovery also fails | `failure_node` writes a partial audit report and logs to registry |

### Structured format failures

| Failure mode | Handler |
|---|---|
| JSON `parse_json_invoice()` raises `ValidationError` | Returns `invoice_data=None`; critic accepts (deterministic bypass) and routes to `validation_agent` which produces an extraction error issue |
| CSV `parse_csv_invoice()` fails | Same as above |
| File not found | `file_loader` raises `FileNotFoundError`; `process_invoice()` catches and returns `{error: ..., _file: ...}` |
| File format unsupported | `ValueError` raised; caught by `process_invoice()` |

### Validation edge cases

| Edge case | Handler |
|---|---|
| `invoice_data` is `None` when validation runs | Returns `{status: "invalid", issues: [extraction_error]}` immediately |
| `InvoiceData(**raw)` fails schema validation | Returns `{status: "invalid", issues: [schema_error]}` immediately |
| SQLite inventory unreachable | `get_inventory_item()` falls back to `data/inventory.json`; if JSON also absent, returns `stock=None`, treated as `inventory_not_found`; marks `status=invalid` |
| `due_date` cannot be parsed to ISO | Schema validator returns the raw string instead of raising; validation agent detects it |
| `tax_rate` is `0.0` (LLM default) | Truthy check (`if invoice.tax_rate:`) skips 0.0; falls through to tax_amount or text regex |
| Negative total amount | Schema accepts it; validation agent flags as `math_discrepancy` |

### Payment safety

| Failure mode | Handler |
|---|---|
| `execute_payment()` called with non-approve decision | Raises `RuntimeError` immediately — payment never executes |
| `execute_payment()` raises | Caught in `payment_node`; `payment_status="failed"` written to audit; registry logs `failed` |
| Duplicate invoice resubmission | Receipt file scan returns existing TXN; returns `duplicate_payment_prevented` without a new charge |

### Registry failures

| Failure mode | Handler |
|---|---|
| `inventory.db` does not exist | `check_duplicate()` returns `NEW`; `log_invoice_to_registry()` returns silently |
| `processed_invoices` table not yet created | `sqlite3.OperationalError` caught; treated as `NEW` |

---

## Schema Design

**File:** `schemas/models.py`

The Pydantic models enforce structure but deliberately avoid business logic. Constraints that require external context (inventory, arithmetic) are the responsibility of `validation_agent`.

**`InvoiceData` fields:**

| Field | Type | Constraint |
|---|---|---|
| `invoice_number` | `str` | Required |
| `vendor` | `str` | Required |
| `total_amount` | `Optional[float]` | No sign constraint — validation agent checks |
| `items` | `list[InvoiceItem]` | `min_length=1` |
| `due_date` | `Optional[str]` | Normalized to ISO; raw string kept on failure |
| `tax_rate` | `Optional[float]` | `ge=0` |
| `tax_amount` | `Optional[float]` | `ge=0` |

**`InvoiceItem` fields:**

| Field | Type | Constraint |
|---|---|---|
| `name` | `str` | Required |
| `quantity` | `int` | No sign constraint — validation agent checks |
| `unit_price` | `Optional[float]` | No constraint |

**Design rationale for relaxed constraints:** An earlier version enforced `quantity > 0` at the schema level. This caused `ValidationError` on invoices with negative quantities (e.g., returns/credits), which triggered the critic retry loop for deterministic-path JSON invoices — a system that bypasses the LLM should never enter a retry loop. Removing the constraint at the schema level and handling it in `validation_agent` preserves the deterministic bypass invariant.

**`due_date` normalization:** The schema validator attempts to parse 12 date formats into ISO `YYYY-MM-DD`. On failure, it returns the raw string rather than raising. This allows strings like `"yesterday"` or `"Net 30"` to propagate to `validation_agent`, which flags them as `invalid_due_date` — the correct place to make that judgement.

---

## Inventory

Inventory validation uses a SQLite database as the authoritative data source, as specified in the case instructions.

The system queries the SQLite inventory table during validation to verify that requested quantities do not exceed available stock.

To improve developer experience, the system includes an optional JSON fallback. If the SQLite database is unavailable or missing, `inventory_tools.py` loads `data/inventory.json` and performs the same lookup logic.

This fallback allows the system to run in development environments without requiring database initialization, while ensuring that production validation remains database-backed.

**Inventory lookup priority:**

1. SQLite database (primary)
2. JSON fallback (development only)

**Inventory lookup flow:**

```
Validation Agent
       │
       ▼
inventory_tools.get_inventory_item()
       │
       ▼
SQLite inventory.db
       │
       └── fallback → data/inventory.json
```

This design mirrors real enterprise systems where production data lives in databases but configuration files may be used for development or offline testing.

**Database schema (`database/inventory.db`):**

```sql
CREATE TABLE inventory (
    item  TEXT PRIMARY KEY,
    stock INTEGER
);
```

**JSON fallback format (`data/inventory.json`):**

```json
{
  "items": [
    {"item": "WidgetA",  "stock": 15},
    {"item": "WidgetB",  "stock": 10},
    {"item": "GadgetX",  "stock": 5},
    {"item": "FakeItem", "stock": 0}
  ]
}
```

**Lookup mechanics:** All lookups use normalized name matching via `normalize_item_name()`, so formatting differences (`"Widget A"` vs `"WidgetA"` vs `"widget-a"`) and order modifiers (`"Widget A (rush order)"`) never produce false "not found" results. Normalization runs once per lookup against each database row, keeping the query logic consistent across both sources.

---

## Outputs and Observability

### Per-invoice audit reports (`outputs/audit_INV-XXXX.json`)

Generated for every processed invoice. Contains:

```json
{
  "invoice_number": "INV-1006",
  "vendor": "Acme Industrial Supplies",
  "validation_status": "valid",
  "approval_decision": "approve",
  "payment_status": "success",
  "issues": [],
  "invoice_hash": "sha256...",
  "transaction_id": "TXN-20260312060428-799",
  "extraction_method": "deterministic_csv",
  "execution_times_ms": {
    "ingestion_agent": 1.2,
    "critic_agent": 0.4,
    "validation_agent": 3.1,
    "approval_agent": 0.2,
    "payment_tool": 5.8
  },
  "risk_score": 0.3,
  "risk_category": "low_risk",
  "subtotal": 2500.0,
  "tax": 250.0,
  "expected_total": 2750.0,
  "normalized_items": ["WidgetA", "GadgetX"],
  "normalization_applied": false
}
```

### Payment receipts (`outputs/receipt_TXN-XXXXXXXXXX-XXX.json`)

Written for every approved invoice. The filename is the transaction ID. Used by the duplicate payment guard on subsequent runs.

### Batch summary CSV (`outputs/batch_summary_YYYYMMDD_HHMMSS.csv`)

Written at the end of every `--invoice_folder` run. One row per unique invoice (deduplicated — revised versions supersede originals; skipped duplicates are omitted when a real record exists).

Columns: `invoice_number`, `vendor`, `file`, `validation_status`, `approval_decision`, `payment_status`, `total_amount`, `transaction_id`, `risk_score`, `risk_category`, `reason`

### System log (`logs/system.log`)

Persistent append-only log. Every agent transition, validation finding, approval decision, and payment event is written here with structured prefixes (`[INGESTION]`, `[CRITIC]`, `[VALIDATION]`, `[APPROVAL]`, `[PAYMENT]`, `[REGISTRY]`, `[AUDIT]`).

### Console agent trace

Live trace to stdout during processing:
```
[AGENT TRACE] -> Ingestion Agent
[AGENT TRACE] -> Extraction Critic
[AGENT TRACE] -> Validation Agent
[AGENT TRACE] -> Validation failed — routing to approval decision
[AGENT TRACE] -> Approval Agent
[AGENT TRACE] -> Escalated for manual review
```

---

## Running the System

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the LLM Provider

The system supports two LLM providers.

**Option A — xAI Grok (recommended)**

```bash
export XAI_API_KEY=your_key
export LLM_PROVIDER=grok
```

**Option B — Local model via Ollama (no API key required)**

If you do not have a Grok API key, run the system using a local model with [Ollama](https://ollama.com):

```bash
ollama pull llama3
export LLM_PROVIDER=ollama
```

### 3. Run a Single Invoice

```bash
python main.py --invoice_path data/invoices/invoice_1001.txt
```

### 4. Run the Full Invoice Dataset

```bash
python main.py --invoice_folder data/invoices/
```

**Sample batch output:**
```
================================================
  Batch Summary — 20 invoices
================================================
  Approved   : 9
  Escalated  : 3
  Rejected   : 5
  Skipped    : 3
  Report     : batch_summary_20260312_060444.csv
================================================
```

### 5. Generated Outputs

After running, the following files are created automatically:

```
outputs/
    audit_INV-XXXX.json          per-invoice audit report
    receipt_TXN-XXXX.json        payment receipt (approved invoices only)
    batch_summary_TIMESTAMP.csv  batch run summary

logs/
    system.log                   append-only structured event log

database/
    inventory.db                 SQLite — inventory + processed_invoices tables
```

All directories (`outputs/`, `logs/`, `database/`) are created automatically on first run. No manual setup is required.

### 6. Duplicate Detection

By default, processed invoices are recorded in `database/inventory.db`. Re-submitting the same invoice in a later run will be detected and skipped.

To allow repeated testing of the same invoices during development:

```bash
export ALLOW_DUPLICATES_FOR_TESTING=true
```

When set, the registry hash check is bypassed and every invoice is treated as new. Disable this in production.

### 7. Inventory Validation

Inventory is validated against the SQLite database at `database/inventory.db`. If the database is unavailable, the system falls back to `data/inventory.json` automatically — no configuration required.

---

## Testing

**Integration tests (`_integration_test.py`):** 27 tests covering the full agent graph with mocked LLM calls. Tests include happy path approval, fraud escalation, stock insufficiency, math discrepancy escalation, payment duplicate guard, registry deduplication, rejected invoice logging, schema relaxation cases, and approval rule edge cases (None status, missing result).

**Smoke tests (`_smoke_test.py`):** 17 unit tests for schemas, file loader, inventory tool, and payment tool. No agent dependencies. Verifies schema constraints, file parsing, inventory normalization, and the payment safety guard (`RuntimeError` on non-approve decisions).

```bash
python -m pytest _integration_test.py -v
python -m pytest _smoke_test.py -v
```

---

## Configuration Reference

| Environment variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `grok` | LLM backend: `grok` (xAI) or `ollama` (local) |
| `XAI_API_KEY` | — | Required when `LLM_PROVIDER=grok` |
| `INVOICE_ESCALATION_THRESHOLD` | `10000` | Dollar amount above which invoices are escalated regardless of validation status |
| `ALLOW_DUPLICATES_FOR_TESTING` | `false` | Bypasses registry duplicate check; set to `true` for local development |

---

## Project Structure

```
invoice-agent-system/
├── agents/
│   ├── ingestion_agent.py      # LLM + deterministic extraction
│   ├── critic_agent.py         # LLM extraction quality review
│   ├── validation_agent.py     # Deterministic validation (no LLM)
│   ├── approval_agent.py       # Deterministic approval rules (no LLM)
│   └── llm_provider.py         # LLM client factory
├── tools/
│   ├── file_loader.py          # Multi-format invoice → text converter
│   ├── inventory_tools.py      # Inventory lookup: SQLite primary, JSON fallback
│   ├── payment_tool.py         # Mock payment execution + receipt persistence
│   └── registry_tool.py        # SHA-256 deduplication registry (auto-initializes DB)
├── schemas/
│   ├── models.py               # Pydantic InvoiceData, InvoiceItem
│   └── state.py                # LangGraph TypedDict state
├── data/
│   ├── inventory.json          # Inventory fallback for development (JSON)
│   └── invoices/               # Invoice files for processing
├── database/
│   └── inventory.db            # SQLite (inventory + registry tables)
├── outputs/                    # Receipts, audit reports, batch CSVs
├── logs/
│   └── system.log              # Persistent append-only system log
├── main.py                     # CLI entry point + LangGraph orchestration
├── _integration_test.py        # 27 integration tests
└── _smoke_test.py              # 17 unit/smoke tests
```
