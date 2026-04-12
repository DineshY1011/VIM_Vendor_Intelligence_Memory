# VendorPulse — Vendor Intelligence Agent

> Drop in documents about a vendor. Ask a question. Get an answer that proves the agent remembers history and detected something you didn't know.

---

## What It Does

VendorPulse is a procurement intelligence agent that ingests vendor documents (contracts, invoices, incident reports, renewal letters, call transcripts), extracts structured facts, stores them as persistent memory, and answers natural language questions with full historical context.

It automatically detects **contract drift** — when a vendor quietly changes pricing, removes SLA clauses, or makes verbal commitments they haven't honoured.

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| LLM | [Groq](https://groq.com) | `qwen/qwen3-32b`, free tier |
| Agent Memory | [Hindsight Cloud](https://hindsight.cloud) | Promo code: `MEMHACK409` ($50 credits) |
| PDF Parsing | `pdfplumber` | Handles contracts and invoices |
| Backend | Python | Pipeline + agent logic |
| UI | Streamlit | Three-screen demo interface |

---

## Project Structure

```
vendorpulse/
├── data/
│   └── synthetic/
│       ├── contract_2022.txt
│       ├── invoice_2023_01.txt  →  invoice_2024_06.txt
│       ├── incident_jun2023.txt
│       ├── renewal_feb2024.txt
│       └── call_may2024.txt
├── pipeline/
│   ├── extractor.py        # extract_facts() — LLM-powered fact extraction
│   ├── memory_writer.py    # write_baseline(), write_event()
│   └── drift_detector.py  # check_drift() — compares events to baseline
├── agent/
│   ├── recall.py           # ask_vendorpulse() — retrieves + reasons over memory
│   └── brief.py            # Structured negotiation brief generator
├── app.py                  # Streamlit UI
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install groq pdfplumber streamlit requests
```

### 2. Set environment variables

```bash
export GROQ_API_KEY=your_groq_key
export HINDSIGHT_API_KEY=your_hindsight_key   # use promo: MEMHACK409
```

### 3. Generate synthetic data

Create the following files in `data/synthetic/`. Make them feel real — use real dollar amounts, plausible names, and accurate dates.

| File | Contents |
|---|---|
| `contract_2022.txt` | Original Twilio contract. Price: $0.0075/SMS, SLA: 99.95% uptime, signed March 2022 |
| `invoice_2023_01.txt` | First invoice with a quiet 8% price increase (no notice given) |
| `invoice_2023_02.txt` → `invoice_2024_06.txt` | Monthly invoices at the new rate |
| `incident_jun2023.txt` | Email thread about a 4-hour outage — no SLA credit issued |
| `renewal_feb2024.txt` | Renewal doc with the uptime SLA clause silently removed |
| `call_may2024.txt` | Sales call transcript — rep (Jordan Mills) promises rate lock through Q2 |

---

## Core Components

### Fact Extractor (`pipeline/extractor.py`)

```python
def extract_facts(text: str, date: str, doc_type: str) -> list[dict]:
    """
    Calls Groq to extract structured facts from a document excerpt.
    Returns a list of facts with type, value, date, confidence, and summary.
    Discards any fact with confidence < 0.7.
    """
```

Groq prompt instructs the model to return JSON only with this schema:

```json
{
  "facts": [
    {
      "fact_type": "price | sla | commitment | incident | term_change",
      "value": "the specific value or statement",
      "date": "YYYY-MM-DD",
      "confidence": 0.95,
      "summary": "One-sentence description"
    }
  ]
}
```

### Memory Writer (`pipeline/memory_writer.py`)

```python
def write_baseline(vendor_id: str, facts: list[dict]) -> None:
    """
    Called once, on the original contract only.
    Stores agreed price, SLA terms, renewal date, and rep name.
    This memory is never overwritten.
    """

def write_event(vendor_id: str, fact: dict) -> None:
    """
    Called for every subsequent fact found in later documents.
    Appends a timestamped event to the vendor's event log in Hindsight.
    """
```

### Drift Detector (`pipeline/drift_detector.py`)

```python
def check_drift(vendor_id: str, new_fact: dict) -> dict:
    """
    Recalls baseline from Hindsight, compares with new_fact,
    and returns a drift report with severity 1–3.
    Severity 3 = material breach (triggers console/UI alert).
    """
```

Drift report schema:

```json
{
  "is_drift": true,
  "severity": 3,
  "delta_summary": "Price increased 8% with no notice, from $0.0075 to $0.0081/SMS",
  "action": "Request retroactive credit and demand written justification"
}
```

### Agent (`agent/recall.py`)

```python
def ask_vendorpulse(vendor_id: str, question: str) -> str:
    """
    1. Recalls all memories for vendor from Hindsight (baseline + events)
    2. Formats them into a structured context block
    3. Passes context + question to Groq
    4. Returns a specific, date-cited, drift-aware answer
    """
```

Context block format fed to the LLM:

```
VENDOR: Twilio
BASELINE (signed March 2022):
- Price: $0.0075 per SMS
- SLA: 99.95% uptime guaranteed
- Renewal: June 2024

EVENT LOG:
- Jan 2023: Price raised to $0.0081/SMS (8% increase, no notice) [DRIFT - severity 3]
- Jun 2023: 4-hour outage, no SLA credit issued [DRIFT - severity 2]
- Feb 2024: Uptime SLA removed from renewal document [DRIFT - severity 3]
- May 2024: Rep (Jordan Mills) verbally promised rate lock through Q2 [commitment - unverified]
```

---

## Running the Pipeline

```bash
# Ingest all synthetic documents in chronological order
python pipeline/run_pipeline.py --vendor twilio --data-dir data/synthetic/

# Verify what was stored in Hindsight
python pipeline/verify_memory.py --vendor twilio
```

After a successful run you should see:
- **1 baseline memory** for Twilio (original contract terms)
- **~8 event memories** covering the price change, outage, SLA removal, and rate lock promise

---

## Running the App

```bash
streamlit run app.py
```

### Screen 1 — Vendor Overview
Vendor name, baseline terms, and a timeline of drift events colour-coded by severity (🔴 3 · 🟡 2 · ⚪ 1).

### Screen 2 — Chat Interface
Natural language Q&A against the full vendor memory. Demo questions to try:

- *"Brief me on Twilio before my renewal call"*
- *"What has Twilio promised that they haven't delivered?"*
- *"How much has our price changed since we signed?"*

### Screen 3 — Drift Alerts
All severity 2 and 3 events sorted by date — the *"what you didn't know you were losing"* view.

---

## Memory Schema (shared contract between pipeline and agent)

Both teammates must code to this and not change it without telling each other.

```python
VENDOR_ID = "twilio"   # always lowercase

BASELINE_KEYS = {
    "price_per_sms": float,        # e.g. 0.0075
    "sla_uptime_pct": float,       # e.g. 99.95
    "renewal_date": str,            # "YYYY-MM-DD"
    "signed_date": str,             # "YYYY-MM-DD"
    "rep_name": str,                # "Jordan Mills"
}

EVENT_SCHEMA = {
    "fact_type": str,               # "price | sla | commitment | incident | term_change"
    "value": str,                   # the specific value or statement
    "date": str,                    # "YYYY-MM-DD"
    "summary": str,                 # one sentence
    "drift_severity": int | None,   # 1, 2, 3, or None if no drift
}
```

---

## Demo Flow (60 seconds)

Rehearse this until it runs without fumbling:

1. Open the app. Show the empty state — *"No history loaded."*
2. Click **Ingest Twilio history**. Watch the pipeline run. Events populate the timeline.
3. Go to **Drift Alerts**. Show the three red alerts.
4. Go to **Chat**. Type *"Brief me on Twilio before my renewal call."* Read the output aloud as it streams.
5. Type *"What has Twilio promised that they haven't delivered?"* Show the rate lock answer.

---

## Team Split

| | Person 1 — Pipeline | Person 2 — Agent |
|---|---|---|
| **Day 1 AM** | Generate synthetic data | Set up Hindsight + Groq, test recall with dummy memories |
| **Day 1 PM** | Build and test `extract_facts()` | Build `ask_vendorpulse()`, test with dummy memories |
| **Day 1 Eve** | Write baseline + events to Hindsight | Build negotiation brief output, integrate with real memories |
| **Day 2 AM** | Build drift detection, test all alerts | Build Streamlit screens 1 and 2 |
| **Day 2 PM** | End-to-end pipeline test, fix extraction misses | Build screen 3, wire everything together |
| **Day 2 Eve** | 🤝 Rehearse demo together, fix anything that breaks the flow ||

---

## Goal

One vendor. One perfect demo flow. A clear story.

That beats a half-built multi-vendor dashboard every time.