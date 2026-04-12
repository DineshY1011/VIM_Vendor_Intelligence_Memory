"""
memory.py — VendorPulse Hindsight Memory Layer
Exact schema from VendorPulse_Memory_Structure.docx

Two namespaces per vendor:
  {vendor_id}:baseline  — written once, never mutated
  {vendor_id}:events    — append-only list of timestamped event dicts

The four Hindsight operations:
  write_baseline(vendor_id, baseline, doc_id)   — Person 1 / pipeline only
  append_event(vendor_id, event, doc_id)        — Person 1 / pipeline only
  get_baseline(vendor_id)                       — Person 2 / agent only
  get_events(vendor_id)                         — Person 2 / agent only
"""

import json
import asyncio
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Try to import Hindsight client; fall back gracefully if not installed ──────
try:
    from hindsight_client import Hindsight
    HINDSIGHT_AVAILABLE = True
except ImportError:
    HINDSIGHT_AVAILABLE = False

_client = None

def get_client():
    global _client
    if not HINDSIGHT_AVAILABLE:
        raise RuntimeError(
            "hindsight_client not installed. Run: pip install hindsight-client"
        )
    if _client is None:
        _client = Hindsight(
            base_url=os.getenv("HINDSIGHT_URL", "https://api.hindsight.vectorize.io"),
            api_key=os.getenv("HINDSIGHT_API_KEY", "")
        )
    return _client

BANK_ID = os.getenv("BANK_NAME", "vendorpulse")

# ── Schema constants ───────────────────────────────────────────────────────────
BASELINE_FIELDS  = ["vendor_name", "agreed_price", "sla_terms",
                    "contract_start", "renewal_date", "rep_name"]
VALID_FACT_TYPES = {"price_change", "sla_violation", "term_change",
                    "commitment", "incident"}


def _baseline_key(vendor_id: str) -> str:
    return f"{vendor_id}:baseline"

def _events_key(vendor_id: str) -> str:
    return f"{vendor_id}:events"

def _normalise_baseline(data: dict) -> dict:
    return {field: str(data.get(field, "")) for field in BASELINE_FIELDS}

def _normalise_event(event: dict) -> dict:
    fact_type = event.get("fact_type", "incident")
    if fact_type not in VALID_FACT_TYPES:
        fact_type = "incident"
    return {
        "date":       str(event.get("date", "")),
        "fact_type":  fact_type,
        "value":      str(event.get("value", "")),
        "summary":    str(event.get("summary", "")),
        "severity":   int(event.get("severity", 1)),
        "source_doc": str(event.get("source_doc", "")),
    }


# ── WRITE BASELINE (Person 1 only) ─────────────────────────────────────────────
async def write_baseline(vendor_id: str, baseline: dict, doc_id: str = None) -> dict:
    """Write vendor baseline to Hindsight. Call ONCE per vendor only."""
    client     = get_client()
    normalised = _normalise_baseline(baseline)
    document_id = doc_id or _baseline_key(vendor_id)

    content = (
        f"VENDOR BASELINE — {vendor_id.upper()}\n"
        f"Key: {_baseline_key(vendor_id)}\n"
        f"Document ID: {document_id}\n\n"
        + "\n".join(f"{k}: {v}" for k, v in normalised.items())
        + f"\n\nJSON:\n{json.dumps(normalised, indent=2)}"
    )

    await client.aretain(
        bank_id=BANK_ID,
        content=content,
        context=_baseline_key(vendor_id),
        document_id=document_id,
    )
    return normalised


# ── APPEND EVENT (Person 1 only) ───────────────────────────────────────────────
async def append_event(vendor_id: str, event: dict, doc_id: str = None) -> dict:
    """Append one event to the vendor's event log. Append-only — never delete."""
    client     = get_client()
    normalised = _normalise_event(event)
    severity_label = {1: "MINOR", 2: "NOTABLE", 3: "BREACH"}.get(
        normalised["severity"], "MINOR"
    )
    document_id = doc_id or _events_key(vendor_id)

    content = (
        f"VENDOR EVENT — {vendor_id.upper()}\n"
        f"Key: {_events_key(vendor_id)}\n"
        f"Document ID: {document_id}\n"
        f"Severity: {normalised['severity']} ({severity_label})\n"
        f"JSON:\n{json.dumps(normalised, indent=2)}"
    )

    await client.aretain(
        bank_id=BANK_ID,
        content=content,
        context=_events_key(vendor_id),
        document_id=document_id,
        timestamp=f"{normalised['date']}T00:00:00Z" if normalised["date"] else None,
    )
    return normalised


# ── GET BASELINE (Person 2 only) ───────────────────────────────────────────────
async def get_baseline(vendor_id: str) -> Optional[dict]:
    """Retrieve vendor baseline from Hindsight. Returns dict or None."""
    client = get_client()
    result = await client.arecall(
        bank_id=BANK_ID,
        query=f"vendor baseline {vendor_id}"
    )

    if not result or not hasattr(result, "results") or not result.results:
        return None

    baseline_key = _baseline_key(vendor_id)
    for r in result.results:
        text = r.text if hasattr(r, "text") else str(r)
        if baseline_key in text or vendor_id.upper() in text:
            try:
                json_start = text.find('{\n')
                if json_start != -1:
                    return json.loads(text[json_start:])
            except (json.JSONDecodeError, ValueError):
                pass
            # Fallback: parse key:value lines
            baseline = {}
            for line in text.split("\n"):
                for field in BASELINE_FIELDS:
                    if line.startswith(f"{field}:"):
                        baseline[field] = line.split(":", 1)[1].strip()
            if baseline:
                return _normalise_baseline(baseline)
    return None


# ── GET EVENTS (Person 2 only) ─────────────────────────────────────────────────
async def get_events(vendor_id: str) -> list:
    """Retrieve all events for a vendor from Hindsight, oldest first."""
    client = get_client()
    result = await client.arecall(
        bank_id=BANK_ID,
        query=f"vendor events {vendor_id}"
    )

    if not result or not hasattr(result, "results") or not result.results:
        return []

    events_key = _events_key(vendor_id)
    events     = []
    seen       = set()

    for r in result.results:
        text = r.text if hasattr(r, "text") else str(r)
        if events_key not in text and vendor_id.upper() not in text:
            continue
        try:
            json_start = text.find('{\n')
            if json_start != -1:
                event = json.loads(text[json_start:])
                dedup_key = (
                    event.get("fact_type", ""),
                    event.get("value", ""),
                    event.get("date", "")
                )
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    events.append(_normalise_event(event))
        except (json.JSONDecodeError, ValueError):
            continue

    events.sort(key=lambda e: e.get("date", ""))
    return events


# ── SYNC WRAPPERS (used by agent.py which is sync) ─────────────────────────────
def get_baseline_sync(vendor_id: str) -> Optional[dict]:
    return asyncio.run(get_baseline(vendor_id))

def get_events_sync(vendor_id: str) -> list:
    return asyncio.run(get_events(vendor_id))


# ── LOCAL FILE FALLBACK (used when USE_HINDSIGHT = False) ──────────────────────
def recall_from_file(vendor_id: str,
                     filepath: str = "data/dummy_memories.json") -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    return {
        "baseline": data.get(f"{vendor_id}:baseline"),
        "events":   data.get(f"{vendor_id}:events", [])
    }