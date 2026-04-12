"""
memory.py — VendorPulse Hindsight Memory Layer
Exact schema from VendorPulse_Memory_Structure.docx

Two namespaces per vendor:
  {vendor_id}:baseline  — written once, never mutated
  {vendor_id}:events    — append-only list of timestamped event dicts

The four Hindsight operations used by VendorPulse:
  write_baseline(vendor_id, baseline)  — Person 1 / pipeline only
  append_event(vendor_id, event)       — Person 1 / pipeline only
  get_baseline(vendor_id)              — Person 2 / agent only
  get_events(vendor_id)                — Person 2 / agent only
"""

import json
import asyncio
import os
from typing import Optional
from hindsight_client import Hindsight

# ── client singleton ──────────────────────────────────────────────────────────
_client: Optional[Hindsight] = None

def get_client() -> Hindsight:
    global _client
    if _client is None:
        _client = Hindsight(
            base_url=os.getenv("HINDSIGHT_URL", "https://api.hindsight.vectorize.io"),
            api_key=os.getenv("HINDSIGHT_API_KEY", "")
        )
    return _client

BANK_ID = os.getenv("BANK_NAME", "vendorpulse")

# ── async-safe runner ─────────────────────────────────────────────────────────
def _run(coro):
    """Run a Hindsight coroutine safely whether inside or outside asyncio."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context (FastAPI) — just return the coro
            # caller must await it
            return coro
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# ── baseline memory ───────────────────────────────────────────────────────────
# Baseline fields (exact spec from §2.1):
#   vendor_name    : str  — full legal name from contract
#   agreed_price   : str  — e.g. "$0.0075 per SMS"
#   sla_terms      : str  — verbatim quote from contract
#   contract_start : str  — YYYY-MM-DD
#   renewal_date   : str  — YYYY-MM-DD
#   rep_name       : str  — vendor account rep who signed

BASELINE_FIELDS = ["vendor_name", "agreed_price", "sla_terms",
                   "contract_start", "renewal_date", "rep_name"]

# Event fact_type valid values (exact spec from §3.2):
VALID_FACT_TYPES = {"price_change", "sla_violation", "term_change",
                    "commitment", "incident"}

def _baseline_key(vendor_id: str) -> str:
    return f"{vendor_id}:baseline"

def _events_key(vendor_id: str) -> str:
    return f"{vendor_id}:events"

def _normalise_baseline(data: dict) -> dict:
    """Ensure all six baseline fields are present; missing → empty string."""
    return {field: str(data.get(field, "")) for field in BASELINE_FIELDS}

def _normalise_event(event: dict) -> dict:
    """Validate and normalise one event object per §3.1 spec."""
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

# -- write_baseline (Person 1 only) --------------------------------------------
async def write_baseline(vendor_id: str, baseline: dict) -> dict:
    client = get_client()
    normalised = _normalise_baseline(baseline)
    content = (
        f"VENDOR BASELINE — {vendor_id.upper()}\n"
        f"Key: {_baseline_key(vendor_id)}\n\n"
        + "\n".join(f"{k}: {v}" for k, v in normalised.items())
        + f"\n\nJSON:\n{json.dumps(normalised, indent=2)}"
    )
    # Change .retain() to .aretain() and await it directly
    await client.aretain(
        bank_id=BANK_ID,
        content=content,
        context=_baseline_key(vendor_id),
        document_id=_baseline_key(vendor_id),
    )
    return normalised

# -- append_event (Person 1 only) ----------------------------------------------
async def append_event(vendor_id: str, event: dict) -> dict:
    client = get_client()
    normalised = _normalise_event(event)
    severity_label = {1: "MINOR", 2: "NOTABLE", 3: "BREACH"}.get(normalised["severity"], "MINOR")
    content = (
        f"VENDOR EVENT — {vendor_id.upper()}\n"
        f"Key: {_events_key(vendor_id)}\n"
        f"JSON:\n{json.dumps(normalised, indent=2)}"
    )
    # Change .retain() to .aretain()
    await client.aretain(
        bank_id=BANK_ID,
        content=content,
        context=_events_key(vendor_id),
        document_id=doc_id, # Link fact to the numbered doc ID
        timestamp=f"{normalised['date']}T00:00:00Z" if normalised["date"] else None,
    )
    return normalised

# ── get_baseline (Person 2 only) ──────────────────────────────────────────────
async def get_baseline(vendor_id: str) -> Optional[dict]:
    client = get_client()
    # Change .recall() to .arecall()
    result = await client.arecall(
        bank_id=BANK_ID,
        query=f"vendor baseline {vendor_id}"
    )
    if asyncio.iscoroutine(result):
        result = await result

    if not result or not hasattr(result, "results") or not result.results:
        return None

    # Find the most relevant result that mentions this vendor's baseline key
    baseline_key = _baseline_key(vendor_id)
    for r in result.results:
        text = r.text if hasattr(r, "text") else str(r)
        if baseline_key in text or vendor_id.upper() in text:
            # Try to extract the JSON block from the stored content
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

# -- get_events (Person 2 only) ------------------------------------------------
async def get_events(vendor_id: str) -> list:
    client = get_client()
    # Change .recall() to .arecall()
    result = await client.arecall(
        bank_id=BANK_ID,
        query=f"vendor events {vendor_id}"
    )
    if asyncio.iscoroutine(result):
        result = await result

    if not result or not hasattr(result, "results") or not result.results:
        return []

    events_key = _events_key(vendor_id)
    events = []
    seen = set()  # duplicate detection: fact_type + value + date

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

    # Sort oldest first
    events.sort(key=lambda e: e.get("date", ""))
    return events
