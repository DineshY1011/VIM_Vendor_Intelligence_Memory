"""
server.py — VendorPulse Person 1 API

Person 1 owns the write side only. This server exposes a single endpoint:

  POST /upload   — ingest one or more documents in baseline or normal mode
  GET  /health   — liveness check

Person 2 (agent / chat) runs its own server and reads from Hindsight directly
via memory.get_baseline() and memory.get_events(). There is no overlap.

Environment variables (set in .env):
  GROQ_API_KEY      — Groq LLM key
  HINDSIGHT_API_KEY — Hindsight memory key
  HINDSIGHT_URL     — Hindsight base URL (optional, defaults to vectorize endpoint)
  BANK_NAME         — Hindsight bank ID (optional, defaults to "vendorpulse")
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

import memory
import pipeline

app = FastAPI(title="VendorPulse — Person 1 Ingestion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "bank": memory.BANK_ID}


# ── upload ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    vendor_id: str = Form(...),
    mode: str = Form(...),          # "baseline" or "normal"
    files: list[UploadFile] = File(...),
):
    """
    Ingest one or more documents for a vendor.

    Form fields:
      vendor_id  — lowercase vendor shorthand e.g. "twilio"
      mode       — "baseline" writes {vendor_id}:baseline (once, frozen)
                   "normal"   appends to {vendor_id}:events (append-only)
      files      — one or more uploaded files (PDF, TXT, DOCX, CSV, EML)

    Document ID convention (written to Hindsight):
      baseline docs → "{vendor_id}:baseline_{n}"
      event docs    → "{vendor_id}:events_{n}"
    where n is the 1-based index within this upload batch.

    Returns per-file results including extracted fields / events added.
    """
    vendor_id = vendor_id.strip().lower()

    if not vendor_id:
        raise HTTPException(status_code=422, detail="vendor_id must not be empty.")

    if mode not in ("baseline", "normal"):
        raise HTTPException(status_code=422, detail="mode must be 'baseline' or 'normal'.")

    results = []

    # Fetch existing state once before the loop — used by all normal-mode docs
    # in this batch for drift classification and deduplication.
    existing_baseline = None
    existing_events = []
    if mode == "normal":
        existing_baseline = await memory.get_baseline(vendor_id)
        existing_events   = await memory.get_events(vendor_id)

    for i, file in enumerate(files, start=1):
        file_bytes = await file.read()
        filename   = file.filename

        # Document ID: uniquely identifies this file inside Hindsight
        # Pattern: "{vendor_id}:baseline_{n}" or "{vendor_id}:events_{n}"
        doc_id = f"{vendor_id}:{mode}_{i}"

        if mode == "baseline":
            result = await pipeline.process_baseline_doc(
                vendor_id=vendor_id,
                file_bytes=file_bytes,
                filename=filename,
                doc_id=doc_id,
            )
        else:
            result = await pipeline.process_normal_doc(
                vendor_id=vendor_id,
                file_bytes=file_bytes,
                filename=filename,
                doc_id=doc_id,
                existing_baseline=existing_baseline,
                existing_events=existing_events,
            )

        results.append({
            "filename": filename,
            "doc_id":   doc_id,
            "ok":       result.get("ok", False),
            "error":    result.get("error"),
            # baseline mode — the six extracted fields
            "baseline": result.get("baseline"),
            # normal mode — facts written and skipped
            "doc_date":       result.get("doc_date"),
            "events_added":   result.get("events_added"),
            "events_skipped": result.get("events_skipped"),
        })

    overall_ok = all(r["ok"] for r in results)
    return {"ok": overall_ok, "vendor_id": vendor_id, "mode": mode, "results": results}