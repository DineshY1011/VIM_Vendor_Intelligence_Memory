"""
server.py — VendorPulse FastAPI Backend

Endpoints:
  POST /upload          — upload a doc in baseline or normal mode
  GET  /baseline/{vid}  — get vendor baseline
  GET  /events/{vid}    — get vendor event log
  POST /reflect         — ask a question against vendor memory
  GET  /health          — health check
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

import memory
import pipeline

app = FastAPI(title="VendorPulse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── serve frontend ────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "VendorPulse API running. Frontend at /static/index.html"}

# ── health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "bank": memory.BANK_ID}

# ── upload ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    vendor_id: str = Form(...),
    mode: str = Form(...),
    files: list[UploadFile] = File(...)  # Support multiple files
):
    vendor_id = vendor_id.strip().lower()
    results = []
    
    for i, file in enumerate(files, start=1):
        file_bytes = await file.read()
        filename = file.filename
        
        # New Naming Convention: vendor:events (or baseline) with numbering
        # This acts as the unique document identifier in Hindsight
        doc_id = f"{vendor_id}:{mode}_{i}" 
        
        if mode == "baseline":
            result = await pipeline.process_baseline_doc(
                vendor_id, file_bytes, filename, doc_id=doc_id
            )
        else:
            # Fetch existing state for drift/dedup
            existing_baseline = await memory.get_baseline(vendor_id)
            existing_events = await memory.get_events(vendor_id)
            
            result = await pipeline.process_normal_doc(
                vendor_id, file_bytes, filename,
                existing_baseline=existing_baseline,
                existing_events=existing_events,
                doc_id=doc_id
            )
        results.append({"filename": filename, "status": "processed", "doc_id": doc_id})
        
    return {"ok": True, "results": results}

# ── get baseline ──────────────────────────────────────────────────────────────
@app.get("/baseline/{vendor_id}")
async def get_baseline(vendor_id: str):
    vendor_id = vendor_id.strip().lower()
    baseline = await memory.get_baseline(vendor_id)
    if baseline is None:
        raise HTTPException(404, f"No baseline found for vendor '{vendor_id}'")
    return {"vendor_id": vendor_id, "baseline": baseline}

# ── get events ────────────────────────────────────────────────────────────────
@app.get("/events/{vendor_id}")
async def get_events(vendor_id: str):
    vendor_id = vendor_id.strip().lower()
    events = await memory.get_events(vendor_id)
    return {
        "vendor_id": vendor_id,
        "count": len(events),
        "events": events,
    }

# ── reflect / ask ─────────────────────────────────────────────────────────────
class ReflectRequest(BaseModel):
    vendor_id: str
    query: str

@app.post("/reflect")
async def reflect(req: ReflectRequest):
    vendor_id = req.vendor_id.strip().lower()
    query = req.query.strip()
    
    client = memory.get_client()
    # Change .reflect() to .areflect() and await it
    result = await client.areflect(bank_id=memory.BANK_ID, query=f"{vendor_id}: {query}")
    
    answer = result.text if hasattr(result, "text") else str(result)
    return {"vendor_id": vendor_id, "query": query, "answer": answer}
