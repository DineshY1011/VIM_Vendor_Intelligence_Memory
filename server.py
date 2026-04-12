"""
server.py — VendorPulse Flask API
Run with: python server.py  →  open http://localhost:5000
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os, sys, asyncio

sys.path.insert(0, os.path.dirname(__file__))

from agent import (ask_vendorpulse, get_negotiation_brief,
                   get_drift_alerts, get_overview, USE_HINDSIGHT)
from memory import (write_baseline, append_event,
                    recall_from_file, VALID_FACT_TYPES)

app = Flask(__name__, static_folder="ui", static_url_path="")
CORS(app)

VENDOR_ID = "twilio"


# ── Serve frontend ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


# ── Overview ───────────────────────────────────────────────────
@app.route("/api/overview")
def overview():
    mem = get_overview(VENDOR_ID)
    # Normalise keys for the frontend
    return jsonify({
        "twilio:baseline": mem.get("baseline"),
        "twilio:events":   mem.get("events", [])
    })


# ── Drift alerts ───────────────────────────────────────────────
@app.route("/api/alerts")
def alerts():
    return jsonify(get_drift_alerts(VENDOR_ID, min_severity=2))


# ── Chat ───────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    body     = request.get_json()
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    return jsonify({"answer": ask_vendorpulse(VENDOR_ID, question)})


# ── Negotiation brief ──────────────────────────────────────────
@app.route("/api/brief", methods=["POST"])
def brief():
    return jsonify({"brief": get_negotiation_brief(VENDOR_ID)})


# ── INGEST DOCUMENT ────────────────────────────────────────────
# Accepts JSON body:
#   {
#     "doc_type":   "baseline" | "event",
#     "vendor_id":  "twilio",              (optional, defaults to VENDOR_ID)
#     "doc_id":     "twilio:baseline_1",   (optional)
#     "data":       { ...baseline or event fields... }
#   }
#
# When USE_HINDSIGHT=False the payload is validated and echoed back
# so the UI can confirm the schema without a live Hindsight connection.
# When USE_HINDSIGHT=True it calls write_baseline / append_event async.
@app.route("/api/ingest", methods=["POST"])
def ingest():
    body     = request.get_json()
    doc_type = body.get("doc_type", "").strip()
    vendor   = body.get("vendor_id", VENDOR_ID).strip().lower()
    doc_id   = body.get("doc_id", "").strip() or None
    data     = body.get("data", {})

    if doc_type not in ("baseline", "event"):
        return jsonify({"error": "doc_type must be 'baseline' or 'event'"}), 400

    if not data:
        return jsonify({"error": "data field is required"}), 400

    if not USE_HINDSIGHT:
        # Offline mode — validate schema and return preview
        from memory import _normalise_baseline, _normalise_event
        if doc_type == "baseline":
            normalised = _normalise_baseline(data)
        else:
            normalised = _normalise_event(data)
        return jsonify({
            "status":  "preview",
            "message": "USE_HINDSIGHT is False — schema validated, not stored. "
                       "Set USE_HINDSIGHT=True in agent.py to write to Hindsight.",
            "doc_type":   doc_type,
            "vendor_id":  vendor,
            "doc_id":     doc_id or f"{vendor}:{doc_type}_preview",
            "normalised": normalised
        })

    # Live Hindsight write
    try:
        if doc_type == "baseline":
            result = asyncio.run(write_baseline(vendor, data, doc_id))
            return jsonify({
                "status":   "stored",
                "doc_type": "baseline",
                "stored":   result
            })
        else:
            result = asyncio.run(append_event(vendor, data, doc_id))
            return jsonify({
                "status":   "stored",
                "doc_type": "event",
                "stored":   result
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  VendorPulse running at → http://localhost:5000\n")
    app.run(debug=True, port=5000)