"""
agent.py — VendorPulse agent reasoning layer
Person 2 owns this file entirely.

Memory reads use get_baseline / get_events from memory.py
(async Hindsight calls wrapped in sync helpers).
"""

import os
import re
from groq import Groq
from collections import defaultdict
from dotenv import load_dotenv
from memory import get_baseline_sync, get_events_sync, recall_from_file

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "qwen/qwen3-32b"

# ── Flip to True once Hindsight is set up and Person 1 has ingested docs ──────
USE_HINDSIGHT = True


# ─────────────────────────────────────────────────────────────
# Clean raw LLM output
# ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("\\$", "$")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load memory (Hindsight or local file)
# ─────────────────────────────────────────────────────────────

def _load_memory(vendor_id: str) -> dict:
    """Returns {"baseline": {...}, "events": [...]}"""
    if USE_HINDSIGHT:
        return {
            "baseline": get_baseline_sync(vendor_id),
            "events":   get_events_sync(vendor_id)
        }
    return recall_from_file(vendor_id)


# ─────────────────────────────────────────────────────────────
# STEP 2 — Build structured context block from memory
# ─────────────────────────────────────────────────────────────

def build_context_block(vendor_id: str) -> str:
    mem      = _load_memory(vendor_id)
    baseline = mem.get("baseline")
    events   = mem.get("events", [])

    if not baseline:
        return f"No memory found for vendor: {vendor_id}"

    context = f"""VENDOR: {baseline["vendor_name"].upper()}

BASELINE (contract start {baseline["contract_start"]}):
- Agreed price:  {baseline["agreed_price"]}
- SLA terms:     {baseline["sla_terms"]}
- Renewal date:  {baseline["renewal_date"]}
- Rep name:      {baseline["rep_name"]}

EVENT LOG (chronological):"""

    if not events:
        context += "\n- No events recorded yet."
        return context

    grouped        = defaultdict(list)
    severity_label = {1: "minor", 2: "notable", 3: "BREACH"}

    for ev in sorted(events, key=lambda x: x["date"]):
        grouped[ev["fact_type"]].append(ev)

    for fact_type in ["price_change", "sla_violation", "term_change",
                      "commitment", "incident"]:
        if fact_type not in grouped:
            continue
        context += f"\n\n  [{fact_type.upper()}]"
        for ev in grouped[fact_type]:
            sev = severity_label.get(ev["severity"], "?")
            context += (
                f"\n  - {ev['date']} | {ev['summary']} "
                f"| severity={sev} | source={ev['source_doc']}"
            )

    return context


# ─────────────────────────────────────────────────────────────
# STEP 3 — Free-form question
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are VendorPulse, a procurement intelligence agent.
You have the complete memory of a vendor relationship.

STRICT OUTPUT RULES:
- Reply in clean markdown only.
- Start directly with the answer. No preamble.
- Use **bold** for all key values: prices, dates, percentages, names.
- Use bullet points for lists of facts.
- Use ### heading when switching topic within one answer.
- Write numbers in plain text only — e.g. $0.0081 - $0.0075 = **$0.0006**. Never LaTeX.
- Do not repeat the question.
- End with a one-line > blockquote of the most important takeaway.
- Answer only from the memory provided. Never guess."""


def ask_vendorpulse(vendor_id: str, question: str) -> str:
    context  = build_context_block(vendor_id)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Vendor memory:\n\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return _clean(response.choices[0].message.content)


# ─────────────────────────────────────────────────────────────
# STEP 4 — Structured negotiation brief
# ─────────────────────────────────────────────────────────────

BRIEF_PROMPT = """You are VendorPulse, a procurement intelligence agent.
Write a pre-renewal negotiation brief using the vendor memory below.

STRICT OUTPUT RULES:
- Clean markdown only. No preamble.
- **Bold** every specific value: price, date, percentage, name.
- One fact per bullet. No LaTeX. No escaped dollar signs.

Produce EXACTLY these four sections:

---

## 1. Pricing history
- Original agreed price and contract start date
- Every price change: date, old value → new value, % change, authorised or not
- Total cumulative overpayment

## 2. SLA compliance
- Original SLA commitment
- Every violation: date, what happened, credit issued or not
- Overall compliance verdict in one sentence

## 3. Broken commitments
- Every unkept promise: what, when, by whom, current status
- Flag verbal-only commitments explicitly

## 4. Suggested opening position
Three numbered, specific asks backed by the evidence above.

---

Vendor memory:
{context}"""


def get_negotiation_brief(vendor_id: str) -> str:
    context  = build_context_block(vendor_id)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": BRIEF_PROMPT.format(context=context)}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    return _clean(response.choices[0].message.content)


# ─────────────────────────────────────────────────────────────
# STEP 5 — Drift alerts
# ─────────────────────────────────────────────────────────────

def get_drift_alerts(vendor_id: str, min_severity: int = 2) -> list:
    mem    = _load_memory(vendor_id)
    events = mem.get("events", [])
    alerts = [ev for ev in events if ev.get("severity", 0) >= min_severity]
    return sorted(alerts, key=lambda x: x["date"], reverse=True)


# ─────────────────────────────────────────────────────────────
# STEP 6 — Overview data (baseline + events combined)
# ─────────────────────────────────────────────────────────────

def get_overview(vendor_id: str) -> dict:
    return _load_memory(vendor_id)


# ─────────────────────────────────────────────────────────────
# CLI test — python agent.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SEP = "=" * 60
    print(SEP); print("TEST 1: Context block"); print(SEP)
    print(build_context_block("twilio"))

    print(f"\n{SEP}"); print("TEST 2: Question"); print(SEP)
    print(ask_vendorpulse("twilio",
        "What is the difference between the old price and new price?"))

    print(f"\n{SEP}"); print("TEST 3: Brief"); print(SEP)
    print(get_negotiation_brief("twilio"))

    print(f"\n{SEP}"); print("TEST 4: Drift alerts"); print(SEP)
    badge = {1: "⚪ minor", 2: "🟡 notable", 3: "🔴 BREACH"}
    for a in get_drift_alerts("twilio"):
        print(f"{badge.get(a['severity'],'?')} | {a['date']} | {a['summary']}")