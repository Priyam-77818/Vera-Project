"""
Vera Bot — magicpin AI Challenge
FastAPI server exposing all 5 required endpoints.
"""
from __future__ import annotations
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from composer import compose
from conversation_handlers import respond, is_auto_reply, is_hostile

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

START_TIME = time.time()

# (scope, context_id) → {"version": int, "payload": dict}
contexts: Dict[tuple, Dict] = {}

# conversation_id → full state dict
conversations: Dict[str, Dict] = {}

# merchant_id → consecutive auto-reply count (persists across conv IDs)
# The judge sends each auto-reply turn on a DIFFERENT conv_id, so we track
# at merchant level to properly detect the pattern.
merchant_auto_reply_counts: Dict[str, int] = {}

# suppression keys already sent this session (dedup)
sent_suppression_keys: set = set()

app = FastAPI(title="Vera Bot", version="1.0.0")

# CORS — allow the frontend served at the same origin + any localhost dev port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the dashboard at /
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.get("/")
def dashboard():
    index = _os.path.join(_static_dir, "index.html")
    if _os.path.isfile(index):
        return FileResponse(index)
    return {"message": "Vera Bot is running. Dashboard at /static/index.html"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _get_payload(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None

def _get_merchant(merchant_id: str) -> Optional[dict]:
    return _get_payload("merchant", merchant_id)

def _get_category_for_merchant(merchant: dict) -> Optional[dict]:
    return _get_payload("category", merchant.get("category_slug", ""))

def _get_customer(customer_id: str) -> Optional[dict]:
    return _get_payload("customer", customer_id)

def _get_trigger(trigger_id: str) -> Optional[dict]:
    return _get_payload("trigger", trigger_id)

def _context_counts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: str
    turn_number: int = 1

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _context_counts(),
    }


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": "Vera AI",
        "team_members": ["Challenger"],
        "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        "approach": (
            "4-context LLM composer (category+merchant+trigger+customer) with "
            "per-trigger-kind prompt dispatch. Rule-based auto-reply detection "
            "(merchant-level counter survives conv_id changes), intent-transition "
            "routing, hostile-message handling. Adaptive to context version updates."
        ),
        "contact_email": "challenger@example.com",
        "version": "1.1.0",
        "submitted_at": "2026-04-26T08:00:00Z",
    }


@app.post("/v1/context")
def push_context(body: ContextBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope",
                "details": f"scope must be one of {valid_scopes}"}

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current and current["version"] > body.version:
        return {"accepted": False, "reason": "stale_version",
                "current_version": current["version"]}

    if current and current["version"] == body.version:
        return {"accepted": False, "reason": "stale_version",
                "current_version": current["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


@app.post("/v1/tick")
def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        trg = _get_trigger(trg_id)
        if not trg:
            continue

        sup_key = trg.get("suppression_key", "")
        if sup_key in sent_suppression_keys:
            continue

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")
        if not merchant_id:
            continue

        conv_id = f"conv_{merchant_id}_{trg_id}"
        existing_conv = conversations.get(conv_id)
        if existing_conv and (existing_conv.get("ended") or
                              len(existing_conv.get("turns", [])) > 0):
            continue

        merchant = _get_merchant(merchant_id)
        if not merchant:
            continue

        category = _get_category_for_merchant(merchant)
        if not category:
            continue

        customer = _get_customer(customer_id) if customer_id else None

        try:
            result = compose(category, merchant, trg, customer)
        except Exception as e:
            print(f"[TICK] Compose error for {trg_id}: {e}")
            continue

        body_text = result.get("body", "").strip()
        if not body_text:
            continue

        kind = trg.get("kind", "generic")
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        mname = merchant.get("identity", {}).get("name", "")
        send_as = result.get("send_as", "vera")
        if customer_id:
            send_as = "merchant_on_behalf"

        action_entry = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{kind}_v1",
            "template_params": [owner or mname, body_text[:80], ""],
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", sup_key),
            "rationale": result.get("rationale", ""),
        }
        actions.append(action_entry)

        conversations[conv_id] = {
            "turns": [{"from": "bot", "msg": body_text}],
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "trigger_kind": kind,
            "ended": False,
            "turn_number": 1,
            "auto_reply_count": 0,
        }

        if sup_key:
            sent_suppression_keys.add(sup_key)

        if len(actions) >= 20:
            break

    return {"actions": actions}


@app.post("/v1/reply")
def reply(body: ReplyBody):
    conv_id = body.conversation_id
    merchant_message = body.message.strip()
    merchant_id = body.merchant_id
    customer_id = body.customer_id

    # Fetch or create conversation state
    conv = conversations.get(conv_id)
    if not conv:
        conv = {
            "turns": [],
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": "",
            "trigger_kind": "unknown",
            "ended": False,
            "turn_number": body.turn_number,
            "auto_reply_count": 0,
        }
        conversations[conv_id] = conv

    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation already ended."}

    # Safety: end after 10 turns
    if body.turn_number > 10:
        conv["ended"] = True
        return {"action": "end", "rationale": "Maximum conversation turns reached."}

    resolved_merchant_id = conv.get("merchant_id") or merchant_id

    # --- Merchant-level auto-reply tracking ---
    # CRITICAL: the judge sends each turn on a different conv_id, so we must
    # track consecutive auto-replies at the merchant level, not per-conversation.
    if resolved_merchant_id:
        if is_auto_reply(merchant_message):
            merchant_auto_reply_counts[resolved_merchant_id] = \
                merchant_auto_reply_counts.get(resolved_merchant_id, 0) + 1
        else:
            merchant_auto_reply_counts[resolved_merchant_id] = 0
        # Write it into conv so conversation_handlers.py sees it
        conv["auto_reply_count"] = merchant_auto_reply_counts[resolved_merchant_id]

    # Record incoming turn
    conv["turns"].append({"from": body.from_role, "msg": merchant_message})
    conv["turn_number"] = body.turn_number
    conv["from_role"] = body.from_role

    # Fetch fresh context
    resolved_customer_id = conv.get("customer_id") or customer_id
    merchant = _get_merchant(resolved_merchant_id) if resolved_merchant_id else None
    category = _get_category_for_merchant(merchant) if merchant else None
    customer = _get_customer(resolved_customer_id) if resolved_customer_id else None

    # Pass conv directly — mutations inside respond() persist (e.g. ended flag)
    result = respond(conv, merchant_message, merchant=merchant, category=category)

    action = result.get("action", "send")
    if action == "send":
        conv["turns"].append({"from": "bot", "msg": result.get("body", "")})
    elif action == "end":
        conv["ended"] = True
        # Also clear the merchant auto-reply count so future conversations start fresh
        if resolved_merchant_id:
            merchant_auto_reply_counts.pop(resolved_merchant_id, None)

    response = {"action": action, "rationale": result.get("rationale", "")}
    if action == "send":
        response["body"] = result.get("body", "")
        response["cta"] = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 3600)

    return response


@app.post("/v1/teardown")
def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    merchant_auto_reply_counts.clear()
    return {"wiped": True}
