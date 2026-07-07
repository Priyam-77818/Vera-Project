"""
Multi-turn conversation handler for Vera.
Handles: auto-reply detection, intent transitions, graceful exits, hostile messages.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

from composer import _llm_complete, SYSTEM_PROMPT

# Auto-reply detection
AUTO_REPLY_PATTERNS = [
    r"thank you for contact",
    r"automated (reply|response|message)",
    r"i am (away|unavailable|out of office)",
    r"i.*ll (get back|respond|reply) (to you )?(shortly|soon|asap)",
    r"(business|working) hours",
    r"aapki (jaankari|madad)",
    r"aapka (sandesh|message) (hamare|hamari) team",
    r"main ek automated",
    r"yeh ek swachalit",
    r"out of office",
    r"will respond.*within.*hour",
    r"currently (unavailable|busy)",
    r"this is an automatic",
    r"auto.reply",
    r"do not reply to this",
    r"noreply",
]

HOSTILE_PATTERNS = [
    r"\bstop\b.*\bmessag",
    r"\bstop (messaging|sending|contacting)\b",
    r"\bnot interested\b",
    r"\bspam\b",
    r"remove (me|my number)",
    r"don'?t (message|contact|bother|call) (me|us)",
    r"\bblock\b",
    r"band karo",
    r"mat bhejo",
    r"mujhe nahi chahiye",
    r"\bbother\b",
    r"\buseless\b",
    r"waste of time",
    r"\birritating\b",
    r"\bannoying\b",
    r"bahut ho gaya",
    r"please remove",
    r"unsubscribe",
    r"opt.?out",
    r"do not (contact|message|send)",
]

INTENT_POSITIVE_PATTERNS = [
    r"\b(yes|ok|okay|sure|alright|haan|ha|yep|yup)\b",
    r"let'?s do it",
    r"go ahead",
    r"sounds good",
    r"what'?s next",
    r"please (proceed|continue|do it|send)",
    r"kar do",
    r"\bkaro\b",
    r"theek hai",
    r"bahut acha",
    r"bilkul",
    r"zaroor",
    r"i want to (join|start|proceed|sign up)",
    r"join (karna|chahta|chahti)",
    r"sign (me )?up",
    r"let'?s (proceed|go|start|do)",
    r"\bconfirm\b",
    r"\bapproved\b",
    r"please do",
    r"please send",
    r"send it",
    r"do it",
    r"proceed",
    r"sahi hai",
    r"chal jata hai",
    r"ho jayega",
    r"perfect",
    r"great.*let",
    r"sounds.*good",
    r"interested",
    r"tell me more",     # soft positive — escalate to action
    r"share kar",
    r"bhejo",
]


def is_auto_reply(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg) for p in AUTO_REPLY_PATTERNS)


def is_hostile(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg) for p in HOSTILE_PATTERNS)


def is_positive_intent(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg) for p in INTENT_POSITIVE_PATTERNS)


# Conversation state helpers
def get_auto_reply_count(turns: list[dict]) -> int:
    """Count consecutive auto-replies from the end of turn list."""
    count = 0
    for turn in reversed(turns):
        if turn.get("from") == "merchant" and is_auto_reply(turn.get("msg", "")):
            count += 1
        else:
            break
    return count


def get_last_bot_body(turns: list[dict]) -> str:
    for turn in reversed(turns):
        if turn.get("from") == "bot":
            return turn.get("msg", "")
    return ""


# ---------------------------------------------------------------------------
# Reply composer prompt
# ---------------------------------------------------------------------------

REPLY_SYSTEM = """You are Vera, magicpin's AI merchant assistant in an ongoing WhatsApp conversation.
You ALWAYS return valid JSON with keys: {"action": "send"|"wait"|"end", "body": "...", "cta": "...", "rationale": "..."}

RULES:
- "action": "send" — compose the next reply
- "action": "wait" — back off; add "wait_seconds" key (int)
- "action": "end" — close the conversation gracefully
- body required only for "send"; omit for "wait"/"end"
- body: concise WhatsApp reply, max ~220 chars
- NO preamble, NO re-introduction, NO "Hope you're well"
- Honor the conversation thread — answer exactly what was asked
- If merchant said YES/confirmed/intent_transition → IMMEDIATELY switch to ACTION mode; deliver the concrete artifact; DO NOT ask qualifying questions
- If merchant gave a curveball (off-topic) → politely decline in 1 sentence, redirect back in 1 sentence
- Match merchant's language from their most recent turn (switch to Hindi if they switched)
- Anti-repetition: NEVER repeat the same body as a previous bot turn; say something NEW
- Return JSON only, no markdown
"""

REPLY_PROMPT = """=== CONVERSATION HISTORY ===
{history}

=== LATEST MESSAGE (Turn {turn_number}) ===
From: {from_role}
Message: "{message}"

=== MERCHANT CONTEXT ===
Merchant: {merchant_name}, {locality}, {city}
Category: {category_slug}
Languages: {languages}
Verified: {verified}
Subscription: {subscription_status} ({subscription_days}d remaining)
Performance: views={views}, calls={calls}, CTR={ctr}%
Active offers: {active_offers}
Signals: {signals}
Trigger that started this: {trigger_kind}
Trigger payload summary: {trigger_summary}
Customer aggregate: lapsed={cust_lapsed}, retention={cust_retention}%

=== SITUATION ===
{situation}

=== TASK ===
This is turn {turn_number} of the conversation.
The merchant/customer just said: "{message}"

Compose the ideal next move (send/wait/end).
If sending: deliver concrete value, advance the conversation, honor their intent.
If intent_transition detected: deliver the artifact/plan IMMEDIATELY — no more qualifying.
Return JSON only."""


def respond(
    state: dict,
    merchant_message: str,
    merchant: Optional[dict] = None,
    category: Optional[dict] = None,
) -> dict:
    """
    Given conversation state + the merchant's latest message → produce reply.
    state = {"conversation_id": str, "turns": [...], "trigger_kind": str, ...}
    """
    turns = state.get("turns", [])
    turn_number = state.get("turn_number", len(turns) + 1)

    # hard-rule checks first

    # 1. Hostile message → end immediately
    if is_hostile(merchant_message):
        return {
            "action": "end",
            "rationale": "Merchant expressed frustration/opt-out. Closing conversation and suppressing.",
        }

    # 2. Auto-reply detection
    if is_auto_reply(merchant_message):
        new_count = state.get("auto_reply_count", 1)  # already incremented by main.py
        if new_count >= 3:
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {new_count} times in a row. Owner not at phone. Closing.",
            }
        elif new_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Second consecutive auto-reply. Backing off 24 hours.",
            }
        else:
            mname = merchant.get("identity", {}).get("owner_first_name", "") if merchant else ""
            biz_name = merchant.get("identity", {}).get("name", "") if merchant else ""
            return {
                "action": "send",
                "body": (
                    f"Looks like an auto-reply 😊 "
                    f"When {mname or 'the owner'} is free, just reply YES to continue — "
                    f"I had something useful about {biz_name or 'your business'} to share."
                ),
                "cta": "binary_yes_no",
                "rationale": "Detected first auto-reply; flagged it with a curiosity hook so the real owner engages when they see the message.",
            }
    else:
        state["auto_reply_count"] = 0

    # 3. Positive intent transition → switch to action mode
    if is_positive_intent(merchant_message) and turn_number <= 4:
        state["intent_confirmed"] = True

    # --- LLM-based reply for everything else ---
    situation = _classify_situation(merchant_message, state)

    # Build history summary (last 8 turns for better context)
    history_lines = []
    for t in turns[-8:]:
        role = "VERA" if t.get("from") == "bot" else "MERCHANT"
        history_lines.append(f"[{role}]: {t.get('msg', '')[:180]}")
    history_str = "\n".join(history_lines) if history_lines else "(conversation start)"

    m_identity = merchant.get("identity", {}) if merchant else {}
    m_perf = merchant.get("performance", {}) if merchant else {}
    m_sub = merchant.get("subscription", {}) if merchant else {}
    m_cust = merchant.get("customer_aggregate", {}) if merchant else {}
    active_offers = [o["title"] for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", []) if merchant else []

    # Summarize trigger payload for reply context
    trigger_id = state.get("trigger_id", "")
    trigger_payload_summary = ""
    if trigger_id and merchant:
        # Pull key numbers from the conversation start message if we have them
        first_bot_turn = next((t for t in turns if t.get("from") == "bot"), None)
        if first_bot_turn:
            trigger_payload_summary = first_bot_turn.get("msg", "")[:120]

    prompt = REPLY_PROMPT.format(
        history=history_str,
        turn_number=turn_number,
        from_role=state.get("from_role", "merchant"),
        message=merchant_message,
        merchant_name=m_identity.get("name", ""),
        locality=m_identity.get("locality", ""),
        city=m_identity.get("city", ""),
        languages=m_identity.get("languages", ["en"]),
        verified=m_identity.get("verified", False),
        subscription_status=m_sub.get("status", ""),
        subscription_days=m_sub.get("days_remaining", "?"),
        views=m_perf.get("views", 0),
        calls=m_perf.get("calls", 0),
        ctr=round(m_perf.get("ctr", 0) * 100, 2),
        active_offers=active_offers or ["(none)"],
        signals=signals,
        category_slug=category.get("slug", "") if category else "",
        trigger_kind=state.get("trigger_kind", ""),
        trigger_summary=trigger_payload_summary,
        cust_lapsed=m_cust.get("lapsed_180d_plus", 0),
        cust_retention=round(m_cust.get("retention_6mo_pct", 0) * 100, 0),
        situation=situation,
    )

    try:
        raw = _llm_complete(REPLY_SYSTEM, prompt, temperature=0.0)
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw.strip())
        result = json.loads(raw)

        # Safety: if body is the same as the last bot message, nudge differently
        if result.get("action") == "send":
            last = get_last_bot_body(turns)
            if last and result.get("body", "").strip() == last.strip():
                result["body"] = result["body"] + " — shall I go ahead?"

        return result

    except Exception as e:
        # Safe fallback
        if turn_number >= 5:
            return {"action": "end", "rationale": f"Max turns reached. [err: {e}]"}
        return {
            "action": "send",
            "body": "Got it — want me to go ahead?",
            "cta": "binary_yes_no",
            "rationale": f"Fallback reply. [LLM error: {e}]",
        }


def _classify_situation(message: str, state: dict) -> str:
    """Return a plain-text situation label to guide the LLM."""
    if is_positive_intent(message):
        return (
            "INTENT TRANSITION — merchant has CONFIRMED/ACCEPTED. "
            "Switch to action mode immediately. Do NOT ask qualifying questions. "
            "Deliver the artifact or next concrete step."
        )
    msg_lower = message.lower()
    off_topic_signals = ["gst", "tax", "legal", "police", "government", "loan", "bank", "insurance"]
    if any(s in msg_lower for s in off_topic_signals):
        return (
            "OFF-TOPIC request. Politely decline (it's outside Vera's scope), "
            "then redirect back to the original conversation topic."
        )
    question_signals = ["?", "kya", "kaise", "when", "how", "what", "why", "kitna", "kaisa"]
    if any(s in msg_lower for s in question_signals):
        return "QUESTION from merchant. Answer directly with facts from context. Then advance."
    negative_signals = ["no", "nahi", "nope", "not now", "later", "baad mein", "abhi nahi"]
    if any(s in msg_lower for s in negative_signals):
        return (
            "SOFT DECLINE. Acknowledge gracefully, offer to come back later or "
            "offer a lower-friction alternative. Don't push hard."
        )
    return "CONTINUING conversation. Advance naturally — answer, deliver, or invite next step."
