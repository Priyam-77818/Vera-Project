"""
Vera Message Composer

LLM-powered composer that takes the 4 contexts and produces a high-scoring
WhatsApp message. Dispatches by trigger kind for best results.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

# LLM client (supports OpenAI + Anthropic)
def _llm_complete(system: str, user: str, temperature: float = 0.0) -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        resp = client.messages.create(
            model=model, max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    elif provider == "groq":
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment / .env file")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        # Groq doesn't support response_format=json_object for all models,
        # so we ask for JSON in the prompt and parse manually.
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    else:  # default openai
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        model = os.getenv("LLM_MODEL", "gpt-4o")
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content


# Prompt templates per trigger kind
SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages that
engage Indian merchants or their customers. You ALWAYS return valid JSON with these exact keys:
{"body": "...", "cta": "...", "send_as": "...", "suppression_key": "...", "rationale": "..."}

RULES (violating any costs heavy scoring penalty):
1. body — the WhatsApp message text. Concise, no preamble ("Hope you're well" etc.). Max ~280 chars.
2. cta — one of: "binary_yes_no", "binary_confirm_cancel", "open_ended", "multi_choice_slot", "none"
3. send_as — "vera" for merchant-facing, "merchant_on_behalf" for customer-facing
4. suppression_key — copy from the trigger's suppression_key
5. rationale — 2-3 sentences explaining WHICH compulsion lever(s) you used, what specific data from context anchors it, and why this message fits THIS merchant/customer
6. NEVER fabricate data not present in the context JSON provided
7. NEVER use taboo words from the category voice profile
8. ALWAYS use the merchant/customer's actual name (owner first name for merchants, customer first name for customers)
9. ALWAYS anchor on at least TWO specific numbers, dates, or source citations from the contexts
10. Hindi-English code-mix is ENCOURAGED for hi/hi-en merchants and customers — match their language exactly
11. The LAST sentence must be the CTA — buried CTAs score 0
12. "X% off" generic discounts score LOWER than "Service @ ₹price" specifics — always prefer service+price
13. For customer-facing messages: honor language preference, preferred slot times, relationship state, consent scope
14. For research/compliance triggers: include the source citation at end (e.g. — JIDA Oct 2026 p.14)
15. Emoji: 1 max, only if it fits the category (🦷 dental, 💇 salon, 🏋️ gym — NEVER for pharmacies)
16. SOCIAL PROOF lever (underused by production Vera, high scoring): "X merchants in your locality/category did Y this month" — use when peer_stats or signals support it
17. EFFORT EXTERNALIZATION lever: "I've already drafted X — just say GO" — dramatically improves reply rate
18. LOSS AVERSION lever: frame as what the merchant is currently missing, not what they could gain
19. DO NOT re-introduce yourself after the first message in a conversation
20. DECISION QUALITY: pick the single most compelling angle from all the context — do not list everything
"""

COMPOSE_PROMPT = """=== CONTEXT ===

CATEGORY ({slug}):
Voice tone: {voice_tone}
Register: {voice_register}
Vocab allowed: {vocab_allowed}
Vocab taboos: {taboos}
Salutation style: {salutation_examples}
Active offers (catalog): {offer_catalog}
Peer stats: {peer_stats}
Digest (latest — USE SPECIFIC NUMBERS AND SOURCES): {digest}
Patient/customer content library (shareable): {content_library}
Seasonal beats: {seasonal_beats}
Trend signals: {trend_signals}

MERCHANT:
ID: {merchant_id}
Name: {merchant_name}
Owner first name: {owner_name}
City: {city}, Locality: {locality}
Established: {established_year}
Verified on GBP: {verified}
Languages: {languages}
Subscription: {subscription}
Performance (30d): views={views}, calls={calls}, directions={directions}, CTR={ctr}%, leads={leads}
Peer benchmarks: avg_views={peer_avg_views}, avg_calls={peer_avg_calls}, avg_CTR={peer_ctr}%, avg_reviews={peer_avg_reviews}, avg_photos={peer_avg_photos}
CTR vs peer: {ctr_vs_peer}
Calls vs peer: {calls_vs_peer}
7-day delta: views {delta_views_pct}%, calls {delta_calls_pct}%
Active offers: {active_offers}
Expired offers (could reactivate): {expired_offers}
Merchant signals: {signals}
Last conversation: {convo_history}
Last conversation engagement: {convo_engagement}
Customer aggregate: total={cust_total}, lapsed_180d={cust_lapsed}, retention_6mo={cust_retention}%, high_risk_count={high_risk_count}
Review themes (30d): {review_themes}

TRIGGER:
ID: {trigger_id}
Kind: {trigger_kind}
Source: {trigger_source}
Urgency: {urgency}/5
Payload: {trigger_payload}
Suppression key: {suppression_key}
Expires: {expires_at}

CUSTOMER (if present):
{customer_block}

=== SOCIAL PROOF DATA ===
(Use this to add the "3 merchants in your locality did X" lever — high scoring, underused by competitors)
{social_proof_block}

=== DECISION QUALITY GUIDANCE ===
From all the data above, pick the SINGLE most compelling angle for this merchant+trigger combination.
Do NOT list everything — pick one sharp hook that would make THIS merchant stop and reply.
Ask yourself: "Why would THIS specific merchant care about this RIGHT NOW?"

=== TASK ===
Compose the best possible Vera message for this (merchant, trigger) combination.
Trigger kind = "{trigger_kind}" — specific guidance:
{kind_guidance}

COMPULSION LEVER SELECTION (pick 1-2, state which you chose in rationale):
- SPECIFICITY: anchor on a number/date/source the merchant can verify
- LOSS AVERSION: frame as what they're currently losing/missing
- SOCIAL PROOF: "X merchants in {{locality}}/{{category}} did Y this month" (HIGH IMPACT — USE THIS)
- EFFORT EXTERNALIZATION: "I've already drafted/prepared X — just say GO"
- CURIOSITY: "Want to see who?" / "Want the full breakdown?"
- RECIPROCITY: "I noticed X about your account and wanted you to know first"
- ASK THE MERCHANT: one specific question that's also a soft CTA (HIGH IMPACT — USE THIS)

Return JSON only. No markdown, no extra keys.
"""

KIND_GUIDANCE = {
    "research_digest": (
        "Lead with the specific research finding (exact trial N + % result + source). "
        "In the SECOND sentence, name the specific patient segment in THIS merchant's roster it applies to "
        "(check customer_aggregate.high_risk_count or customer state distribution). "
        "End with effort-externalization: 'I can pull the abstract + draft a patient-ed WhatsApp — say GO.'"
    ),
    "regulation_change": (
        "Lead with the exact deadline date and what SPECIFICALLY changes (e.g., dose limit mSv values). "
        "Tell them the ONE concrete action they must take before that date. "
        "Use loss-aversion framing — non-compliance has real consequences. "
        "Single YES/NO CTA. Include the regulatory body name and source circular."
    ),
    "cde_opportunity": (
        "Lead with the CDE credit count, cost, and speaker name. "
        "State the tangible practice benefit (e.g., 'CAD/CAM ROI for solo practices'). "
        "Mention the registration deadline if available. Single yes/no CTA."
    ),
    "perf_dip": (
        "Name the EXACT metric (calls/CTR/views) and the exact % drop vs baseline. "
        "Offer a specific diagnosis — look at signals, review_themes, and peer data to suggest a likely cause. "
        "Deliver ONE concrete fix they can do today (not a list). "
        "Use loss-aversion: X calls/week lost = X potential patients."
    ),
    "perf_spike": (
        "Lead with the exact spike number and timeframe. "
        "Credit a likely driver if visible in the data (offer launched, post published, etc). "
        "Use effort-externalization: 'I can prep a post to keep this going — want me to draft?' "
        "Ask them to capitalize on the momentum with ONE specific action."
    ),
    "milestone_reached": (
        "Name the EXACT milestone number. "
        "Add social proof: how does this compare to peer avg_review_count? "
        "('You're now {X}x the locality average of {peer_avg} reviews') "
        "Suggest one specific forward action that turns this momentum into new bookings."
    ),
    "dormant_with_vera": (
        "Do NOT mention the dormancy period — that's guilt, not value. "
        "Lead with a fresh piece of value from the digest or trend_signals relevant to their category. "
        "Use curiosity lever: 'Noticed something interesting about your category this week — want a quick look?'"
    ),
    "review_theme_emerged": (
        "Name the theme and the exact occurrence count (e.g., '4 reviews in 30d mention wait time'). "
        "Position as 'I noticed X — thought you'd want to know' (reciprocity lever). "
        "Offer to draft a response template OR suggest the concrete fix. "
        "Do NOT be accusatory — be collegial."
    ),
    "competitor_opened": (
        "Name the distance (e.g., 1.3 km) but do NOT invent the competitor name if not in payload. "
        "Frame as opportunity: what does THIS merchant have that the new entrant likely doesn't? "
        "(Check reviews, offers, rating, established_year.) "
        "Suggest a specific counter-move anchored in THIS merchant's verified strengths."
    ),
    "festival_upcoming": (
        "Name the festival + exact days until. "
        "Suggest a specific service+price campaign from the offer_catalog relevant to the category and festival. "
        "Use effort-externalization: 'I've drafted a GBP post + WhatsApp blast — want to review?' "
        "Reference seasonal_beats if there's a known demand spike for this period."
    ),
    "ipl_match_today": (
        "Name the specific match (teams + venue + match_time). "
        "Be SMART about recommendation — weeknight match = different advice than weekend. "
        "Recommend the specific existing offer to push OR advise NOT to push if it won't convert. "
        "Loss-aversion: 'match-day orders 2x for restaurants that push at 6pm — window closes in X hours.'"
    ),
    "renewal_due": (
        "Be direct — X days remaining. "
        "Show SPECIFIC loss: 'Profile paused = estimated Y views/week gone dark' (use peer avg as proxy). "
        "Do NOT beg — frame as their business continuity decision. "
        "Single YES CTA. If days_remaining < 7, add urgency: 'after expiry, reactivation takes 48h.'"
    ),
    "curious_ask_due": (
        "Ask the merchant ONE sharp, specific question about their business RIGHT NOW. "
        "Examples by category: dentist='Which treatment is your most common new-patient ask this month?', "
        "salon='What color service is trending at your chair this week?', "
        "restaurant='What is your busiest order in the last 7 days?', "
        "gym='Which class fills first — weekday morning or evening?'. "
        "Offer to turn their answer into a ready-made artifact: post, reply template, GBP update. "
        "This is the ASK-THE-MERCHANT lever — high scoring, rarely used by competitors."
    ),
    "winback_eligible": (
        "Lead with a specific metric of what they've missed since expiry "
        "(use days_since_expiry + peer avg to estimate missed views/calls). "
        "Make re-subscribing feel instant: 'Back live in 2 minutes — just reply YES.' "
        "Mention the lapsed_customers_added_since_expiry if available — loss aversion."
    ),
    "active_planning_intent": (
        "CRITICAL: Merchant already confirmed. DO NOT ask qualifying questions. "
        "DELIVER the concrete artifact they asked for immediately. "
        "Structure: '[Here is the thing] — [one specific next step]. Confirm to execute.' "
        "End with a single confirm/execute CTA."
    ),
    "seasonal_perf_dip": (
        "Normalize the dip with specific peer data: 'Most {{category}} in {{city}} see this in {{season}}'. "
        "Reframe as the RIGHT time for retention focus (not acquisition). "
        "Give ONE specific retention action they can take THIS WEEK (use existing offers or lapsed customer count). "
        "Reference high_risk_count or lapsed customer data for specificity."
    ),
    "gbp_unverified": (
        "Name the specific CTR uplift % that verified profiles get vs unverified (use peer data). "
        "Tell them the EXACT verification path: 'Google will send a postcard to {locality} OR "
        "you can verify by phone call if eligible.' "
        "Loss-aversion: 'Unverified profiles rank below verified in local search.' Single CTA."
    ),
    "recall_due": (
        "Customer-facing. Open with the merchant's name (send_as = merchant_on_behalf). "
        "Name the specific service + exact months since last visit. "
        "Offer the exact available slots from the trigger payload matching their preferred time. "
        "Include the real catalog price. Use hi-en mix if customer language_pref says so. "
        "Multi-choice slot booking is ALLOWED here."
    ),
    "customer_lapsed_soft": (
        "Customer-facing. No-shame, warm tone. "
        "Reference something specific about their history (services_received, lifetime_value). "
        "Name a new or relevant service/offer they haven't tried. "
        "Single no-obligation CTA: 'Interested? Just reply and I'll hold a slot.'"
    ),
    "customer_lapsed_hard": (
        "Customer-facing. Acknowledge the gap with warmth, not guilt. "
        "Give a SPECIFIC new reason to return (new class, new offer, new capability — from digest or signals). "
        "No-commitment framing: 'No pressure — just wanted to share this.'"
    ),
    "trial_followup": (
        "Customer-facing. Reference the specific trial they did (from services_received). "
        "Give ONE specific next session slot with the price from the catalog. "
        "Mention the wedding_date if available and count days remaining. Single YES CTA."
    ),
    "chronic_refill_due": (
        "Customer-facing. List the specific molecule names + estimated runout date from payload. "
        "Show the price and any applicable offers. "
        "If free_delivery applies, mention it. Reply CONFIRM CTA."
    ),
    "appointment_tomorrow": (
        "Customer-facing. Confirm date + time + specific service name. "
        "Add one prep reminder relevant to the service (what to bring, what to avoid). "
        "Keep it warm and brief — this is a confirmation, not a sales message."
    ),
    "supply_alert": (
        "URGENT compliance message. Name exact batch numbers + molecule from payload. "
        "Use the customer_aggregate to state how many of THEIR customers are likely affected. "
        "Do NOT panic — be clinical and solution-focused. "
        "Offer: 'I can draft the customer notification + outline the replacement workflow — say GO.'"
    ),
    "category_seasonal": (
        "Name 2 specific demand shifts with numbers from seasonal_beats or trend_signals. "
        "Suggest ONE concrete shelf or service action for each. Quick wins only — no strategy lectures."
    ),
    "wedding_package_followup": (
        "Customer-facing. Reference the specific trial they did + the wedding date + days remaining. "
        "Suggest the next program with specific price from offer_catalog. "
        "Single booking CTA with a specific slot time."
    ),
    "competitor_opened": (
        "Acknowledge the new competitor (distance from payload) without naming if not in context. "
        "Lead with THIS merchant's strongest verified differentiators "
        "(rating, review count, established_year, specific services). "
        "Frame as opportunity: 'This is actually good timing to — [specific action].' "
        "End with effort-externalization: 'I can update your GBP to highlight [differentiator] — say GO.'"
    ),
}


def _build_customer_block(customer: Optional[dict]) -> str:
    if not customer:
        return "None — this is a merchant-facing message."
    idn = customer.get("identity", {})
    rel = customer.get("relationship", {})
    prefs = customer.get("preferences", {})
    consent = customer.get("consent", {})
    # Calculate months since last visit if possible
    last_visit = rel.get("last_visit", "")
    return (
        f"Name: {idn.get('name', '?')}\n"
        f"Language pref: {idn.get('language_pref', 'en')}\n"
        f"Age band: {idn.get('age_band', '?')}\n"
        f"Customer state: {customer.get('state', '?')}\n"
        f"First visit: {rel.get('first_visit', '?')}\n"
        f"Last visit: {last_visit}\n"
        f"Total visits: {rel.get('visits_total', '?')}\n"
        f"Services received (history): {rel.get('services_received', [])}\n"
        f"Lifetime value: ₹{rel.get('lifetime_value', '?')}\n"
        f"Preferred slots: {prefs.get('preferred_slots', '?')}\n"
        f"Preferred stylist/provider: {prefs.get('preferred_stylist', 'none specified')}\n"
        f"Channel: {prefs.get('channel', 'whatsapp')}\n"
        f"Consent scope: {consent.get('scope', [])}\n"
        f"Wedding date (if applicable): {prefs.get('wedding_date', 'N/A')}"
    )


def _build_social_proof_block(category: dict, merchant: dict) -> str:
    """Build a social proof paragraph the LLM can use directly in the message."""
    peer_stats = category.get("peer_stats", {})
    slug = category.get("slug", "")
    locality = merchant.get("identity", {}).get("locality", "")
    city = merchant.get("identity", {}).get("city", "")
    perf = merchant.get("performance", {})

    lines = []
    avg_reviews = peer_stats.get("avg_review_count", peer_stats.get("avg_reviews", 0))
    merchant_reviews = perf.get("review_count", 0)

    # Generic social proof sentences the LLM can adapt
    lines.append(
        f"Peer benchmark ({slug}, {city}): avg_ctr={peer_stats.get('avg_ctr', 0):.1%}, "
        f"avg_calls={peer_stats.get('avg_calls_30d', peer_stats.get('avg_calls', '?'))}/mo, "
        f"avg_reviews={avg_reviews}, avg_photos={peer_stats.get('avg_photos', '?')}, "
        f"avg_post_freq={peer_stats.get('avg_post_freq_days', '?')} days"
    )
    lines.append(
        f"Social proof framing examples: "
        f"'Most {slug} in {locality or city} post every {peer_stats.get('avg_post_freq_days', 14)} days'; "
        f"'Top-rated {slug} in {city} average {avg_reviews} reviews'; "
        f"'3 {slug} in your area activated their {slug} offer this month'"
    )
    return "\n".join(lines)


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Main composition entry point.
    Returns dict with keys: body, cta, send_as, suppression_key, rationale
    """
    slug = category.get("slug", "unknown")
    voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})
    digest = category.get("digest", [])
    seasonal_beats = category.get("seasonal_beats", [])
    trend_signals = category.get("trend_signals", [])
    content_library = category.get("patient_content_library", [])

    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    all_offers = merchant.get("offers", [])
    active_offers = [o["title"] for o in all_offers if o.get("status") == "active"]
    expired_offers = [o["title"] for o in all_offers if o.get("status") == "expired"]
    signals = merchant.get("signals", [])
    convo = merchant.get("conversation_history", [])
    review_themes = merchant.get("review_themes", [])
    cust_agg = merchant.get("customer_aggregate", {})

    # Last conversation with engagement tag
    convo_summary = []
    last_engagement = "(no recent conversation)"
    if convo:
        for t in convo[-3:]:
            convo_summary.append(
                f"[{t.get('from','?')} @ {t.get('ts','')}]: {t.get('body','')[:120]}"
            )
        last_engagement = convo[-1].get("engagement", "") if convo else ""
    else:
        convo_summary = ["(no recent conversation)"]

    # CTR comparison
    peer_ctr = peer_stats.get("avg_ctr", 0.03)
    merchant_ctr = perf.get("ctr", 0)
    ctr_delta = round((merchant_ctr - peer_ctr) / peer_ctr * 100, 0) if peer_ctr else 0
    ctr_vs_peer = (
        f"BELOW peer by {abs(ctr_delta):.0f}% ({merchant_ctr:.1%} vs peer {peer_ctr:.1%})"
        if ctr_delta < -5 else
        f"ABOVE peer by {ctr_delta:.0f}% ({merchant_ctr:.1%} vs peer {peer_ctr:.1%})"
        if ctr_delta > 5 else
        f"AT peer median ({merchant_ctr:.1%})"
    )

    # Calls comparison
    peer_calls = peer_stats.get("avg_calls_30d", peer_stats.get("avg_calls", 0))
    merchant_calls = perf.get("calls", 0)
    if peer_calls and merchant_calls:
        calls_delta = round((merchant_calls - peer_calls) / peer_calls * 100, 0)
        calls_vs_peer = (
            f"BELOW peer by {abs(calls_delta):.0f}% ({merchant_calls} vs peer avg {peer_calls})"
            if calls_delta < -10 else
            f"ABOVE peer by {calls_delta:.0f}% ({merchant_calls} vs peer avg {peer_calls})"
            if calls_delta > 10 else
            f"AT peer avg ({merchant_calls} vs {peer_calls})"
        )
    else:
        calls_vs_peer = f"{merchant_calls} calls/mo"

    trigger_kind = trigger.get("kind", "")
    kind_guidance = KIND_GUIDANCE.get(trigger_kind, "Compose a relevant, specific, compelling message.")

    # Find the relevant digest item if trigger references one
    top_item_id = (
        trigger.get("payload", {}).get("top_item_id")
        or trigger.get("payload", {}).get("digest_item_id")
    )
    relevant_digest = []
    if top_item_id:
        relevant_digest = [d for d in digest if d.get("id") == top_item_id]
    if not relevant_digest:
        relevant_digest = digest[:3]  # top 3 by default

    # Content library — pick 1-2 most relevant items
    relevant_content = content_library[:2]

    # Format review themes for prompt
    review_themes_fmt = []
    for rt in review_themes:
        review_themes_fmt.append(
            f"theme='{rt.get('theme')}', sentiment={rt.get('sentiment')}, "
            f"occurrences={rt.get('occurrences_30d', 0)}x in 30d, "
            f"example quote: '{rt.get('common_quote', '')}'"
        )

    prompt = COMPOSE_PROMPT.format(
        slug=slug,
        voice_tone=voice.get("tone", ""),
        voice_register=voice.get("register", ""),
        vocab_allowed=voice.get("vocab_allowed", [])[:8],
        taboos=voice.get("vocab_taboo", []),
        salutation_examples=voice.get("salutation_examples", []),
        offer_catalog=[o["title"] for o in category.get("offer_catalog", [])[:6]],
        peer_stats={k: v for k, v in peer_stats.items() if k in (
            "avg_ctr", "avg_rating", "avg_review_count", "avg_reviews",
            "avg_views_30d", "avg_calls_30d", "avg_photos", "avg_post_freq_days",
            "retention_6mo_pct", "scope"
        )},
        digest=relevant_digest,
        content_library=relevant_content,
        seasonal_beats=seasonal_beats,
        trend_signals=trend_signals[:3],
        merchant_id=merchant.get("merchant_id", ""),
        merchant_name=identity.get("name", ""),
        owner_name=identity.get("owner_first_name", ""),
        city=identity.get("city", ""),
        locality=identity.get("locality", ""),
        established_year=identity.get("established_year", ""),
        verified=identity.get("verified", False),
        languages=identity.get("languages", ["en"]),
        subscription=merchant.get("subscription", {}),
        views=perf.get("views", 0),
        calls=perf.get("calls", 0),
        directions=perf.get("directions", 0),
        ctr=round(perf.get("ctr", 0) * 100, 2),
        leads=perf.get("leads", 0),
        peer_avg_views=peer_stats.get("avg_views_30d", "?"),
        peer_avg_calls=peer_stats.get("avg_calls_30d", peer_stats.get("avg_calls", "?")),
        peer_avg_reviews=peer_stats.get("avg_review_count", peer_stats.get("avg_reviews", "?")),
        peer_avg_photos=peer_stats.get("avg_photos", "?"),
        peer_ctr=round(peer_ctr * 100, 2),
        ctr_vs_peer=ctr_vs_peer,
        calls_vs_peer=calls_vs_peer,
        delta_views_pct=round(perf.get("delta_7d", {}).get("views_pct", 0) * 100, 0),
        delta_calls_pct=round(perf.get("delta_7d", {}).get("calls_pct", 0) * 100, 0),
        active_offers=active_offers or ["(none)"],
        expired_offers=expired_offers or ["(none)"],
        signals=signals,
        convo_history=convo_summary,
        convo_engagement=last_engagement,
        cust_total=cust_agg.get("total_unique_ytd", 0),
        cust_lapsed=cust_agg.get("lapsed_180d_plus", 0),
        cust_retention=round(cust_agg.get("retention_6mo_pct", 0) * 100, 0),
        high_risk_count=cust_agg.get("high_risk_adult_count", 0),
        review_themes=review_themes_fmt or ["(none)"],
        trigger_id=trigger.get("id", ""),
        trigger_kind=trigger_kind,
        trigger_source=trigger.get("source", ""),
        urgency=trigger.get("urgency", 1),
        trigger_payload=json.dumps(trigger.get("payload", {}), ensure_ascii=False),
        suppression_key=trigger.get("suppression_key", ""),
        expires_at=trigger.get("expires_at", ""),
        customer_block=_build_customer_block(customer),
        social_proof_block=_build_social_proof_block(category, merchant),
        kind_guidance=kind_guidance,
    )

    try:
        raw = _llm_complete(SYSTEM_PROMPT, prompt, temperature=0.0)
        # strip markdown fences if any
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw.strip())
        # extract first JSON object if there's surrounding text
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            raw = m.group()
        result = json.loads(raw)
    except Exception as e:
        # fallback — rule-based message
        result = _fallback_compose(category, merchant, trigger, customer)
        result["rationale"] += f" [LLM error: {e}]"

    # Ensure suppression_key is always set
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get(
            "suppression_key", f"msg:{merchant.get('merchant_id')}:{trigger_kind}"
        )

    # Ensure send_as is correct
    if customer and not result.get("send_as"):
        result["send_as"] = "merchant_on_behalf"
    elif not result.get("send_as"):
        result["send_as"] = "vera"

    return result


def _fallback_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    """Rule-based fallback when LLM fails."""
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name", "there")
    kind = trigger.get("kind", "")
    slug = category.get("slug", "")

    if customer:
        cname = customer.get("identity", {}).get("name", "")
        mname = identity.get("name", "")
        body = f"Hi {cname}, {mname} here. It's been a while since your last visit. Reply YES to book a slot this week."
        return {
            "body": body, "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": "Fallback: customer re-engagement message."
        }

    body = f"Hi {name}, quick update from Vera — there's a {kind.replace('_', ' ')} relevant to your {slug} business. Reply YES to know more."
    return {
        "body": body, "cta": "binary_yes_no",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": f"Fallback: generic {kind} message for {slug}."
    }
