# Vera Bot — magicpin AI Challenge Submission

## Live Bot URL
```
https://vera-bot-production-7e46.up.railway.app
```

---

## For the Evaluator — How to Test

The bot is live on Railway.

### Quick health check
```bash
curl https://vera-bot-production-7e46.up.railway.app/v1/healthz
```
Expected response:
```json
{"status": "ok", "uptime_seconds": 1234, "contexts_loaded": {...}}
```

### Run the full judge simulator
Point `judge_simulator.py` at the live URL and run it:

```python
# In judge_simulator.py, set:
BOT_URL = "https://vera-bot-production-7e46.up.railway.app"
LLM_PROVIDER = "openai"   # or whichever provider you use
LLM_API_KEY = "your-key"
```

```bash
python judge_simulator.py
```

### All 5 endpoints are live
| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/healthz` | GET | Liveness check |
| `/v1/metadata` | GET | Bot identity |
| `/v1/context` | POST | Push category / merchant / customer / trigger data |
| `/v1/tick` | POST | Wake bot — returns proactive messages to send |
| `/v1/reply` | POST | Send merchant reply — bot responds with next action |

---

## Approach

**4-context LLM composer with trigger-kind dispatch.**

Every message is composed from the full `(category, merchant, trigger, customer?)` quadruple.
The composer dispatches to a **kind-specific guidance block** per trigger kind — `research_digest`,
`regulation_change`, `recall_due`, `competitor_opened` etc. each get different instructions
on what compulsion lever to use and what structure to follow.

### Key design decisions

1. **Trigger-kind routing** — 20 trigger kinds each have a dedicated prompt guidance paragraph.
   A `research_digest` message leads with source + trial N + patient segment.
   A `recall_due` customer message leads with slot options + price.
   A `perf_dip` message leads with the exact metric delta and a concrete fix.

2. **Specificity by construction** — the prompt instructs the LLM to anchor on verifiable numbers
   from the context (CTR vs peer median, review count, batch numbers, days until expiry).
   If the number isn't in the context, it doesn't get used — no hallucination.

3. **Multi-turn handler** — `conversation_handlers.py` uses regex pattern matching before any LLM call:
   - Auto-reply detected → 1st: polite flag, 2nd: 24h wait, 3rd: end
   - Hostile / opt-out → immediate `end`
   - Positive intent ("yes", "let's do it", "kar do") → switches to action mode immediately
   - Off-topic (GST, loans) → polite redirect back to topic

4. **Merchant-level auto-reply tracking** — the judge sends each auto-reply turn on a different
   `conversation_id`. The bot tracks consecutive auto-replies at the merchant level so the
   3-strike pattern is detected correctly regardless of conv_id changes.

5. **Adaptive to new context** — `/v1/context` stores the latest version atomically.
   The tick handler always reads the freshest payload at compose time. New digest items
   injected mid-test are used in the next composition automatically.

6. **Suppression dedup** — suppression keys tracked per session; duplicate triggers skipped.

### LLM
- Provider: Groq (`llama-3.3-70b-versatile`)
- Temperature: 0 (deterministic)
- Fallback: rule-based message if LLM call fails

### What I'd improve with more time
- Retrieval over digest items: embed all digest items, retrieve by cosine similarity to trigger payload
- Per-merchant conversation memory across sessions (Redis persistence)
- Social proof lever: "3 dentists in your locality did X this month" — currently underused

---

## Running Locally (optional)

```bash
# 1. Clone the repo
git clone https://github.com/CrowdContract/magicpin_bot.git
cd magicpin_bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env — add GROQ_API_KEY

# 4. Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8080

# 5. Test it
curl http://localhost:8080/v1/healthz
```

---

## Judge Simulator Results (against live Railway URL)

```
[PASS] healthz
[PASS] metadata — Team: Vera AI, Model: llama-3.3-70b-versatile
[PASS] category/dentists
[PASS] category/gyms
[PASS] category/pharmacies
[PASS] category/restaurants
[PASS] category/salons
[PASS] merchant/001 through 005
[PASS] auto_reply  — Turn 1: polite flag → Turn 2: wait 24h → Turn 3: ENDED
[PASS] intent      — switched to action mode on "Ok lets do it"
[PASS] hostile     — ended immediately on opt-out
```
