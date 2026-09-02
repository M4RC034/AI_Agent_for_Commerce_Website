"""Runtime configuration for the agentic customer-service CLI.

Paths are resolved relative to the repository root so the CLI works from any
working directory. Nothing here reads the network or loads a model — import
cost stays near zero so the eval harness can import config without paying for
the embedding model.
"""

from __future__ import annotations

import os
from datetime import date
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Mirrors the source project's convention (backend/main.py loads both), so a key
# placed in either location works. Values already in the environment win.
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "api_key.env")
load_dotenv(BASE_DIR / "backend" / "api_key.env")

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

#: Sonnet 5 ($2/$10 per MTok) is the default: customer service is a tool-routing
#: and grounding task, not a hard-reasoning one, and Opus 5 ($5/$25) buys depth
#: this workload does not use.
#:
#: One behavioural consequence, not just a price change — Sonnet 5 does NOT
#: support mid-conversation ``role: "system"`` messages, so the entity-ledger and
#: step-budget operator notes fall back to a tagged user turn. See
#: SUPPORTS_MIDCONV_SYSTEM below and agent/memory.py._operator_message.
#:
#: Override with AGENT_MODEL. Note that on the Claude API *older* models are not
#: cheaper: Sonnet 4.5/4.6 cost $3/$15 while the newer Sonnet 5 costs $2/$10.
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5")

#: Per-turn output ceiling. Customer-service answers are short; the loop makes
#: several calls per turn, so a lower cap keeps latency predictable.
MAX_TOKENS = 8_000

#: Models that accept adaptive thinking and `output_config.effort`. Older models
#: (Haiku 4.5, Sonnet 4.5 and earlier) reject both with a 400 — they use the
#: retired `budget_tokens` form instead. Since AGENT_MODEL lets any model be
#: swapped in, these two parameters have to follow the model rather than be
#: hardcoded to the Opus 5 defaults.
_ADAPTIVE_THINKING_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

SUPPORTS_ADAPTIVE = MODEL.startswith(_ADAPTIVE_THINKING_MODELS)

#: Mid-conversation ``role: "system"`` messages are a narrower feature than
#: adaptive thinking — notably NOT available on Sonnet 5. Where unsupported the
#: operator note falls back to a tagged user message (see agent/memory.py).
_MIDCONV_SYSTEM_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
)

SUPPORTS_MIDCONV_SYSTEM = MODEL.startswith(_MIDCONV_SYSTEM_MODELS)

#: Customer service is not a hard-reasoning workload. `medium` keeps tool
#: selection sharp without paying for depth the task never uses. `None` on
#: models that do not accept the parameter, in which case it is omitted.
EFFORT = "medium" if SUPPORTS_ADAPTIVE else None

#: Thinking is on by default on Opus 5, and `display` defaults to "omitted" —
#: which shows up in a streaming CLI as a long silent pause before the first
#: token, exactly the hang FR18 forbids. Summarised thinking gives the renderer
#: something to show; the spinner covers the rest. `None` on older models, where
#: thinking is simply left off — the loop omits the parameter entirely.
THINKING = {"type": "adaptive", "display": "summarized"} if SUPPORTS_ADAPTIVE else None

# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

#: FR13. Enough for the three-tool return chain plus two retrieval attempts and
#: a final answer; tight enough that a cycle is caught within one turn.
MAX_STEPS = 8

#: FR3 guard. Self-correcting retrieval is desirable; unbounded re-querying is
#: not. Enforced inside the retrieval tool, per user turn.
MAX_RETRIEVAL_ATTEMPTS = 3

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class Mode(str, Enum):
    """FR5 / NFR4 — selected once at startup, never branched on inside the loop."""

    RAG = "rag"
    CAG = "cag"


#: Swapped from the source project's `all-MiniLM-L6-v2`, which has effectively
#: no Chinese capability (FR22). The wrapper around it is unchanged.
EMBED_MODEL = os.environ.get("AGENT_EMBED_MODEL", "BAAI/bge-m3")

#: Fallback if bge-m3's ~2.2GB download is impractical. Requires the
#: "query: " / "passage: " prefixes — see retrieval/embedder.py.
EMBED_MODEL_FALLBACK = "intfloat/multilingual-e5-base"

#: Swapped from `ms-marco-MiniLM-L-6-v2`, which reranks English pairs only.
RERANK_MODEL = os.environ.get("AGENT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

#: Two-stage retrieve-then-rerank, inherited from the source project's design:
#: cast wide with the bi-encoder, then let the cross-encoder slice precisely.
RETRIEVE_K = 15
RERANK_TOP_N = 4

#: FR4 thresholds on the reranker's score, calibrated against 18 real zh-TW
#: probes — see scripts/calibrate_thresholds.py, which reproduces the numbers.
#:
#:   covered queries (10)   0.635 – 0.993   min 0.635
#:   deliberate KB gaps (5) 0.0003 – 0.042  max 0.042
#:   out-of-scope (3)       0.0000 – 0.003  max 0.003
#:
#: bge-reranker-v2-m3 separates these by ~0.59, so the thresholds sit in a wide
#: empty band rather than being tuned to the edge of either class. SUFFICIENT is
#: set well below the weakest covered query (0.635) to tolerate phrasings the
#: probe set does not cover, and ~10x above the strongest gap query (0.042) so
#: AC4 cannot pass by accident.
SUFFICIENT_THRESHOLD = 0.40
PARTIAL_THRESHOLD = 0.12

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

KB_DIR = BASE_DIR / "data" / "kb"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
KB_INDEX_PATH = PROCESSED_DIR / "kb.index"
KB_CHUNKS_PATH = PROCESSED_DIR / "kb_chunks.jsonl"
PRODUCT_INDEX_PATH = PROCESSED_DIR / "products.index"
PRODUCT_META_PATH = PROCESSED_DIR / "products.jsonl"
FIXTURES_PATH = BASE_DIR / "data" / "fixtures" / "orders.json"

#: Candidates returned by image search. Recall@5 measured at 91.5% on a 2,000
#: product catalog; the agent sees several and picks, rather than being handed
#: a single answer it cannot sanity-check.
PRODUCT_TOP_K = 5
LOG_DIR = BASE_DIR / "logs"

# --------------------------------------------------------------------------
# Business policy constants
# --------------------------------------------------------------------------
# These mirror data/kb/*.md. The KB is what the agent quotes to the customer;
# these are what the tools compute with. They must agree — evals/run_evals.py
# checks that the refund figures the agent states match what the tools return.

RETURN_WINDOW_DAYS = 7
RESTOCKING_FEE_RATE = 0.05
FREE_SHIPPING_THRESHOLD = 1_000
STANDARD_SHIPPING_FEE = 80
REFUND_ETA_BUSINESS_DAYS = 7

#: Reasons the merchant is at fault: full refund including shipping, no fee.
MERCHANT_FAULT_REASONS = frozenset({"商品瑕疵", "與描述不符", "運送破損"})


# --------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------


def today() -> "date":
    """Today's date, pinnable for reproducible evals.

    FR10 wants fixed fixtures, but return-window arithmetic needs a real clock —
    absolute fixture dates would drift out of the retention window within a
    week and stop exercising the eligibility logic. The fixtures therefore store
    day offsets, and this is the single point where "now" enters the system.
    Set ``AGENT_TODAY=2026-09-01`` to pin it.

    Note this is deliberately *not* interpolated into the system prompt — that
    would invalidate the prompt cache on every request. It reaches the model
    only through tool return values.
    """
    pinned = os.environ.get("AGENT_TODAY")
    return date.fromisoformat(pinned) if pinned else date.today()
