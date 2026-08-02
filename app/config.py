"""
Shared configuration for Property AI: filesystem paths, model names, and
the small set of tunable limits used across app.py, ingest.py, and
eval.py. Centralizing these here means there's exactly one place to
change, say, the embedding model or the chroma_store location — app.py
and ingest.py used to each redefine their own copy of ROOT/DATA_DIR/
PERSIST_DIR/EMBEDDING_MODEL, which is exactly the kind of duplication
that drifts out of sync silently.
"""
import logging
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = ROOT / "chroma_store"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
REPO_URL = "https://github.com/michael-bellido/property-ai-rag"

# How many prior user+assistant exchanges get sent back to the LLM as
# conversation history for follow-up questions ("what about cheaper ones?").
# Kept small on purpose: LLM_MODEL is a free-tier 8B model, so an unbounded
# chat history would both slow it down and dilute the system prompt's
# grounding instructions the longer a conversation runs.
MAX_HISTORY_TURNS = 3

# Maximum questions a single browser session can ask this demo. This public
# demo shares one free-tier Groq API key across every visitor, so this caps
# how much of that shared quota any one session can consume — past the
# limit, the app stops calling the LLM and shows a friendly message
# instead. It resets the moment someone reloads into a new session, so
# it's a courtesy limit, not a security control. See
# llm._has_reached_session_limit() / app.handle_question().
QUESTION_LIMIT_PER_SESSION = 20

# Real failures (bad/expired API key, Groq rate limits, network hiccups) are
# logged server-side with the full exception so they're visible in hosted
# logs (e.g. Streamlit Community Cloud's log viewer) — the chat UI itself
# only ever shows the short, friendly message built by
# llm._friendly_llm_error, never a raw traceback. See app.handle_question().
logger = logging.getLogger("property_ai")
logging.basicConfig(level=logging.INFO)
