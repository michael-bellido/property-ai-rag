"""
Everything about safely calling the LLM: constructing the Groq client,
translating a raw API exception into a short user-facing message instead
of a stack trace, and the per-session question cap that protects this
demo's shared, free-tier API quota.
"""
import os

from langchain_groq import ChatGroq

from config import LLM_MODEL, QUESTION_LIMIT_PER_SESSION
from ui import t


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0.2)


def _friendly_llm_error(error: Exception) -> str:
    """Turn a raw Groq/LLM exception into a short, user-facing message
    instead of letting a stack trace reach the chat UI. Matches on the
    exception's own text rather than importing groq's exception classes
    directly, so this keeps working even if the underlying HTTP client
    library's class names change. The real exception is always logged
    server-side by the caller — this only controls what the visitor sees."""
    text = str(error).lower()
    if "401" in text or "invalid_api_key" in text or "authentication" in text:
        return t("llm_error_auth")
    if "429" in text or "rate_limit" in text or "rate limit" in text:
        return t("llm_error_rate_limit")
    return t("llm_error_generic")


def _has_reached_session_limit(
    questions_asked: int, limit: int = QUESTION_LIMIT_PER_SESSION
) -> bool:
    """Pure predicate behind the per-session question cap — takes a plain
    count instead of touching st.session_state directly so it's trivially
    unit-testable."""
    return questions_asked >= limit
