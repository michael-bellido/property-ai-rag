"""Unit tests for the per-session question cap that protects the public
demo's shared, free-tier Groq quota from a single session asking an
unbounded number of questions. See llm._has_reached_session_limit() and
its use in app.handle_question(). Pure integer comparison, no LLM call,
no Streamlit session_state needed.
"""
import llm


def test_zero_questions_asked_has_not_reached_the_limit():
    assert llm._has_reached_session_limit(0, limit=20) is False


def test_below_the_limit_has_not_reached_it():
    assert llm._has_reached_session_limit(19, limit=20) is False


def test_at_the_limit_has_reached_it():
    # The 21st question is blocked once 20 have already been asked and
    # answered — the check runs before a question is counted, so
    # questions_asked == limit means "no more".
    assert llm._has_reached_session_limit(20, limit=20) is True


def test_past_the_limit_has_reached_it():
    assert llm._has_reached_session_limit(25, limit=20) is True


def test_uses_the_apps_default_limit_when_not_overridden():
    just_under = llm.QUESTION_LIMIT_PER_SESSION - 1
    assert llm._has_reached_session_limit(just_under) is False
    assert llm._has_reached_session_limit(llm.QUESTION_LIMIT_PER_SESSION) is True
