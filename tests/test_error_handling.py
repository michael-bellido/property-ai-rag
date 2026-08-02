"""Unit tests for the friendly-error mapping used when a Groq/LLM call
fails (bad API key, rate limit, network hiccup). This is what stands
between a visitor on the public demo and a raw Python traceback — see
_friendly_llm_error() and its use in handle_question() in app.py.
Pure string-matching logic, no LLM call or network needed.
"""
import app


def test_maps_401_to_auth_error():
    error = Exception("Error code: 401 - {'error': {'code': 'invalid_api_key'}}")
    assert app._friendly_llm_error(error) == app.t("llm_error_auth")


def test_maps_429_to_rate_limit_error():
    error = Exception("Error code: 429 - {'error': {'code': 'rate_limit_exceeded'}}")
    assert app._friendly_llm_error(error) == app.t("llm_error_rate_limit")


def test_maps_unknown_error_to_generic_message():
    error = Exception("Connection reset by peer")
    assert app._friendly_llm_error(error) == app.t("llm_error_generic")


def test_never_leaks_raw_exception_text_to_the_user():
    # Whatever the underlying error says, the message shown to the user
    # must be one of the canned, friendly strings — never the raw
    # exception text, which could contain internal details.
    raw = "Traceback: something internal broke at line 42"
    result = app._friendly_llm_error(Exception(raw))
    assert raw not in result
