"""Unit tests for the dependency-free EN/ES question-language heuristic.

This is what fixes the bug where the LLM (a small, free-tier 8B model)
would sometimes answer in Spanish even when the question was clearly in
English: instead of trusting the model's own language detection, app.py
detects the question's language in Python and injects an explicit,
per-question directive naming that language. These tests run instantly —
no LLM call, no network, no API key required.
"""
import app


def test_detects_clearly_english_question():
    assert app.guess_question_language("Can I buy property as a non-resident?") == "English"


def test_detects_clearly_spanish_question():
    assert app.guess_question_language("¿Puedo comprar una propiedad siendo no residente?") == "Spanish"


def test_detects_spanish_from_accented_characters_alone():
    # "Está" and "mañana" alone are enough of a signal even before any
    # stopword scoring kicks in.
    assert app.guess_question_language("¿Está disponible mañana?") == "Spanish"


def test_ambiguous_short_input_returns_none():
    # "ok" has zero hits in either stopword list and no signal characters —
    # guessing wrong here would be worse than admitting we don't know.
    assert app.guess_question_language("ok") is None


def test_empty_or_blank_input_returns_none():
    assert app.guess_question_language("") is None
    assert app.guess_question_language("   ") is None


def test_build_language_directive_names_the_detected_language():
    directive = app.build_language_directive("What villas do you have with a pool?")
    assert "English" in directive


def test_build_language_directive_falls_back_when_undetectable():
    directive = app.build_language_directive("ok")
    assert "same language" in directive.lower()
