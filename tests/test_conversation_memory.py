"""Unit tests for the conversation-memory helpers (app.py's "CONVERSATION
MEMORY" section): bounding + formatting prior turns for the follow-up
question condensation prompt, and for the final answer-generation message
list sent to the LLM. The LLM itself is stubbed out everywhere here, so
these tests run instantly with no Groq API key and no network call.
"""
import app


SAMPLE_HISTORY = [
    {"role": "user", "content": "What villas do you have with a pool?"},
    {
        "role": "assistant",
        "content": "We have Villa Sunset (SR-102) and Villa Azul (SR-105), both with private pools.",
        "sources": [],
    },
]


def test_format_history_for_condensing_labels_each_speaker():
    formatted = app._format_history_for_condensing(SAMPLE_HISTORY)
    assert formatted.startswith("User: What villas do you have with a pool?")
    assert "Assistant: We have Villa Sunset (SR-102) and Villa Azul (SR-105)" in formatted


def test_format_history_for_condensing_truncates_long_turns():
    # A previous answer this long shouldn't be allowed to dominate the
    # (otherwise small) condensation prompt sent to the free-tier model.
    long_history = [{"role": "assistant", "content": "word " * 200}]
    formatted = app._format_history_for_condensing(long_history)
    assert formatted.endswith("...")
    assert len(formatted) < 320


def test_format_history_for_condensing_skips_empty_turns():
    history = [{"role": "user", "content": ""}, {"role": "user", "content": "Hi"}]
    assert app._format_history_for_condensing(history) == "User: Hi"


def test_bounded_conversation_messages_maps_roles_for_the_llm():
    messages = app._bounded_conversation_messages(SAMPLE_HISTORY)
    assert messages == [
        ("human", "What villas do you have with a pool?"),
        ("assistant", "We have Villa Sunset (SR-102) and Villa Azul (SR-105), both with private pools."),
    ]


def test_bounded_conversation_messages_caps_at_max_history_turns():
    # 5 user+assistant exchanges = 10 messages; only the most recent
    # MAX_HISTORY_TURNS exchanges should survive, so a long chat can't
    # blow up the prompt sent to the LLM on every single turn.
    long_history = []
    for i in range(5):
        long_history.append({"role": "user", "content": f"question {i}"})
        long_history.append({"role": "assistant", "content": f"answer {i}", "sources": []})

    messages = app._bounded_conversation_messages(long_history)
    assert len(messages) == app.MAX_HISTORY_TURNS * 2
    first_kept_index = 5 - app.MAX_HISTORY_TURNS
    assert messages[0] == ("human", f"question {first_kept_index}")
    assert messages[-1] == ("assistant", "answer 4")


def test_condense_follow_up_question_skips_llm_when_no_history():
    # No prior turns means there's nothing to condense against — the LLM
    # should never even be called (passing llm=None proves it wasn't).
    result = app.condense_follow_up_question(llm=None, history=[], question="What about cheaper ones?")
    assert result == "What about cheaper ones?"


def test_condense_follow_up_question_falls_back_on_llm_error():
    class ExplodingLLM:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("Groq is down")

    result = app.condense_follow_up_question(ExplodingLLM(), SAMPLE_HISTORY, "What about cheaper ones?")
    assert result == "What about cheaper ones?"


def test_condense_follow_up_question_uses_the_llms_rewrite():
    class StubResponse:
        content = '"What cheaper villas do you have?"'

    class StubLLM:
        def invoke(self, *_args, **_kwargs):
            return StubResponse()

    result = app.condense_follow_up_question(StubLLM(), SAMPLE_HISTORY, "What about cheaper ones?")
    assert result == "What cheaper villas do you have?"
