"""
Property AI — a Retrieval-Augmented Generation (RAG) chat assistant for a
fictional real-estate agency ("Sunset Realty Group").

The user asks questions in natural language (e.g. "What villas do you have
under 500k?" or "Can non-residents get a mortgage?"). The app retrieves the
most relevant chunks from a local Chroma vector store (built by ingest.py)
and passes them to an LLM (via Groq's free-tier API) as context, so answers
are grounded in the actual listings/FAQ data instead of the model's own
guesses.

Run:
    streamlit run app/app.py

Requires a GROQ_API_KEY environment variable (see .env.example).

This file is intentionally thin: it wires together the pieces defined in
the sibling modules below and owns only the two things that don't belong
anywhere else — handle_question() (per-turn orchestration) and main() (the
Streamlit page itself). Everything else lives in a focused module:
    config.py  — paths, model names, tunable limits, logger
    ui.py      — bilingual UI text, CSS, and every render_*() function
    prompts.py — the system prompt, language detection, conversation memory
    rag.py     — vector store loading, retrieval, source formatting
    llm.py     — LLM client construction, friendly error mapping, session cap
"""
import inspect

import streamlit as st

from config import QUESTION_LIMIT_PER_SESSION, logger
from llm import _friendly_llm_error, _has_reached_session_limit, get_llm
from prompts import (
    SYSTEM_PROMPT,
    _bounded_conversation_messages,
    build_language_directive,
    condense_follow_up_question,
)
from rag import load_vector_store, retrieve_context
from ui import (
    CUSTOM_CSS,
    MINIMAL_CHAT_INPUT_CSS,
    render_bubble,
    render_conversation_header,
    render_landing,
    render_loading_screen,
    render_sidebar,
    render_sources_expander,
    t,
)


def handle_question(prompt, vector_store, llm):
    # Per-session question cap check FIRST, before any retrieval or LLM
    # work — a blocked question shouldn't cost an embedding call either.
    # Shown as a toast (not a chat bubble) and never written to
    # session_state.messages, so it doesn't linger in the conversation or
    # get replayed to the LLM as fake history on a later question.
    questions_asked = st.session_state.get("questions_asked", 0)
    if _has_reached_session_limit(questions_asked):
        st.toast(
            t("session_limit_message").format(limit=QUESTION_LIMIT_PER_SESSION),
            icon="🚦",
        )
        return
    st.session_state.questions_asked = questions_asked + 1

    # Snapshot prior turns BEFORE appending the current question, so the
    # conversation-memory helpers below see only what was already
    # discussed, not this new question appearing twice.
    prior_history = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        render_bubble(prompt, "user")

    with st.chat_message("assistant"):
        if llm is None:
            answer = (
                "I can't generate an answer yet because no GROQ_API_KEY is "
                "configured. Once it's set, I'll be able to respond using the "
                "retrieved listings/FAQ context below."
            )
            context, sources = retrieve_context(vector_store, prompt)
        else:
            # A bare follow-up ("what about cheaper ones?") retrieves badly
            # on its own — condense it into a standalone query first, using
            # the recent conversation for anything it depends on.
            search_query = condense_follow_up_question(llm, prior_history, prompt)
            context, sources = retrieve_context(vector_store, search_query)
            messages = [
                (
                    "system",
                    SYSTEM_PROMPT.format(
                        context=context,
                        language_directive=build_language_directive(prompt),
                    ),
                ),
                *_bounded_conversation_messages(prior_history),
                ("human", prompt),
            ]
            with st.spinner(t("thinking")):
                try:
                    response = llm.invoke(messages)
                    answer = response.content
                except Exception as e:
                    # Never let a raw API exception (bad/expired key, Groq
                    # rate limit, network hiccup) reach the chat UI as a
                    # stack trace — log the real error server-side and show
                    # the visitor a short, friendly message instead.
                    logger.exception("LLM call failed for question: %r", prompt)
                    answer = _friendly_llm_error(e)

        render_bubble(answer, "assistant")
        if sources:
            render_sources_expander(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


def main():
    st.set_page_config(page_title="Property AI — Sunset Realty Group", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_sidebar()

    if "app_loaded" not in st.session_state:
        loading_slot = st.empty()
        with loading_slot.container():
            render_loading_screen()
        vector_store = load_vector_store()
        loading_slot.empty()
        st.session_state.app_loaded = True
    else:
        vector_store = load_vector_store()

    if vector_store is None:
        st.error(
            "Couldn't load or build the vector store. If you're running this "
            "locally, run `python app/ingest.py` first, then reload this page. "
            "If this is a hosted deploy, check the app logs for the underlying "
            "error."
        )
        st.stop()

    llm = get_llm()
    if llm is None:
        st.warning(
            "No GROQ_API_KEY found. Set it in a `.env` file (see `.env.example`) "
            "or as an environment variable to enable chat responses. "
            "Get a free key at https://console.groq.com/keys."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    started = len(st.session_state.messages) > 0

    if started:
        render_conversation_header()
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                render_bubble(message["content"], message["role"])
                if message["role"] == "assistant" and message.get("sources"):
                    render_sources_expander(message["sources"])
    else:
        render_landing()

    pending = st.session_state.pop("pending_question", None)
    if pending:
        handle_question(pending, vector_store, llm)

    st.markdown(
        f'<div style="text-align:center; color:#7a7a7a; font-size:0.78rem; '
        f'margin: 0.4rem 0 0.2rem 0;">{t("helper_text")}</div>',
        unsafe_allow_html=True,
    )
    # Must stay at the top level of main() — not inside a container, column,
    # tab, or sidebar — so Streamlit pins it to the bottom on its own.
    #
    # `width="stretch"` / `height="content"` let the native widget handle
    # sizing and auto-growth itself (no more hand-patched min/max-height on
    # an internal textarea), but both parameters only exist on Streamlit
    # >=1.56.0. Rather than hard-requiring that version, detect support at
    # runtime via inspect.signature and only pass what the installed
    # st.chat_input actually accepts — this way the app works whether the
    # environment has been upgraded yet or not, instead of crashing with a
    # TypeError on older installs. Check the installed version/signature
    # with: python -c "import streamlit as st, inspect; print(st.__version__); print(inspect.signature(st.chat_input))"
    #
    # The only remaining CSS (ui.MINIMAL_CHAT_INPUT_CSS) just centers and
    # caps the width of the stable `.st-key-property_chat` wrapper; it
    # never touches stBottom, stChatInput, BaseWeb internals, the textarea,
    # or the button.
    st.html(MINIMAL_CHAT_INPUT_CSS)
    chat_input_kwargs = {
        "key": "property_chat",
        "max_chars": 1500,
    }
    supported_params = inspect.signature(st.chat_input).parameters
    if "width" in supported_params:
        chat_input_kwargs["width"] = "stretch"
    if "height" in supported_params:
        chat_input_kwargs["height"] = "content"
    if prompt := st.chat_input(t("chat_placeholder"), **chat_input_kwargs):
        handle_question(prompt, vector_store, llm)


if __name__ == "__main__":
    main()
