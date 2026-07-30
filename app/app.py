"""
PropertyBot — a Retrieval-Augmented Generation (RAG) chat assistant for a
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
"""
import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = ROOT / "chroma_store"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
REPO_URL = "https://github.com/michael-bellido/propertybot-rag"

# =========================================================
# BILINGUAL UI TEXT (EN default / ES) — single source of truth for every
# static string in the interface, so a stray mixed-language string can't
# happen again. "PropertyBot" is a brand name and is deliberately identical
# in both languages — never translate it.
# =========================================================

DEFAULT_LANGUAGE = "en"

UI_TEXT_BY_LANG = {
    "en": {
        "app_name": "PropertyBot",
        "sidebar_heading": "Try asking",
        "clear_chat": "Clear chat",
        "welcome_title": "Welcome to PropertyBot.",
        "welcome_subtitle": (
            "Ask about our Tenerife listings or our buying/renting process"
        ),
        "conversation_subtitle": (
            "Ask about our Tenerife listings or our buying/renting process — "
            "every answer is grounded in real (fictional) agency data."
        ),
        "helper_text": (
            "Try a question above, or ask anything about our listings"
        ),
        "chat_placeholder": "Ask about a property or our services...",
        "sources": "Sources used",
        "source_label": "Source",
        "thinking": "Thinking...",
        "loading_text": "Loading knowledge base…",
    },
    "es": {
        "app_name": "PropertyBot",
        "sidebar_heading": "Prueba a preguntar",
        "clear_chat": "Borrar chat",
        "welcome_title": "Bienvenido a PropertyBot.",
        "welcome_subtitle": (
            "Pregunta sobre nuestras propiedades en Tenerife o el proceso de compra/alquiler"
        ),
        "conversation_subtitle": (
            "Pregunta sobre nuestras propiedades en Tenerife o el proceso de compra/alquiler — "
            "cada respuesta se basa en datos reales (ficticios) de la agencia."
        ),
        "helper_text": (
            "Prueba una pregunta arriba, o pregunta lo que quieras sobre nuestras propiedades"
        ),
        "chat_placeholder": "Pregunta sobre una propiedad o nuestros servicios...",
        "sources": "Fuentes usadas",
        "source_label": "Fuente",
        "thinking": "Pensando...",
        "loading_text": "Cargando base de conocimiento…",
    },
}

# Canonical list of example questions per language. The sidebar and the two
# landing-page marquee rows are all derived from this one list (by index)
# instead of retyping the strings, which is what let a stray Spanish string
# drift out of sync with the rest of the (English-default) UI before.
SUGGESTIONS_BY_LANG = {
    "en": [
        "What villas do you have with a pool?",          # 0
        "Can I buy property as a non-resident?",          # 1
        "What are the rental management fees?",           # 2
        "Show me something under €200,000",                # 3
        "Tell me about the Golf del Sur duplex",           # 4
        "Show me apartments in Costa Adeje",               # 5
        "What's your cheapest listing?",                   # 6
        "Do you have anything with a sea view?",           # 7
        "How long does the buying process take?",          # 8
        "What extra costs should I budget for?",           # 9
        "Can I book a virtual viewing?",                   # 10
    ],
    "es": [
        "¿Qué villas tenéis con piscina?",                          # 0
        "¿Puedo comprar propiedad siendo no residente?",             # 1
        "¿Cuáles son las tarifas de gestión de alquiler?",           # 2
        "Muéstrame algo por menos de €200,000",                      # 3
        "Cuéntame sobre el dúplex de Golf del Sur",                  # 4
        "Muéstrame apartamentos en Costa Adeje",                     # 5
        "¿Cuál es vuestra propiedad más económica?",                 # 6
        "¿Tenéis algo con vistas al mar?",                           # 7
        "¿Cuánto dura el proceso de compra?",                        # 8
        "¿Qué costes adicionales debería tener en cuenta?",          # 9
        "¿Puedo reservar una visita virtual?",                       # 10
    ],
}


def current_lang() -> str:
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def t(key: str) -> str:
    """Look up a UI string in the currently selected language."""
    return UI_TEXT_BY_LANG[current_lang()][key]


def get_suggestions() -> list[str]:
    return SUGGESTIONS_BY_LANG[current_lang()]


SYSTEM_PROMPT = """You are PropertyBot, a helpful assistant for Sunset Realty Group, \
a fictional real-estate agency in Tenerife. Answer the user's question using ONLY \
the context provided below (property listings and FAQ). If the answer is not in \
the context, say you don't have that information and suggest contacting the agency \
directly — do not make up prices, listings, or policies. Keep answers concise and \
mention specific listing IDs when relevant.

Language requirements:
- Always respond in English, even if the user writes in Spanish or another language.
- Translate any Spanish-language context into natural English in your answer.
- Never mix English and Spanish in the same response.
- Property, street, and place names may stay in their original form.

Context:
{context}
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Page background: pure black — including the bottom-pinned input bar */
[data-testid="stAppViewContainer"] > .main {
    background: #000000;
}
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
body {
    background-color: #000000 !important;
}

/* Sidebar — minimal, monochrome */
[data-testid="stSidebar"] {
    background-color: #000000;
    border-right: 1px solid #2a2a2a;
}
/* "PropertyBot" lives in the sidebar's real header, next to the built-in
   collapse arrow, instead of as a separate title element pushed down with
   manual margins. */
[data-testid="stSidebarHeader"] {
    height: 72px !important;
    padding: 0 22px !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stSidebarHeader"]::before {
    content: "PropertyBot";
    margin-right: auto;
    color: #ffffff;
    font-size: 20px;
    font-weight: 700;
    line-height: 1;
}
[data-testid="stSidebarCollapseButton"] {
    position: static !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    background-color: #0a0a0a;
    border: 1px solid #3a3a3a;
    color: #d8d8d8;
    border-radius: 10px;
    font-size: 0.82rem;
    padding: 0.5rem 0.7rem;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #ffffff;
    color: #ffffff;
}

/* EN / ES language toggle — compact, centered pills, current language
   shown as a disabled (filled) button instead of a separate active state */
.st-key-lang_en button,
.st-key-lang_es button {
    text-align: center !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}
.st-key-lang_en button:disabled,
.st-key-lang_es button:disabled {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #ffffff !important;
    opacity: 1 !important;
}

/* Hide Streamlit's auto-generated heading anchor-link icon everywhere */
[data-testid="stHeaderActionElements"],
[data-testid="stElementToolbar"],
a.anchor-link {
    display: none !important;
}

/* Hide Streamlit's top decoration bar (was showing the old orange theme color) */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Streamlit's default focus/active color is orange/red — force green everywhere */
button:focus,
button:focus-visible,
button:active {
    box-shadow: 0 0 0 1px #4CAF6D !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stButton button:focus,
[data-testid="stSidebar"] .stButton button:active {
    border-color: #4CAF6D !important;
    color: #ffffff !important;
}

/* Compact header (shown once a conversation has started) */
.pb-hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.0rem;
    margin-bottom: 0.1rem;
    color: #F2ECE4;
}
.pb-hero p {
    color: #B8ADA0;
    font-size: 0.95rem;
    margin-top: 0;
}
.pb-divider {
    height: 1px;
    background: linear-gradient(90deg, #4CAF6D 0%, transparent 80%);
    margin: 1.1rem 0 1.4rem 0;
    border: none;
}

/* Landing / welcome screen */
.pb-landing-wrap {
    text-align: center;
    padding: 2.2rem 0 1.6rem 0;
}
.pb-landing-wrap h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    color: #F2ECE4;
    margin-bottom: 0.5rem;
}
.pb-landing-wrap p {
    color: #B8ADA0;
    font-size: 1.05rem;
    margin: 0;
}

/* Typewriter effect on the landing-page title: types the full title out,
   pauses with it fully visible, then deletes it back down — forever.

   Each character of the title gets its OWN tiny <span> and its OWN
   per-character @keyframes rule (generated in render_landing(), since the
   exact percentages depend on the title's length). Every character
   snaps — instantly, via `steps(1, end)` — from `max-width: 0` (not
   rendered at all) to `max-width: 2em` (i.e. its own natural width, since
   2em is comfortably larger than any single glyph) at its own scheduled
   moment, and back to 0 when it's "deleted". Because the transition for
   each character is a single instant jump rather than a value smoothly
   sliding across a whole multi-character string, no glyph is ever caught
   mid-reveal/mid-delete — and because the revealed width is always that
   character's own real, browser-computed size, this works correctly with
   the proportional Fraunces font (no per-character width estimate is
   needed at all). The wrapping .pb-typewriter is NOT itself animated —
   its width is simply whatever its currently-visible children add up to,
   so the border-right cursor naturally sits right after the last visible
   character with no separate tracking needed. The blink animates
   `border-color`, not `opacity` — opacity would fade the whole element,
   including all the character spans inside it, making the text itself
   flicker along with the cursor. border-color only affects the
   border-right line, so the text stays solid and only the cursor blinks. */
.pb-typewriter {
    display: inline-block;
    white-space: nowrap;
    vertical-align: bottom;
    border-right: 0.08em solid #F2ECE4;
    animation: pb-caret-blink 0.75s step-end infinite;
}
.pb-char {
    display: inline-block;
    overflow: hidden;
    vertical-align: bottom;
    white-space: pre;
    max-width: 0;
}
@keyframes pb-caret-blink {
    50% { border-color: transparent; }
}

/* Infinite scrolling question marquee */
.pb-marquee {
    overflow: hidden;
    position: relative;
    width: 100%;
    -webkit-mask-image: linear-gradient(90deg, transparent, black 8%, black 92%, transparent);
    mask-image: linear-gradient(90deg, transparent, black 8%, black 92%, transparent);
}
.pb-marquee-track {
    display: flex;
    width: max-content;
    gap: 0.6rem;
    padding: 0.35rem 0;
}
.pb-marquee-track.pb-left { animation: pb-scroll-left 34s linear infinite; }
.pb-marquee-track.pb-right { animation: pb-scroll-right 34s linear infinite; }
@keyframes pb-scroll-left {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}
@keyframes pb-scroll-right {
    from { transform: translateX(-50%); }
    to { transform: translateX(0); }
}
.pb-chip {
    flex: 0 0 auto;
    background: #000000;
    border: 1px solid #3a3a3a;
    color: #d8d8d8;
    border-radius: 999px;
    padding: 0.45rem 1rem;
    font-size: 0.82rem;
    white-space: nowrap;
}

/* Chat message row: no background of its own, no avatar icon — the bubble (below) carries the styling */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"] {
    display: none !important;
}

/* Message bubble — sized to its own content, not the full row width */
.pb-bubble-row {
    display: flex;
    width: 100%;
}
.pb-bubble-row-user {
    justify-content: flex-end;
}
.pb-bubble-row-assistant {
    justify-content: flex-start;
}
.pb-bubble {
    display: inline-block;
    max-width: 75%;
    width: fit-content;
    background-color: #0a0a0a;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 0.6rem 0.95rem;
    color: #e5e5e5;
    line-height: 1.4rem;
    font-size: 0.92rem;
}

/* Chat input — the width fix targets Streamlit's own full-width bottom
   container (stBottom) and its direct child wrapper, not stChatInput
   itself: stChatInput always stretches to fill its parent, so narrowing
   stChatInput alone left the *container* full-width and the input just
   sat off-center inside it. Narrowing the container is what actually works. */

/* Full-width bottom container: make it a flex row so its child can be
   centered inside it */
div[data-testid="stBottom"] {
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: flex-end !important;
    background: transparent !important;
    padding: 0 !important;
}
/* The direct child wrapper — this is what gets sized and centered */
div[data-testid="stBottom"] > div {
    width: 60vw !important;
    max-width: 820px !important;
    min-width: 0 !important;
    margin: 0 0 22px 0 !important;
    padding: 0 !important;
}
/* The input itself just fills that sized wrapper */
div[data-testid="stBottom"] div[data-testid="stChatInput"] {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    background: transparent !important;
}
@media (max-width: 800px) {
    div[data-testid="stBottom"] > div {
        width: calc(100vw - 32px) !important;
        max-width: none !important;
    }
}
/* ===== MAIN CONTAINER =====
   Compact start, grows upward as the user types, caps out around 5-6
   lines, then scrolls internally. Single background + single border on
   the outer container only — no more double pill/inner-div layering. */
div[data-testid="stChatInput"] {
    position: relative !important;
    min-height: 56px !important;
    max-height: 190px !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: #212121 !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 28px !important;
    box-shadow: none !important;
}
/* Strip Streamlit's own internal borders/backgrounds so only the outer
   container above shows */
div[data-testid="stChatInput"] [data-baseweb="textarea"],
div[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: #4CAF6D !important;
    box-shadow: none !important;
}

/* ===== TEXT ===== */
textarea[data-testid="stChatInputTextArea"],
textarea[data-testid="stChatInput"],
div[data-testid="stChatInput"] textarea {
    min-height: 56px !important;
    max-height: 170px !important;
    box-sizing: border-box !important;
    resize: none !important;
    overflow-y: auto !important;
    background: transparent !important;
    color: #f4f4f4 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-size: 16px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    line-height: 24px !important;
    /* room for the text on the left and the send button on the right */
    padding: 16px 64px 16px 20px !important;
    margin: 0 !important;
    scrollbar-width: thin;
    scrollbar-color: #565656 transparent;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: #8e8e9f !important;
    opacity: 1 !important;
}

/* ===== SEND BUTTON — pinned to the bottom-right corner ===== */
div[data-testid="stChatInput"] button {
    position: absolute !important;
    right: 10px !important;
    bottom: 10px !important;
    width: 36px !important;
    min-width: 36px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    border-radius: 50% !important;
    background: #ffffff !important;
    color: #000000 !important;
    box-shadow: none !important;
    z-index: 10 !important;
}
div[data-testid="stChatInput"] button:hover {
    background: #e8e8e8 !important;
}
div[data-testid="stChatInput"] button:disabled {
    background: #3a3a3a !important;
    color: #8b8b8b !important;
    opacity: 1 !important;
}
div[data-testid="stChatInput"] button svg {
    width: 21px !important;
    height: 21px !important;
    color: inherit !important;
    fill: currentColor !important;
}

/* ===== HIDE THE 0/1500 CHARACTER COUNTER ===== */
div[data-testid="stChatInput"] [data-testid="stTextAreaCharacterCount"],
div[data-testid="stChatInput"] [aria-live="polite"],
div[data-testid="stChatInput"] small {
    display: none !important;
}

/* ===== INNER SCROLLBAR ===== */
div[data-testid="stChatInput"] textarea::-webkit-scrollbar {
    width: 6px;
}
div[data-testid="stChatInput"] textarea::-webkit-scrollbar-track {
    background: transparent;
    margin-top: 18px;
    margin-bottom: 18px;
}
div[data-testid="stChatInput"] textarea::-webkit-scrollbar-thumb {
    background: #565656;
    border-radius: 999px;
}

/* Professional branded loading screen */
.pb-loading-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 7rem 0 6rem 0;
    text-align: center;
}
.pb-loading-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.5rem;
    color: #FFFFFF;
    letter-spacing: 0.02em;
    margin-bottom: 1.6rem;
}
.pb-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #2a2a2a;
    border-radius: 50%;
    animation: pb-pulse 1.1s ease-in-out infinite;
    margin-bottom: 1rem;
}
@keyframes pb-pulse {
    0%, 100% { opacity: 0.35; transform: scale(0.92); }
    50% { opacity: 1; transform: scale(1); }
}
.pb-loading-text {
    color: #8a8a8a;
    font-size: 0.82rem;
    letter-spacing: 0.01em;
}

/* ===== Chat input accent — white, not green (overrides earlier rules;
   kept last in the stylesheet so it wins on specificity ties) ===== */
div[data-testid="stChatInput"] {
    border: 1px solid #FFFFFF !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: #FFFFFF !important;
    outline: none !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] [data-baseweb="textarea"],
div[data-testid="stChatInput"] [data-baseweb="textarea"] > div,
div[data-testid="stChatInput"] textarea {
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea {
    caret-color: #FFFFFF !important;
}
div[data-testid="stChatInput"] button:not(:disabled) {
    background: #FFFFFF !important;
    color: #000000 !important;
    border: none !important;
}
div[data-testid="stChatInput"] button:not(:disabled):hover {
    background: #E8E8E8 !important;
}
div[data-testid="stChatInput"] button:disabled {
    background: #3A3A3A !important;
    color: #A1A1A1 !important;
    border: none !important;
}

/* "Sources used" expander — st.expander instead of st.popover, since a
   popover picks its own open direction (up or down depending on space)
   with no way to force it, while an expander always opens downward,
   inline, inside the assistant's own chat_message. */
div[data-testid="stChatMessage"] div[data-testid="stExpander"] {
    width: fit-content !important;
    min-width: 180px !important;
    max-width: 520px !important;
    margin-top: 8px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stChatMessage"] div[data-testid="stExpander"] details {
    border: none !important;
    background: transparent !important;
}
div[data-testid="stChatMessage"] div[data-testid="stExpander"] summary {
    width: fit-content !important;
    min-height: 32px !important;
    padding: 5px 12px !important;
    background: transparent !important;
    color: #a8a8a8 !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 999px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}
div[data-testid="stChatMessage"] div[data-testid="stExpander"] summary:hover {
    color: #ffffff !important;
    background: #171717 !important;
    border-color: #666666 !important;
}
div[data-testid="stChatMessage"] div[data-testid="stExpander"] details > div {
    width: 420px !important;
    max-width: calc(100vw - 48px) !important;
    margin-top: 7px !important;
    padding: 14px !important;
    background: #111111 !important;
    border: 1px solid #343434 !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}
div[data-testid="stChatMessage"] div[data-testid="stExpander"] hr {
    border-color: #2f2f2f !important;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_vector_store():
    if not PERSIST_DIR.exists():
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


def render_loading_screen():
    st.markdown(
        f"""
        <div class="pb-loading-wrap">
            <div class="pb-loading-title">{t('app_name')}</div>
            <div class="pb-spinner"></div>
            <div class="pb-loading-text">{t('loading_text')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0.2)


def _build_sources(docs) -> list[dict]:
    """Turn retrieved Chroma documents into the small dicts the sources
    popover renders (title / url / excerpt). These are demo listings with
    no real product page, so url is always None."""
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        if meta.get("source") == "listings":
            title = f"Listing {meta.get('listing_id', '')}".strip()
        elif meta.get("source") == "faq":
            title = "Agency FAQ"
        else:
            title = "Knowledge base"
        excerpt = " ".join(doc.page_content.split())
        if len(excerpt) > 160:
            excerpt = excerpt[:157].rstrip() + "..."
        sources.append({"title": title, "url": None, "excerpt": excerpt})
    return sources


def retrieve_context(vector_store, query: str, k: int = 4) -> tuple[str, list[dict]]:
    results = vector_store.similarity_search(query, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in results)
    return context, _build_sources(results)


def render_sources_expander(sources: list[dict]):
    with st.expander(f"{t('sources')} · {len(sources)}", expanded=False):
        for index, source in enumerate(sources, start=1):
            title = source.get("title") or f"{t('source_label')} {index}"
            url = source.get("url")
            excerpt = source.get("excerpt")

            if url:
                st.markdown(f"**{index}. [{title}]({url})**")
            else:
                st.markdown(f"**{index}. {title}**")

            if excerpt:
                st.caption(excerpt)

            if index < len(sources):
                st.divider()


def render_sidebar():
    with st.sidebar:
        # Language selector — only the interface text (this file's
        # UI_TEXT_BY_LANG / SUGGESTIONS_BY_LANG) switches with it. The
        # assistant's own answers stay in English regardless (see
        # SYSTEM_PROMPT), so this only affects "what you see", not
        # "what the bot writes back". "PropertyBot" itself never translates.
        lang = current_lang()
        col_en, col_es = st.columns(2)
        with col_en:
            if st.button("EN", key="lang_en", disabled=(lang == "en"), use_container_width=True):
                st.session_state.language = "en"
                st.rerun()
        with col_es:
            if st.button("ES", key="lang_es", disabled=(lang == "es"), use_container_width=True):
                st.session_state.language = "es"
                st.rerun()

        st.markdown("---")

        st.markdown(f"**{t('sidebar_heading')}**")
        for q in get_suggestions()[:4]:
            if st.button(q, key=f"sidebar_{q}"):
                st.session_state.pending_question = q

        st.markdown("---")
        if st.button(t("clear_chat"), key="clear_chat"):
            st.session_state.messages = []
            st.session_state.pop("pending_question", None)
            st.rerun()


def render_marquee_row(questions, direction, key):
    chips = "".join(f'<span class="pb-chip">{q}</span>' for q in questions)
    st.markdown(
        f'<div class="pb-marquee"><div class="pb-marquee-track {direction}">{chips}{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_landing():
    title = t("welcome_title")
    # ~220ms per character while typing/deleting, plus a long pause with the
    # full title visible before it starts erasing. These three fractions of
    # the loop must sum to 1.0 and must match the percentages used below
    # when building each character's individual keyframes.
    seconds_per_char = 0.22
    type_fraction = 0.35
    hold_fraction = 0.30
    delete_fraction = 1.0 - type_fraction - hold_fraction
    char_count = max(len(title), 1)
    duration = round((char_count * seconds_per_char) / type_fraction, 2)

    # One @keyframes rule per character: each one instantly snaps that
    # single character between hidden (max-width: 0) and its own natural
    # width (max-width: 2em, comfortably larger than any real glyph) at
    # its own scheduled moment, then instantly back to hidden. Typing
    # sweeps left to right; deleting sweeps right to left (last-typed
    # character disappears first), like an actual backspace.
    keyframes = []
    chars_html = []
    for i, ch in enumerate(title):
        show_pct = round((i / char_count) * type_fraction * 100, 3)
        hide_pct = round(
            (type_fraction + hold_fraction + ((char_count - 1 - i) / char_count) * delete_fraction) * 100,
            3,
        )
        hide_pct_end = min(hide_pct + 0.1, 100)
        keyframes.append(
            f"""@keyframes pb-char-{i} {{
                0% {{ max-width: 0; }}
                {show_pct}% {{ max-width: 0; }}
                {show_pct}% {{ max-width: 2em; }}
                {hide_pct}% {{ max-width: 2em; }}
                {hide_pct_end}% {{ max-width: 0; }}
                100% {{ max-width: 0; }}
            }}"""
        )
        safe_char = html.escape(ch)
        chars_html.append(
            f'<span class="pb-char" style="animation: pb-char-{i} {duration}s steps(1, end) infinite;">{safe_char}</span>'
        )

    st.markdown(
        f"""
        <style>
        {"".join(keyframes)}
        </style>
        <div class="pb-landing-wrap">
            <h1><span class="pb-typewriter">{"".join(chars_html)}</span></h1>
            <p>{t('welcome_subtitle')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = get_suggestions()
    marquee_row_1 = [suggestions[0], suggestions[5], suggestions[6], suggestions[7], suggestions[4]]
    marquee_row_2 = [suggestions[1], suggestions[2], suggestions[8], suggestions[9], suggestions[10]]
    render_marquee_row(marquee_row_1, "pb-left", "row1")
    st.write("")
    render_marquee_row(marquee_row_2, "pb-right", "row2")
    st.write("")


def render_conversation_header():
    st.markdown(
        f"""
        <div class="pb-hero">
            <h1>{t('app_name')}</h1>
            <p>{t('conversation_subtitle')}</p>
        </div>
        <hr class="pb-divider" />
        """,
        unsafe_allow_html=True,
    )


def render_bubble(text: str, role: str = "assistant"):
    safe = html.escape(text).replace("\n", "<br>")
    row_class = "pb-bubble-row-user" if role == "user" else "pb-bubble-row-assistant"
    st.markdown(
        f'<div class="pb-bubble-row {row_class}"><div class="pb-bubble">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def handle_question(prompt, vector_store, llm):
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
            context, sources = retrieve_context(vector_store, prompt)
            messages = [
                ("system", SYSTEM_PROMPT.format(context=context)),
                ("human", prompt),
            ]
            with st.spinner(t("thinking")):
                response = llm.invoke(messages)
            answer = response.content

        render_bubble(answer, "assistant")
        if sources:
            render_sources_expander(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


def main():
    st.set_page_config(page_title="PropertyBot — Sunset Realty Group", layout="wide")
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
            "No vector store found. Run `python app/ingest.py` first to build it, "
            "then reload this page."
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
    # tab, or sidebar — so Streamlit pins it to the bottom on its own. Width
    # and centering are handled purely via CSS on [data-testid="stChatInput"]
    # (see CUSTOM_CSS above).
    if prompt := st.chat_input(
        t("chat_placeholder"),
        key="property_chat",
        max_chars=1500,
    ):
        handle_question(prompt, vector_store, llm)


if __name__ == "__main__":
    main()
