"""
Retrieval layer: loading the Chroma vector store, running similarity
search, and formatting retrieved documents into the small dicts the UI
renders as "Sources used". No prompt text or LLM-calling logic lives
here — see prompts.py and llm.py for that.
"""
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, PERSIST_DIR


@st.cache_resource(show_spinner=False)
def load_vector_store():
    if not PERSIST_DIR.exists():
        # chroma_store/ is gitignored on purpose (it's a derived artifact
        # rebuilt from data/, not source data) — locally that's fine
        # because the README has you run `python app/ingest.py` once. A
        # fresh hosted deploy (e.g. Streamlit Community Cloud) has no way
        # to run that separate step first, so build it here on first load
        # instead. @st.cache_resource means this only runs once per running
        # app instance, not per user session; the branded loading screen
        # in app.main() is what covers this extra first-run latency.
        try:
            from ingest import build_vector_store

            build_vector_store()
        except Exception:
            return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


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
