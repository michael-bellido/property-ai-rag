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
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = ROOT / "chroma_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """You are PropertyBot, a helpful assistant for Sunset Realty Group, \
a fictional real-estate agency in Tenerife. Answer the user's question using ONLY \
the context provided below (property listings and FAQ). If the answer is not in \
the context, say you don't have that information and suggest contacting the agency \
directly — do not make up prices, listings, or policies. Keep answers concise and \
mention specific listing IDs when relevant.

Context:
{context}
"""


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vector_store():
    if not PERSIST_DIR.exists():
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0.2)


def retrieve_context(vector_store, query: str, k: int = 4) -> str:
    results = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in results)


def main():
    st.set_page_config(page_title="PropertyBot — Sunset Realty Group", page_icon="🏡")
    st.title("🏡 PropertyBot")
    st.caption(
        "RAG demo chatbot for a fictional real-estate agency. "
        "All listings and FAQ content are synthetic placeholder data."
    )

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
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm PropertyBot. Ask me about our Tenerife listings "
                    "(price, location, bedrooms, features) or our buying/renting "
                    "process for non-residents."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about a property or our services..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if llm is None:
                answer = (
                    "I can't generate an answer yet because no GROQ_API_KEY is "
                    "configured. Once it's set, I'll be able to respond using the "
                    "retrieved listings/FAQ context below."
                )
                context = retrieve_context(vector_store, prompt)
                with st.expander("Retrieved context (would be sent to the LLM)"):
                    st.text(context)
            else:
                context = retrieve_context(vector_store, prompt)
                messages = [
                    ("system", SYSTEM_PROMPT.format(context=context)),
                    ("human", prompt),
                ]
                with st.spinner("Thinking..."):
                    response = llm.invoke(messages)
                answer = response.content
                with st.expander("Sources used"):
                    st.text(context)

            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
