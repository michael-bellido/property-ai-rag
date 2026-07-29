# PropertyBot — RAG Chatbot for Real Estate

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
property listings and agency policies by retrieving relevant information
from a local knowledge base before generating a response — instead of
relying purely on an LLM's own (often outdated or hallucinated) knowledge.

> **Note on the data:** the property listings and agency FAQ used here are
> entirely fictional, created for demonstration purposes. They are **not**
> derived from any real client, agency, or confidential data. "Sunset Realty
> Group" does not exist.

## Why this project

Most public LLMs know nothing about a specific business's current listings,
prices, or policies, and will either refuse to answer or invent plausible
sounding but wrong details. RAG fixes this by grounding every answer in
retrieved source documents, which is the same pattern used in production
customer-support and internal-knowledge chatbots.

## How it works

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion (run once — ingest.py)"]
        A[listings.json] --> C[Chunk & format documents]
        B[agency_faq.md] --> C
        C --> D[Local embeddings\nsentence-transformers/all-MiniLM-L6-v2]
        D --> E[(Chroma vector store\non disk)]
    end

    subgraph Query["Chat (app.py, Streamlit)"]
        F[User question] --> G[Embed question]
        G --> H[Similarity search]
        E --> H
        H --> I[Top-k relevant chunks]
        I --> J[Prompt: system instructions\n+ retrieved context\n+ user question]
        J --> K[LLM via Groq API\nllama-3.1-8b-instant]
        K --> L[Grounded answer\nshown in chat UI]
    end
```

1. **Ingestion** (`app/ingest.py`, run once): loads the property listings and
   FAQ, splits the FAQ into section-sized chunks, embeds everything locally
   with a HuggingFace `sentence-transformers` model (no API key needed for
   this step), and persists the vectors to a local Chroma database.
2. **Retrieval + generation** (`app/app.py`, Streamlit UI): when the user
   asks a question, the app embeds the question, retrieves the most similar
   chunks from Chroma, and sends them as context to an LLM (Groq's free-tier
   API, model `llama-3.1-8b-instant`) so the answer is grounded in the
   retrieved data rather than the model's own guesses.

## Tech stack

- **LangChain** — orchestration (document loading, splitting, vector store, LLM calls)
- **Chroma** — local vector database
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, free and offline
- **Groq API** (`llama-3.1-8b-instant`) — fast, free-tier LLM inference
- **Streamlit** — chat UI

## Setup

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)
   (no credit card required).

3. Copy `.env.example` to `.env` and paste in your key:

   ```bash
   cp .env.example .env
   ```

4. Build the vector store (run once, or again after editing the data files):

   ```bash
   python app/ingest.py
   ```

5. Launch the chat app:

   ```bash
   streamlit run app/app.py
   ```

## Project structure

```
propertybot-rag/
├── app/
│   ├── ingest.py       # builds the local vector store from data/
│   └── app.py           # Streamlit chat UI (retrieval + LLM call)
├── data/
│   ├── listings.json    # fictional property listings
│   └── agency_faq.md     # fictional agency FAQ
├── requirements.txt
├── .env.example
└── .gitignore
```

## Possible extensions

- Swap Chroma for a hosted vector DB (e.g. Pinecone, Qdrant) for a production deployment.
- Add conversation memory so follow-up questions ("what about cheaper ones?") use prior turns.
- Add source citations with clickable listing links.
- Deploy on Streamlit Community Cloud or a small VPS for a live demo link.

## Disclaimer

This is a portfolio/demo project. The listings, prices, and FAQ content are
synthetic and created solely to illustrate the RAG pattern. Any resemblance
to real properties or agencies is coincidental.
