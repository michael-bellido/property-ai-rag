"""
Ingestion pipeline: loads property listings + agency FAQ, chunks them,
embeds them locally (no API key needed for embeddings), and persists
them into a Chroma vector store on disk.

Run once before starting the app:
    python app/ingest.py
"""
import json
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = ROOT / "chroma_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_listings() -> list[Document]:
    """Turn each structured listing into a self-contained text document."""
    with open(DATA_DIR / "listings.json", encoding="utf-8") as f:
        listings = json.load(f)

    docs = []
    for item in listings:
        text = (
            f"Listing {item['id']}: {item['title']}\n"
            f"Location: {item['location']}\n"
            f"Type: {item['type']}\n"
            f"Price: €{item['price_eur']:,}\n"
            f"Bedrooms: {item['bedrooms']} | Bathrooms: {item['bathrooms']} | Size: {item['size_m2']} m2\n"
            f"Features: {', '.join(item['features'])}\n"
            f"Description: {item['description']}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={"source": "listings", "listing_id": item["id"], "price_eur": item["price_eur"]},
            )
        )
    return docs


def load_faq() -> list[Document]:
    """Split the agency FAQ markdown into section-sized chunks."""
    text = (DATA_DIR / "agency_faq.md").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=60, separators=["\n## ", "\n\n", "\n", " "]
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": "faq"}) for chunk in chunks]


def build_vector_store() -> None:
    docs = load_listings() + load_faq()
    print(f"Loaded {len(docs)} documents ({len(load_listings())} listings + {len(load_faq())} FAQ chunks).")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if PERSIST_DIR.exists():
        import shutil

        shutil.rmtree(PERSIST_DIR)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    print(f"Vector store built and persisted to {PERSIST_DIR} ({vector_store._collection.count()} vectors).")


if __name__ == "__main__":
    build_vector_store()
