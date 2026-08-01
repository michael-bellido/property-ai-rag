"""Unit tests for the ingestion pipeline's document-building functions
(app/ingest.py). These only exercise plain file I/O and text splitting —
no embeddings get computed, so no model download and no GROQ_API_KEY are
needed to run them. build_vector_store() itself (which does compute
embeddings) is intentionally NOT covered here, to keep this test suite
fast enough to run on every push.
"""
import json

import ingest


def test_load_listings_returns_one_document_per_listing():
    with open(ingest.DATA_DIR / "listings.json", encoding="utf-8") as f:
        raw_listings = json.load(f)

    docs = ingest.load_listings()
    assert len(docs) == len(raw_listings)
    assert len(docs) > 0


def test_load_listings_document_content_includes_key_fields():
    docs = ingest.load_listings()
    first = docs[0]

    assert first.metadata["source"] == "listings"
    assert "listing_id" in first.metadata
    assert first.metadata["listing_id"] in first.page_content
    assert "Price: €" in first.page_content


def test_load_faq_splits_into_multiple_chunks():
    chunks = ingest.load_faq()
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "faq" for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_load_faq_chunks_stay_close_to_the_configured_size():
    chunks = ingest.load_faq()
    # chunk_size=500 in the splitter is a soft target, not a hard cap, so
    # allow some slack instead of asserting an exact <= 500 everywhere.
    assert all(len(chunk.page_content) <= 600 for chunk in chunks)
