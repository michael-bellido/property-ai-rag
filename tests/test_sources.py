"""Unit tests for _build_sources(), which turns retrieved Chroma documents
into the small dicts the "Sources used" expander renders. Uses a minimal
fake document instead of a real Chroma result — the function only ever
touches .page_content and .metadata, so nothing Chroma-specific is needed
to exercise it.
"""
import app


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


def test_listing_source_gets_a_listing_id_title():
    docs = [FakeDoc("Listing SR-101: Sea-View Apartment...", {"source": "listings", "listing_id": "SR-101"})]
    sources = app._build_sources(docs)
    assert sources[0]["title"] == "Listing SR-101"
    assert sources[0]["url"] is None


def test_faq_source_gets_a_generic_title():
    docs = [FakeDoc("Viewings can be scheduled Monday to Saturday...", {"source": "faq"})]
    sources = app._build_sources(docs)
    assert sources[0]["title"] == "Agency FAQ"


def test_unknown_source_falls_back_to_generic_title():
    docs = [FakeDoc("Some text.", {})]
    sources = app._build_sources(docs)
    assert sources[0]["title"] == "Knowledge base"


def test_long_excerpt_gets_truncated_with_ellipsis():
    long_text = "word " * 100  # well over the 160-char excerpt limit
    sources = app._build_sources([FakeDoc(long_text, {"source": "faq"})])
    excerpt = sources[0]["excerpt"]
    assert len(excerpt) <= 160
    assert excerpt.endswith("...")


def test_excerpt_collapses_internal_whitespace():
    docs = [FakeDoc("Line one.\n\n  Line   two.", {"source": "faq"})]
    sources = app._build_sources(docs)
    assert sources[0]["excerpt"] == "Line one. Line two."


def test_builds_one_source_per_document_in_order():
    docs = [
        FakeDoc("A", {"source": "listings", "listing_id": "SR-1"}),
        FakeDoc("B", {"source": "faq"}),
    ]
    sources = app._build_sources(docs)
    assert len(sources) == 2
    assert sources[0]["title"] == "Listing SR-1"
    assert sources[1]["title"] == "Agency FAQ"
