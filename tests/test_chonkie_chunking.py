"""
TDD tests for Chonkie SemanticChunker integration in KeyDocIngester.

These tests define the desired behavior BEFORE implementation:
1. SemanticChunker produces chunks with semantic boundaries
2. Heading pre-split is preserved (heading metadata in chunks)
3. Config reads new keys (chunk_token_size, similarity_threshold)
4. Graceful fallback when chonkie is unavailable
5. Empty/short text handled correctly
"""

from unittest.mock import patch, MagicMock

from fastcode.doc_ingester import KeyDocIngester


class _DummyEmbedder:
    def embed_text(self, text: str):
        if not text:
            return None
        return [0.1, 0.2, 0.3]


def _make_config(**overrides):
    """Build a docs_integration config with sensible defaults."""
    cfg = {
        "docs_integration": {
            "enabled": True,
            "curated_paths": ["docs/design/**"],
            "allow_paths": [],
            "deny_paths": [],
            "chunk_token_size": 512,
            "similarity_threshold": 0.5,
        }
    }
    cfg["docs_integration"].update(overrides)
    return cfg


# ---- Test 1: SemanticChunker produces semantic chunks ----

def test_chunk_document_produces_semantic_boundaries():
    """SemanticChunker should split at semantic boundaries, not arbitrary word counts."""
    text = """# Architecture Overview

The system uses a microservices pattern with event-driven communication.
Each service owns its data and exposes APIs through a gateway.

## Data Flow

Requests enter through the API gateway which routes to appropriate services.
The gateway handles authentication, rate limiting, and request transformation.
Services communicate asynchronously via a message broker.

## Storage Layer

We use PostgreSQL for relational data and Redis for caching.
Vector embeddings are stored in pgvector with HNSW indexing.
Full-text search uses GIN indexes on materialized document views.
"""
    ingester = KeyDocIngester(_make_config(), _DummyEmbedder())
    chunks = ingester._chunk_document(text)

    # Should produce multiple chunks
    assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"

    # Each chunk must have required keys
    for chunk in chunks:
        assert "text" in chunk
        assert "heading" in chunk
        assert "start_line" in chunk
        assert "end_line" in chunk
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) > 0

    # Chunks should preserve heading metadata
    headings = [c["heading"] for c in chunks if c["heading"]]
    assert len(headings) >= 1, "At least one chunk should have heading metadata"


def test_chunk_document_does_not_split_mid_sentence():
    """Semantic chunking should respect sentence boundaries."""
    # A single paragraph with clear sentences
    text = (
        "First sentence is about initialization. "
        "Second sentence covers configuration loading. "
        "Third sentence describes validation. "
        "Fourth sentence handles error reporting. "
        "Fifth sentence manages cleanup. "
        "Sixth sentence deals with shutdown procedures."
    )
    ingester = KeyDocIngester(_make_config(chunk_token_size=20), _DummyEmbedder())
    chunks = ingester._chunk_document(text)

    # No chunk should end mid-word or mid-sentence in a broken way
    for chunk in chunks:
        text = chunk["text"].strip()
        if text:
            # Should end with sentence-ending punctuation or be the last chunk
            assert text[-1] in ".!?\n", (
                f"Chunk ends without sentence boundary: '...{text[-30:]}'"
            )


# ---- Test 2: Heading pre-split preserved ----

def test_heading_preserved_in_chunk_metadata():
    """Chunks under a heading should carry that heading in metadata."""
    text = """# Main Title

Content under main title.

## Subsection A

Content under subsection A.

## Subsection B

Content under subsection B.
"""
    ingester = KeyDocIngester(_make_config(), _DummyEmbedder())
    chunks = ingester._chunk_document(text)

    # At least one chunk should reference each heading
    headings = {c["heading"] for c in chunks if c["heading"]}
    assert "Main Title" in headings, f"'Main Title' not in {headings}"
    assert "Subsection A" in headings, f"'Subsection A' not in {headings}"
    assert "Subsection B" in headings, f"'Subsection B' not in {headings}"


def test_line_numbers_are_valid():
    """Chunk start_line and end_line should be within document bounds."""
    text = """# Title

First paragraph.

## Section

Second paragraph.
Third line.
"""
    ingester = KeyDocIngester(_make_config(), _DummyEmbedder())
    chunks = ingester._chunk_document(text)

    total_lines = len(text.splitlines())
    for chunk in chunks:
        assert 1 <= chunk["start_line"] <= total_lines, (
            f"start_line {chunk['start_line']} out of range [1, {total_lines}]"
        )
        assert chunk["start_line"] <= chunk["end_line"], (
            f"start_line {chunk['start_line']} > end_line {chunk['end_line']}"
        )


# ---- Test 3: Config reads new keys ----

def test_chunk_token_size_affects_chunk_granularity():
    """Smaller chunk_token_size should produce more chunks from the same text."""
    text = "Word. ".join(f"Sentence {i} with enough content to matter." for i in range(20))

    ingester_small = KeyDocIngester(_make_config(chunk_token_size=64), _DummyEmbedder())
    ingester_large = KeyDocIngester(_make_config(chunk_token_size=2048), _DummyEmbedder())

    chunks_small = ingester_small._chunk_document(text)
    chunks_large = ingester_large._chunk_document(text)

    # Smaller token size should produce more (or equal) chunks
    assert len(chunks_small) >= len(chunks_large), (
        f"token_size=64 produced {len(chunks_small)} chunks, "
        f"token_size=2048 produced {len(chunks_large)} — smaller size should yield more chunks"
    )


def test_similarity_threshold_affects_split_sensitivity():
    """Higher similarity_threshold should produce fewer splits (harder to break)."""
    text = (
        "Section about databases. PostgreSQL stores relational data. "
        "Redis handles caching layer. "
        "Section about messaging. Kafka processes events. RabbitMQ routes queues."
    )

    ingester_strict = KeyDocIngester(_make_config(similarity_threshold=0.99), _DummyEmbedder())
    ingester_loose = KeyDocIngester(_make_config(similarity_threshold=0.01), _DummyEmbedder())

    chunks_strict = ingester_strict._chunk_document(text)
    chunks_loose = ingester_loose._chunk_document(text)

    # Very high threshold means almost nothing splits; very low means more splits
    assert len(chunks_strict) <= len(chunks_loose), (
        f"threshold=0.99 produced {len(chunks_strict)} chunks, "
        f"threshold=0.01 produced {len(chunks_loose)} — higher threshold should yield fewer chunks"
    )


def test_config_defaults_when_missing():
    """KeyDocIngester should use sensible defaults for new config keys."""
    cfg = {"docs_integration": {"enabled": True}}
    ingester = KeyDocIngester(cfg, _DummyEmbedder())
    # Verify defaults are reasonable numbers (not None, not zero, not negative)
    assert isinstance(ingester.chunk_token_size, int)
    assert ingester.chunk_token_size > 0
    assert isinstance(ingester.similarity_threshold, float)
    assert 0.0 < ingester.similarity_threshold <= 1.0


# ---- Test 4: Fallback when chonkie unavailable ----

def test_fallback_when_chonkie_import_fails():
    """Should fall back to word-based chunking if chonkie import fails."""
    text = "# Title\n\nSome content here. More content follows."

    with patch.dict("sys.modules", {"chonkie": None}):
        ingester = KeyDocIngester(_make_config(), _DummyEmbedder())
        # Force re-initialization to trigger import failure
        ingester._chunker = None
        chunks = ingester._chunk_document(text)

    # Should still produce chunks via fallback
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk
        assert "heading" in chunk


def test_fallback_when_chunker_init_fails():
    """Should fall back to word-based chunking if SemanticChunker init fails."""
    text = "# Title\n\nSome content here. More content follows."

    ingester = KeyDocIngester(_make_config(), _DummyEmbedder())

    # Simulate chunker.chunk() raising an exception
    ingester._chunker = MagicMock()
    ingester._chunker.chunk.side_effect = RuntimeError("model load failed")

    chunks = ingester._chunk_document(text)

    # Should still produce chunks via per-section fallback
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk


# ---- Test 5: Edge cases ----

def test_chunk_empty_text():
    """Empty text should produce no chunks."""
    ingester = KeyDocIngester(_make_config(), _DummyEmbedder())
    chunks = ingester._chunk_document("")
    assert chunks == []


def test_chunk_whitespace_only():
    """Whitespace-only text should produce no chunks."""
    ingester = KeyDocIngester(_make_config(), _DummyEmbedder())
    chunks = ingester._chunk_document("   \n\n  \n  ")
    assert chunks == []


def test_chunk_single_line():
    """A single line of text should produce at least one chunk."""
    ingester = KeyDocIngester(_make_config(), _DummyEmbedder())
    chunks = ingester._chunk_document("Just a single line of text.")
    assert len(chunks) >= 1
    assert chunks[0]["text"].strip() == "Just a single line of text."


def test_chunk_very_long_section():
    """A long section should be split into multiple chunks."""
    # Generate a long text with clear paragraph breaks
    paragraphs = []
    for i in range(20):
        paragraphs.append(
            f"Paragraph {i}: This is a discussion about topic {i}. "
            f"It covers the main points and provides examples. "
            f"The conclusion summarizes the key takeaways."
        )
    text = "\n\n".join(paragraphs)

    ingester = KeyDocIngester(_make_config(chunk_token_size=64), _DummyEmbedder())
    chunks = ingester._chunk_document(text)

    assert len(chunks) >= 2, f"Expected >= 2 chunks for long text, got {len(chunks)}"
