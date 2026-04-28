"""Tests for doc_ingester module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Never
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.doc_ingester import KeyDocIngester
from fastcode.semantic_ir import IRSnapshot, IRSymbol

# --- Helpers ---


class _DummyEmbedder:
    def embed_text(self, text: str) -> Any:
        if not text:
            return None
        return [0.1, 0.2, 0.3]


class _FakeEmbedder:
    """Minimal embedder that returns fixed vectors."""

    def embed_text(self, text: str) -> Any:
        return [0.1] * 8


def _make_ingester(**overrides) -> Any:
    defaults = {
        "config": {"embedding": {}},
        "embedder": _FakeEmbedder(),
    }
    defaults.update(overrides)
    return KeyDocIngester(**defaults)


def _make_config(**overrides) -> Any:
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


small_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ",
    min_size=1,
    max_size=30,
)


# --- Basic tests ---


def test_doc_ingester_selects_curated_files_and_extracts_mentions():
    with tempfile.TemporaryDirectory(prefix="fc_doc_ingest_") as tmp:
        root = Path(tmp)
        (root / "docs" / "design").mkdir(parents=True, exist_ok=True)
        (root / "docs" / "private").mkdir(parents=True, exist_ok=True)
        (root / "docs" / "design" / "pipeline.md").write_text(
            "# Pipeline Design\nThe BranchIndexer coordinates indexing.\n",
            encoding="utf-8",
        )
        (root / "docs" / "private" / "secret.md").write_text(
            "# Secret\nShould not be ingested.\n", encoding="utf-8"
        )

        cfg = {
            "docs_integration": {
                "enabled": True,
                "curated_paths": ["docs/design/**"],
                "allow_paths": [],
                "deny_paths": ["docs/private/**"],
                "chunk_size": 64,
                "chunk_overlap": 8,
            }
        }
        ingester = KeyDocIngester(cfg, _DummyEmbedder())
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:1",
            symbols=[
                IRSymbol(
                    symbol_id="sym:branchindexer",
                    external_symbol_id=None,
                    path="fastcode/main.py",
                    display_name="BranchIndexer",
                    kind="class",
                    language="python",
                    source_set={"ast"},
                    metadata={"source": "ast"},
                )
            ],
        )
        result = ingester.ingest(
            repo_path=str(root),
            repo_name="repo",
            snapshot_id="snap:repo:1",
            snapshot=snapshot,
        )

        assert result["chunks"]
        assert result["elements"]
        assert all(e["type"] == "design_document" for e in result["elements"])
        assert all("docs/design/" in e["relative_path"] for e in result["elements"])
        assert all(
            "docs/private/" not in e["relative_path"] for e in result["elements"]
        )
        assert any(m["symbol_id"] == "sym:branchindexer" for m in result["mentions"])


# --- Chonkie chunking tests ---


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


def test_chunk_token_size_affects_chunk_granularity():
    """Smaller chunk_token_size should produce more chunks from the same text."""
    text = "Word. ".join(
        f"Sentence {i} with enough content to matter." for i in range(20)
    )

    ingester_small = KeyDocIngester(_make_config(chunk_token_size=64), _DummyEmbedder())
    ingester_large = KeyDocIngester(
        _make_config(chunk_token_size=2048), _DummyEmbedder()
    )

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

    ingester_strict = KeyDocIngester(
        _make_config(similarity_threshold=0.99), _DummyEmbedder()
    )
    ingester_loose = KeyDocIngester(
        _make_config(similarity_threshold=0.01), _DummyEmbedder()
    )

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


# --- Fallback tests ---


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


# --- Property tests ---


class TestReadFileContent:
    def test_read_utf8_file_property(self):
        """HAPPY: reads UTF-8 file content."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == "hello world"

    @pytest.mark.edge
    def test_read_binary_extension_returns_empty_property(self):
        """EDGE: binary extension returns empty string."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("not a pdf")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == ""

    @pytest.mark.edge
    def test_read_nonexistent_file_returns_empty_property(self):
        """EDGE: missing file returns empty string."""
        ingester = _make_ingester()
        content = ingester._read_text("/nonexistent/file.txt")
        assert content == ""

    @pytest.mark.edge
    def test_read_non_utf8_returns_content_property(self):
        """EDGE: non-UTF-8 file uses lossy fallback."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(b"\xff\xfe invalid utf8 \x80")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert isinstance(content, str)

    @given(ext=st.sampled_from([".pdf", ".zip", ".gz", ".pyc", ".so", ".dll", ".woff"]))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_binary_extensions_return_empty_property(self, ext: Any):
        """EDGE: known binary extensions return empty string."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
            f.write("data")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == ""

    @given(text=small_text)
    @settings(max_examples=10)
    def test_read_roundtrip_property(self, text: str):
        """HAPPY: written content is readable."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == text


class TestChunkSectionFallback:
    def test_short_text_single_piece_property(self):
        """HAPPY: short text produces single piece."""
        ingester = _make_ingester()
        pieces = ingester._chunk_section_fallback("hello world")
        assert len(pieces) >= 1

    @pytest.mark.edge
    def test_empty_text_no_pieces_property(self):
        """EDGE: empty text produces no pieces."""
        ingester = _make_ingester()
        pieces = ingester._chunk_section_fallback("")
        assert pieces == []

    def test_long_text_multiple_pieces_property(self):
        """HAPPY: long text produces multiple overlapping pieces."""
        ingester = _make_ingester()
        words = " ".join(f"word{i}" for i in range(500))
        pieces = ingester._chunk_section_fallback(words)
        assert len(pieces) >= 2

    @given(n_words=st.integers(min_value=1, max_value=50))
    @settings(max_examples=15)
    def test_fallback_produces_pieces_property(self, n_words: Any):
        """HAPPY: fallback produces at least one piece for any non-empty text."""
        ingester = _make_ingester()
        text = " ".join(f"w{i}" for i in range(n_words))
        pieces = ingester._chunk_section_fallback(text)
        assert len(pieces) >= 1
        # All content is captured in the pieces
        joined = " ".join(pieces).strip()
        assert joined == text.strip() or joined in text.strip()


class TestDetectDocType:
    def test_design_path_property(self):
        assert KeyDocIngester._detect_doc_type("docs/design/arch.md") == "design"

    def test_research_path_property(self):
        assert KeyDocIngester._detect_doc_type("docs/research/paper.md") == "research"

    def test_adr_path_property(self):
        assert KeyDocIngester._detect_doc_type("docs/adr/001-choice.md") == "adr"

    def test_decisions_path_property(self):
        assert KeyDocIngester._detect_doc_type("docs/decisions/001.md") == "adr"

    def test_rfc_path_property(self):
        assert KeyDocIngester._detect_doc_type("docs/rfc/prop.md") == "rfc"

    def test_readme_property(self):
        assert KeyDocIngester._detect_doc_type("README.md") == "readme"

    def test_readme_lowercase_property(self):
        assert KeyDocIngester._detect_doc_type("readme.txt") == "readme"

    def test_default_doc_property(self):
        assert KeyDocIngester._detect_doc_type("notes/misc.txt") == "doc"

    @given(path=small_text)
    @settings(max_examples=15)
    def test_always_returns_known_type_property(self, path: str):
        """HAPPY: _detect_doc_type always returns a known type."""
        result = KeyDocIngester._detect_doc_type(path)
        assert result in ("design", "research", "adr", "rfc", "readme", "doc")


class TestChunkId:
    def test_chunk_id_deterministic_property(self):
        """HAPPY: same inputs produce same chunk ID."""
        id1 = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        id2 = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        assert id1 == id2

    def test_chunk_id_starts_with_prefix_property(self):
        """HAPPY: chunk ID starts with docchunk:."""
        cid = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        assert cid.startswith("docchunk:")

    def test_chunk_id_length_property(self):
        """HAPPY: chunk ID has consistent format."""
        cid = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        # "docchunk:" (9) + 24 hex chars
        assert len(cid) == 33

    @given(
        snap_id=small_text, path=small_text, idx=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=15)
    def test_chunk_id_unique_per_inputs_property(
        self, snap_id: str, path: str, idx: int
    ):
        """HAPPY: different inputs produce different IDs."""
        id1 = KeyDocIngester._chunk_id(snap_id, path, idx)
        id2 = KeyDocIngester._chunk_id(snap_id + "x", path, idx)
        assert id1 != id2


class TestEmbed:
    def test_embed_returns_vector_property(self):
        """HAPPY: _embed returns float list from embedder."""
        ingester = _make_ingester()
        result = ingester._embed("test text")
        assert result is not None
        assert len(result) == 8

    @pytest.mark.edge
    def test_embed_returns_none_on_error_property(self):
        """EDGE: _embed returns None when embedder raises."""

        class BrokenEmbedder:
            def embed_text(self, text: str) -> Never:
                raise RuntimeError("broken")

        ingester = _make_ingester(embedder=BrokenEmbedder())
        result = ingester._embed("test")
        assert result is None

    @pytest.mark.edge
    def test_embed_returns_none_on_none_property(self):
        """EDGE: _embed returns None when embedder returns None."""

        class NoneEmbedder:
            def embed_text(self, text: str) -> None:
                return None

        ingester = _make_ingester(embedder=NoneEmbedder())
        result = ingester._embed("test")
        assert result is None


class TestDocIngesterEdgeCases:
    @pytest.mark.edge
    def test_read_empty_file_property(self):
        """EDGE: empty file returns empty string."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == ""

    @pytest.mark.edge
    def test_chunk_section_fallback_whitespace_only_property(self):
        """EDGE: whitespace-only text produces no pieces or empty."""
        ingester = _make_ingester()
        pieces = ingester._chunk_section_fallback("   \n\t  ")
        assert isinstance(pieces, list)

    @pytest.mark.edge
    def test_chunk_id_different_index_property(self):
        """EDGE: different index produces different chunk ID."""
        id1 = KeyDocIngester._chunk_id("snap:1", "a.py", 0)
        id2 = KeyDocIngester._chunk_id("snap:1", "a.py", 1)
        assert id1 != id2

    @pytest.mark.edge
    def test_detect_doc_type_changelog_property(self):
        """EDGE: CHANGELOG files detected."""
        result = KeyDocIngester._detect_doc_type("CHANGELOG.md")
        assert isinstance(result, str)

    @pytest.mark.edge
    def test_read_symlink_to_file_property(self):
        """EDGE: reading symlink resolves to target."""
        ingester = _make_ingester()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "real.txt")
            link = os.path.join(tmpdir, "link.txt")
            with open(target, "w") as f:
                f.write("symlinked")
            os.symlink(target, link)
            content = ingester._read_text(link)
            assert content == "symlinked"


# --- Chunking edge cases ---


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
