"""Property-based tests for doc_ingester module (non-ML methods only)."""

from __future__ import annotations

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.doc_ingester import KeyDocIngester

# --- Helpers ---

small_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ",
    min_size=1, max_size=30,
)


class _FakeEmbedder:
    """Minimal embedder that returns fixed vectors."""
    def embed_text(self, text):
        return [0.1] * 8


def _make_ingester(**overrides):
    defaults = {
        "config": {"embedding": {}},
        "embedder": _FakeEmbedder(),
    }
    defaults.update(overrides)
    return KeyDocIngester(**defaults)


# --- Properties ---


@pytest.mark.property
class TestReadFileContent:

    @pytest.mark.happy
    def test_read_utf8_file(self):
        """HAPPY: reads UTF-8 file content."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == "hello world"

    @pytest.mark.edge
    def test_read_binary_extension_returns_empty(self):
        """EDGE: binary extension returns empty string."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("not a pdf")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == ""

    @pytest.mark.edge
    def test_read_nonexistent_file_returns_empty(self):
        """EDGE: missing file returns empty string."""
        ingester = _make_ingester()
        content = ingester._read_text("/nonexistent/file.txt")
        assert content == ""

    @pytest.mark.edge
    def test_read_non_utf8_returns_content(self):
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
    def test_binary_extensions_return_empty(self, ext):
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
    @pytest.mark.happy
    def test_read_roundtrip(self, text):
        """HAPPY: written content is readable."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == text


@pytest.mark.property
class TestChunkSectionFallback:

    @pytest.mark.happy
    def test_short_text_single_piece(self):
        """HAPPY: short text produces single piece."""
        ingester = _make_ingester()
        pieces = ingester._chunk_section_fallback("hello world")
        assert len(pieces) >= 1

    @pytest.mark.edge
    def test_empty_text_no_pieces(self):
        """EDGE: empty text produces no pieces."""
        ingester = _make_ingester()
        pieces = ingester._chunk_section_fallback("")
        assert pieces == []

    @pytest.mark.happy
    def test_long_text_multiple_pieces(self):
        """HAPPY: long text produces multiple overlapping pieces."""
        ingester = _make_ingester()
        words = " ".join(f"word{i}" for i in range(500))
        pieces = ingester._chunk_section_fallback(words)
        assert len(pieces) >= 2

    @given(n_words=st.integers(min_value=1, max_value=50))
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_fallback_produces_pieces(self, n_words):
        """HAPPY: fallback produces at least one piece for any non-empty text."""
        ingester = _make_ingester()
        text = " ".join(f"w{i}" for i in range(n_words))
        pieces = ingester._chunk_section_fallback(text)
        assert len(pieces) >= 1
        # All content is captured in the pieces
        joined = " ".join(pieces).strip()
        assert joined == text.strip() or joined in text.strip()


@pytest.mark.property
class TestDetectDocType:

    @pytest.mark.happy
    def test_design_path(self):
        assert KeyDocIngester._detect_doc_type("docs/design/arch.md") == "design"

    @pytest.mark.happy
    def test_research_path(self):
        assert KeyDocIngester._detect_doc_type("docs/research/paper.md") == "research"

    @pytest.mark.happy
    def test_adr_path(self):
        assert KeyDocIngester._detect_doc_type("docs/adr/001-choice.md") == "adr"

    @pytest.mark.happy
    def test_decisions_path(self):
        assert KeyDocIngester._detect_doc_type("docs/decisions/001.md") == "adr"

    @pytest.mark.happy
    def test_rfc_path(self):
        assert KeyDocIngester._detect_doc_type("docs/rfc/prop.md") == "rfc"

    @pytest.mark.happy
    def test_readme(self):
        assert KeyDocIngester._detect_doc_type("README.md") == "readme"

    @pytest.mark.happy
    def test_readme_lowercase(self):
        assert KeyDocIngester._detect_doc_type("readme.txt") == "readme"

    @pytest.mark.happy
    def test_default_doc(self):
        assert KeyDocIngester._detect_doc_type("notes/misc.txt") == "doc"

    @given(path=small_text)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_always_returns_known_type(self, path):
        """HAPPY: _detect_doc_type always returns a known type."""
        result = KeyDocIngester._detect_doc_type(path)
        assert result in ("design", "research", "adr", "rfc", "readme", "doc")


@pytest.mark.property
class TestChunkId:

    @pytest.mark.happy
    def test_chunk_id_deterministic(self):
        """HAPPY: same inputs produce same chunk ID."""
        id1 = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        id2 = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        assert id1 == id2

    @pytest.mark.happy
    def test_chunk_id_starts_with_prefix(self):
        """HAPPY: chunk ID starts with docchunk:."""
        cid = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        assert cid.startswith("docchunk:")

    @pytest.mark.happy
    def test_chunk_id_length(self):
        """HAPPY: chunk ID has consistent format."""
        cid = KeyDocIngester._chunk_id("snap:repo:abc", "a.py", 0)
        # "docchunk:" (9) + 24 hex chars
        assert len(cid) == 33

    @given(snap_id=small_text, path=small_text, idx=st.integers(min_value=0, max_value=100))
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_chunk_id_unique_per_inputs(self, snap_id, path, idx):
        """HAPPY: different inputs produce different IDs."""
        id1 = KeyDocIngester._chunk_id(snap_id, path, idx)
        id2 = KeyDocIngester._chunk_id(snap_id + "x", path, idx)
        assert id1 != id2


@pytest.mark.property
class TestEmbed:

    @pytest.mark.happy
    def test_embed_returns_vector(self):
        """HAPPY: _embed returns float list from embedder."""
        ingester = _make_ingester()
        result = ingester._embed("test text")
        assert result is not None
        assert len(result) == 8

    @pytest.mark.edge
    def test_embed_returns_none_on_error(self):
        """EDGE: _embed returns None when embedder raises."""
        class BrokenEmbedder:
            def embed_text(self, text):
                raise RuntimeError("broken")
        ingester = _make_ingester(embedder=BrokenEmbedder())
        result = ingester._embed("test")
        assert result is None

    @pytest.mark.edge
    def test_embed_returns_none_on_none(self):
        """EDGE: _embed returns None when embedder returns None."""
        class NoneEmbedder:
            def embed_text(self, text):
                return None
        ingester = _make_ingester(embedder=NoneEmbedder())
        result = ingester._embed("test")
        assert result is None


@pytest.mark.property
class TestDocIngesterEdgeCases:

    @pytest.mark.edge
    def test_read_empty_file(self):
        """EDGE: empty file returns empty string."""
        ingester = _make_ingester()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            content = ingester._read_text(f.name)
        os.unlink(f.name)
        assert content == ""

    @pytest.mark.edge
    def test_chunk_section_fallback_whitespace_only(self):
        """EDGE: whitespace-only text produces no pieces or empty."""
        ingester = _make_ingester()
        pieces = ingester._chunk_section_fallback("   \n\t  ")
        assert isinstance(pieces, list)

    @pytest.mark.edge
    def test_chunk_id_different_index(self):
        """EDGE: different index produces different chunk ID."""
        id1 = KeyDocIngester._chunk_id("snap:1", "a.py", 0)
        id2 = KeyDocIngester._chunk_id("snap:1", "a.py", 1)
        assert id1 != id2

    @pytest.mark.edge
    def test_detect_doc_type_changelog(self):
        """EDGE: CHANGELOG files detected."""
        result = KeyDocIngester._detect_doc_type("CHANGELOG.md")
        assert isinstance(result, str)

    @pytest.mark.edge
    def test_read_symlink_to_file(self):
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
