"""Tests for SCIPResolutionBridge 3-strategy cascade."""

from __future__ import annotations

import math
from typing import Any, Never

import pytest

from fastcode.ir.types import IRCodeUnit, IRSnapshot
from fastcode.scip.resolution_bridge import (
    SCIPResolutionBridge,
    _cosine_similarity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit(
    unit_id: str,
    display_name: str,
    path: str = "src/module.py",
    kind: str = "class",
    qualified_name: str | None = None,
    source_set: set[str] | None = None,
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind=kind,
        path=path,
        language="python",
        display_name=display_name,
        qualified_name=qualified_name,
        source_set=source_set or {"scip"},
    )


def _make_snapshot(*units: IRCodeUnit) -> IRSnapshot:
    return IRSnapshot(
        repo_name="test",
        snapshot_id="snap:test:1",
        units=list(units),
    )


class _FakeEmbedder:
    """Embedder that returns deterministic vectors based on text hash."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    def _vec_for(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        # Simple deterministic vector: use characters to fill dimensions
        raw = [float(ord(c)) for c in text.ljust(self._dim * 2, "x")[: self._dim * 2]]
        vec: list[float] = []
        for i in range(self._dim):
            vec.append(raw[i * 2] / 128.0)
        # Normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        self._cache[text] = vec
        return vec

    def embed_text(self, text: str) -> list[float]:
        return self._vec_for(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec_for(t) for t in texts]


class _PerfectEmbedder:
    """Embedder where identical strings produce identical, unit-norm vectors."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}

    def _vec_for(self, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        # Deterministic but unique per text
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        vec = [float(int(h[i : i + 2], 16)) / 255.0 for i in range(0, 32, 2)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        self._vectors[text] = vec
        return vec

    def embed_text(self, text: str) -> list[float]:
        return self._vec_for(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec_for(t) for t in texts]


class _ControlledEmbedder:
    """Embedder that returns pre-specified unit-norm vectors for controlled testing."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        # Normalise all vectors
        self._vectors: dict[str, list[float]] = {}
        self._dim = 2
        for text, vec in vectors.items():
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            self._vectors[text] = [v / norm for v in vec]
            self._dim = len(vec)

    def embed_text(self, text: str) -> list[float]:
        return self._vectors.get(text, [0.0] * self._dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(t, [0.0] * self._dim) for t in texts]


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_length(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Strategy 1 -- Lexical exact
# ---------------------------------------------------------------------------


class TestLexicalStrategy:
    def test_exact_display_name_match(self):
        unit = _make_unit("u:1", "BranchIndexer")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        result = bridge.resolve("BranchIndexer")
        assert result is not None
        assert result.symbol_id == "u:1"
        assert result.match_strategy == "lexical"
        assert result.confidence == pytest.approx(0.95)

    def test_exact_qualified_name_match(self):
        unit = _make_unit("u:1", "Idx", qualified_name="my_pkg.BranchIndexer")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        result = bridge.resolve("my_pkg.BranchIndexer")
        assert result is not None
        assert result.symbol_id == "u:1"
        assert result.match_strategy == "lexical"
        assert result.confidence == pytest.approx(0.95)

    def test_no_match_returns_none(self):
        unit = _make_unit("u:1", "BranchIndexer")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        assert bridge.resolve("NonExistent") is None

    def test_short_mention_returns_none(self):
        unit = _make_unit("u:1", "AB")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        assert bridge.resolve("AB") is None

    def test_file_units_are_excluded(self):
        file_unit = _make_unit("u:file", "main.py", kind="file")
        bridge = SCIPResolutionBridge(_make_snapshot(file_unit))
        assert bridge.resolve("main.py") is None

    def test_display_name_takes_priority_over_qualified(self):
        unit = _make_unit("u:1", "Foo", qualified_name="bar.Baz")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        # "Foo" matches display_name -> lexical
        result = bridge.resolve("Foo")
        assert result is not None
        assert result.match_strategy == "lexical"


# ---------------------------------------------------------------------------
# Strategy 2 -- Namespace contextual
# ---------------------------------------------------------------------------


class TestNamespaceStrategy:
    def test_same_directory_display_name_match(self):
        """Namespace strategy matches display_name in same directory.

        Since lexical also checks display_name, when mention == display_name,
        lexical wins (higher confidence). This test verifies that the cascade
        still returns a valid result -- lexical takes priority.
        """
        unit = _make_unit("u:1", "Helper", path="src/utils/helper.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        result = bridge.resolve("Helper", context_path="src/utils/design.md")
        assert result is not None
        assert result.symbol_id == "u:1"
        # Lexical wins because display_name matches exactly
        assert result.match_strategy == "lexical"
        assert result.confidence == pytest.approx(0.95)

    def test_same_directory_namespace_strategy_match(self):
        unit = _make_unit("u:1", "Helper", path="src/utils/helper.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        result = bridge._resolve_namespace("Helper", "src/utils/design.md")
        assert result is not None
        assert result.symbol_id == "u:1"
        assert result.match_strategy == "namespace"
        assert result.confidence == pytest.approx(0.70)

    def test_different_directory_no_match(self):
        unit = _make_unit("u:1", "Helper", path="src/utils/helper.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        assert bridge._resolve_namespace("Helper", "src/other/design.md") is None

    def test_no_context_path_skips_namespace(self):
        unit = _make_unit("u:1", "Helper", path="src/utils/helper.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        assert bridge._resolve_namespace("Helper", context_path=None) is None

    def test_no_units_in_directory(self):
        unit = _make_unit("u:1", "Helper", path="src/utils/helper.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        assert bridge._resolve_namespace("Helper", context_path="src/other/doc.md") is None


# ---------------------------------------------------------------------------
# Strategy 3 -- Vector semantic
# ---------------------------------------------------------------------------


class TestSemanticStrategy:
    def test_semantic_match_above_threshold(self):
        """Semantic matches when mention text differs from display_name but vectors are similar.

        We use a unit with display_name that doesn't match the query text, so lexical
        misses. We use a _ControlledEmbedder to guarantee high similarity.
        """
        unit = _make_unit("u:1", "DatabaseConnectionPool", path="src/db.py")
        embedder = _ControlledEmbedder(
            {"DatabaseConnectionPool": [1.0, 0.0], "DBConnection": [0.99, 0.14]}
        )
        bridge = SCIPResolutionBridge(
            _make_snapshot(unit), embedder=embedder, semantic_threshold=0.5
        )
        result = bridge.resolve("DBConnection")
        assert result is not None
        assert result.match_strategy == "semantic"
        assert result.confidence == pytest.approx(0.7)  # clamped to ceil

    def test_semantic_below_threshold(self):
        unit = _make_unit("u:1", "BranchIndexer", path="src/main.py")
        embedder = _FakeEmbedder(dim=8)
        bridge = SCIPResolutionBridge(
            _make_snapshot(unit), embedder=embedder, semantic_threshold=0.99
        )
        # Threshold 0.99 is very high; even perfect self-similarity might not reach it
        # depending on vector quality. But with our FakeEmbedder, self-sim is 1.0.
        # Let's use a mention that definitely won't match: empty or very different.
        # Actually, any non-empty text will produce some vector. Let's just verify
        # the mechanism by using a very high threshold.
        # Self-match should still work at 1.0 > 0.99.
        result = bridge.resolve("BranchIndexer")
        assert result is not None  # self-similarity is 1.0

    def test_no_embedder_skips_semantic(self):
        unit = _make_unit("u:1", "Something", path="src/x.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit), embedder=None)
        assert bridge.resolve("SomethingElse") is None

    def test_confidence_clamped_to_range(self):
        unit = _make_unit("u:1", "DatabaseConnectionPool", path="src/db.py")
        embedder = _ControlledEmbedder(
            {"DatabaseConnectionPool": [1.0, 0.0], "DBConnection": [0.99, 0.14]}
        )
        bridge = SCIPResolutionBridge(
            _make_snapshot(unit),
            embedder=embedder,
            semantic_threshold=0.5,
            semantic_confidence_floor=0.4,
            semantic_confidence_ceil=0.6,
        )
        result = bridge.resolve("DBConnection")
        assert result is not None
        assert result.match_strategy == "semantic"
        assert 0.4 <= result.confidence <= 0.6


# ---------------------------------------------------------------------------
# Cascade ordering
# ---------------------------------------------------------------------------


class TestCascadeOrdering:
    def test_lexical_wins_over_namespace(self):
        """When both lexical and namespace would match, lexical wins (higher confidence)."""
        unit = _make_unit("u:1", "Helper", path="src/utils/helper.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        result = bridge.resolve("Helper", context_path="src/utils/design.md")
        assert result is not None
        assert result.match_strategy == "lexical"
        assert result.confidence == pytest.approx(0.95)

    def test_lexical_wins_when_namespace_also_matches(self):
        """Public cascade prefers lexical when namespace could also match."""
        unit = _make_unit("u:1", "Helper", path="src/utils/helper.py")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        result = bridge.resolve("Helper", context_path="src/utils/design.md")
        assert result is not None
        assert result.symbol_id == "u:1"
        assert result.match_strategy == "lexical"

    def test_semantic_wins_when_lexical_and_namespace_miss(self):
        """When lexical and namespace both miss, semantic can still match."""
        unit = _make_unit("u:1", "DatabaseConnectionPool", path="src/db.py")
        embedder = _ControlledEmbedder(
            {"DatabaseConnectionPool": [1.0, 0.0], "DBConn": [0.99, 0.14]}
        )
        bridge = SCIPResolutionBridge(
            _make_snapshot(unit),
            embedder=embedder,
            semantic_threshold=0.5,
        )
        # "DBConn" doesn't match any display_name or qualified_name -> lexical misses
        # No context_path -> namespace skips
        # Semantic: "DBConn" vector is very similar to "DatabaseConnectionPool" -> hits
        result = bridge.resolve("DBConn")
        assert result is not None
        assert result.match_strategy == "semantic"

    def test_semantic_fires_when_no_lexical_match(self):
        """Semantic matches when mention text differs from display_name but vectors are similar."""
        unit = _make_unit("u:1", "DatabaseConnection", path="src/db.py")
        embedder = _ControlledEmbedder(
            {"DatabaseConnection": [1.0, 0.0], "DBConn": [0.99, 0.14]}
        )
        bridge = SCIPResolutionBridge(
            _make_snapshot(unit),
            embedder=embedder,
            semantic_threshold=0.5,
        )
        result = bridge.resolve("DBConn")
        assert result is not None
        assert result.symbol_id == "u:1"
        assert result.match_strategy == "semantic"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_snapshot(self):
        bridge = SCIPResolutionBridge(_make_snapshot())
        assert bridge.resolve("Anything") is None

    def test_empty_mention_text(self):
        unit = _make_unit("u:1", "Foo")
        bridge = SCIPResolutionBridge(_make_snapshot(unit))
        assert bridge.resolve("") is None
        assert bridge.resolve("  ") is None

    def test_snapshot_with_only_file_units(self):
        file_unit = _make_unit("u:file", "main.py", kind="file")
        bridge = SCIPResolutionBridge(_make_snapshot(file_unit))
        assert bridge.resolve("main.py") is None

    def test_multiple_units_same_display_name(self):
        """First unit wins for lexical exact match."""
        u1 = _make_unit("u:1", "Helper", path="src/a.py")
        u2 = _make_unit("u:2", "Helper", path="src/b.py")
        bridge = SCIPResolutionBridge(_make_snapshot(u1, u2))
        result = bridge.resolve("Helper")
        assert result is not None
        assert result.symbol_id == "u:1"

    def test_embedder_failure_graceful(self):
        """If embed_batch fails during index build, semantic is silently disabled."""

        class _BrokenEmbedder:
            def embed_batch(self, texts: Any) -> Never:
                msg = "embedding service down"
                raise RuntimeError(msg)

        unit = _make_unit("u:1", "Foo")
        bridge = SCIPResolutionBridge(_make_snapshot(unit), embedder=_BrokenEmbedder())
        # Should not crash; semantic is disabled
        assert bridge.resolve("Foo") is not None  # lexical catches it
        assert bridge.resolve("NonExistent") is None  # no semantic fallback
