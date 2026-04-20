"""Property-based tests for ProjectionTransformer determinism and structure."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol
from fastcode.projection_transform import ProjectionTransformer
from fastcode.projection_models import ProjectionScope


# --- Helpers ---

def _make_scope() -> ProjectionScope:
    """Create a snapshot-scoped ProjectionScope."""
    return ProjectionScope(
        scope_kind="snapshot",
        snapshot_id="snap:repo:abc",
        scope_key="repo",
    )


def _make_transformer() -> ProjectionTransformer:
    """Create a ProjectionTransformer with LLM disabled for deterministic output."""
    return ProjectionTransformer(config={"projection": {"llm_enabled": False}})


def _make_snapshot_with_docs(n_docs: int) -> IRSnapshot:
    """Create a snapshot with n_docs documents, each with one symbol and one edge."""
    docs = [
        IRDocument(doc_id=f"d{i}", path=f"src/mod{i}.py", language="python", source_set={"ast"})
        for i in range(n_docs)
    ]
    syms = [
        IRSymbol(
            symbol_id=f"sym:{i}", external_symbol_id=None,
            path=f"src/mod{i}.py", display_name=f"func_{i}",
            kind="function", language="python",
            start_line=1, source_priority=10, source_set={"ast"},
        )
        for i in range(n_docs)
    ]
    edges = [
        IREdge(
            edge_id=f"e:{i}", src_id=f"d{i}", dst_id=f"sym:{i}",
            edge_type="contain", source="ast", confidence="resolved",
        )
        for i in range(n_docs)
    ]
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:abc",
        branch="main",
        commit_id="abc123",
        documents=docs,
        symbols=syms,
        edges=edges,
        metadata={"source_modes": ["ast"]},
    )


doc_count_st = st.integers(min_value=1, max_value=5)


def _strip_timestamps(d: dict) -> dict:
    """Remove non-deterministic 'updated_at' keys from meta dicts."""
    out = dict(d)
    if "meta" in out and isinstance(out["meta"], dict):
        meta = {k: v for k, v in out["meta"].items() if k != "updated_at"}
        out["meta"] = meta
    return out


@pytest.mark.property
class TestProjectionProperties:

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_projection_deterministic(self, n_docs: int):
        """HAPPY: same input always produces same output (excluding timestamps)."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result1 = transformer.build(scope=scope, snapshot=snapshot)
        result2 = transformer.build(scope=scope, snapshot=snapshot)
        assert result1.projection_id == result2.projection_id
        assert _strip_timestamps(result1.l0) == _strip_timestamps(result2.l0)
        assert _strip_timestamps(result1.l1) == _strip_timestamps(result2.l1)
        assert _strip_timestamps(result1.l2_index) == _strip_timestamps(result2.l2_index)
        assert len(result1.chunks) == len(result2.chunks)

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_l0_summary_nonempty(self, n_docs: int):
        """HAPPY: L0 content is always non-empty for non-empty snapshot."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        l0_content = result.l0.get("content", {})
        summary = l0_content.get("summary", "")
        assert summary is not None
        assert len(str(summary)) > 0

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_l1_sections_populated(self, n_docs: int):
        """HAPPY: L1 sections are populated for non-empty snapshot."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        l1_content = result.l1.get("content", {})
        sections = l1_content.get("sections", [])
        assert len(sections) >= 1

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_chunks_count_gte_one(self, n_docs: int):
        """HAPPY: at least one L2 chunk is produced for non-empty snapshot."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert len(result.chunks) >= 1

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_projection_id_stable(self, n_docs: int):
        """HAPPY: projection_id is a stable hash derived from scope + algo version."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result1 = transformer.build(scope=scope, snapshot=snapshot)
        result2 = transformer.build(scope=scope, snapshot=snapshot)
        assert result1.projection_id == result2.projection_id
        assert result1.projection_id.startswith("proj_")

    @given(n_docs=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_single_doc_snapshot(self, n_docs: int):
        """EDGE: projection handles single-document snapshot."""
        snapshot = _make_snapshot_with_docs(1)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None
        assert len(result.chunks) >= 1

    @given(n_docs=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_snapshot_no_edges(self, n_docs: int):
        """EDGE: projection handles snapshot with symbols but no edges."""
        docs = [
            IRDocument(doc_id=f"d{i}", path=f"src/mod{i}.py", language="python", source_set={"ast"})
            for i in range(n_docs)
        ]
        syms = [
            IRSymbol(
                symbol_id=f"sym:{i}", external_symbol_id=None,
                path=f"src/mod{i}.py", display_name=f"func_{i}",
                kind="function", language="python",
                start_line=1, source_priority=10, source_set={"ast"},
            )
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            branch="main", commit_id="abc123",
            documents=docs, symbols=syms, edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None
        assert len(result.chunks) >= 1

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_l2_index_has_chunks_list(self, n_docs: int):
        """HAPPY: L2 index content contains a non-empty chunks list."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        l2_chunks = result.l2_index.get("content", {}).get("chunks", [])
        assert len(l2_chunks) >= 1

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_scope_kind_propagated(self, n_docs: int):
        """HAPPY: scope_kind and scope_key are propagated into the result."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.scope_kind == "snapshot"
        assert result.scope_key == "repo"
        assert result.snapshot_id == "snap:repo:abc"

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_result_structure_has_required_keys(self, n_docs: int):
        """HAPPY: result dict has all required keys per ProjectionBuildResult schema."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        result_dict = result.to_dict()
        for key in ("projection_id", "snapshot_id", "scope_kind", "scope_key",
                     "l0", "l1", "l2_index", "chunks", "warnings", "created_at"):
            assert key in result_dict, f"missing key: {key}"

    @given(n_docs=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_no_symbols(self, n_docs: int):
        """EDGE: projection handles docs with no symbols."""
        docs = [
            IRDocument(doc_id=f"d{i}", path=f"src/mod{i}.py", language="python", source_set={"ast"})
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            branch="main", commit_id="abc123",
            documents=docs, symbols=[], edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @pytest.mark.edge
    def test_projection_single_symbol_only(self):
        """EDGE: projection with exactly one symbol and no edges."""
        doc = IRDocument(doc_id="d1", path="a.py", language="python", source_set={"ast"})
        sym = IRSymbol(
            symbol_id="sym:1", external_symbol_id=None,
            path="a.py", display_name="f", kind="function",
            language="python", start_line=1, source_priority=10,
            source_set={"ast"},
        )
        snapshot = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=[doc], symbols=[sym], edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @given(n_docs=doc_count_st)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_all_same_kind(self, n_docs: int):
        """EDGE: all symbols same kind — Counter.most_common() must not flip."""
        docs = [
            IRDocument(doc_id=f"d{i}", path=f"f{i}.py", language="python", source_set={"ast"})
            for i in range(n_docs)
        ]
        syms = [
            IRSymbol(
                symbol_id=f"sym:{i}", external_symbol_id=None,
                path=f"f{i}.py", display_name=f"fn_{i}",
                kind="function", language="python",
                start_line=1, source_priority=10, source_set={"ast"},
            )
            for i in range(n_docs)
        ]
        edges = [
            IREdge(
                edge_id=f"e:{i}", src_id=f"d{i}", dst_id=f"sym:{i}",
                edge_type="contain", source="ast", confidence="resolved",
            )
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=docs, symbols=syms, edges=edges,
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        r1 = transformer.build(scope=scope, snapshot=snapshot)
        r2 = transformer.build(scope=scope, snapshot=snapshot)
        assert r1.projection_id == r2.projection_id

    @pytest.mark.edge
    def test_projection_empty_repo_name(self):
        """EDGE: projection handles empty string repo_name gracefully."""
        doc = IRDocument(doc_id="d1", path="a.py", language="python", source_set={"ast"})
        sym = IRSymbol(
            symbol_id="sym:1", external_symbol_id=None,
            path="a.py", display_name="f", kind="function",
            language="python", start_line=1, source_priority=10,
            source_set={"ast"},
        )
        edge = IREdge(
            edge_id="e:1", src_id="d1", dst_id="sym:1",
            edge_type="contain", source="ast", confidence="resolved",
        )
        snapshot = IRSnapshot(
            repo_name="", snapshot_id="snap::abc",
            documents=[doc], symbols=[sym], edges=[edge],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @given(n_docs=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_mixed_languages(self, n_docs: int):
        """EDGE: projection handles documents with mixed languages."""
        langs = ["python", "javascript", "go"]
        docs = [
            IRDocument(doc_id=f"d{i}", path=f"src/mod{i}.{langs[i % 3][:2]}", language=langs[i % 3], source_set={"ast"})
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            branch="main", commit_id="abc123",
            documents=docs, symbols=[], edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None
        assert len(result.chunks) >= 1

    @pytest.mark.edge
    def test_projection_large_symbol_set(self):
        """EDGE: projection handles many symbols without crash."""
        docs = [IRDocument(doc_id="d0", path="big.py", language="python", source_set={"ast"})]
        syms = [
            IRSymbol(
                symbol_id=f"sym:{i}", external_symbol_id=None,
                path="big.py", display_name=f"func_{i}",
                kind="function", language="python",
                start_line=i + 1, source_priority=10, source_set={"ast"},
            )
            for i in range(50)
        ]
        snapshot = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=docs, symbols=syms, edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @given(n_docs=doc_count_st)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_warnings_field_exists(self, n_docs: int):
        """EDGE: warnings field is always a list (may be empty)."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert isinstance(result.warnings, list)
