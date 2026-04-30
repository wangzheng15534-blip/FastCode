"""Tests for projection_transform module."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.ir_graph_builder import IRGraphs
from fastcode.projection_models import ProjectionBuildResult, ProjectionScope
from fastcode.projection_transform import ProjectionTransformer
from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol

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
        IRDocument(
            doc_id=f"d{i}", path=f"src/mod{i}.py", language="python", source_set={"ast"}
        )
        for i in range(n_docs)
    ]
    syms = [
        IRSymbol(
            symbol_id=f"sym:{i}",
            external_symbol_id=None,
            path=f"src/mod{i}.py",
            display_name=f"func_{i}",
            kind="function",
            language="python",
            start_line=1,
            source_priority=10,
            source_set={"ast"},
        )
        for i in range(n_docs)
    ]
    edges = [
        IREdge(
            edge_id=f"e:{i}",
            src_id=f"d{i}",
            dst_id=f"sym:{i}",
            edge_type="contain",
            source="ast",
            confidence="resolved",
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


def _strip_timestamps(obj: Any) -> Any:
    """Recursively remove non-deterministic timestamp and pagerank fields from dicts/lists."""
    if isinstance(obj, dict):
        return {
            k: _strip_timestamps(v)
            for k, v in obj.items()
            if k not in ("updated_at", "created_at", "pagerank")
        }
    if isinstance(obj, list):
        return [_strip_timestamps(item) for item in obj]
    return obj


def _strip_timestamps_shallow(d: dict[str, Any]) -> dict[str, Any]:
    """Remove non-deterministic 'updated_at' keys from top-level meta dicts."""
    out = dict(d)
    if "meta" in out and isinstance(out["meta"], dict):
        meta = {k: v for k, v in out["meta"].items() if k != "updated_at"}
        out["meta"] = meta
    return out


def _sample_snapshot_pipeline() -> IRSnapshot:
    """Create a sample snapshot with full symbol IDs for pipeline tests."""
    doc = IRDocument(
        doc_id="doc:1", path="app/service.py", language="python", source_set={"ast"}
    )
    sym_a = IRSymbol(
        symbol_id="ast:snap:repo:abc:python:app/service.py:function:login:10:20",
        external_symbol_id=None,
        path="app/service.py",
        display_name="login",
        kind="function",
        language="python",
        start_line=10,
        end_line=20,
        source_priority=10,
        source_set={"ast"},
        metadata={"source": "ast"},
    )
    sym_b = IRSymbol(
        symbol_id="ast:snap:repo:abc:python:app/service.py:function:validate:22:30",
        external_symbol_id=None,
        path="app/service.py",
        display_name="validate",
        kind="function",
        language="python",
        start_line=22,
        end_line=30,
        source_priority=10,
        source_set={"ast"},
        metadata={"source": "ast"},
    )
    edges = [
        IREdge(
            edge_id="edge:contain:1",
            src_id=doc.doc_id,
            dst_id=sym_a.symbol_id,
            edge_type="contain",
            source="ast",
            confidence="resolved",
            doc_id=doc.doc_id,
            metadata={},
        ),
        IREdge(
            edge_id="edge:contain:2",
            src_id=doc.doc_id,
            dst_id=sym_b.symbol_id,
            edge_type="contain",
            source="ast",
            confidence="resolved",
            doc_id=doc.doc_id,
            metadata={},
        ),
        IREdge(
            edge_id="edge:call:1",
            src_id=sym_a.symbol_id,
            dst_id=sym_b.symbol_id,
            edge_type="call",
            source="ast",
            confidence="heuristic",
            doc_id=doc.doc_id,
            metadata={},
        ),
    ]
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:abc",
        branch="main",
        commit_id="abc",
        documents=[doc],
        symbols=[sym_a, sym_b],
        edges=edges,
        metadata={"source_modes": ["ast"]},
    )


def _sample_snapshot_v2() -> IRSnapshot:
    """Create a sample snapshot for v2 schema tests."""
    doc = IRDocument(
        doc_id="doc:1", path="app/service.py", language="python", source_set={"ast"}
    )
    sym_a = IRSymbol(
        symbol_id="ast:snap:repo:abc:python:app/service.py:function:login:10:20",
        external_symbol_id=None,
        path="app/service.py",
        display_name="login",
        kind="function",
        language="python",
        start_line=10,
        end_line=20,
        source_priority=10,
        source_set={"ast"},
        metadata={"source": "ast"},
    )
    sym_b = IRSymbol(
        symbol_id="ast:snap:repo:abc:python:app/service.py:function:validate:22:30",
        external_symbol_id=None,
        path="app/service.py",
        display_name="validate",
        kind="function",
        language="python",
        start_line=22,
        end_line=30,
        source_priority=10,
        source_set={"ast"},
        metadata={"source": "ast"},
    )
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:abc",
        branch="main",
        commit_id="abc",
        documents=[doc],
        symbols=[sym_a, sym_b],
        edges=[
            IREdge(
                edge_id="edge:contain:1",
                src_id=doc.doc_id,
                dst_id=sym_a.symbol_id,
                edge_type="contain",
                source="ast",
                confidence="resolved",
            ),
            IREdge(
                edge_id="edge:call:1",
                src_id=sym_a.symbol_id,
                dst_id=sym_b.symbol_id,
                edge_type="call",
                source="ast",
                confidence="heuristic",
            ),
        ],
    )


def _projection_snapshot() -> IRSnapshot:
    """Create a snapshot with 1 doc, 2 symbols, 5 edges.

    Structure is carefully designed for deterministic clustering output:
    - doc:1 has 4 incident edges (highest degree/weight), making it the
      clear cluster representative regardless of set iteration order.
    - Both symbols are kind="function" to avoid Counter.most_common()
      non-determinism when counts tie (class:1 vs method:1).
    - Edges use high-weight "contain" (4.0) and low-weight "call" (2.0).
    """
    doc = IRDocument(
        doc_id="doc:1", path="app/service.py", language="python", source_set={"ast"}
    )
    sym_a = IRSymbol(
        symbol_id="sym:a",
        external_symbol_id=None,
        path="app/service.py",
        display_name="create_user",
        kind="function",
        language="python",
        start_line=5,
        source_priority=10,
        source_set={"ast"},
    )
    sym_b = IRSymbol(
        symbol_id="sym:b",
        external_symbol_id=None,
        path="app/service.py",
        display_name="get_user",
        kind="function",
        language="python",
        start_line=10,
        source_priority=10,
        source_set={"ast"},
    )
    edges = [
        IREdge(
            edge_id="e:1",
            src_id="doc:1",
            dst_id="sym:a",
            edge_type="contain",
            source="ast",
            confidence="resolved",
        ),
        IREdge(
            edge_id="e:2",
            src_id="doc:1",
            dst_id="sym:b",
            edge_type="contain",
            source="ast",
            confidence="resolved",
        ),
        IREdge(
            edge_id="e:3",
            src_id="sym:a",
            dst_id="sym:b",
            edge_type="contain",
            source="ast",
            confidence="resolved",
        ),
        # Extra edges to make doc:1 the clear representative (highest degree/pagerank)
        IREdge(
            edge_id="e:4",
            src_id="sym:b",
            dst_id="doc:1",
            edge_type="call",
            source="ast",
            confidence="resolved",
        ),
        IREdge(
            edge_id="e:5",
            src_id="sym:a",
            dst_id="doc:1",
            edge_type="import",
            source="ast",
            confidence="resolved",
        ),
    ]
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:abc",
        branch="main",
        commit_id="abc",
        documents=[doc],
        symbols=[sym_a, sym_b],
        edges=edges,
        metadata={"source_modes": ["ast"]},
    )


def _sample_graphs() -> IRGraphs:
    """Create empty IRGraphs for tests that pass graph objects."""
    return IRGraphs(
        dependency_graph=nx.DiGraph(),
        call_graph=nx.DiGraph(),
        inheritance_graph=nx.DiGraph(),
        reference_graph=nx.DiGraph(),
        containment_graph=nx.DiGraph(),
    )


def _build_with_doc_mentions(
    transformer: Any, snapshot: IRSnapshot, doc_mentions: Any = None
) -> None:
    """Build projection with doc_mentions support."""
    scope = ProjectionScope(
        scope_kind="snapshot",
        snapshot_id=snapshot.snapshot_id,
        scope_key=snapshot.snapshot_id,
    )
    return transformer.build(
        scope=scope,
        snapshot=snapshot,
        ir_graphs=_sample_graphs(),
        doc_mentions=doc_mentions,
    )


def _build_result(snapshot: IRSnapshot) -> ProjectionBuildResult:
    """Build a projection result using standard transformer and scope."""
    return _make_transformer().build(scope=_make_scope(), snapshot=snapshot)


def _supporting_docs_config() -> dict[str, Any]:
    """Config for supporting_docs tests."""
    return {
        "projection": {"enable_leiden": True, "leiden_resolutions": [1.0]},
        "embedding": {"enabled": False},
    }


doc_count_st = st.integers(min_value=1, max_value=5)


# --- Property-based tests (from test_projection_properties.py) ---


class TestProjectionProperties:
    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    def test_projection_deterministic_property(self, n_docs: int):
        """HAPPY: same input always produces same output (excluding timestamps)."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result1 = transformer.build(scope=scope, snapshot=snapshot)
        result2 = transformer.build(scope=scope, snapshot=snapshot)
        assert result1.projection_id == result2.projection_id
        assert _strip_timestamps_shallow(result1.l0) == _strip_timestamps_shallow(
            result2.l0
        )
        assert _strip_timestamps_shallow(result1.l1) == _strip_timestamps_shallow(
            result2.l1
        )
        assert _strip_timestamps_shallow(result1.l2_index) == _strip_timestamps_shallow(
            result2.l2_index
        )
        assert len(result1.chunks) == len(result2.chunks)

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    def test_l0_summary_nonempty_property(self, n_docs: int):
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
    def test_l1_sections_populated_property(self, n_docs: int):
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
    def test_chunks_count_gte_one_property(self, n_docs: int):
        """HAPPY: at least one L2 chunk is produced for non-empty snapshot."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert len(result.chunks) >= 1

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    def test_projection_id_stable_property(self, n_docs: int):
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
    def test_projection_single_doc_snapshot_property(self, n_docs: int):
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
    def test_projection_snapshot_no_edges_property(self, n_docs: int):
        """EDGE: projection handles snapshot with symbols but no edges."""
        docs = [
            IRDocument(
                doc_id=f"d{i}",
                path=f"src/mod{i}.py",
                language="python",
                source_set={"ast"},
            )
            for i in range(n_docs)
        ]
        syms = [
            IRSymbol(
                symbol_id=f"sym:{i}",
                external_symbol_id=None,
                path=f"src/mod{i}.py",
                display_name=f"func_{i}",
                kind="function",
                language="python",
                start_line=1,
                source_priority=10,
                source_set={"ast"},
            )
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            branch="main",
            commit_id="abc123",
            documents=docs,
            symbols=syms,
            edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None
        assert len(result.chunks) >= 1

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    def test_l2_index_has_chunks_list_property(self, n_docs: int):
        """HAPPY: L2 index content contains a non-empty chunks list."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        l2_chunks = result.l2_index.get("content", {}).get("chunks", [])
        assert len(l2_chunks) >= 1

    @given(n_docs=doc_count_st)
    @settings(max_examples=20)
    def test_scope_kind_propagated_property(self, n_docs: int):
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
    def test_result_structure_has_required_keys_property(self, n_docs: int):
        """HAPPY: result dict has all required keys per ProjectionBuildResult schema."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        result_dict = result.to_dict()
        for key in (
            "projection_id",
            "snapshot_id",
            "scope_kind",
            "scope_key",
            "l0",
            "l1",
            "l2_index",
            "chunks",
            "warnings",
            "created_at",
        ):
            assert key in result_dict, f"missing key: {key}"

    @given(n_docs=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_no_symbols_property(self, n_docs: int):
        """EDGE: projection handles docs with no symbols."""
        docs = [
            IRDocument(
                doc_id=f"d{i}",
                path=f"src/mod{i}.py",
                language="python",
                source_set={"ast"},
            )
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            branch="main",
            commit_id="abc123",
            documents=docs,
            symbols=[],
            edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @pytest.mark.edge
    def test_projection_single_symbol_only_property(self):
        """EDGE: projection with exactly one symbol and no edges."""
        doc = IRDocument(
            doc_id="d1", path="a.py", language="python", source_set={"ast"}
        )
        sym = IRSymbol(
            symbol_id="sym:1",
            external_symbol_id=None,
            path="a.py",
            display_name="f",
            kind="function",
            language="python",
            start_line=1,
            source_priority=10,
            source_set={"ast"},
        )
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[sym],
            edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @given(n_docs=doc_count_st)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_all_same_kind_property(self, n_docs: int):
        """EDGE: all symbols same kind -- Counter.most_common() must not flip."""
        docs = [
            IRDocument(
                doc_id=f"d{i}", path=f"f{i}.py", language="python", source_set={"ast"}
            )
            for i in range(n_docs)
        ]
        syms = [
            IRSymbol(
                symbol_id=f"sym:{i}",
                external_symbol_id=None,
                path=f"f{i}.py",
                display_name=f"fn_{i}",
                kind="function",
                language="python",
                start_line=1,
                source_priority=10,
                source_set={"ast"},
            )
            for i in range(n_docs)
        ]
        edges = [
            IREdge(
                edge_id=f"e:{i}",
                src_id=f"d{i}",
                dst_id=f"sym:{i}",
                edge_type="contain",
                source="ast",
                confidence="resolved",
            )
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=docs,
            symbols=syms,
            edges=edges,
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        r1 = transformer.build(scope=scope, snapshot=snapshot)
        r2 = transformer.build(scope=scope, snapshot=snapshot)
        assert r1.projection_id == r2.projection_id

    @pytest.mark.edge
    def test_projection_empty_repo_name_property(self):
        """EDGE: projection handles empty string repo_name gracefully."""
        doc = IRDocument(
            doc_id="d1", path="a.py", language="python", source_set={"ast"}
        )
        sym = IRSymbol(
            symbol_id="sym:1",
            external_symbol_id=None,
            path="a.py",
            display_name="f",
            kind="function",
            language="python",
            start_line=1,
            source_priority=10,
            source_set={"ast"},
        )
        edge = IREdge(
            edge_id="e:1",
            src_id="d1",
            dst_id="sym:1",
            edge_type="contain",
            source="ast",
            confidence="resolved",
        )
        snapshot = IRSnapshot(
            repo_name="",
            snapshot_id="snap::abc",
            documents=[doc],
            symbols=[sym],
            edges=[edge],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @given(n_docs=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_mixed_languages_property(self, n_docs: int):
        """EDGE: projection handles documents with mixed languages."""
        langs = ["python", "javascript", "go"]
        docs = [
            IRDocument(
                doc_id=f"d{i}",
                path=f"src/mod{i}.{langs[i % 3][:2]}",
                language=langs[i % 3],
                source_set={"ast"},
            )
            for i in range(n_docs)
        ]
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            branch="main",
            commit_id="abc123",
            documents=docs,
            symbols=[],
            edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None
        assert len(result.chunks) >= 1

    @pytest.mark.edge
    def test_projection_large_symbol_set_property(self):
        """EDGE: projection handles many symbols without crash."""
        docs = [
            IRDocument(
                doc_id="d0", path="big.py", language="python", source_set={"ast"}
            )
        ]
        syms = [
            IRSymbol(
                symbol_id=f"sym:{i}",
                external_symbol_id=None,
                path="big.py",
                display_name=f"func_{i}",
                kind="function",
                language="python",
                start_line=i + 1,
                source_priority=10,
                source_set={"ast"},
            )
            for i in range(50)
        ]
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=docs,
            symbols=syms,
            edges=[],
            metadata={"source_modes": ["ast"]},
        )
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert result.projection_id is not None

    @given(n_docs=doc_count_st)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_projection_warnings_field_exists_property(self, n_docs: int):
        """EDGE: warnings field is always a list (may be empty)."""
        snapshot = _make_snapshot_with_docs(n_docs)
        scope = _make_scope()
        transformer = _make_transformer()
        result = transformer.build(scope=scope, snapshot=snapshot)
        assert isinstance(result.warnings, list)


# --- Pipeline tests (from test_projection_pipeline.py) ---


def test_projection_transform_emits_required_layers_and_meta():
    snapshot = _sample_snapshot_pipeline()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k1"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    assert result.l0["layer"] == "L0"
    assert result.l1["layer"] == "L1"
    assert result.l2_index["layer"] == "L2"
    assert "covers_nodes" in result.l0["meta"]
    assert "covers_edges" in result.l1["meta"]
    assert "projection_method" in result.l2_index["meta"]
    assert result.l1["meta"]["algo_version"] == transformer.ALGO_VERSION
    assert result.chunks


def test_scip_adapter_uses_snapshot_prefixed_symbol_ids_and_ref_edges():
    scip = {
        "indexer_name": "scip-python",
        "indexer_version": "0.1.0",
        "documents": [
            {
                "path": "app/service.py",
                "language": "python",
                "symbols": [
                    {
                        "symbol": "pkg app/service.py login().",
                        "name": "login",
                        "kind": "function",
                    }
                ],
                "occurrences": [
                    {
                        "symbol": "pkg app/service.py login().",
                        "role": "reference",
                        "range": [10, 0, 10, 5],
                    },
                    {
                        "symbol": "pkg app/service.py login().",
                        "role": "type_definition",
                        "range": [12, 0, 12, 5],
                    },
                ],
            }
        ],
    }
    snap = build_ir_from_scip(
        repo_name="repo", snapshot_id="snap:repo:abc", scip_index=scip
    )
    assert snap.symbols
    assert snap.symbols[0].symbol_id.startswith("scip:snap:repo:abc:")
    assert any(e.edge_type == "contain" and e.source == "scip" for e in snap.edges)
    assert any(e.edge_type == "ref" and e.source == "scip" for e in snap.edges)
    assert any(o.role == "type_definition" for o in snap.occurrences)


def test_steiner_prune_removes_non_terminal_leaves():
    transformer = ProjectionTransformer(
        config={"projection": {"enable_leiden": False, "steiner_prune": True}}
    )
    tree = nx.Graph()
    tree.add_edge("t1", "mid")
    tree.add_edge("mid", "t2")
    tree.add_edge("mid", "leaf")
    pruned = transformer._prune_steiner_leaves(tree, {"t1", "t2"})
    assert "leaf" not in pruned.nodes
    assert {"t1", "mid", "t2"} == set(pruned.nodes)


# --- Snapshot contract tests (from test_projection_contract.py) ---


@pytest.mark.snapshot
class TestProjectionContract:
    def test_l0_summary_format_snapshot(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_l0 = _strip_timestamps(result.l0)
        snapshot.assert_match(safe_l0)

    def test_l1_sections_format_snapshot(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_l1 = _strip_timestamps(result.l1)
        snapshot.assert_match(safe_l1)

    def test_l2_chunks_format_snapshot(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_chunks = _strip_timestamps(result.chunks)
        snapshot.assert_match(safe_chunks)

    def test_l2_index_format_snapshot(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_l2 = _strip_timestamps(result.l2_index)
        snapshot.assert_match(safe_l2)

    def test_result_dict_keys_snapshot(self, snapshot: IRSnapshot):
        """Full result dict (minus timestamps) has stable structure."""
        result = _build_result(_projection_snapshot())
        safe_dict = _strip_timestamps(result.to_dict())
        snapshot.assert_match(safe_dict)

    @pytest.mark.edge
    def test_projection_minimal_symbol_deterministic_snapshot(
        self, snapshot: IRSnapshot
    ):
        """EDGE: minimal snapshot where sym:1 is the clear cluster representative.

        Three nodes: doc:1, doc:2, sym:1. sym:1 is connected to both docs,
        giving it degree 2 while each doc has degree 1. sym:1 wins on all
        centrality metrics deterministically.
        """
        doc1 = IRDocument(
            doc_id="doc:1", path="main.py", language="python", source_set={"ast"}
        )
        doc2 = IRDocument(
            doc_id="doc:2", path="util.py", language="python", source_set={"ast"}
        )
        sym = IRSymbol(
            symbol_id="sym:1",
            external_symbol_id=None,
            path="main.py",
            display_name="process",
            kind="function",
            language="python",
            start_line=5,
            source_priority=10,
            source_set={"ast"},
        )
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            branch="main",
            commit_id="abc",
            documents=[doc1, doc2],
            symbols=[sym],
            edges=[
                IREdge(
                    edge_id="e:1",
                    src_id="doc:1",
                    dst_id="sym:1",
                    edge_type="contain",
                    source="ast",
                    confidence="resolved",
                ),
                IREdge(
                    edge_id="e:2",
                    src_id="doc:2",
                    dst_id="sym:1",
                    edge_type="ref",
                    source="ast",
                    confidence="resolved",
                ),
            ],
            metadata={"source_modes": ["ast"]},
        )
        result = _build_result(snap)
        safe_chunks = _strip_timestamps(result.chunks)
        snapshot.assert_match(safe_chunks)


# --- Supporting docs tests (from test_projection_supporting_docs.py) ---


def test_l2_no_supporting_docs_when_no_mentions():
    """supporting_docs should be absent when doc_mentions is None."""
    transformer = ProjectionTransformer(_supporting_docs_config())
    snapshot = _sample_snapshot_pipeline()
    result = _build_with_doc_mentions(transformer, snapshot)
    chunks = result.chunks
    assert len(chunks) > 0
    for chunk in chunks:
        assert "supporting_docs" not in chunk.get("content", {})


def test_l2_supporting_docs_with_mentions():
    """supporting_docs should appear when doc_mentions reference cluster symbols."""
    transformer = ProjectionTransformer(_supporting_docs_config())
    snapshot = _sample_snapshot_pipeline()
    sym_a = snapshot.symbols[0]
    sym_b = snapshot.symbols[1]
    doc_mentions = [
        {
            "chunk_id": "doc_adr_001",
            "symbol_id": sym_a.symbol_id,
            "symbol_name": "login",
            "confidence": "exact_name",
        },
        {
            "chunk_id": "doc_adr_002",
            "symbol_id": sym_b.symbol_id,
            "symbol_name": "validate",
            "confidence": "exact_name",
        },
    ]
    result = _build_with_doc_mentions(transformer, snapshot, doc_mentions=doc_mentions)
    chunks = result.chunks
    assert len(chunks) > 0
    chunks_with_docs = [c for c in chunks if "supporting_docs" in c.get("content", {})]
    assert len(chunks_with_docs) > 0
    for chunk in chunks_with_docs:
        docs = chunk["content"]["supporting_docs"]
        assert isinstance(docs, list)
        for d in docs:
            assert "chunk_id" in d
            assert "mentioned_symbols" in d


def test_l2_supporting_docs_capping():
    """supporting_docs should be capped at max_supporting_docs_per_cluster."""
    config = _supporting_docs_config()
    config["projection"]["max_supporting_docs_per_cluster"] = 2
    transformer = ProjectionTransformer(config)
    snapshot = _sample_snapshot_pipeline()
    sym_a = snapshot.symbols[0]
    doc_mentions = [
        {
            "chunk_id": f"doc_{i}",
            "symbol_id": sym_a.symbol_id,
            "symbol_name": "login",
            "confidence": "exact_name",
        }
        for i in range(5)
    ]
    result = _build_with_doc_mentions(transformer, snapshot, doc_mentions=doc_mentions)
    chunks = result.chunks
    for chunk in chunks:
        docs = chunk.get("content", {}).get("supporting_docs", [])
        assert len(docs) <= 2


def test_l2_supporting_docs_no_match():
    """supporting_docs should be absent when doc_mentions reference unknown symbols."""
    transformer = ProjectionTransformer(_supporting_docs_config())
    snapshot = _sample_snapshot_pipeline()
    doc_mentions = [
        {
            "chunk_id": "doc_unrelated",
            "symbol_id": "nonexistent_symbol_id",
            "symbol_name": "phantom",
            "confidence": "exact_name",
        },
    ]
    result = _build_with_doc_mentions(transformer, snapshot, doc_mentions=doc_mentions)
    chunks = result.chunks
    for chunk in chunks:
        assert "supporting_docs" not in chunk.get("content", {})


# --- V2 schema tests (from test_projection_v2_schema.py) ---


def test_projection_schema_has_hierarchy_and_relations():
    snapshot = _sample_snapshot_v2()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k2"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    l1_content = result.l1["content"]
    assert "relations" in l1_content
    assert "xref" in l1_content["relations"]
    assert "hierarchy" in l1_content["relations"]
    assert "backbone" in l1_content["relations"]
    assert "related_code" in l1_content
    assert "related_memory" in l1_content

    chunk = result.chunks[0]
    assert "chunk_id" in chunk
    assert "content" in chunk
    assert chunk["version"] == "v1"
    assert chunk["layer"] == "L2"
    assert "source" in chunk
    assert "render" in chunk
    assert "meta" in chunk
    assert "aggregation" in chunk["meta"]


def test_projection_l2_chunk_has_all_required_metadata():
    snapshot = _sample_snapshot_v2()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k3"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    for chunk in result.chunks:
        assert chunk["version"] == "v1", f"chunk {chunk['chunk_id']} missing version"
        assert chunk["layer"] == "L2", f"chunk {chunk['chunk_id']} missing layer"
        assert "id" in chunk, f"chunk {chunk['chunk_id']} missing id"
        assert "path" in chunk, f"chunk {chunk['chunk_id']} missing path"
        assert "title" in chunk, f"chunk {chunk['chunk_id']} missing title"
        assert "source" in chunk, f"chunk {chunk['chunk_id']} missing source"
        assert "render" in chunk, f"chunk {chunk['chunk_id']} missing render"
        assert "meta" in chunk, f"chunk {chunk['chunk_id']} missing meta"


def test_projection_l1_xref_relations_have_confidence():
    snapshot = _sample_snapshot_v2()
    transformer = ProjectionTransformer(config={"projection": {"enable_leiden": False}})
    scope = ProjectionScope(
        scope_kind="snapshot", snapshot_id=snapshot.snapshot_id, scope_key="k4"
    )
    result = transformer.build(
        scope=scope, snapshot=snapshot, ir_graphs=_sample_graphs()
    )

    l1_content = result.l1["content"]
    xrefs = l1_content["relations"]["xref"]
    for rel in xrefs:
        assert "id" in rel
        assert "type" in rel
        assert 0.0 <= rel["confidence"] <= 1.0
