"""Snapshot tests for projection output data contracts."""

from __future__ import annotations

from typing import Any

import pytest

from fastcode.projection_models import ProjectionBuildResult, ProjectionScope
from fastcode.projection_transform import ProjectionTransformer
from fastcode.semantic_ir import IRDocument, IREdge, IRSnapshot, IRSymbol


def _make_transformer() -> ProjectionTransformer:
    """Create a ProjectionTransformer with LLM disabled for deterministic output."""
    return ProjectionTransformer(config={"projection": {"llm_enabled": False}})


def _make_scope() -> ProjectionScope:
    """Create a snapshot-scoped ProjectionScope."""
    return ProjectionScope(
        scope_kind="snapshot",
        snapshot_id="snap:repo:abc",
        scope_key="repo",
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


def _build_result(snapshot: IRSnapshot) -> ProjectionBuildResult:
    """Build a projection result using standard transformer and scope."""
    return _make_transformer().build(scope=_make_scope(), snapshot=snapshot)


@pytest.mark.snapshot
@pytest.mark.basic
class TestProjectionContract:
    def test_l0_summary_format(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_l0 = _strip_timestamps(result.l0)
        snapshot.assert_match(safe_l0)

    def test_l1_sections_format(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_l1 = _strip_timestamps(result.l1)
        snapshot.assert_match(safe_l1)

    def test_l2_chunks_format(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_chunks = _strip_timestamps(result.chunks)
        snapshot.assert_match(safe_chunks)

    def test_l2_index_format(self, snapshot: IRSnapshot):
        result = _build_result(_projection_snapshot())
        safe_l2 = _strip_timestamps(result.l2_index)
        snapshot.assert_match(safe_l2)

    def test_result_dict_keys(self, snapshot: IRSnapshot):
        """Full result dict (minus timestamps) has stable structure."""
        result = _build_result(_projection_snapshot())
        safe_dict = _strip_timestamps(result.to_dict())
        snapshot.assert_match(safe_dict)

    @pytest.mark.edge
    def test_projection_minimal_symbol_deterministic(self, snapshot: IRSnapshot):
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
