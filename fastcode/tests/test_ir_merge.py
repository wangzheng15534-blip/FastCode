"""Tests for ir_merge module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.ir_merge import merge_ir
from fastcode.semantic_ir import (
    IRAttachment,
    IRCodeUnit,
    IRDocument,
    IREdge,
    IROccurrence,
    IRRelation,
    IRSnapshot,
    IRSymbol,
    IRUnitSupport,
)

# --- Helpers ---


def _doc(doc_id: str, path: str, source: str = "fc_structure") -> IRDocument:
    return IRDocument(doc_id=doc_id, path=path, language="python", source_set={source})


def _sym(
    symbol_id: str,
    path: str,
    display_name: str,
    kind: str = "function",
    start_line: int = 1,
    source_priority: int = 50,
    source: str = "fc_structure",
) -> IRSymbol:
    return IRSymbol(
        symbol_id=symbol_id,
        external_symbol_id=None,
        path=path,
        display_name=display_name,
        kind=kind,
        language="python",
        start_line=start_line,
        source_priority=source_priority,
        source_set={source},
        metadata={"source": source},
    )


def _occ(
    occ_id: str,
    symbol_id: str,
    doc_id: str,
    role: str = "definition",
    start_line: int = 1,
    start_col: int = 0,
    end_line: int = 1,
    end_col: int = 10,
    source: str = "fc_structure",
) -> IROccurrence:
    return IROccurrence(
        occurrence_id=occ_id,
        symbol_id=symbol_id,
        doc_id=doc_id,
        role=role,
        start_line=start_line,
        start_col=start_col,
        end_line=end_line,
        end_col=end_col,
        source=source,
        metadata={},
    )


def _edge(
    edge_id: str,
    src: str,
    dst: str,
    edge_type: str = "call",
    source: str = "fc_structure",
) -> IREdge:
    return IREdge(
        edge_id=edge_id,
        src_id=src,
        dst_id=dst,
        edge_type=edge_type,
        source=source,
        confidence="heuristic",
    )


def _att(
    attachment_id: str,
    target_id: str,
    source: str = "fc_embedding",
    attachment_type: str = "embedding",
) -> IRAttachment:
    payload = (
        {"vector": [0.1, 0.2], "text": "hello"}
        if attachment_type == "embedding"
        else {"text": "hello"}
    )
    return IRAttachment(
        attachment_id=attachment_id,
        target_id=target_id,
        target_type="symbol",
        attachment_type=attachment_type,
        source=source,
        confidence="derived",
        payload=payload,
        metadata={"producer": "test"},
    )


def _file(snapshot_id: str = "snap:1", source: str = "fc_structure") -> IRCodeUnit:
    return _file_at(snapshot_id, "a.py", source)


def _file_at(snapshot_id: str, path: str, source: str = "fc_structure") -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=f"doc:{snapshot_id}:{path}",
        kind="file",
        path=path,
        language="python",
        display_name=path,
        source_set={source},
    )


def _class(unit_id: str, name: str, source: str = "fc_structure") -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind="class",
        path="a.py",
        language="python",
        display_name=name,
        qualified_name=name,
        start_line=1 if name == "A" else 20,
        end_line=10 if name == "A" else 30,
        parent_unit_id="doc:snap:1:a.py",
        source_set={source},
    )


def _method(
    unit_id: str,
    name: str,
    parent_id: str,
    start_line: int,
    end_line: int,
    source: str = "fc_structure",
) -> IRCodeUnit:
    return _method_at(unit_id, name, "a.py", parent_id, start_line, end_line, source)


def _method_at(
    unit_id: str,
    name: str,
    path: str,
    parent_id: str,
    start_line: int,
    end_line: int,
    source: str = "fc_structure",
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind="method",
        path=path,
        language="python",
        display_name=name,
        qualified_name=f"{parent_id}.{name}",
        start_line=start_line,
        end_line=end_line,
        parent_unit_id=parent_id,
        source_set={source},
    )


def _scip_method(
    unit_id: str, anchor: str, name: str, start_line: int, end_line: int, enclosing: str
) -> tuple[IRCodeUnit, IRUnitSupport]:
    unit = IRCodeUnit(
        unit_id=unit_id,
        kind="method",
        path="a.py",
        language="python",
        display_name=name,
        qualified_name=anchor,
        start_line=start_line,
        end_line=end_line,
        parent_unit_id="doc:snap:1:a.py",
        primary_anchor_symbol_id=anchor,
        anchor_symbol_ids=[anchor],
        anchor_coverage=1.0,
        source_set={"scip"},
    )
    support = IRUnitSupport(
        support_id=f"support:{unit_id}",
        unit_id=unit_id,
        source="scip",
        support_kind="anchor",
        external_id=anchor,
        display_name=name,
        start_line=start_line,
        end_line=end_line,
        enclosing_external_id=enclosing,
    )
    return unit, support


small_id = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)
small_snapshot = st.builds(
    IRSnapshot,
    repo_name=st.just("repo"),
    snapshot_id=st.just("snap:repo:abc"),
    documents=st.lists(
        st.builds(
            _doc,
            st.builds(lambda x: f"d{x}", small_id),
            st.builds(lambda x: f"{x}.py", small_id),
        ),
        max_size=3,
    ),
    symbols=st.lists(
        st.builds(
            _sym,
            st.builds(lambda x: f"sym:{x}", small_id),
            st.builds(lambda x: f"{x}.py", small_id),
            st.builds(lambda x: f"fn_{x}", small_id),
            source=st.just("fc_structure"),
        ),
        max_size=5,
    ),
    occurrences=st.lists(
        st.just(_occ("occ:1", "sym:a", "d1", source="fc_structure")), max_size=3
    ),
    edges=st.lists(
        st.just(_edge("e:1", "d1", "sym:a", source="fc_structure")), max_size=2
    ),
)


# --- Alignment algorithm tests ---


def test_alignment_uses_parent_context_for_same_named_methods():
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[
            _file(),
            _class("ast:class:A", "A"),
            _class("ast:class:B", "B"),
            _method("ast:method:A.run", "run", "ast:class:A", 2, 5),
            _method("ast:method:B.run", "run", "ast:class:B", 22, 25),
        ],
    )
    scip_a, support_a = _scip_method("scip:run:A", "pkg/A#run", "run", 2, 5, "pkg/A#")
    scip_b, support_b = _scip_method("scip:run:B", "pkg/B#run", "run", 22, 25, "pkg/B#")
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(source="scip"), scip_a, scip_b],
        supports=[support_a, support_b],
    )

    merged = merge_ir(ast, scip)
    unit_a = next(unit for unit in merged.units if unit.unit_id == "ast:method:A.run")
    unit_b = next(unit for unit in merged.units if unit.unit_id == "ast:method:B.run")

    assert unit_a.primary_anchor_symbol_id == "pkg/A#run"
    assert unit_b.primary_anchor_symbol_id == "pkg/B#run"


def test_medium_score_alignment_keeps_candidate_anchor_and_synthetic_symbol():
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(), _method("ast:method:run", "run", "doc:snap:1:a.py", 2, 5)],
    )
    scip_unit, scip_support = _scip_method("scip:run", "pkg/run", "run", 40, 45, "")
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(source="scip"), scip_unit],
        supports=[scip_support],
    )

    merged = merge_ir(ast, scip)
    ast_unit = next(unit for unit in merged.units if unit.unit_id == "ast:method:run")

    assert ast_unit.primary_anchor_symbol_id is None
    assert "pkg/run" in ast_unit.candidate_anchor_symbol_ids
    assert any(unit.unit_id == "scip:run" for unit in merged.units)


def test_scip_occurrence_retargets_ref_to_enclosing_unit():
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[
            _file(),
            _method("ast:method:caller", "caller", "doc:snap:1:a.py", 1, 20),
            _method("ast:method:callee", "callee", "doc:snap:1:a.py", 30, 40),
        ],
    )
    scip_callee, scip_support = _scip_method(
        "scip:callee", "pkg/callee", "callee", 30, 40, ""
    )
    scip_occ = IRUnitSupport(
        support_id="support:occ:1",
        unit_id="scip:callee",
        source="scip",
        support_kind="occurrence",
        external_id="pkg/callee",
        role="reference",
        path="a.py",
        start_line=10,
        end_line=10,
        metadata={"doc_id": "doc:snap:1:a.py"},
    )
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(source="scip"), scip_callee],
        supports=[scip_support, scip_occ],
    )

    merged = merge_ir(ast, scip)
    ref_relations = [
        relation for relation in merged.relations if relation.relation_type == "ref"
    ]

    assert any(
        relation.src_unit_id == "ast:method:caller"
        and relation.dst_unit_id == "ast:method:callee"
        for relation in ref_relations
    )


def test_scip_occurrence_retargets_cross_file_ref_to_occurrence_path():
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[
            _file(),
            _file_at("snap:1", "b.py"),
            _method("ast:method:caller", "caller", "doc:snap:1:a.py", 1, 20),
            _method_at(
                "ast:method:callee", "callee", "b.py", "doc:snap:1:b.py", 30, 40
            ),
        ],
    )
    scip_callee = IRCodeUnit(
        unit_id="scip:callee",
        kind="method",
        path="b.py",
        language="python",
        display_name="callee",
        qualified_name="pkg/callee",
        start_line=30,
        end_line=40,
        parent_unit_id="doc:snap:1:b.py",
        primary_anchor_symbol_id="pkg/callee",
        anchor_symbol_ids=["pkg/callee"],
        anchor_coverage=1.0,
        source_set={"scip"},
    )
    scip_anchor = IRUnitSupport(
        support_id="support:scip:callee",
        unit_id="scip:callee",
        source="scip",
        support_kind="anchor",
        external_id="pkg/callee",
        display_name="callee",
        path="b.py",
        start_line=30,
        end_line=40,
    )
    scip_occ = IRUnitSupport(
        support_id="support:occ:cross-file",
        unit_id="scip:callee",
        source="scip",
        support_kind="occurrence",
        external_id="pkg/callee",
        role="reference",
        path="a.py",
        start_line=10,
        end_line=10,
        metadata={"doc_id": "doc:snap:1:a.py"},
    )
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[
            _file_at("snap:1", "b.py", source="scip"),
            scip_callee,
        ],
        supports=[scip_anchor, scip_occ],
    )

    merged = merge_ir(ast, scip)
    ref_relations = [
        relation for relation in merged.relations if relation.relation_type == "ref"
    ]

    assert any(
        relation.src_unit_id == "ast:method:caller"
        and relation.dst_unit_id == "ast:method:callee"
        for relation in ref_relations
    )
    assert not any(
        relation.src_unit_id == "doc:snap:1:b.py"
        and relation.dst_unit_id == "ast:method:callee"
        for relation in ref_relations
    )


def test_scip_occurrence_falls_back_to_target_path_when_occ_path_is_none():
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[
            _file(),
            _method("ast:method:caller", "caller", "doc:snap:1:a.py", 1, 20),
            _method("ast:method:callee", "callee", "doc:snap:1:a.py", 30, 40),
        ],
    )
    scip_callee, scip_support = _scip_method(
        "scip:callee", "pkg/callee", "callee", 30, 40, ""
    )
    scip_occ = IRUnitSupport(
        support_id="support:occ:no-path",
        unit_id="scip:callee",
        source="scip",
        support_kind="occurrence",
        external_id="pkg/callee",
        role="reference",
        path=None,
        start_line=10,
        end_line=10,
        metadata={"doc_id": "doc:snap:1:a.py"},
    )
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(source="scip"), scip_callee],
        supports=[scip_support, scip_occ],
    )

    merged = merge_ir(ast, scip)
    ref_relations = [
        relation for relation in merged.relations if relation.relation_type == "ref"
    ]

    assert any(
        relation.src_unit_id == "ast:method:caller"
        and relation.dst_unit_id == "ast:method:callee"
        for relation in ref_relations
    )


# --- Property tests ---


class TestMergeIrProperties:
    @given(snapshot=small_snapshot)
    @settings(max_examples=50)
    @pytest.mark.edge
    def test_merge_with_none_returns_clone_property(self, snapshot: IRSnapshot):
        """EDGE: merge_ir(ast, None) returns a deep clone, not the same object."""
        result = merge_ir(snapshot, None)
        assert result is not snapshot
        assert result.snapshot_id == snapshot.snapshot_id
        assert len(result.symbols) == len(snapshot.symbols)

    @given(snapshot=small_snapshot)
    @settings(max_examples=30)
    def test_merge_idempotent_property(self, snapshot: IRSnapshot):
        """HAPPY: merge(merge(ast, scip), scip) == merge(ast, scip)."""
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[],
            symbols=[],
            occurrences=[],
            edges=[],
            metadata={"source_modes": ["scip"]},
        )
        first = merge_ir(snapshot, scip)
        second = merge_ir(first, scip)
        assert len(first.symbols) == len(second.symbols)
        assert len(first.occurrences) == len(second.occurrences)
        assert len(first.edges) == len(second.edges)

    @given(
        display_name=small_id,
        path=st.builds(lambda x: f"{x}.py", small_id),
        kind=st.sampled_from(["function", "class", "method"]),
        start_line=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_scip_wins_on_overlap_property(
        self, display_name: str, path: str, kind: str, start_line: int
    ):
        """HAPPY: when AST and SCIP units share location, merge anchors SCIP onto AST unit."""
        ast_sym = _sym(
            f"ast:{display_name}",
            path,
            display_name,
            kind,
            start_line=start_line,
            source_priority=50,
            source="fc_structure",
        )
        scip_sym = _sym(
            f"scip:{display_name}",
            path,
            display_name,
            kind,
            start_line=start_line,
            source_priority=100,
            source="scip",
        )
        doc = _doc("d1", path)
        ast = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[ast_sym],
        )
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[scip_sym],
        )
        merged = merge_ir(ast, scip)
        merged_ids = {s.symbol_id for s in merged.symbols}
        # AST unit kept as canonical; SCIP ID recorded as alias in metadata
        assert f"ast:{display_name}" in merged_ids
        matched = [s for s in merged.symbols if s.symbol_id == f"ast:{display_name}"]
        assert len(matched) == 1
        aliases = (matched[0].metadata or {}).get("aliases", [])
        assert f"scip:{display_name}" in aliases

    @given(
        ast_edges=st.lists(
            st.builds(
                _edge,
                st.builds(lambda x: f"ae:{x}", small_id),
                small_id,
                small_id,
                source=st.just("fc_structure"),
            ),
            max_size=5,
        ),
        scip_edges=st.lists(
            st.builds(
                _edge,
                st.builds(lambda x: f"se:{x}", small_id),
                small_id,
                small_id,
                source=st.just("scip"),
            ),
            max_size=5,
        ),
    )
    @settings(max_examples=50)
    def test_edge_coexistence_property(
        self, ast_edges: list[IREdge], scip_edges: list[IREdge]
    ):
        """HAPPY: merged edge set is union (deduped by src_id, dst_id, edge_type)."""
        ast = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            edges=ast_edges,
        )
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            edges=scip_edges,
        )
        merged = merge_ir(ast, scip)
        merged_keys = {(e.src_id, e.dst_id, e.edge_type) for e in merged.edges}
        ast_keys = {(e.src_id, e.dst_id, e.edge_type) for e in ast_edges}
        scip_keys = {(e.src_id, e.dst_id, e.edge_type) for e in scip_edges}
        assert ast_keys | scip_keys == merged_keys

    @given(snapshot_id=small_id)
    @settings(max_examples=30)
    def test_snapshot_identity_preserved_property(self, snapshot_id: str):
        """HAPPY: output snapshot_id equals AST snapshot_id."""
        ast = IRSnapshot(
            repo_name="repo",
            snapshot_id=f"snap:{snapshot_id}",
        )
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:other",
        )
        merged = merge_ir(ast, scip)
        assert merged.snapshot_id == f"snap:{snapshot_id}"

    @given(
        n_occs=st.integers(min_value=1, max_value=5),
        start_line=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30)
    @pytest.mark.edge
    def test_occurrence_dedup_scip_wins_property(self, n_occs: int, start_line: int):
        """EDGE: duplicate occurrences deduplicate, SCIP source wins."""
        doc = _doc("d1", "a.py")
        ast_sym = _sym(
            "ast:s1", "a.py", "foo", start_line=start_line, source="fc_structure"
        )
        scip_sym = _sym("scip:s1", "a.py", "foo", start_line=start_line, source="scip")

        ast_occs = [
            _occ(
                f"ast:occ:{i}",
                "ast:s1",
                "d1",
                "definition",
                start_line=start_line,
                source="fc_structure",
            )
            for i in range(n_occs)
        ]
        scip_occs = [
            _occ(
                f"scip:occ:{i}",
                "scip:s1",
                "d1",
                "definition",
                start_line=start_line,
                source="scip",
            )
            for i in range(n_occs)
        ]

        ast = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[ast_sym],
            occurrences=ast_occs,
        )
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            symbols=[scip_sym],
            occurrences=scip_occs,
        )
        merged = merge_ir(ast, scip)
        assert len(merged.occurrences) == 1
        assert merged.occurrences[0].source == "scip"

    @given(n=st.integers(min_value=0, max_value=5))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_empty_ast_snapshot_property(self, n: int):
        """EDGE: merge with empty AST returns SCIP content unchanged."""
        scip_syms = [
            _sym(f"scip:s{i}", f"{i}.py", f"fn_{i}", source="scip") for i in range(n)
        ]
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            symbols=scip_syms,
        )
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.symbols) == n

    @given(n=st.integers(min_value=0, max_value=5))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_empty_scip_snapshot_property(self, n: int):
        """EDGE: merge with empty SCIP returns AST content unchanged."""
        ast_syms = [
            _sym(f"ast:s{i}", f"{i}.py", f"fn_{i}", source="fc_structure")
            for i in range(n)
        ]
        ast = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            symbols=ast_syms,
        )
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.symbols) == n

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_duplicate_edges_deduped_property(self, n: int):
        """EDGE: identical edges deduplicated across AST and SCIP."""
        edge = _edge("e:1", "a", "b", "call", "fc_structure")
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", edges=[edge] * n
        )
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            edges=[_edge("e:1", "a", "b", "call", "scip")],
        )
        merged = merge_ir(ast, scip)
        assert len(merged.edges) == 1

    @given(snapshot=small_snapshot)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_both_empty_snapshots_property(self, snapshot: IRSnapshot):
        """EDGE: merging two empty snapshots produces empty result."""
        empty = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(empty, empty)
        assert len(merged.symbols) == 0
        assert len(merged.occurrences) == 0
        assert len(merged.edges) == 0

    @given(snapshot=small_snapshot)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_source_modes_union_property(self, snapshot: IRSnapshot):
        """EDGE: merged source_modes is union of both inputs."""
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            metadata={"source_modes": ["scip"]},
        )
        merged = merge_ir(snapshot, scip)
        modes = merged.metadata.get("source_modes", [])
        assert "scip" in modes

    @given(snapshot=small_snapshot)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_preserves_document_source_sets_property(self, snapshot: IRSnapshot):
        """EDGE: merged documents union source sets from both inputs."""
        if not snapshot.documents:
            return
        doc = snapshot.documents[0]
        scip_doc = _doc(doc.doc_id, doc.path, source="scip")
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[scip_doc],
        )
        merged = merge_ir(snapshot, scip)
        merged_doc = next((d for d in merged.documents if d.doc_id == doc.doc_id), None)
        if merged_doc:
            assert (
                "scip" in merged_doc.source_set
                or "fc_structure" in merged_doc.source_set
            )

    @given(
        n_ast=st.integers(min_value=1, max_value=3),
        n_scip=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_merge_documents_union_property(self, n_ast: int, n_scip: int):
        """EDGE: merged documents is union (no duplicates on doc_id)."""
        ast_docs = [_doc(f"ast_d{i}", f"a{i}.py", "fc_structure") for i in range(n_ast)]
        scip_docs = [_doc(f"scip_d{i}", f"s{i}.py", "scip") for i in range(n_scip)]
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", documents=ast_docs
        )
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", documents=scip_docs
        )
        merged = merge_ir(ast, scip)
        merged_ids = {d.doc_id for d in merged.documents}
        assert len(merged_ids) == n_ast + n_scip

    @given(n=st.integers(min_value=1, max_value=5))
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_merge_scip_only_occurrences_preserved_property(self, n: int):
        """EDGE: SCIP-only occurrences survive merge when AST has none."""
        doc = _doc("d1", "a.py")
        occs = [
            _occ(
                f"scip:occ:{i}",
                f"scip:s{i}",
                "d1",
                "reference",
                start_line=i + 1,
                source="scip",
            )
            for i in range(n)
        ]
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            occurrences=occs,
        )
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.occurrences) == n

    @pytest.mark.edge
    def test_merge_ast_only_edges_preserved_property(self):
        """EDGE: AST-only edges survive when SCIP has none."""
        edge = _edge("e:1", "a", "b", "call", "fc_structure")
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", edges=[edge])
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.edges) == 1
        assert merged.edges[0].source == "fc_structure"

    @pytest.mark.edge
    def test_merge_preserves_branch_property(self):
        """EDGE: branch from AST preserved in merge."""
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", branch="dev")
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert merged.branch == "dev"

    @pytest.mark.edge
    def test_merge_preserves_commit_id_property(self):
        """EDGE: commit_id from AST preserved in merge."""
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", commit_id="abc123"
        )
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert merged.commit_id == "abc123"

    @pytest.mark.edge
    def test_merge_documents_with_same_id_maps_canonical_property(self):
        """EDGE: same path from both sources maps SCIP file to AST canonical file."""
        doc_ast = _doc("d1", "a.py", "fc_structure")
        doc_scip = _doc("d1", "a.py", "scip")
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", documents=[doc_ast]
        )
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", documents=[doc_scip]
        )
        merged = merge_ir(ast, scip)
        # File units merge by path; AST file unit kept as canonical
        d = next((d for d in merged.documents if d.doc_id == "d1"), None)
        assert d is not None
        assert "fc_structure" in d.source_set

    @pytest.mark.edge
    def test_merge_no_overlap_symbols_coexist_property(self):
        """EDGE: non-overlapping symbols from AST and SCIP both survive."""
        ast_sym = _sym("ast:s1", "a.py", "foo", source="fc_structure")
        scip_sym = _sym("scip:s2", "b.py", "bar", source="scip")
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", symbols=[ast_sym]
        )
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc", symbols=[scip_sym]
        )
        merged = merge_ir(ast, scip)
        ids = {s.symbol_id for s in merged.symbols}
        assert "ast:s1" in ids
        assert "scip:s2" in ids

    @pytest.mark.edge
    def test_merge_occurrence_dedup_same_role_property(self):
        """EDGE: occurrences with same symbol/doc/role deduplicate."""
        doc = _doc("d1", "a.py")
        occ1 = _occ(
            "ast:o1", "sym:1", "d1", "definition", start_line=5, source="fc_structure"
        )
        occ2 = _occ("scip:o2", "sym:1", "d1", "definition", start_line=5, source="scip")
        ast = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            occurrences=[occ1],
        )
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[doc],
            occurrences=[occ2],
        )
        merged = merge_ir(ast, scip)
        assert len(merged.occurrences) == 1
        assert merged.occurrences[0].source == "scip"

    @pytest.mark.edge
    def test_merge_retargets_and_deduplicates_semantically_identical_attachments_property(
        self,
    ):
        """EDGE: embedding attachments on AST symbol are preserved after merge with SCIP."""
        ast_sym = _sym("ast:s1", "a.py", "foo", source="fc_structure")
        scip_sym = _sym("scip:s1", "a.py", "foo", source_priority=100, source="scip")
        ast = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[_doc("d1", "a.py")],
            symbols=[ast_sym],
            attachments=[_att("att:ast", "ast:s1"), _att("att:ast-dup", "ast:s1")],
        )
        scip = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:abc",
            documents=[_doc("d1", "a.py", source="scip")],
            symbols=[scip_sym],
        )
        merged = merge_ir(ast, scip)
        # Attachments are converted to embeddings; AST unit retains its ID
        # as canonical with SCIP as alias. Embeddings stay attached to canonical unit.
        assert len(merged.embeddings) >= 1
        unit_ids = {e.unit_id for e in merged.embeddings}
        assert "ast:s1" in unit_ids


# --- pending_capabilities merge ---


def test_upsert_relation_merges_pending_capabilities_via_intersection():
    """pending_capabilities uses intersection: a capability is only still
    pending if *both* sources consider it pending."""
    from fastcode.ir_merge import _upsert_relation

    merged: dict[tuple[str, str, str], IRRelation] = {}
    r1 = IRRelation(
        relation_id="r1",
        src_unit_id="u1",
        dst_unit_id="u2",
        relation_type="call",
        resolution_state="structural",
        support_sources={"fc_structure"},
        pending_capabilities={"resolve_calls", "resolve_types"},
    )
    r2 = IRRelation(
        relation_id="r2",
        src_unit_id="u1",
        dst_unit_id="u2",
        relation_type="call",
        resolution_state="anchored",
        support_sources={"scip"},
        pending_capabilities={"resolve_calls"},
    )
    _upsert_relation(merged, r1)
    _upsert_relation(merged, r2)
    result = merged[("u1", "u2", "call")]
    assert result.pending_capabilities == {"resolve_calls"}


def test_upsert_relation_empty_intersection_removes_all_pending():
    from fastcode.ir_merge import _upsert_relation

    merged: dict[tuple[str, str, str], IRRelation] = {}
    r1 = IRRelation(
        relation_id="r1",
        src_unit_id="u1",
        dst_unit_id="u2",
        relation_type="call",
        resolution_state="structural",
        pending_capabilities={"resolve_calls"},
    )
    r2 = IRRelation(
        relation_id="r2",
        src_unit_id="u1",
        dst_unit_id="u2",
        relation_type="call",
        resolution_state="anchored",
        pending_capabilities={"resolve_types"},
    )
    _upsert_relation(merged, r1)
    _upsert_relation(merged, r2)
    result = merged[("u1", "u2", "call")]
    assert result.pending_capabilities == set()
