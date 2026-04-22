"""Property-based tests for ir_merge.merge_ir invariants."""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from fastcode.ir_merge import merge_ir
from fastcode.semantic_ir import (
    IRAttachment,
    IRDocument,
    IREdge,
    IROccurrence,
    IRSnapshot,
    IRSymbol,
)


# --- Helpers ---

def _doc(doc_id: str, path: str, source: str = "fc_structure") -> IRDocument:
    return IRDocument(doc_id=doc_id, path=path, language="python", source_set={source})


def _sym(symbol_id: str, path: str, display_name: str, kind: str = "function",
         start_line: int = 1, source_priority: int = 50,
         source: str = "fc_structure") -> IRSymbol:
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


def _occ(occ_id: str, symbol_id: str, doc_id: str, role: str = "definition",
         start_line: int = 1, start_col: int = 0,
         end_line: int = 1, end_col: int = 10,
         source: str = "fc_structure") -> IROccurrence:
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


def _edge(edge_id: str, src: str, dst: str, edge_type: str = "call",
           source: str = "fc_structure") -> IREdge:
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
    payload = {"vector": [0.1, 0.2], "text": "hello"} if attachment_type == "embedding" else {"text": "hello"}
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


small_id = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)
small_snapshot = st.builds(
    IRSnapshot,
    repo_name=st.just("repo"),
    snapshot_id=st.just("snap:repo:abc"),
    documents=st.lists(
        st.builds(_doc, st.builds(lambda x: f"d{x}", small_id), st.builds(lambda x: f"{x}.py", small_id)),
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
    occurrences=st.lists(st.just(_occ("occ:1", "sym:a", "d1", source="fc_structure")), max_size=3),
    edges=st.lists(st.just(_edge("e:1", "d1", "sym:a", source="fc_structure")), max_size=2),
)


# --- Properties ---

@pytest.mark.property
class TestMergeIrProperties:

    @given(snapshot=small_snapshot)
    @settings(max_examples=50)
    @pytest.mark.edge
    def test_merge_with_none_returns_original(self, snapshot: IRSnapshot):
        """EDGE: merge_ir(ast, None) returns ast unchanged (None input)."""
        result = merge_ir(snapshot, None)
        assert result is snapshot

    @given(snapshot=small_snapshot)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_merge_idempotent(self, snapshot: IRSnapshot):
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
    @pytest.mark.happy
    def test_scip_wins_on_overlap(self, display_name: str, path: str,
                                   kind: str, start_line: int):
        """HAPPY: when AST and SCIP symbols share a key, SCIP survives."""
        ast_sym = _sym(
            f"ast:{display_name}", path, display_name, kind,
            start_line=start_line, source_priority=50, source="fc_structure",
        )
        scip_sym = _sym(
            f"scip:{display_name}", path, display_name, kind,
            start_line=start_line, source_priority=100, source="scip",
        )
        doc = _doc("d1", path)
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=[doc], symbols=[ast_sym],
        )
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=[doc], symbols=[scip_sym],
        )
        merged = merge_ir(ast, scip)
        merged_ids = {s.symbol_id for s in merged.symbols}
        assert f"scip:{display_name}" in merged_ids
        assert f"ast:{display_name}" not in merged_ids

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
    @pytest.mark.happy
    def test_edge_coexistence(self, ast_edges: list[IREdge], scip_edges: list[IREdge]):
        """HAPPY: merged edge set is union (deduped by src_id, dst_id, edge_type)."""
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            edges=ast_edges,
        )
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            edges=scip_edges,
        )
        merged = merge_ir(ast, scip)
        merged_keys = {(e.src_id, e.dst_id, e.edge_type) for e in merged.edges}
        ast_keys = {(e.src_id, e.dst_id, e.edge_type) for e in ast_edges}
        scip_keys = {(e.src_id, e.dst_id, e.edge_type) for e in scip_edges}
        assert ast_keys | scip_keys == merged_keys

    @given(snapshot_id=small_id)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_snapshot_identity_preserved(self, snapshot_id: str):
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
    def test_occurrence_dedup_scip_wins(self, n_occs: int, start_line: int):
        """EDGE: duplicate occurrences deduplicate, SCIP source wins."""
        doc = _doc("d1", "a.py")
        ast_sym = _sym("ast:s1", "a.py", "foo", start_line=start_line, source="fc_structure")
        scip_sym = _sym("scip:s1", "a.py", "foo", start_line=start_line, source="scip")

        ast_occs = [
            _occ(f"ast:occ:{i}", "ast:s1", "d1", "definition",
                 start_line=start_line, source="fc_structure")
            for i in range(n_occs)
        ]
        scip_occs = [
            _occ(f"scip:occ:{i}", "scip:s1", "d1", "definition",
                 start_line=start_line, source="scip")
            for i in range(n_occs)
        ]

        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=[doc], symbols=[ast_sym], occurrences=ast_occs,
        )
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=[doc], symbols=[scip_sym], occurrences=scip_occs,
        )
        merged = merge_ir(ast, scip)
        assert len(merged.occurrences) == 1
        assert merged.occurrences[0].source == "scip"

    @given(n=st.integers(min_value=0, max_value=5))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_empty_ast_snapshot(self, n: int):
        """EDGE: merge with empty AST returns SCIP content unchanged."""
        scip_syms = [
            _sym(f"scip:s{i}", f"{i}.py", f"fn_{i}", source="scip")
            for i in range(n)
        ]
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            symbols=scip_syms,
        )
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.symbols) == n

    @given(n=st.integers(min_value=0, max_value=5))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_empty_scip_snapshot(self, n: int):
        """EDGE: merge with empty SCIP returns AST content unchanged."""
        ast_syms = [
            _sym(f"ast:s{i}", f"{i}.py", f"fn_{i}", source="fc_structure")
            for i in range(n)
        ]
        ast = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            symbols=ast_syms,
        )
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.symbols) == n

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_duplicate_edges_deduped(self, n: int):
        """EDGE: identical edges deduplicated across AST and SCIP."""
        edge = _edge("e:1", "a", "b", "call", "fc_structure")
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", edges=[edge] * n)
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            edges=[_edge("e:1", "a", "b", "call", "scip")],
        )
        merged = merge_ir(ast, scip)
        assert len(merged.edges) == 1

    @given(snapshot=small_snapshot)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_both_empty_snapshots(self, snapshot: IRSnapshot):
        """EDGE: merging two empty snapshots produces empty result."""
        empty = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(empty, empty)
        assert len(merged.symbols) == 0
        assert len(merged.occurrences) == 0
        assert len(merged.edges) == 0

    @given(snapshot=small_snapshot)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_source_modes_union(self, snapshot: IRSnapshot):
        """EDGE: merged source_modes is union of both inputs."""
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            metadata={"source_modes": ["scip"]},
        )
        merged = merge_ir(snapshot, scip)
        modes = merged.metadata.get("source_modes", [])
        assert "scip" in modes

    @given(snapshot=small_snapshot)
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_merge_preserves_document_source_sets(self, snapshot: IRSnapshot):
        """EDGE: merged documents union source sets from both inputs."""
        if not snapshot.documents:
            return
        doc = snapshot.documents[0]
        scip_doc = _doc(doc.doc_id, doc.path, source="scip")
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=[scip_doc],
        )
        merged = merge_ir(snapshot, scip)
        merged_doc = next((d for d in merged.documents if d.doc_id == doc.doc_id), None)
        if merged_doc:
            assert "scip" in merged_doc.source_set or "fc_structure" in merged_doc.source_set

    @given(n_ast=st.integers(min_value=1, max_value=3), n_scip=st.integers(min_value=1, max_value=3))
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_merge_documents_union(self, n_ast: int, n_scip: int):
        """EDGE: merged documents is union (no duplicates on doc_id)."""
        ast_docs = [_doc(f"ast_d{i}", f"a{i}.py", "fc_structure") for i in range(n_ast)]
        scip_docs = [_doc(f"scip_d{i}", f"s{i}.py", "scip") for i in range(n_scip)]
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", documents=ast_docs)
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", documents=scip_docs)
        merged = merge_ir(ast, scip)
        merged_ids = {d.doc_id for d in merged.documents}
        assert len(merged_ids) == n_ast + n_scip

    @given(n=st.integers(min_value=1, max_value=5))
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_merge_scip_only_occurrences_preserved(self, n: int):
        """EDGE: SCIP-only occurrences survive merge when AST has none."""
        doc = _doc("d1", "a.py")
        occs = [
            _occ(f"scip:occ:{i}", f"scip:s{i}", "d1", "reference",
                 start_line=i + 1, source="scip")
            for i in range(n)
        ]
        scip = IRSnapshot(
            repo_name="repo", snapshot_id="snap:repo:abc",
            documents=[doc], occurrences=occs,
        )
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.occurrences) == n

    @pytest.mark.edge
    def test_merge_ast_only_edges_preserved(self):
        """EDGE: AST-only edges survive when SCIP has none."""
        edge = _edge("e:1", "a", "b", "call", "fc_structure")
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", edges=[edge])
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert len(merged.edges) == 1
        assert merged.edges[0].source == "fc_structure"

    @pytest.mark.edge
    def test_merge_preserves_branch(self):
        """EDGE: branch from AST preserved in merge."""
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", branch="dev")
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert merged.branch == "dev"

    @pytest.mark.edge
    def test_merge_preserves_commit_id(self):
        """EDGE: commit_id from AST preserved in merge."""
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", commit_id="abc123")
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc")
        merged = merge_ir(ast, scip)
        assert merged.commit_id == "abc123"

    @pytest.mark.edge
    def test_merge_documents_with_same_id_union_source(self):
        """EDGE: same doc_id from both sources unions source_set."""
        doc_ast = _doc("d1", "a.py", "fc_structure")
        doc_scip = _doc("d1", "a.py", "scip")
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", documents=[doc_ast])
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", documents=[doc_scip])
        merged = merge_ir(ast, scip)
        d = next(d for d in merged.documents if d.doc_id == "d1")
        assert "fc_structure" in d.source_set
        assert "scip" in d.source_set

    @pytest.mark.edge
    def test_merge_no_overlap_symbols_coexist(self):
        """EDGE: non-overlapping symbols from AST and SCIP both survive."""
        ast_sym = _sym("ast:s1", "a.py", "foo", source="fc_structure")
        scip_sym = _sym("scip:s2", "b.py", "bar", source="scip")
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", symbols=[ast_sym])
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc", symbols=[scip_sym])
        merged = merge_ir(ast, scip)
        ids = {s.symbol_id for s in merged.symbols}
        assert "ast:s1" in ids
        assert "scip:s2" in ids

    @pytest.mark.edge
    def test_merge_occurrence_dedup_same_role(self):
        """EDGE: occurrences with same symbol/doc/role deduplicate."""
        doc = _doc("d1", "a.py")
        occ1 = _occ("ast:o1", "sym:1", "d1", "definition", start_line=5, source="fc_structure")
        occ2 = _occ("scip:o2", "sym:1", "d1", "definition", start_line=5, source="scip")
        ast = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc",
                         documents=[doc], occurrences=[occ1])
        scip = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:abc",
                         documents=[doc], occurrences=[occ2])
        merged = merge_ir(ast, scip)
        assert len(merged.occurrences) == 1
        assert merged.occurrences[0].source == "scip"

    @pytest.mark.edge
    def test_merge_retargets_and_deduplicates_semantically_identical_attachments(self):
        """EDGE: structure attachments retarget to SCIP symbols and collapse by semantic key."""
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
        assert len(merged.attachments) == 1
        assert merged.attachments[0].target_id == "scip:s1"
