from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.ir_merge import merge_ir
from fastcode.ir_validators import validate_snapshot
from fastcode.semantic_ir import IRDocument, IREdge, IROccurrence, IRSnapshot, IRSymbol


def _doc(doc_id: str, path: str) -> IRDocument:
    return IRDocument(doc_id=doc_id, path=path, language="python", source_set={"ast"})


def test_merge_prefers_scip_symbol_and_preserves_alias():
    ast_sym = IRSymbol(
        symbol_id="ast:s1",
        external_symbol_id=None,
        path="a.py",
        display_name="foo",
        kind="function",
        language="python",
        start_line=1,
        source_priority=10,
        source_set={"ast"},
        metadata={"source": "ast"},
    )
    scip_sym = IRSymbol(
        symbol_id="scip:snap:1:foo",
        external_symbol_id="foo",
        path="a.py",
        display_name="foo",
        kind="function",
        language="python",
        start_line=1,
        source_priority=100,
        source_set={"scip"},
        metadata={"source": "scip"},
    )
    ast = IRSnapshot(repo_name="r", snapshot_id="snap:1", documents=[_doc("d1", "a.py")], symbols=[ast_sym])
    scip = IRSnapshot(repo_name="r", snapshot_id="snap:1", documents=[_doc("d1", "a.py")], symbols=[scip_sym])
    merged = merge_ir(ast, scip)
    assert any(s.symbol_id == "scip:snap:1:foo" for s in merged.symbols)
    s = next(s for s in merged.symbols if s.symbol_id == "scip:snap:1:foo")
    assert "ast:s1" in (s.metadata.get("aliases") or [])


def test_validate_snapshot_catches_missing_nodes_and_provenance():
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="s",
        documents=[_doc("d1", "a.py")],
        symbols=[
            IRSymbol(
                symbol_id="sym:1",
                external_symbol_id=None,
                path="a.py",
                display_name="x",
                kind="function",
                language="python",
                source_priority=0,
                source_set=set(),
                metadata={},
            )
        ],
        occurrences=[
            IROccurrence(
                occurrence_id="o1",
                symbol_id="sym:missing",
                doc_id="d1",
                role="reference",
                start_line=1,
                start_col=0,
                end_line=1,
                end_col=1,
                source="ast",
                metadata={},
            )
        ],
        edges=[
            IREdge(
                edge_id="e1",
                src_id="d1",
                dst_id="sym:missing",
                edge_type="ref",
                source="",
                confidence="",
            )
        ],
    )
    errors = validate_snapshot(snap)
    assert any("missing symbol_id" in e for e in errors)
    assert any("edge dst not found" in e for e in errors)
    assert any("edge source missing" in e for e in errors)


def test_merge_deduplicates_occurrences_scip_wins():
    """Rule D: when AST and SCIP produce the same occurrence, SCIP wins."""
    doc = _doc("d1", "a.py")
    scip_occ = IROccurrence(
        occurrence_id="occ:scip:1",
        symbol_id="scip:snap:1:foo",
        doc_id="d1",
        role="definition",
        start_line=10,
        start_col=0,
        end_line=20,
        end_col=0,
        source="scip",
        metadata={"source": "scip"},
    )
    ast_occ = IROccurrence(
        occurrence_id="occ:ast:1",
        symbol_id="ast:s1",
        doc_id="d1",
        role="definition",
        start_line=10,
        start_col=0,
        end_line=20,
        end_col=0,
        source="ast",
        metadata={"source": "ast"},
    )
    ast_sym = IRSymbol(
        symbol_id="ast:s1", external_symbol_id=None, path="a.py",
        display_name="foo", kind="function", language="python",
        start_line=10, source_priority=10, source_set={"ast"},
        metadata={"source": "ast"},
    )
    scip_sym = IRSymbol(
        symbol_id="scip:snap:1:foo", external_symbol_id="foo", path="a.py",
        display_name="foo", kind="function", language="python",
        start_line=10, source_priority=100, source_set={"scip"},
        metadata={"source": "scip"},
    )
    ast = IRSnapshot(repo_name="r", snapshot_id="snap:1", documents=[doc], symbols=[ast_sym], occurrences=[ast_occ])
    scip = IRSnapshot(repo_name="r", snapshot_id="snap:1", documents=[doc], symbols=[scip_sym], occurrences=[scip_occ])
    merged = merge_ir(ast, scip)
    # After merge, both occurrences map to canonical symbol "scip:snap:1:foo"
    # They share the same (symbol_id, doc_id, role, start_line, start_col, end_line, end_col)
    # so only one should survive, and it should be the SCIP-sourced one
    assert len(merged.occurrences) == 1
    assert merged.occurrences[0].source == "scip"


def test_ir_graph_builder_routes_edge_types():
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="s",
        documents=[_doc("d1", "a.py")],
        symbols=[
            IRSymbol(
                symbol_id="sym:1",
                external_symbol_id=None,
                path="a.py",
                display_name="x",
                kind="function",
                language="python",
                source_priority=0,
                source_set={"ast"},
                metadata={"source": "ast"},
            )
        ],
        edges=[
            IREdge(edge_id="e1", src_id="d1", dst_id="sym:1", edge_type="contain", source="ast", confidence="resolved"),
            IREdge(edge_id="e2", src_id="sym:1", dst_id="sym:1", edge_type="call", source="ast", confidence="heuristic"),
        ],
    )
    graphs = IRGraphBuilder().build_graphs(snap)
    assert graphs.containment_graph.number_of_edges() == 1
    assert graphs.call_graph.number_of_edges() == 1
