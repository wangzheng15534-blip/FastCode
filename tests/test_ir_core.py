from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.ir_merge import merge_ir
from fastcode.ir_validators import validate_snapshot
from fastcode.semantic_ir import IRDocument, IRAttachment, IREdge, IROccurrence, IRSnapshot, IRSymbol


def _doc(doc_id: str, path: str) -> IRDocument:
    return IRDocument(doc_id=doc_id, path=path, language="python", source_set={"fc_structure"})


def _att(
    attachment_id: str,
    target_id: str,
    target_type: str = "symbol",
    attachment_type: str = "embedding",
    source: str = "fc_embedding",
) -> IRAttachment:
    payload = {"text": "hello"} if attachment_type == "summary" else {"vector": [0.1, 0.2], "text": "hello"}
    return IRAttachment(
        attachment_id=attachment_id,
        target_id=target_id,
        target_type=target_type,
        attachment_type=attachment_type,
        source=source,
        confidence="derived",
        payload=payload,
        metadata={"producer": "test"},
    )


def test_merge_prefers_scip_symbol_and_preserves_alias():
    ast_sym = IRSymbol(
        symbol_id="ast:s1",
        external_symbol_id=None,
        path="a.py",
        display_name="foo",
        kind="function",
        language="python",
        start_line=1,
        source_priority=50,
        source_set={"fc_structure"},
        metadata={"source": "fc_structure"},
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
                source="fc_structure",
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
        source="fc_structure",
        metadata={"source": "fc_structure"},
    )
    ast_sym = IRSymbol(
        symbol_id="ast:s1", external_symbol_id=None, path="a.py",
        display_name="foo", kind="function", language="python",
        start_line=10, source_priority=50, source_set={"fc_structure"},
        metadata={"source": "fc_structure"},
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


def test_merge_retargets_symbol_attachments_to_canonical_symbol():
    ast_sym = IRSymbol(
        symbol_id="ast:s1",
        external_symbol_id=None,
        path="a.py",
        display_name="foo",
        kind="function",
        language="python",
        start_line=1,
        source_priority=50,
        source_set={"fc_structure"},
        metadata={"source": "fc_structure"},
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
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        documents=[_doc("d1", "a.py")],
        symbols=[ast_sym],
        attachments=[_att("att:ast:1", "ast:s1")],
    )
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        documents=[_doc("d1", "a.py")],
        symbols=[scip_sym],
    )
    merged = merge_ir(ast, scip)
    assert len(merged.attachments) == 1
    assert merged.attachments[0].target_id == "scip:snap:1:foo"
    assert merged.attachments[0].target_type == "symbol"


def test_ast_symbol_id_uses_qualified_name_and_start_col():
    """Spec: ast:{snapshot_id}:{language}:{file_path}:{kind}:{qualified_name}:{start_line}:{start_col}"""
    from fastcode.indexer import CodeElement
    from fastcode.adapters.ast_to_ir import _ast_symbol_id

    elem = CodeElement(
        id="el1", name="MyClass.my_method", type="method",
        language="python", relative_path="src/app.py",
        file_path="/repo/src/app.py",
        start_line=42, end_line=50,
        code="", signature=None, docstring=None, summary=None,
        metadata={
            "qualified_name": "pkg.src.app.MyClass.my_method",
            "start_col": 8,
        },
    )
    sid = _ast_symbol_id("snap:abc", elem)
    # Must contain qualified_name value and start_col, not name and end_line
    assert "pkg.src.app.MyClass.my_method" in sid
    assert ":42:8" in sid
    assert ":50:" not in sid  # end_line should not appear as position


def test_ast_symbol_id_falls_back_to_name_when_no_qualified_name():
    """When qualified_name is absent from metadata, fall back to elem.name."""
    from fastcode.indexer import CodeElement
    from fastcode.adapters.ast_to_ir import _ast_symbol_id

    elem = CodeElement(
        id="el2", name="my_func", type="function",
        language="python", relative_path="lib/util.py",
        file_path="/repo/lib/util.py",
        start_line=10, end_line=20,
        code="", signature=None, docstring=None, summary=None,
        metadata={},
    )
    sid = _ast_symbol_id("snap:xyz", elem)
    assert "my_func" in sid
    assert ":10:0" in sid  # start_col defaults to 0


def test_validate_catches_duplicate_doc_paths():
    """Spec: document paths must be unique inside snapshot."""
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="s",
        documents=[
            IRDocument(doc_id="d1", path="a.py", language="python", source_set={"fc_structure"}),
            IRDocument(doc_id="d2", path="a.py", language="python", source_set={"fc_structure"}),
        ],
        symbols=[
            IRSymbol(
                symbol_id="s1", external_symbol_id=None, path="a.py",
                display_name="x", kind="function", language="python",
                source_priority=0, source_set={"fc_structure"}, metadata={"source": "fc_structure"},
            )
        ],
    )
    errors = validate_snapshot(snap)
    assert any("duplicate document path" in e for e in errors)


def test_validate_snapshot_catches_missing_attachment_target():
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
                source_set={"fc_structure"},
                metadata={"source": "fc_structure"},
            )
        ],
        attachments=[_att("att:missing", "sym:missing")],
    )
    errors = validate_snapshot(snap)
    assert any("attachment target not found" in e for e in errors)


def test_scip_edges_include_extractor_field():
    """SCIP edges should carry extractor field for consistency with AST edges."""
    from fastcode.adapters.scip_to_ir import build_ir_from_scip

    scip = {
        "indexer_name": "scip-python",
        "indexer_version": "0.1.0",
        "documents": [
            {
                "path": "a.py",
                "language": "python",
                "symbols": [{"symbol": "pkg a/Foo.", "name": "Foo", "kind": "class"}],
                "occurrences": [
                    {"symbol": "pkg a/Foo.", "role": "reference", "range": [5, 0, 5, 3]},
                ],
            }
        ],
    }
    snap = build_ir_from_scip(repo_name="r", snapshot_id="s:1", scip_index=scip)
    for edge in snap.edges:
        assert "extractor" in (edge.metadata or {}), f"missing extractor in {edge.edge_type} edge"
        assert edge.metadata["extractor"] == "fastcode.adapters.scip_to_ir"


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
                source_set={"fc_structure"},
                metadata={"source": "fc_structure"},
            )
        ],
        edges=[
            IREdge(
                edge_id="e1",
                src_id="d1",
                dst_id="sym:1",
                edge_type="contain",
                source="fc_structure",
                confidence="resolved",
            ),
            IREdge(
                edge_id="e2",
                src_id="sym:1",
                dst_id="sym:1",
                edge_type="call",
                source="fc_structure",
                confidence="heuristic",
            ),
        ],
    )
    graphs = IRGraphBuilder().build_graphs(snap)
    assert graphs.containment_graph.number_of_edges() == 1
    assert graphs.call_graph.number_of_edges() == 1
