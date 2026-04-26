from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.ir_graph_builder import IRGraphBuilder
from fastcode.ir_merge import merge_ir
from fastcode.ir_validators import validate_snapshot
from fastcode.semantic_ir import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)


def _file(
    unit_id: str = "doc:snap:1:a.py", path: str = "a.py", source: str = "fc_structure"
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind="file",
        path=path,
        language="python",
        display_name=path,
        source_set={source},
    )


def _structure_unit(
    unit_id: str = "ast:s1",
    path: str = "a.py",
    name: str = "foo",
    kind: str = "function",
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind=kind,
        path=path,
        language="python",
        display_name=name,
        qualified_name=name,
        start_line=10,
        end_line=20,
        parent_unit_id="doc:snap:1:a.py",
        source_set={"fc_structure"},
        metadata={"source": "fc_structure"},
    )


def _scip_unit(
    unit_id: str = "scip:snap:1:foo",
    path: str = "a.py",
    name: str = "foo",
    kind: str = "function",
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind=kind,
        path=path,
        language="python",
        display_name=name,
        qualified_name=name,
        start_line=10,
        end_line=20,
        parent_unit_id="doc:snap:1:a.py",
        primary_anchor_symbol_id="pkg/foo",
        anchor_symbol_ids=["pkg/foo"],
        anchor_coverage=1.0,
        source_set={"scip"},
        metadata={"source": "scip"},
    )


def test_merge_grounds_scip_anchor_onto_structure_unit():
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(), _structure_unit()],
        supports=[
            IRUnitSupport(
                support_id="support:ast:1",
                unit_id="ast:s1",
                source="fc_structure",
                support_kind="structure",
                display_name="foo",
                start_line=10,
                end_line=20,
            )
        ],
        embeddings=[
            IRUnitEmbedding(
                embedding_id="emb:1",
                unit_id="ast:s1",
                source="fc_embedding",
                embedding_text="foo function",
                vector=[0.1, 0.2],
            )
        ],
    )
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(unit_id="doc:snap:1:a.py", source="scip"), _scip_unit()],
        supports=[
            IRUnitSupport(
                support_id="support:scip:anchor",
                unit_id="scip:snap:1:foo",
                source="scip",
                support_kind="anchor",
                external_id="pkg/foo",
                display_name="foo",
                start_line=10,
                end_line=20,
            )
        ],
    )

    merged = merge_ir(ast, scip)
    grounded = next(unit for unit in merged.units if unit.unit_id == "ast:s1")

    assert grounded.primary_anchor_symbol_id == "pkg/foo"
    assert grounded.source_set == {"fc_structure", "scip"}
    assert "scip:snap:1:foo" in grounded.metadata["aliases"]
    assert all(
        unit.unit_id != "scip:snap:1:foo"
        for unit in merged.units
        if unit.kind != "file"
    )


def test_merge_preserves_unmatched_scip_unit_as_synthetic():
    ast = IRSnapshot(repo_name="r", snapshot_id="snap:1", units=[_file()])
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(unit_id="doc:snap:1:a.py", source="scip"), _scip_unit(name="bar")],
    )

    merged = merge_ir(ast, scip)
    synthetic = next(unit for unit in merged.units if unit.unit_id == "scip:snap:1:foo")

    assert synthetic.primary_anchor_symbol_id == "pkg/foo"
    assert synthetic.parent_unit_id == "doc:snap:1:a.py"


def test_merge_deduplicates_overlapping_occurrences_on_canonical_unit():
    ast = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(), _structure_unit()],
        supports=[
            IRUnitSupport(
                support_id="occ:ast:1",
                unit_id="ast:s1",
                source="fc_structure",
                support_kind="occurrence",
                role="definition",
                path="a.py",
                start_line=10,
                end_line=20,
                metadata={"doc_id": "doc:snap:1:a.py"},
            )
        ],
    )
    scip = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(unit_id="doc:snap:1:a.py", source="scip"), _scip_unit()],
        supports=[
            IRUnitSupport(
                support_id="occ:scip:1",
                unit_id="scip:snap:1:foo",
                source="scip",
                support_kind="occurrence",
                role="definition",
                path="a.py",
                start_line=10,
                end_line=20,
                metadata={"doc_id": "doc:snap:1:a.py", "source": "scip"},
            )
        ],
    )

    merged = merge_ir(ast, scip)
    occurrences = merged.occurrences

    assert len(occurrences) == 1
    assert occurrences[0].symbol_id == "ast:s1"
    assert occurrences[0].source == "scip"


def test_validate_snapshot_catches_missing_unit_refs_and_duplicate_files():
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[
            _file(unit_id="doc:1", path="a.py"),
            _file(unit_id="doc:2", path="a.py"),
            IRCodeUnit(
                unit_id="u:1",
                kind="function",
                path="a.py",
                language="python",
                display_name="foo",
                parent_unit_id="missing",
                source_set=set(),
            ),
        ],
        supports=[
            IRUnitSupport(
                support_id="support:1",
                unit_id="missing",
                source="",
                support_kind="structure",
            )
        ],
        relations=[
            IRRelation(
                relation_id="rel:1",
                src_unit_id="doc:1",
                dst_unit_id="missing",
                relation_type="ref",
                resolution_state="anchored",
                support_ids=["support:missing"],
            )
        ],
    )

    errors = validate_snapshot(snap)
    assert any("duplicate file paths" in error for error in errors)
    assert any("unit parent not found" in error for error in errors)
    assert any("support references missing unit_id" in error for error in errors)
    assert any("relation dst not found" in error for error in errors)


def test_legacy_attachments_project_from_summary_and_embedding():
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[
            _file(),
            IRCodeUnit(
                unit_id="ast:s1",
                kind="function",
                path="a.py",
                language="python",
                display_name="foo",
                summary="summary text",
                source_set={"fc_structure"},
            ),
        ],
        embeddings=[
            IRUnitEmbedding(
                embedding_id="emb:1",
                unit_id="ast:s1",
                source="fc_embedding",
                embedding_text="foo summary",
                vector=[0.1, 0.2],
            )
        ],
    )

    attachments = snap.attachments
    assert {attachment.attachment_type for attachment in attachments} == {
        "summary",
        "embedding",
    }


def test_scip_adapter_projects_ref_and_contain_edges():
    snap = build_ir_from_scip(
        repo_name="r",
        snapshot_id="snap:1",
        scip_index={
            "documents": [
                {
                    "path": "a.py",
                    "language": "python",
                    "symbols": [
                        {
                            "symbol": "pkg/foo",
                            "name": "foo",
                            "kind": "function",
                            "range": [1, 0, 2, 0],
                        }
                    ],
                    "occurrences": [
                        {
                            "symbol": "pkg/foo",
                            "role": "reference",
                            "range": [5, 0, 5, 3],
                        }
                    ],
                }
            ]
        },
    )

    edge_types = {edge.edge_type for edge in snap.edges}
    assert "contain" in edge_types
    assert "ref" in edge_types
    assert all("extractor" in edge.metadata for edge in snap.edges)


def test_ir_graph_builder_routes_relation_types():
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        units=[_file(unit_id="doc:1"), _structure_unit(unit_id="u:1")],
        relations=[
            IRRelation(
                relation_id="rel:contain",
                src_unit_id="doc:1",
                dst_unit_id="u:1",
                relation_type="contain",
                resolution_state="structural",
                support_sources={"fc_structure"},
            ),
            IRRelation(
                relation_id="rel:call",
                src_unit_id="u:1",
                dst_unit_id="u:1",
                relation_type="call",
                resolution_state="structural",
                support_sources={"fc_structure"},
            ),
        ],
    )

    graphs = IRGraphBuilder().build_graphs(snap)
    assert graphs.containment_graph.number_of_edges() == 1
    assert graphs.call_graph.number_of_edges() == 1
