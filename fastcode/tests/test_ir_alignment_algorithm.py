from fastcode.ir_merge import merge_ir
from fastcode.semantic_ir import IRCodeUnit, IRSnapshot, IRUnitSupport


def _file(snapshot_id: str = "snap:1", source: str = "fc_structure") -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=f"doc:{snapshot_id}:a.py",
        kind="file",
        path="a.py",
        language="python",
        display_name="a.py",
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


def _method(unit_id: str, name: str, parent_id: str, start_line: int, end_line: int, source: str = "fc_structure") -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind="method",
        path="a.py",
        language="python",
        display_name=name,
        qualified_name=f"{parent_id}.{name}",
        start_line=start_line,
        end_line=end_line,
        parent_unit_id=parent_id,
        source_set={source},
    )


def _scip_method(unit_id: str, anchor: str, name: str, start_line: int, end_line: int, enclosing: str) -> tuple[IRCodeUnit, IRUnitSupport]:
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
        units=[_file(), _method("ast:method:caller", "caller", "doc:snap:1:a.py", 1, 20), _method("ast:method:callee", "callee", "doc:snap:1:a.py", 30, 40)],
    )
    scip_callee, scip_support = _scip_method("scip:callee", "pkg/callee", "callee", 30, 40, "")
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
    ref_relations = [relation for relation in merged.relations if relation.relation_type == "ref"]

    assert any(
        relation.src_unit_id == "ast:method:caller" and relation.dst_unit_id == "ast:method:callee"
        for relation in ref_relations
    )

