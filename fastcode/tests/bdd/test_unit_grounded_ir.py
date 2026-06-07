"""Scenario contracts for the unit-grounded IR design notes."""

from __future__ import annotations

from fastcode.ir.merge import merge_ir
from fastcode.ir.types import IRCodeUnit, IRSnapshot, IRUnitSupport


def _file_unit(unit_id: str, *, source: str) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind="file",
        path="auth.py",
        language="python",
        display_name="auth.py",
        source_set={source},
    )


def _ast_method(unit_id: str, *, start_line: int, end_line: int) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind="method",
        path="auth.py",
        language="python",
        display_name="login",
        qualified_name="AuthService.login",
        start_line=start_line,
        end_line=end_line,
        parent_unit_id="doc:snap:1:auth.py",
        source_set={"fc_structure"},
    )


def _scip_method(
    unit_id: str,
    *,
    anchor: str,
    start_line: int,
    end_line: int,
) -> tuple[IRCodeUnit, IRUnitSupport]:
    unit = IRCodeUnit(
        unit_id=unit_id,
        kind="method",
        path="auth.py",
        language="python",
        display_name="login",
        qualified_name=anchor,
        start_line=start_line,
        end_line=end_line,
        parent_unit_id="doc:snap:1:auth.py",
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
        display_name="login",
        start_line=start_line,
        end_line=end_line,
    )
    return unit, support


def _given_tree_sitter_skeleton_and_matching_scip_anchor() -> tuple[
    IRSnapshot, IRSnapshot
]:
    ast = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:auth",
        units=[
            _file_unit("doc:snap:1:auth.py", source="fc_structure"),
            _ast_method("ast:method:login", start_line=10, end_line=20),
        ],
    )
    scip_method, scip_support = _scip_method(
        "scip:method:login",
        anchor="pkg/AuthService#login().",
        start_line=10,
        end_line=20,
    )
    scip = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:auth",
        units=[
            _file_unit("doc:snap:1:auth.py", source="scip"),
            scip_method,
        ],
        supports=[scip_support],
    )
    return ast, scip


def _when_the_ir_sources_are_merged(ast: IRSnapshot, scip: IRSnapshot) -> IRSnapshot:
    return merge_ir(ast, scip)


def _then_scip_anchors_the_ast_unit_without_replacing_it(
    merged: IRSnapshot,
) -> None:
    units = {unit.unit_id: unit for unit in merged.units}

    assert "ast:method:login" in units
    assert "scip:method:login" not in units
    assert units["ast:method:login"].primary_anchor_symbol_id == (
        "pkg/AuthService#login()."
    )
    assert units["ast:method:login"].source_set == {"fc_structure", "scip"}
    assert units["ast:method:login"].metadata["aliases"] == ["scip:method:login"]
    assert merged.supports[0].unit_id == "ast:method:login"


def test_scip_anchors_tree_sitter_unit_without_replacing_skeleton() -> None:
    """Scenario: SCIP sharpens the tree-sitter unit map instead of winning over it."""
    ast, scip = _given_tree_sitter_skeleton_and_matching_scip_anchor()

    merged = _when_the_ir_sources_are_merged(ast, scip)

    _then_scip_anchors_the_ast_unit_without_replacing_it(merged)


def _given_ambiguous_scip_anchor_far_from_ast_unit() -> tuple[IRSnapshot, IRSnapshot]:
    ast = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:auth",
        units=[
            _file_unit("doc:snap:1:auth.py", source="fc_structure"),
            _ast_method("ast:method:login", start_line=10, end_line=20),
        ],
    )
    scip_method, scip_support = _scip_method(
        "scip:method:login",
        anchor="pkg/login",
        start_line=40,
        end_line=45,
    )
    scip = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:auth",
        units=[
            _file_unit("doc:snap:1:auth.py", source="scip"),
            scip_method,
        ],
        supports=[scip_support],
    )
    return ast, scip


def _then_ambiguous_alignment_keeps_candidate_anchor_and_synthetic_unit(
    merged: IRSnapshot,
) -> None:
    units = {unit.unit_id: unit for unit in merged.units}

    assert units["ast:method:login"].primary_anchor_symbol_id is None
    assert "pkg/login" in units["ast:method:login"].candidate_anchor_symbol_ids
    assert "scip:method:login" in units
    assert units["scip:method:login"].source_set == {"scip"}


def test_ambiguous_scip_alignment_remains_candidate_not_primary() -> None:
    """Scenario: ambiguous SCIP evidence is retained without corrupting the AST unit."""
    ast, scip = _given_ambiguous_scip_anchor_far_from_ast_unit()

    merged = _when_the_ir_sources_are_merged(ast, scip)

    _then_ambiguous_alignment_keeps_candidate_anchor_and_synthetic_unit(merged)
