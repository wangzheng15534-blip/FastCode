"""Tests for semantic resolver patches and multi-language adapter behavior."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from fastcode.indexer import CodeElement
from fastcode.main import FastCode
from fastcode.semantic_ir import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport
from fastcode.semantic_resolvers import (
    CSemanticResolver,
    CppSemanticResolver,
    PYTHON_RESOLVER_EXTRACTOR,
    PYTHON_RESOLVER_SOURCE,
    PythonSemanticResolver,
    ResolutionPatch,
    apply_resolution_patch,
    build_default_semantic_resolver_registry,
)


def _file_unit(path: str, *, language: str = "python") -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=f"doc:snap:1:{path}",
        kind="file",
        path=path,
        language=language,
        display_name=path,
        source_set={"fc_structure"},
        metadata={"source": "fc_structure"},
    )


def _symbol_unit(
    unit_id: str,
    path: str,
    name: str,
    *,
    element_id: str,
    kind: str = "function",
    anchor: str | None = None,
    language: str = "python",
) -> IRCodeUnit:
    return IRCodeUnit(
        unit_id=unit_id,
        kind=kind,
        path=path,
        language=language,
        display_name=name,
        qualified_name=name,
        parent_unit_id=f"doc:snap:1:{path}",
        primary_anchor_symbol_id=anchor,
        anchor_symbol_ids=[anchor] if anchor else [],
        source_set={"fc_structure"} | ({"scip"} if anchor else set()),
        metadata={"ast_element_id": element_id, "source": "fc_structure"},
    )


def _snapshot(
    *,
    units: list[IRCodeUnit],
    supports: list[IRUnitSupport] | None = None,
    relations: list[IRRelation] | None = None,
) -> IRSnapshot:
    return IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:1",
        units=units,
        supports=supports or [],
        relations=relations or [],
    )


def _element(
    *,
    element_id: str,
    element_type: str,
    name: str,
    path: str,
    metadata: dict[str, object] | None = None,
    language: str = "python",
) -> CodeElement:
    return CodeElement(
        id=element_id,
        type=element_type,
        name=name,
        file_path=path,
        relative_path=path,
        language=language,
        start_line=1,
        end_line=5,
        code="pass",
        signature=None,
        docstring=None,
        summary=None,
        metadata=dict(metadata or {}),
    )


def test_apply_resolution_patch_promotes_relation_and_unions_supports():
    src = _symbol_unit("unit:src", "a.py", "caller", element_id="elem:src")
    dst = _symbol_unit(
        "unit:dst", "b.py", "callee", element_id="elem:dst", anchor="scip:callee"
    )
    old_support = IRUnitSupport(
        support_id="sup:ast",
        unit_id="unit:src",
        source="fc_structure",
        support_kind="call_resolution",
        metadata={"source": "fc_structure"},
    )
    old_relation = IRRelation(
        relation_id="rel:old",
        src_unit_id="unit:src",
        dst_unit_id="unit:dst",
        relation_type="call",
        resolution_state="candidate",
        support_sources={"fc_structure"},
        support_ids=["sup:ast"],
        metadata={"call_name": "helper", "source": "fc_structure"},
    )
    new_support = IRUnitSupport(
        support_id="sup:resolver",
        unit_id="unit:src",
        source=PYTHON_RESOLVER_SOURCE,
        support_kind="call_resolution",
        metadata={"source": PYTHON_RESOLVER_SOURCE},
    )
    new_relation = IRRelation(
        relation_id="rel:new",
        src_unit_id="unit:src",
        dst_unit_id="unit:dst",
        relation_type="call",
        resolution_state="anchored",
        support_sources={PYTHON_RESOLVER_SOURCE},
        support_ids=["sup:resolver"],
        metadata={"call_name": "helper", "source": PYTHON_RESOLVER_SOURCE},
    )
    snapshot = _snapshot(
        units=[_file_unit("a.py"), _file_unit("b.py"), src, dst],
        supports=[old_support],
        relations=[old_relation],
    )

    updated = apply_resolution_patch(
        snapshot,
        ResolutionPatch(supports=[new_support], relations=[new_relation]),
    )

    assert len(updated.relations) == 1
    relation = updated.relations[0]
    assert relation.resolution_state == "anchored"
    assert relation.support_sources == {"fc_structure", PYTHON_RESOLVER_SOURCE}
    assert set(relation.support_ids) == {"sup:ast", "sup:resolver"}


def test_apply_resolution_patch_records_resolver_run_metadata():
    snapshot = _snapshot(units=[_file_unit("a.py")])

    updated = apply_resolution_patch(
        snapshot,
        ResolutionPatch(
            metadata_updates={
                "semantic_resolver_runs": [
                    {"language": "python", "source": PYTHON_RESOLVER_SOURCE}
                ]
            }
        ),
    )

    assert updated.metadata["semantic_resolver_runs"] == [
        {"language": "python", "source": PYTHON_RESOLVER_SOURCE}
    ]


def test_apply_resolution_patch_replaces_same_slot_import_target():
    file_a = _file_unit("a.py")
    wrong = _file_unit("wrong.py")
    right = _file_unit("right.py")
    old_relation = IRRelation(
        relation_id="rel:wrong",
        src_unit_id=file_a.unit_id,
        dst_unit_id=wrong.unit_id,
        relation_type="import",
        resolution_state="structural",
        support_sources={"fc_structure"},
        metadata={"module": "b", "source": "fc_structure"},
    )
    new_support = IRUnitSupport(
        support_id="sup:new",
        unit_id=file_a.unit_id,
        source=PYTHON_RESOLVER_SOURCE,
        support_kind="import_resolution",
        metadata={"module": "b"},
    )
    new_relation = IRRelation(
        relation_id="rel:right",
        src_unit_id=file_a.unit_id,
        dst_unit_id=right.unit_id,
        relation_type="import",
        resolution_state="structural",
        support_sources={PYTHON_RESOLVER_SOURCE},
        support_ids=["sup:new"],
        metadata={"module": "b", "source": PYTHON_RESOLVER_SOURCE},
    )
    snapshot = _snapshot(units=[file_a, wrong, right], relations=[old_relation])

    updated = apply_resolution_patch(
        snapshot,
        ResolutionPatch(supports=[new_support], relations=[new_relation]),
    )

    assert len(updated.relations) == 1
    assert updated.relations[0].dst_unit_id == right.unit_id
    assert updated.relations[0].metadata["source"] == PYTHON_RESOLVER_SOURCE


def test_python_resolver_emits_canonical_import_inherit_and_call_relations():
    resolver = PythonSemanticResolver()
    file_a = _file_unit("a.py")
    file_b = _file_unit("b.py")
    child = _symbol_unit(
        "unit:child", "a.py", "Child", element_id="class:child", kind="class"
    )
    base = _symbol_unit(
        "unit:base",
        "b.py",
        "Base",
        element_id="class:base",
        kind="class",
        anchor="scip:Base",
    )
    caller = _symbol_unit(
        "unit:caller", "a.py", "run", element_id="func:caller", kind="function"
    )
    callee = _symbol_unit(
        "unit:callee",
        "b.py",
        "helper",
        element_id="func:callee",
        kind="function",
        anchor="scip:helper",
    )
    snapshot = _snapshot(units=[file_a, file_b, child, base, caller, callee])
    elements = [
        _element(
            element_id="file:a",
            element_type="file",
            name="a.py",
            path="a.py",
            metadata={"imports": [{"module": "b", "level": 0}]},
        ),
        _element(element_id="file:b", element_type="file", name="b.py", path="b.py"),
        _element(
            element_id="class:child",
            element_type="class",
            name="Child",
            path="a.py",
            metadata={"bases": ["Base"]},
        ),
        _element(element_id="class:base", element_type="class", name="Base", path="b.py"),
        _element(
            element_id="func:caller",
            element_type="function",
            name="run",
            path="a.py",
        ),
        _element(
            element_id="func:callee",
            element_type="function",
            name="helper",
            path="b.py",
        ),
    ]

    graph_builder = SimpleNamespace(
        dependency_graph=nx.DiGraph(),
        inheritance_graph=nx.DiGraph(),
        call_graph=nx.DiGraph(),
    )
    graph_builder.dependency_graph.add_edge(
        "file:a",
        "file:b",
        module="b",
        level=0,
        resolution_method="AST ModuleResolver",
    )
    graph_builder.inheritance_graph.add_edge(
        "class:child",
        "class:base",
        base_name="Base",
    )
    graph_builder.call_graph.add_edge(
        "func:caller",
        "func:callee",
        call_name="helper",
        call_type="direct",
        file_path="a.py",
    )

    patch = resolver.resolve(
        snapshot=snapshot,
        elements=elements,
        target_paths={"a.py"},
        legacy_graph_builder=graph_builder,
    )

    assert patch.stats["relations_emitted"] == {"import": 1, "inherit": 1, "call": 1}
    assert len(patch.supports) == 3
    assert len(patch.relations) == 3

    relations_by_type = {relation.relation_type: relation for relation in patch.relations}
    assert relations_by_type["import"].src_unit_id == file_a.unit_id
    assert relations_by_type["import"].dst_unit_id == file_b.unit_id
    assert relations_by_type["import"].resolution_state == "structural"
    assert relations_by_type["inherit"].dst_unit_id == base.unit_id
    assert relations_by_type["inherit"].resolution_state == "anchored"
    assert relations_by_type["call"].dst_unit_id == callee.unit_id
    assert relations_by_type["call"].metadata["extractor"] == PYTHON_RESOLVER_EXTRACTOR
    assert patch.metadata_updates["semantic_resolver_runs"][0]["language"] == "python"


def test_cpp_resolver_emits_include_and_inheritance_relations():
    resolver = CppSemanticResolver()
    file_main = _file_unit("include/derived.hpp", language="cpp")
    file_base = _file_unit("include/base.hpp", language="cpp")
    derived = _symbol_unit(
        "unit:derived",
        "include/derived.hpp",
        "Derived",
        element_id="class:derived",
        kind="class",
        language="cpp",
    )
    base = _symbol_unit(
        "unit:base",
        "include/base.hpp",
        "Base",
        element_id="class:base",
        kind="class",
        anchor="scip:Base",
        language="cpp",
    )
    snapshot = _snapshot(units=[file_main, file_base, derived, base])
    elements = [
        _element(
            element_id="file:derived",
            element_type="file",
            name="include/derived.hpp",
            path="include/derived.hpp",
            metadata={"imports": [{"module": "base.hpp", "level": 0}]},
            language="cpp",
        ),
        _element(
            element_id="file:base",
            element_type="file",
            name="include/base.hpp",
            path="include/base.hpp",
            language="cpp",
        ),
        _element(
            element_id="class:derived",
            element_type="class",
            name="Derived",
            path="include/derived.hpp",
            metadata={"bases": ["ns::Base<int>"]},
            language="cpp",
        ),
        _element(
            element_id="class:base",
            element_type="class",
            name="Base",
            path="include/base.hpp",
            language="cpp",
        ),
    ]

    patch = resolver.resolve(
        snapshot=snapshot,
        elements=elements,
        target_paths={"include/derived.hpp"},
        legacy_graph_builder=None,
    )

    assert patch.stats["relations_emitted"] == {"import": 1, "inherit": 1}
    relations_by_type = {relation.relation_type: relation for relation in patch.relations}
    assert relations_by_type["import"].dst_unit_id == file_base.unit_id
    assert relations_by_type["import"].metadata["resolution_method"] in {
        "relative_include_exact",
        "include_exact",
        "include_basename_match",
    }
    assert relations_by_type["inherit"].dst_unit_id == base.unit_id
    assert relations_by_type["inherit"].resolution_state == "anchored"


def test_c_resolver_emits_relative_include_relation():
    resolver = CSemanticResolver()
    file_main = _file_unit("src/main.c", language="c")
    file_header = _file_unit("include/util.h", language="c")
    snapshot = _snapshot(units=[file_main, file_header])
    elements = [
        _element(
            element_id="file:main",
            element_type="file",
            name="src/main.c",
            path="src/main.c",
            metadata={"imports": [{"module": "../include/util.h", "level": 0}]},
            language="c",
        )
    ]

    patch = resolver.resolve(
        snapshot=snapshot,
        elements=elements,
        target_paths={"src/main.c"},
        legacy_graph_builder=None,
    )

    assert patch.stats["relations_emitted"] == {"import": 1, "inherit": 0}
    assert len(patch.relations) == 1
    assert patch.relations[0].dst_unit_id == file_header.unit_id
    assert patch.relations[0].metadata["resolution_method"] == "relative_include_exact"


def test_default_registry_includes_python_c_and_cpp():
    registry = build_default_semantic_resolver_registry()

    languages = {resolver.language for resolver in registry.all()}

    assert languages == {"python", "c", "cpp"}


def test_fastcode_apply_semantic_resolvers_replaces_heuristic_import_relation():
    fc = FastCode.__new__(FastCode)
    file_a = _file_unit("a.py")
    wrong = _file_unit("wrong.py")
    right = _file_unit("right.py")
    snapshot = _snapshot(
        units=[file_a, wrong, right],
        relations=[
            IRRelation(
                relation_id="rel:heuristic",
                src_unit_id=file_a.unit_id,
                dst_unit_id=wrong.unit_id,
                relation_type="import",
                resolution_state="structural",
                support_sources={"fc_structure"},
                metadata={"module": "b", "source": "fc_structure"},
            )
        ],
    )
    elements = [
        _element(
            element_id="file:a",
            element_type="file",
            name="a.py",
            path="a.py",
            metadata={"imports": [{"module": "b", "level": 0}]},
        ),
        _element(element_id="file:right", element_type="file", name="right.py", path="right.py"),
    ]
    graph_builder = SimpleNamespace(
        dependency_graph=nx.DiGraph(),
        inheritance_graph=nx.DiGraph(),
        call_graph=nx.DiGraph(),
    )
    graph_builder.dependency_graph.add_edge(
        "file:a",
        "file:right",
        module="b",
        level=0,
        resolution_method="AST ModuleResolver",
    )
    warnings: list[str] = []

    updated = fc._apply_semantic_resolvers(
        snapshot=snapshot,
        elements=elements,
        legacy_graph_builder=graph_builder,
        target_paths={"a.py"},
        warnings=warnings,
    )

    assert warnings == []
    assert len(updated.relations) == 1
    assert updated.relations[0].dst_unit_id == right.unit_id
    assert updated.relations[0].metadata["source"] == PYTHON_RESOLVER_SOURCE
    assert updated.metadata["semantic_resolver_runs"][0]["language"] == "python"
