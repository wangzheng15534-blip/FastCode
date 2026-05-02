"""Tests for semantic resolver patches and multi-language adapter behavior."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn
from unittest.mock import patch

import networkx as nx
import pytest

from fastcode.indexer import CodeElement
from fastcode.main import FastCode
from fastcode.pipeline import IndexPipeline
from fastcode.semantic_ir import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport
from fastcode.semantic_resolvers import (
    PYTHON_RESOLVER_EXTRACTOR,
    PYTHON_RESOLVER_SOURCE,
    CppSemanticResolver,
    CSemanticResolver,
    CSharpCompilerResolver,
    FortranCompilerResolver,
    GoCompilerResolver,
    JavaCompilerResolver,
    JavaScriptCompilerResolver,
    JuliaCompilerResolver,
    PythonSemanticResolver,
    ResolutionPatch,
    RustCompilerResolver,
    TypeScriptCompilerResolver,
    ZigCompilerResolver,
    apply_resolution_patch,
    build_default_semantic_resolver_registry,
)
from fastcode.semantic_resolvers._utils import (
    _hash_id,
    _normalize_path,
    validate_helper_paths,
)
from fastcode.semantic_resolvers.helper_backed import HelperBackedSemanticResolver
from fastcode.semantic_resolvers.patching import _source_preference


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


def _wire_pipeline(fc: Any) -> None:
    """Wire a minimal IndexPipeline onto a bare FastCode.__new__() instance.

    Only populates what _apply_semantic_resolvers needs (the registry).
    """
    registry = getattr(fc, "semantic_resolver_registry", None)
    if registry is None:
        registry = build_default_semantic_resolver_registry()
        fc.semantic_resolver_registry = registry
    fc.pipeline = IndexPipeline(
        config=getattr(fc, "config", {}),
        logger=SimpleNamespace(
            info=lambda *a, **kw: None, warning=lambda *a, **kw: None
        ),
        loader=SimpleNamespace(repo_path=None, scan_files=lambda: []),
        snapshot_store=SimpleNamespace(),
        manifest_store=SimpleNamespace(),
        index_run_store=SimpleNamespace(),
        snapshot_symbol_index=SimpleNamespace(),
        vector_store=SimpleNamespace(metadata=[]),
        embedder=SimpleNamespace(),
        indexer=SimpleNamespace(),
        retriever=SimpleNamespace(),
        graph_builder=SimpleNamespace(),
        ir_graph_builder=SimpleNamespace(),
        pg_retrieval_store=None,
        terminus_publisher=SimpleNamespace(),
        doc_ingester=SimpleNamespace(),
        semantic_resolver_registry=registry,
        set_repo_indexed=lambda v: None,
        set_repo_loaded=lambda v: None,
        set_repo_info=lambda v: None,
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
        _element(
            element_id="class:base", element_type="class", name="Base", path="b.py"
        ),
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

    relations_by_type = {
        relation.relation_type: relation for relation in patch.relations
    }
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
    relations_by_type = {
        relation.relation_type: relation for relation in patch.relations
    }
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

    assert languages == {
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
        "csharp",
        "c",
        "cpp",
        "zig",
        "fortran",
        "julia",
    }


def test_resolver_specs_advertise_capabilities_and_frontends():
    registry = build_default_semantic_resolver_registry()

    specs = {resolver.language: resolver.spec for resolver in registry.all()}

    assert "resolve_calls" in specs["typescript"].capabilities
    assert specs["typescript"].frontend_kind == "typescript_compiler_api"
    assert specs["cpp"].required_tools == ("clang++",)
    assert specs["zig"].required_tools == ("zig", "zls")


def test_graph_backed_language_resolver_records_missing_tool_diagnostics():
    resolver = build_default_semantic_resolver_registry().all()[1]
    assert resolver.language == "javascript"
    file_js = _file_unit("app.js", language="javascript")
    snapshot = _snapshot(units=[file_js])
    elements = [
        _element(
            element_id="file:js",
            element_type="file",
            name="app.js",
            path="app.js",
            language="javascript",
        )
    ]

    with patch(
        "fastcode.semantic_resolvers.graph_backed.shutil.which", return_value=None
    ):
        patch_result = resolver.resolve(
            snapshot=snapshot,
            elements=elements,
            target_paths={"app.js"},
            legacy_graph_builder=SimpleNamespace(
                dependency_graph=nx.DiGraph(),
                inheritance_graph=nx.DiGraph(),
                call_graph=nx.DiGraph(),
            ),
        )

    assert patch_result.diagnostics
    assert patch_result.stats["diagnostics"][0]["code"] == "required_tool_missing"


def test_fastcode_apply_semantic_resolvers_replaces_heuristic_import_relation():
    fc = FastCode.__new__(FastCode)
    _wire_pipeline(fc)
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
        _element(
            element_id="file:right",
            element_type="file",
            name="right.py",
            path="right.py",
        ),
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


def test_apply_resolution_patch_merges_pending_capabilities_via_intersection():
    src = _symbol_unit("unit:src", "a.py", "caller", element_id="elem:src")
    dst = _symbol_unit("unit:dst", "b.py", "callee", element_id="elem:dst")
    existing_relation = IRRelation(
        relation_id="rel:1",
        src_unit_id="unit:src",
        dst_unit_id="unit:dst",
        relation_type="call",
        resolution_state="structural",
        support_sources={"fc_structure"},
        pending_capabilities={"resolve_calls", "resolve_types"},
    )
    snapshot = _snapshot(
        units=[_file_unit("a.py"), _file_unit("b.py"), src, dst],
        relations=[existing_relation],
    )
    patch_relation = IRRelation(
        relation_id="rel:2",
        src_unit_id="unit:src",
        dst_unit_id="unit:dst",
        relation_type="call",
        resolution_state="anchored",
        support_sources={"python_resolver"},
        pending_capabilities={"resolve_calls"},
    )
    updated = apply_resolution_patch(
        snapshot, ResolutionPatch(relations=[patch_relation])
    )
    assert len(updated.relations) == 1
    assert updated.relations[0].pending_capabilities == {"resolve_calls"}
    assert updated.relations[0].support_sources == {"fc_structure", "python_resolver"}


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_target_paths_includes_all_languages_on_fresh_index():
    """F-3 regression: target_paths must not filter by language.

    When no incremental changeset is available, all element paths must be
    included so C/C++ resolvers can run on a fresh full-index.
    """
    fc = FastCode.__new__(FastCode)
    _wire_pipeline(fc)
    file_cpp = _file_unit("main.cpp", language="cpp")
    file_py = _file_unit("app.py")
    snapshot = _snapshot(units=[file_cpp, file_py])
    elements = [
        _element(
            element_id="file:cpp",
            element_type="file",
            name="main.cpp",
            path="main.cpp",
            metadata={"imports": [{"module": "util.h", "level": 0}]},
            language="cpp",
        ),
        _element(
            element_id="file:py",
            element_type="file",
            name="app.py",
            path="app.py",
            language="python",
        ),
    ]
    warnings: list[str] = []

    updated = fc._apply_semantic_resolvers(
        snapshot=snapshot,
        elements=elements,
        legacy_graph_builder=None,
        target_paths={"main.cpp", "app.py"},
        warnings=warnings,
    )

    # CppSemanticResolver should be invoked for main.cpp — at minimum, its
    # resolver run metadata should be recorded even if no edges are resolved.
    resolver_runs = updated.metadata.get("semantic_resolver_runs", [])
    languages_run = {run.get("language") for run in resolver_runs}
    assert "cpp" in languages_run, (
        "CppSemanticResolver must be triggered when target_paths includes .cpp files"
    )


@pytest.mark.regression
def test_source_preference_ranks_c_and_cpp_resolvers_above_fc_structure():
    """F-7 regression: _source_preference must rank C/C++ resolver sources
    higher than fc_structure so resolver patches win tiebreaks.
    """
    fc_rel = IRRelation(
        relation_id="rel:1",
        src_unit_id="u:a",
        dst_unit_id="u:b",
        relation_type="import",
        resolution_state="structural",
        support_sources={"fc_structure"},
    )
    c_rel = IRRelation(
        relation_id="rel:2",
        src_unit_id="u:a",
        dst_unit_id="u:b",
        relation_type="import",
        resolution_state="structural",
        support_sources={"c_resolver"},
    )
    cpp_rel = IRRelation(
        relation_id="rel:3",
        src_unit_id="u:a",
        dst_unit_id="u:b",
        relation_type="import",
        resolution_state="structural",
        support_sources={"cpp_resolver"},
    )
    assert _source_preference(c_rel) > _source_preference(fc_rel)
    assert _source_preference(cpp_rel) > _source_preference(fc_rel)


@pytest.mark.regression
def test_resolution_patch_allows_mutation():
    """F-8 regression: ResolutionPatch must allow field mutation.

    Resolvers build patches incrementally via .append() and .update().
    The dataclass must not be frozen.
    """
    patch = ResolutionPatch()
    # These operations must succeed without FrozenInstanceError:
    patch.supports.append(
        IRUnitSupport(
            support_id="sup:1",
            unit_id="u:1",
            source="test",
            support_kind="test",
        )
    )
    patch.stats.update({"language": "python", "count": 1})
    patch.warnings.append("test warning")

    assert len(patch.supports) == 1
    assert patch.stats["language"] == "python"
    assert patch.warnings == ["test warning"]


@pytest.mark.regression
def test_c_family_resolver_populates_doc_id_from_dict_lookup():
    """F-9 regression: _build_relation must use dict lookup for doc_id,
    not a linear scan over snapshot units.
    """
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
        ),
    ]

    patch = resolver.resolve(
        snapshot=snapshot,
        elements=elements,
        target_paths={"src/main.c"},
        legacy_graph_builder=None,
    )

    assert len(patch.relations) == 1
    assert patch.relations[0].metadata.get("doc_id") == file_main.unit_id


@pytest.mark.regression
def test_resolver_failure_gracefully_degrades():
    """Regression: a failing resolver must not crash the pipeline.

    _apply_semantic_resolvers catches exceptions and appends warnings.
    """
    fc = FastCode.__new__(FastCode)
    _wire_pipeline(fc)
    snapshot = _snapshot(units=[_file_unit("fail.py")])
    elements = [
        _element(
            element_id="file:fail",
            element_type="file",
            name="fail.py",
            path="fail.py",
        ),
    ]

    class BrokenResolver:
        language = "python"
        capabilities = frozenset({"resolve_calls"})
        cost_class = "low"

        def applicable(
            self,
            *,
            snapshot: Any,
            elements: Any,
            target_paths: Any,
        ) -> bool:
            return True

        def resolve(
            self,
            *,
            snapshot: Any,
            elements: Any,
            target_paths: Any,
            legacy_graph_builder: Any,
        ) -> NoReturn:
            raise RuntimeError("resolver crashed")

    from fastcode.semantic_resolvers import SemanticResolverRegistry

    fc.semantic_resolver_registry = SemanticResolverRegistry([BrokenResolver()])
    fc.pipeline.semantic_resolver_registry = fc.semantic_resolver_registry
    warnings: list[str] = []

    updated = fc._apply_semantic_resolvers(
        snapshot=snapshot,
        elements=elements,
        legacy_graph_builder=None,
        target_paths={"fail.py"},
        warnings=warnings,
    )

    assert any("resolver_failed" in w or "resolver crashed" in w for w in warnings)
    # Snapshot should be returned unchanged:
    assert updated.units == snapshot.units


# ---------------------------------------------------------------------------
# Workstream 1: Capability gating, resolution tiers, and new types
# ---------------------------------------------------------------------------


def test_semantic_capability_constants_are_strings():
    """SemanticCapability constants must be plain strings."""
    from fastcode.semantic_resolvers.base import SemanticCapability

    caps = [
        SemanticCapability.RESOLVE_CALLS,
        SemanticCapability.RESOLVE_IMPORTS,
        SemanticCapability.RESOLVE_IMPORT_ALIASES,
        SemanticCapability.RESOLVE_TYPES,
        SemanticCapability.RESOLVE_INHERITANCE,
        SemanticCapability.RESOLVE_BINDINGS,
        SemanticCapability.EXPAND_MACROS,
        SemanticCapability.RECOVER_QUALIFIED_NAMES,
        SemanticCapability.RESOLVE_INCLUDES,
    ]
    assert all(isinstance(c, str) for c in caps)
    assert len(set(caps)) == len(caps), "capability constants must be unique"


def test_resolution_tier_constants():
    """ResolutionTier constants must be plain strings."""
    from fastcode.semantic_resolvers.base import ResolutionTier

    assert ResolutionTier.STRUCTURAL_FALLBACK == "structural_fallback"
    assert ResolutionTier.COMPILER_CONFIRMED == "compiler_confirmed"
    assert ResolutionTier.ANCHORED == "anchored"


def test_semantic_resolution_request_is_frozen():
    """SemanticResolutionRequest must be an immutable dataclass."""
    from fastcode.semantic_resolvers.base import SemanticResolutionRequest

    req = SemanticResolutionRequest(
        snapshot_id="snap:1",
        target_paths=frozenset({"a.py", "b.py"}),
        budget="changed_files",
        repo_root="/repo",
    )
    assert req.snapshot_id == "snap:1"
    assert req.target_paths == frozenset({"a.py", "b.py"})
    with pytest.raises(AttributeError):
        req.snapshot_id = "snap:2"  # type: ignore[misc]


def test_resolution_patch_has_resolution_tier_default():
    """ResolutionPatch must default to structural_fallback tier."""
    from fastcode.semantic_resolvers.base import ResolutionTier

    patch = ResolutionPatch()
    assert patch.resolution_tier == ResolutionTier.STRUCTURAL_FALLBACK


@pytest.mark.regression
def test_capability_gating_filters_resolvers_by_pending_capabilities():
    """Registry.applicable_for_capabilities() must only return resolvers
    whose capabilities overlap with the required set.
    """
    registry = build_default_semantic_resolver_registry()
    file_py = _file_unit("app.py")
    snapshot = _snapshot(units=[file_py])
    elements = [
        _element(
            element_id="file:py",
            element_type="file",
            name="app.py",
            path="app.py",
        ),
    ]

    # With a capability that Python resolver has
    resolvers = registry.applicable_for_capabilities(
        snapshot=snapshot,
        elements=elements,
        target_paths={"app.py"},
        required_capabilities=frozenset({"resolve_calls"}),
    )
    languages = {r.language for r in resolvers}
    assert "python" in languages

    # With a made-up capability that no resolver has
    resolvers = registry.applicable_for_capabilities(
        snapshot=snapshot,
        elements=elements,
        target_paths={"app.py"},
        required_capabilities=frozenset({"nonexistent_capability"}),
    )
    assert len(resolvers) == 0


@pytest.mark.regression
def test_capability_gating_runs_all_when_no_pending_capabilities():
    """When no relations have pending_capabilities, _apply_semantic_resolvers
    should run all applicable resolvers (initial index path) without error.
    """
    fc = FastCode.__new__(FastCode)
    _wire_pipeline(fc)
    file_py = _file_unit("app.py")
    file_b = _file_unit("b.py")
    caller = _symbol_unit(
        "unit:caller", "app.py", "run", element_id="func:caller", kind="function"
    )
    callee = _symbol_unit(
        "unit:callee",
        "b.py",
        "helper",
        element_id="func:callee",
        kind="function",
        anchor="scip:helper",
    )
    snapshot = _snapshot(units=[file_py, file_b, caller, callee])
    elements = [
        _element(
            element_id="file:py",
            element_type="file",
            name="app.py",
            path="app.py",
            metadata={"imports": [{"module": "b", "level": 0}]},
        ),
        _element(
            element_id="file:b",
            element_type="file",
            name="b.py",
            path="b.py",
        ),
        _element(
            element_id="func:caller",
            element_type="function",
            name="run",
            path="app.py",
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
        "file:py",
        "file:b",
        module="b",
        level=0,
        resolution_method="AST ModuleResolver",
    )
    graph_builder.call_graph.add_edge(
        "func:caller",
        "func:callee",
        call_name="helper",
        call_type="direct",
        file_path="app.py",
    )
    warnings: list[str] = []

    updated = fc._apply_semantic_resolvers(
        snapshot=snapshot,
        elements=elements,
        legacy_graph_builder=graph_builder,
        target_paths={"app.py", "b.py"},
        warnings=warnings,
    )

    # Python resolver should have run and produced metadata
    resolver_runs = updated.metadata.get("semantic_resolver_runs", [])
    assert any(r.get("language") == "python" for r in resolver_runs)
    assert warnings == []


@pytest.mark.regression
def test_graph_backed_relations_carry_resolution_tier_metadata():
    """Relations emitted by graph-backed resolvers must carry
    resolution_tier: structural_fallback in metadata.
    """
    from fastcode.semantic_resolvers.base import ResolutionTier

    resolver = build_default_semantic_resolver_registry().all()[1]
    # Should be JS compiler resolver wrapping graph-backed
    file_js = _file_unit("app.js", language="javascript")
    file_util = _file_unit("util.js", language="javascript")
    snapshot = _snapshot(units=[file_js, file_util])
    elements = [
        _element(
            element_id="file:js",
            element_type="file",
            name="app.js",
            path="app.js",
            language="javascript",
            metadata={"imports": [{"module": "util.js", "level": 0}]},
        ),
        _element(
            element_id="file:util",
            element_type="file",
            name="util.js",
            path="util.js",
            language="javascript",
        ),
    ]
    graph_builder = SimpleNamespace(
        dependency_graph=nx.DiGraph(),
        inheritance_graph=nx.DiGraph(),
        call_graph=nx.DiGraph(),
    )
    graph_builder.dependency_graph.add_edge(
        "file:js",
        "file:util",
        module="util.js",
        level=0,
        resolution_method="AST ModuleResolver",
    )

    with (
        patch(
            "fastcode.semantic_resolvers.helper_backed.shutil.which", return_value=None
        ),
        patch(
            "fastcode.semantic_resolvers.graph_backed.shutil.which", return_value=None
        ),
    ):
        result = resolver.resolve(
            snapshot=snapshot,
            elements=elements,
            target_paths={"app.js"},
            legacy_graph_builder=graph_builder,
        )

    for relation in result.relations:
        assert (
            relation.metadata.get("resolution_tier")
            == ResolutionTier.STRUCTURAL_FALLBACK
        )


@pytest.mark.regression
def test_graph_backed_relations_carry_pending_capabilities_when_tools_missing():
    """When required tools are missing, graph-backed relations must populate
    pending_capabilities so downstream resolvers can fulfil them.
    """
    resolver = build_default_semantic_resolver_registry().all()[1]
    file_js = _file_unit("app.js", language="javascript")
    file_util = _file_unit("util.js", language="javascript")
    snapshot = _snapshot(units=[file_js, file_util])
    elements = [
        _element(
            element_id="file:js",
            element_type="file",
            name="app.js",
            path="app.js",
            language="javascript",
            metadata={"imports": [{"module": "util.js", "level": 0}]},
        ),
        _element(
            element_id="file:util",
            element_type="file",
            name="util.js",
            path="util.js",
            language="javascript",
        ),
    ]
    graph_builder = SimpleNamespace(
        dependency_graph=nx.DiGraph(),
        inheritance_graph=nx.DiGraph(),
        call_graph=nx.DiGraph(),
    )
    graph_builder.dependency_graph.add_edge(
        "file:js",
        "file:util",
        module="util.js",
        level=0,
        resolution_method="AST ModuleResolver",
    )

    with (
        patch(
            "fastcode.semantic_resolvers.helper_backed.shutil.which", return_value=None
        ),
        patch(
            "fastcode.semantic_resolvers.graph_backed.shutil.which", return_value=None
        ),
    ):
        result = resolver.resolve(
            snapshot=snapshot,
            elements=elements,
            target_paths={"app.js"},
            legacy_graph_builder=graph_builder,
        )

    for relation in result.relations:
        assert len(relation.pending_capabilities) > 0, (
            "graph-backed relations must carry pending_capabilities when tools are missing"
        )


# ---------------------------------------------------------------------------
# Workstream 2: Language adapter tests (compiler-backed with fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("resolver_cls", "language", "tools"),
    [
        ("JavaScriptCompilerResolver", "javascript", ["node", "tsc"]),
        ("TypeScriptCompilerResolver", "typescript", ["node", "tsc"]),
        ("JavaCompilerResolver", "java", ["javac"]),
        ("GoCompilerResolver", "go", ["go"]),
        ("RustCompilerResolver", "rust", ["rust-analyzer", "cargo"]),
        ("CSharpCompilerResolver", "csharp", ["dotnet"]),
        ("ZigCompilerResolver", "zig", ["zig", "zls"]),
        ("FortranCompilerResolver", "fortran", ["fortls"]),
        ("JuliaCompilerResolver", "julia", ["julia"]),
    ],
)
def test_compiler_resolver_emits_diagnostics_when_tools_missing(
    resolver_cls: str, language: str, tools: list[str]
):
    """Each compiler-backed resolver must emit ToolDiagnostic when its tools
    are missing, falling back gracefully to structural evidence.
    """
    import importlib

    # Map resolver class names to their modules
    module_map = {
        "JavaScriptCompilerResolver": "fastcode.semantic_resolvers.js_ts",
        "TypeScriptCompilerResolver": "fastcode.semantic_resolvers.js_ts",
        "JavaCompilerResolver": "fastcode.semantic_resolvers.java",
        "GoCompilerResolver": "fastcode.semantic_resolvers.go",
        "RustCompilerResolver": "fastcode.semantic_resolvers.rust",
        "CSharpCompilerResolver": "fastcode.semantic_resolvers.csharp",
        "ZigCompilerResolver": "fastcode.semantic_resolvers.zig",
        "FortranCompilerResolver": "fastcode.semantic_resolvers.fortran",
        "JuliaCompilerResolver": "fastcode.semantic_resolvers.julia",
    }
    mod = importlib.import_module(module_map[resolver_cls])
    cls = getattr(mod, resolver_cls)
    resolver = cls()

    file_unit = _file_unit(f"test.{language}", language=language)
    snapshot = _snapshot(units=[file_unit])
    elements = [
        _element(
            element_id=f"file:{language}",
            element_type="file",
            name=f"test.{language}",
            path=f"test.{language}",
            language=language,
        )
    ]

    # Ensure all tool checks return None (not installed)
    # _has_tools lives in HelperBackedSemanticResolver (helper_backed.py),
    # so patch shutil.which there.
    with patch(
        "fastcode.semantic_resolvers.helper_backed.shutil.which", return_value=None
    ):
        result = resolver.resolve(
            snapshot=snapshot,
            elements=elements,
            target_paths={f"test.{language}"},
            legacy_graph_builder=SimpleNamespace(
                dependency_graph=nx.DiGraph(),
                inheritance_graph=nx.DiGraph(),
                call_graph=nx.DiGraph(),
            ),
        )

    assert result.diagnostics, (
        f"{resolver_cls} must emit ToolDiagnostic when tools are missing"
    )
    diagnostic_codes = {d.code for d in result.diagnostics}
    assert "required_tool_missing" in diagnostic_codes


@pytest.mark.parametrize(
    ("resolver_cls", "language"),
    [
        ("JavaScriptCompilerResolver", "javascript"),
        ("TypeScriptCompilerResolver", "typescript"),
        ("JavaCompilerResolver", "java"),
        ("GoCompilerResolver", "go"),
        ("RustCompilerResolver", "rust"),
        ("CSharpCompilerResolver", "csharp"),
        ("ZigCompilerResolver", "zig"),
        ("FortranCompilerResolver", "fortran"),
        ("JuliaCompilerResolver", "julia"),
    ],
)
def test_compiler_resolver_spec_matches_language(resolver_cls: str, language: str):
    """Compiler resolver spec.language must match its advertised language."""
    import importlib

    module_map = {
        "JavaScriptCompilerResolver": "fastcode.semantic_resolvers.js_ts",
        "TypeScriptCompilerResolver": "fastcode.semantic_resolvers.js_ts",
        "JavaCompilerResolver": "fastcode.semantic_resolvers.java",
        "GoCompilerResolver": "fastcode.semantic_resolvers.go",
        "RustCompilerResolver": "fastcode.semantic_resolvers.rust",
        "CSharpCompilerResolver": "fastcode.semantic_resolvers.csharp",
        "ZigCompilerResolver": "fastcode.semantic_resolvers.zig",
        "FortranCompilerResolver": "fastcode.semantic_resolvers.fortran",
        "JuliaCompilerResolver": "fastcode.semantic_resolvers.julia",
    }
    mod = importlib.import_module(module_map[resolver_cls])
    cls = getattr(mod, resolver_cls)
    resolver = cls()

    assert resolver.language == language
    assert resolver.spec.language == language
    assert len(resolver.capabilities) > 0
    assert "resolve_calls" in resolver.capabilities


def test_typescript_compiler_facts_emit_semantic_relations():
    resolver = TypeScriptCompilerResolver()
    file_app = _file_unit("src/app.ts", language="typescript")
    file_lib = _file_unit("src/lib.ts", language="typescript")
    caller = _symbol_unit(
        "unit:ts:caller",
        "src/app.ts",
        "run",
        element_id="ts:caller",
        kind="function",
        language="typescript",
    )
    callee = _symbol_unit(
        "unit:ts:callee",
        "src/lib.ts",
        "helper",
        element_id="ts:callee",
        kind="function",
        anchor="scip:ts:helper",
        language="typescript",
    )
    snapshot = _snapshot(units=[file_app, file_lib, caller, callee])
    payload = {
        "imports": [
            {
                "source_path": "src/app.ts",
                "target_path": "src/lib.ts",
                "module": "./lib",
            }
        ],
        "calls": [
            {
                "source_path": "src/app.ts",
                "target_path": "src/lib.ts",
                "call_name": "helper",
                "target_name": "helper",
                "target_symbol": "helper",
                "source_line": 2,
                "source_col": 0,
                "target_line": 1,
                "target_col": 0,
            }
        ],
        "stats": {"files": 2, "imports": 1, "calls": 1},
    }

    with (
        patch.object(resolver, "_has_tools", return_value=True),
        patch.object(
            resolver,
            "_target_files",
            return_value=["src/app.ts", "src/lib.ts"],
        ),
        patch.object(resolver, "_run_semantic_helper", return_value=payload),
    ):
        patch_result = resolver.resolve(
            snapshot=snapshot,
            elements=[],
            target_paths={"src/app.ts", "src/lib.ts"},
            legacy_graph_builder=None,
        )

    assert patch_result.resolution_tier == "compiler_confirmed"
    assert patch_result.stats["relations_emitted"] == {
        "import": 1,
        "call": 1,
        "inherit": 0,
        "type": 0,
    }
    assert {relation.relation_type for relation in patch_result.relations} == {
        "import",
        "call",
    }
    assert all(
        relation.resolution_state == "semantically_resolved"
        for relation in patch_result.relations
    )


def test_typescript_target_files_use_repo_root_for_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    resolver = TypeScriptCompilerResolver()
    target = tmp_path / "src" / "app.ts"
    target.parent.mkdir(parents=True)
    target.write_text("export const x = 1;", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = resolver._target_files({str(target), "src/missing.ts"})
    assert str(target) in result


def test_go_compiler_facts_emit_semantic_relations():
    resolver = GoCompilerResolver()
    file_main = _file_unit("main.go", language="go")
    file_util = _file_unit("util.go", language="go")
    caller = _symbol_unit(
        "unit:go:caller",
        "main.go",
        "run",
        element_id="go:caller",
        kind="function",
        language="go",
    )
    callee = _symbol_unit(
        "unit:go:callee",
        "util.go",
        "helper",
        element_id="go:callee",
        kind="function",
        anchor="scip:go:helper",
        language="go",
    )
    snapshot = _snapshot(units=[file_main, file_util, caller, callee])
    payload = {
        "imports": [
            {"source_path": "main.go", "target_path": "util.go", "import_path": "util"}
        ],
        "calls": [
            {
                "source_path": "main.go",
                "target_path": "util.go",
                "call_name": "helper",
                "target_name": "helper",
                "target_symbol": "helper",
                "source_line": 2,
                "source_col": 0,
                "target_line": 1,
                "target_col": 0,
            }
        ],
        "stats": {"files": 2, "imports": 1, "calls": 1},
    }

    with (
        patch.object(resolver, "_has_tools", return_value=True),
        patch.object(
            resolver,
            "_target_files",
            return_value=["main.go", "util.go"],
        ),
        patch.object(resolver, "_run_semantic_helper", return_value=payload),
    ):
        patch_result = resolver.resolve(
            snapshot=snapshot,
            elements=[],
            target_paths={"main.go", "util.go"},
            legacy_graph_builder=None,
        )

    assert patch_result.stats["relations_emitted"] == {
        "import": 1,
        "call": 1,
        "inherit": 0,
        "type": 0,
    }
    assert len(patch_result.supports) == 2


def test_go_target_files_use_repo_root_for_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    resolver = GoCompilerResolver()
    target = tmp_path / "pkg" / "main.go"
    target.parent.mkdir(parents=True)
    target.write_text("package main", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = resolver._target_files({str(target)})
    assert result == [str(target)]


def test_java_compiler_facts_emit_semantic_relations():
    resolver = JavaCompilerResolver()
    file_app = _file_unit("src/App.java", language="java")
    file_lib = _file_unit("src/Lib.java", language="java")
    caller = _symbol_unit(
        "unit:java:caller",
        "src/App.java",
        "run",
        element_id="java:caller",
        kind="function",
        language="java",
    )
    callee = _symbol_unit(
        "unit:java:callee",
        "src/Lib.java",
        "helper",
        element_id="java:callee",
        kind="function",
        anchor="scip:java:helper",
        language="java",
    )
    snapshot = _snapshot(units=[file_app, file_lib, caller, callee])
    payload = {
        "imports": [
            {
                "source_path": "src/App.java",
                "target_path": "src/Lib.java",
                "import_name": "demo.Lib",
            }
        ],
        "calls": [
            {
                "source_path": "src/App.java",
                "target_path": "src/Lib.java",
                "call_name": "helper",
                "target_name": "helper",
                "target_symbol": "helper",
                "source_line": 2,
                "source_col": 0,
                "target_line": 1,
                "target_col": 0,
            }
        ],
        "stats": {"files": 2, "imports": 1, "calls": 1},
    }

    with (
        patch.object(resolver, "_has_tools", return_value=True),
        patch.object(
            resolver,
            "_target_files",
            return_value=["src/App.java", "src/Lib.java"],
        ),
        patch.object(resolver, "_run_semantic_helper", return_value=payload),
    ):
        patch_result = resolver.resolve(
            snapshot=snapshot,
            elements=[],
            target_paths={"src/App.java", "src/Lib.java"},
            legacy_graph_builder=None,
        )

    assert patch_result.stats["relations_emitted"] == {
        "import": 1,
        "call": 1,
        "inherit": 0,
        "type": 0,
    }
    assert len(patch_result.relations) == 2


def test_java_target_files_use_repo_root_for_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    resolver = JavaCompilerResolver()
    target = tmp_path / "src" / "App.java"
    target.parent.mkdir(parents=True)
    target.write_text("class App {}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = resolver._target_files({str(target)})
    assert result == [str(target)]


@pytest.mark.parametrize(
    ("resolver", "language", "file_a", "file_b"),
    [
        (JavaScriptCompilerResolver(), "javascript", "app.js", "util.js"),
        (RustCompilerResolver(), "rust", "src/main.rs", "src/lib.rs"),
        (CSharpCompilerResolver(), "csharp", "App.cs", "Lib.cs"),
        (ZigCompilerResolver(), "zig", "src/main.zig", "src/lib.zig"),
        (FortranCompilerResolver(), "fortran", "main.f90", "util.f90"),
        (JuliaCompilerResolver(), "julia", "app.jl", "lib.jl"),
    ],
)
def test_additional_compiler_resolvers_map_helper_facts(
    resolver: Any, language: str, file_a: str, file_b: str
):
    file_src = _file_unit(file_a, language=language)
    file_dst = _file_unit(file_b, language=language)
    caller = _symbol_unit(
        f"unit:{language}:caller",
        file_a,
        "run",
        element_id=f"{language}:caller",
        kind="function",
        language=language,
    )
    callee = _symbol_unit(
        f"unit:{language}:callee",
        file_b,
        "helper",
        element_id=f"{language}:callee",
        kind="function",
        anchor=f"scip:{language}:helper",
        language=language,
    )
    snapshot = _snapshot(units=[file_src, file_dst, caller, callee])
    payload = {
        "imports": [
            {
                "source_path": file_a,
                "target_path": file_b,
                "module": file_b,
                "import_name": file_b,
                "import_path": file_b,
            }
        ],
        "calls": [
            {
                "source_path": file_a,
                "target_path": file_b,
                "call_name": "helper",
                "target_name": "helper",
                "target_symbol": "helper",
                "source_line": 2,
                "source_col": 0,
                "target_line": 1,
                "target_col": 0,
            }
        ],
        "stats": {"files": 2, "imports": 1, "calls": 1},
    }

    with (
        patch.object(resolver, "_has_tools", return_value=True),
        patch.object(
            resolver,
            "_target_files",
            return_value=[file_a, file_b],
        ),
        patch.object(resolver, "_run_semantic_helper", return_value=payload),
    ):
        patch_result = resolver.resolve(
            snapshot=snapshot,
            elements=[],
            target_paths={file_a, file_b},
            legacy_graph_builder=None,
        )

    assert patch_result.stats["relations_emitted"]["import"] == 1
    assert patch_result.stats["relations_emitted"]["call"] == 1
    assert all(
        r.metadata.get("resolution_tier") == "compiler_confirmed"
        for r in patch_result.relations
    )


class _DummyHelperResolver(HelperBackedSemanticResolver):
    language = "python"
    capabilities = frozenset(
        {"resolve_calls", "resolve_imports", "resolve_inheritance"}
    )
    cost_class = "low"
    source_name = "dummy_resolver"
    extractor_name = "dummy_extractor"
    frontend_kind = "dummy_frontend"
    required_tools = ()
    helper_filename = "dummy.py"
    file_extensions = (".py",)


class _DummyFallbackResolver:
    def resolve(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        legacy_graph_builder: Any,
    ) -> ResolutionPatch:
        del snapshot, elements, target_paths, legacy_graph_builder
        relation = IRRelation(
            relation_id="rel:fallback",
            src_unit_id="unit:caller",
            dst_unit_id="unit:callee",
            relation_type="call",
            resolution_state="structural",
            support_sources={"dummy_resolver"},
            metadata={
                "source": "dummy_resolver",
                "resolution_tier": "structural_fallback",
            },
        )
        return ResolutionPatch(
            metadata_updates={
                "semantic_resolver_runs": [
                    {
                        "language": "python",
                        "source": "dummy_resolver",
                        "frontend_kind": "dummy_fallback",
                        "fallback": True,
                    }
                ]
            },
            relations=[relation],
            resolution_tier="structural_fallback",
            stats={"fallback_used": True},
        )


def test_helper_backed_resolver_prefers_symbol_containing_source_line() -> None:
    resolver = _DummyHelperResolver()
    file_unit = _file_unit("a.py")
    outer = _symbol_unit("unit:outer", "a.py", "outer", element_id="outer")
    outer.start_line = 1
    outer.end_line = 50
    inner = _symbol_unit("unit:inner", "a.py", "inner", element_id="inner")
    inner.start_line = 10
    inner.end_line = 20
    callee = _symbol_unit("unit:callee", "b.py", "helper", element_id="callee")
    callee.start_line = 1
    callee.end_line = 5
    snapshot = _snapshot(units=[file_unit, outer, inner, _file_unit("b.py"), callee])

    patch_result = ResolutionPatch(
        metadata_updates={"semantic_resolver_runs": [{"language": "python"}]},
        resolution_tier="compiler_confirmed",
    )
    resolver._apply_semantic_facts(
        snapshot=snapshot,
        patch=patch_result,
        payload={
            "calls": [
                {
                    "source_path": "a.py",
                    "target_path": "b.py",
                    "call_name": "helper",
                    "target_name": "helper",
                    "source_line": 12,
                    "target_line": 1,
                }
            ]
        },
    )

    assert len(patch_result.relations) == 1
    assert patch_result.relations[0].src_unit_id == "unit:inner"


def test_helper_backed_resolver_skips_ambiguous_import_target_matches() -> None:
    resolver = _DummyHelperResolver()
    file_src = _file_unit("src/a.py")
    file_x = _file_unit("pkg1/util.py")
    file_y = _file_unit("pkg2/util.py")
    snapshot = _snapshot(units=[file_src, file_x, file_y])
    patch_result = ResolutionPatch(
        metadata_updates={"semantic_resolver_runs": [{"language": "python"}]},
        resolution_tier="compiler_confirmed",
    )

    resolver._apply_semantic_facts(
        snapshot=snapshot,
        patch=patch_result,
        payload={
            "imports": [
                {
                    "source_path": "src/a.py",
                    "module": "util.py",
                }
            ]
        },
    )

    assert patch_result.relations == []


def test_helper_backed_resolver_records_helper_json_failure() -> None:
    resolver = _DummyHelperResolver()
    patch_result = ResolutionPatch()
    with patch("fastcode.semantic_resolvers.helper_backed.subprocess.run") as run_mock:
        run_mock.return_value = SimpleNamespace(returncode=0, stdout="{", stderr="")
        payload = resolver._run_semantic_helper(["/tmp/a.py"], patch_result)

    assert payload == {}
    assert any(d.code == "invalid_helper_json" for d in patch_result.diagnostics)
    assert any("helper_invalid_json" in warning for warning in patch_result.warnings)


def test_helper_backed_resolver_falls_back_on_helper_nonzero_exit(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "a.py"
    source_path.write_text("print('hi')\n", encoding="utf-8")
    resolver = _DummyHelperResolver(fallback=_DummyFallbackResolver())
    snapshot = _snapshot(
        units=[
            _file_unit("a.py"),
            _file_unit("b.py"),
            _symbol_unit("unit:caller", "a.py", "run", element_id="caller"),
            _symbol_unit("unit:callee", "b.py", "helper", element_id="callee"),
        ]
    )
    element = _element(
        element_id="file:a",
        element_type="file",
        name="a.py",
        path="a.py",
    )

    with (
        patch.object(resolver, "_has_tools", return_value=True),
        patch(
            "fastcode.semantic_resolvers.helper_backed.os.getcwd",
            return_value=str(tmp_path),
        ),
        patch("fastcode.semantic_resolvers.helper_backed.subprocess.run") as run_mock,
    ):
        run_mock.return_value = SimpleNamespace(returncode=7, stdout="", stderr="boom")
        patch_result = resolver.resolve(
            snapshot=snapshot,
            elements=[element],
            target_paths={"a.py"},
            legacy_graph_builder=None,
        )

    assert patch_result.resolution_tier == "structural_fallback"
    assert len(patch_result.relations) == 1
    assert patch_result.stats["fallback_used"] is True
    assert patch_result.stats["helper_failed"] is True
    assert patch_result.stats["helper_exit_code"] == 7
    assert "helper_nonzero_exit" in patch_result.stats["helper_failure_codes"]
    resolver_run = patch_result.metadata_updates["semantic_resolver_runs"][0]
    assert resolver_run["helper_backed"] is True
    assert resolver_run["helper_failed"] is True
    assert resolver_run["fallback"] is True
    assert any(d.code == "helper_nonzero_exit" for d in patch_result.diagnostics)
    assert any("helper_error" in warning for warning in patch_result.warnings)


def test_helper_backed_resolver_falls_back_on_helper_timeout(tmp_path: Path) -> None:
    source_path = tmp_path / "a.py"
    source_path.write_text("print('hi')\n", encoding="utf-8")
    resolver = _DummyHelperResolver(fallback=_DummyFallbackResolver())
    snapshot = _snapshot(
        units=[
            _file_unit("a.py"),
            _file_unit("b.py"),
            _symbol_unit("unit:caller", "a.py", "run", element_id="caller"),
            _symbol_unit("unit:callee", "b.py", "helper", element_id="callee"),
        ]
    )
    element = _element(
        element_id="file:a",
        element_type="file",
        name="a.py",
        path="a.py",
    )

    with (
        patch.object(resolver, "_has_tools", return_value=True),
        patch(
            "fastcode.semantic_resolvers.helper_backed.os.getcwd",
            return_value=str(tmp_path),
        ),
        patch(
            "fastcode.semantic_resolvers.helper_backed.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["dummy"], timeout=1),
        ),
    ):
        patch_result = resolver.resolve(
            snapshot=snapshot,
            elements=[element],
            target_paths={"a.py"},
            legacy_graph_builder=None,
        )

    assert patch_result.resolution_tier == "structural_fallback"
    assert patch_result.stats["helper_failed"] is True
    assert any(d.code == "tool_invocation_failed" for d in patch_result.diagnostics)


# ---------------------------------------------------------------------------
# Workstream 3: SCIP indexer tests
# ---------------------------------------------------------------------------


def test_is_scip_available_returns_false_for_unsupported_language():
    """is_scip_available must return False for unknown languages."""
    from fastcode.scip_indexers import is_scip_available

    assert is_scip_available("brainfuck") is False


def test_is_scip_available_checks_binary_presence():
    """is_scip_available must check PATH for the indexer binary."""
    from fastcode.scip_indexers import is_scip_available

    with patch("fastcode.scip_indexers.shutil.which", return_value=None):
        assert is_scip_available("python") is False

    with patch(
        "fastcode.scip_indexers.shutil.which", return_value="/usr/bin/scip-python"
    ):
        assert is_scip_available("python") is True


def test_julia_scip_command_is_valid():
    """Julia SCIP command must not contain exit(1)."""
    from fastcode.scip_indexers import get_indexer_command

    cmd = get_indexer_command("julia", "/tmp/out.scip")
    assert cmd is not None
    assert "exit(1)" not in " ".join(cmd)
    assert "SymbolServer" in " ".join(cmd)


def test_experimental_scip_languages_set():
    """Experimental SCIP languages must include zig, fortran, julia."""
    from fastcode.scip_indexers import _EXPERIMENTAL_SCIP_LANGUAGES

    assert frozenset({"zig", "fortran", "julia"}) == _EXPERIMENTAL_SCIP_LANGUAGES


# ---------------------------------------------------------------------------
# Shared utils tests
# ---------------------------------------------------------------------------


class TestSharedUtils:
    def test_hash_id_is_deterministic(self) -> None:
        result = _hash_id(
            "support", "snapshot:go_resolver:import:src/main.go:pkg/mod.go:fmt"
        )
        assert result == _hash_id(
            "support", "snapshot:go_resolver:import:src/main.go:pkg/mod.go:fmt"
        )

    def test_hash_id_different_inputs_differ(self) -> None:
        a = _hash_id("support", "aaa")
        b = _hash_id("support", "bbb")
        assert a != b

    def test_hash_id_different_prefixes_differ(self) -> None:
        a = _hash_id("support", "same")
        b = _hash_id("rel", "same")
        assert a != b

    def test_normalize_path_strips_dot_slash(self) -> None:
        assert _normalize_path("./src/main.go") == "src/main.go"

    def test_normalize_path_converts_backslashes(self) -> None:
        assert _normalize_path("src\\main.go") == "src/main.go"

    def test_normalize_path_idempotent(self) -> None:
        assert _normalize_path(_normalize_path("src\\main.go")) == _normalize_path(
            "src/main.go"
        )

    def test_validate_helper_paths_rejects_symlink(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("ok")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        safe, rejected = validate_helper_paths([str(link)], str(tmp_path))
        assert len(rejected) == 1
        assert len(safe) == 0

    def test_validate_helper_paths_rejects_outside_repo(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("ok")
        repo = tmp_path / "repo"
        repo.mkdir()
        safe, rejected = validate_helper_paths([str(outside)], str(repo))
        assert len(rejected) == 1
        assert len(safe) == 0

    def test_validate_helper_paths_accepts_valid(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        f = repo / "main.go"
        f.write_text("ok")
        safe, rejected = validate_helper_paths([str(f)], str(repo))
        assert len(safe) == 1
        assert len(rejected) == 0
