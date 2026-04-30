"""Tests for semantic resolver patches and multi-language adapter behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NoReturn
from unittest.mock import patch

import networkx as nx
import pytest

from fastcode.indexer import CodeElement
from fastcode.main import FastCode
from fastcode.semantic_ir import IRCodeUnit, IRRelation, IRSnapshot, IRUnitSupport
from fastcode.semantic_resolvers import (
    PYTHON_RESOLVER_EXTRACTOR,
    PYTHON_RESOLVER_SOURCE,
    CppSemanticResolver,
    CSemanticResolver,
    PythonSemanticResolver,
    ResolutionPatch,
    apply_resolution_patch,
    build_default_semantic_resolver_registry,
)
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
        patch("fastcode.semantic_resolvers.js_ts.shutil.which", return_value=None),
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
        patch("fastcode.semantic_resolvers.js_ts.shutil.which", return_value=None),
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
    with patch(f"{module_map[resolver_cls]}.shutil.which", return_value=None):
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
