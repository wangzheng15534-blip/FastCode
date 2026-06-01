"""Guard hot paths against accidental object/vector materialization."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

HOT_PATHS = [
    PACKAGE_ROOT / "app" / "indexing" / "pipeline" / "service.py",
    PACKAGE_ROOT / "app" / "indexing" / "projection" / "transform.py",
    PACKAGE_ROOT / "graph" / "analysis.py",
    PACKAGE_ROOT / "semantic" / "symbol_index.py",
    PACKAGE_ROOT / "semantic" / "resolvers" / "engine" / "patching.py",
    PACKAGE_ROOT / "app" / "store" / "vectors" / "pg_retrieval.py",
    PACKAGE_ROOT / "app" / "store" / "snapshots" / "snapshot.py",
    PACKAGE_ROOT / "app" / "store" / "snapshots" / "ir_payloads.py",
    PACKAGE_ROOT / "app" / "store" / "vectors" / "vector.py",
    PACKAGE_ROOT / "app" / "query" / "selection" / "retriever.py",
]
VECTOR_INSERTION_PATHS = [
    PACKAGE_ROOT / "main" / "fastcode.py",
    PACKAGE_ROOT / "app" / "indexing" / "pipeline" / "service.py",
    PACKAGE_ROOT / "app" / "store" / "vectors" / "vector.py",
]

ALLOWED_SAFE_JSONABLE: set[tuple[str, int]] = set()
ALLOWED_SAFE_JSONABLE_FUNCTIONS = {
    # Snapshot JSON files are an explicit persistence boundary. These helpers
    # normalize arbitrary metadata/support payloads before json.dump().
    ("app/store/snapshots/snapshot.py", "_json_mapping_payload"),
    ("app/store/snapshots/snapshot.py", "_json_list_payload"),
}
ALLOWED_TOLIST_FUNCTIONS = {
    # Repository overview manifests are a storage/export boundary for string labels,
    # not embedding vectors or ranked candidate rows.
    ("app/store/vectors/vector.py", "_load_repo_overview_embeddings"),
}
ALLOWED_GENERIC_DICT_CALLS = {
    # SCIP cache payloads are an explicit artifact boundary.
    ("app/indexing/pipeline/service.py", "_load_scoped_scip_cache", "from_dict"),
    ("app/indexing/pipeline/service.py", "_save_scoped_scip_cache", "to_dict"),
    # Legacy object fallback and typed config adapters are compatibility/config
    # boundaries, not row-shaped persistence or vector materialization paths.
    ("app/indexing/pipeline/service.py", "_legacy_element_mapping", "to_dict"),
    ("app/query/selection/retriever.py", "_project_doc_priors", "from_dict"),
    ("app/query/selection/retriever.py", "_apply_doc_projection_to_code", "from_dict"),
    ("app/query/selection/retriever.py", "_adaptive_fuse_channels", "from_dict"),
    (
        "app/query/selection/retriever.py",
        "_compute_adaptive_fusion_params",
        "from_dict",
    ),
}
ALLOWED_NETWORKX_IMPORTS = {
    # Compatibility and explicit graph materialization boundaries. New hot paths
    # should use IRGraphView/native handles before this allowlist grows.
    "graph/build.py",
    "app/indexing/projection/transform.py",
    "ir/graph.py",
    "graph/analysis.py",
    "app/query/selection/retriever.py",
    "app/store/snapshots/snapshot.py",
}


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.FunctionDef):
            return current.name
        current = parents.get(current)
    return None


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _qualified_call_name(node: ast.Call) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    value = func.value
    if isinstance(value, ast.Name):
        return f"{value.id}.{func.attr}"
    return func.attr


def test_hot_paths_do_not_use_generic_safe_jsonable() -> None:
    violations: list[str] = []
    for path in HOT_PATHS:
        tree = ast.parse(path.read_text())
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) != "safe_jsonable":
                continue
            key = (_rel(path), node.lineno)
            function_name = _enclosing_function(node, parents) or "<module>"
            if (
                key in ALLOWED_SAFE_JSONABLE
                or (key[0], function_name) in ALLOWED_SAFE_JSONABLE_FUNCTIONS
            ):
                continue
            violations.append(f"{key[0]}:{key[1]}:{function_name}")
    assert not violations, "generic safe_jsonable in hot paths:\n" + "\n".join(
        violations
    )


def test_pg_retrieval_does_not_materialize_embedding_lists_on_active_path() -> None:
    path = PACKAGE_ROOT / "app" / "store" / "vectors" / "pg_retrieval.py"
    tree = ast.parse(path.read_text())
    violations: list[str] = []

    parents = _parents(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != "tolist":
            continue
        if _enclosing_function(node, parents) != "_json_safe_payload":
            violations.append(f"{_rel(path)}:{node.lineno}")
    assert not violations, (
        "embedding/list materialization in pg retrieval:\n" + "\n".join(violations)
    )


def test_hot_paths_do_not_use_unapproved_tolist_materialization() -> None:
    violations: list[str] = []
    for path in HOT_PATHS:
        rel_path = _rel(path)
        tree = ast.parse(path.read_text())
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "tolist":
                continue
            function_name = _enclosing_function(node, parents) or "<module>"
            if (rel_path, function_name) in ALLOWED_TOLIST_FUNCTIONS:
                continue
            violations.append(f"{rel_path}:{node.lineno}:{function_name}")
    assert not violations, "unapproved .tolist() in hot paths:\n" + "\n".join(
        violations
    )


def test_hot_paths_do_not_use_raw_np_array_vectors() -> None:
    violations: list[str] = []
    for path in VECTOR_INSERTION_PATHS:
        rel_path = _rel(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _qualified_call_name(node) != "np.array":
                continue
            first_arg = node.args[0] if node.args else None
            if isinstance(first_arg, ast.Name) and first_arg.id in {
                "vectors",
                "query_vectors",
                "embeddings",
            }:
                violations.append(f"{rel_path}:{node.lineno}")
    assert not violations, "raw np.array(vector-list) in hot paths:\n" + "\n".join(
        violations
    )


def test_hot_paths_do_not_use_generic_row_or_record_round_trips() -> None:
    violations: list[str] = []
    for path in HOT_PATHS:
        rel_path = _rel(path)
        tree = ast.parse(path.read_text())
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in {"row_to_dict", "to_dict", "from_dict"}:
                continue
            function_name = _enclosing_function(node, parents) or "<module>"
            if (rel_path, function_name, name) in ALLOWED_GENERIC_DICT_CALLS:
                continue
            violations.append(f"{rel_path}:{node.lineno}:{function_name}:{name}")
    assert not violations, (
        "generic dict conversion round trips in hot paths:\n" + "\n".join(violations)
    )


def test_incremental_pipeline_does_not_call_full_element_graph_fallback() -> None:
    path = PACKAGE_ROOT / "app" / "indexing" / "pipeline" / "service.py"
    tree = ast.parse(path.read_text())
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) == "_full_elements_for_incremental_fallback":
            violations.append(f"{_rel(path)}:{node.lineno}")
    assert not violations, (
        "incremental artifact graph path must stay delta-first:\n"
        + "\n".join(violations)
    )


def test_incremental_pipeline_does_not_reintroduce_full_element_fallback_helper() -> (
    None
):
    path = PACKAGE_ROOT / "app" / "indexing" / "pipeline" / "service.py"
    tree = ast.parse(path.read_text())
    violations = [
        f"{_rel(path)}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_full_elements_for_incremental_fallback"
    ]
    assert not violations, (
        "full-element incremental fallback helper must stay removed:\n"
        + "\n".join(violations)
    )


def test_semantic_patch_does_not_eagerly_copy_snapshot_collections() -> None:
    path = PACKAGE_ROOT / "semantic" / "resolvers" / "engine" / "patching.py"
    tree = ast.parse(path.read_text())
    parents = _parents(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "list":
            continue
        if _enclosing_function(node, parents) != "apply_resolution_patch":
            continue
        first_arg = node.args[0] if node.args else None
        if not isinstance(first_arg, ast.Attribute):
            continue
        if (
            not isinstance(first_arg.value, ast.Name)
            or first_arg.value.id != "snapshot"
        ):
            continue
        if first_arg.attr in {"units", "supports", "relations", "embeddings"}:
            violations.append(f"{_rel(path)}:{node.lineno}:{first_arg.attr}")
    assert not violations, (
        "semantic patch must not eagerly copy snapshot collections:\n"
        + "\n".join(violations)
    )


def test_shard_native_bm25_retrieval_helpers_do_not_construct_bm25okapi() -> None:
    path = PACKAGE_ROOT / "app" / "query" / "selection" / "retriever.py"
    tree = ast.parse(path.read_text())
    parents = _parents(tree)
    guarded_functions = {
        "load_bm25_sources",
        "load_bm25_legacy_sources",
        "reload_specific_repositories",
        "_keyword_search_sharded",
        "_keyword_search_sharded_runtime",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != "BM25Okapi":
            continue
        function_name = _enclosing_function(node, parents) or "<module>"
        if function_name in guarded_functions:
            violations.append(f"{_rel(path)}:{node.lineno}:{function_name}")
    assert not violations, (
        "shard-native BM25 helpers must stay rebuild-free:\n" + "\n".join(violations)
    )


def test_networkx_imports_stay_explicit_compatibility_boundaries() -> None:
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        rel_path = _rel(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imports_networkx = False
            if isinstance(node, ast.Import):
                imports_networkx = any(alias.name == "networkx" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_networkx = module == "networkx" or module.startswith(
                    "networkx."
                )
            if not imports_networkx:
                continue
            if rel_path not in ALLOWED_NETWORKX_IMPORTS:
                violations.append(f"{rel_path}:{node.lineno}")
    assert not violations, "unapproved NetworkX imports:\n" + "\n".join(violations)
