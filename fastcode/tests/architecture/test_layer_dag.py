"""Enforce the strict FastCode layer dependency DAG."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fastcode"

PREFIX_LAYERS = {
    "api.inbound": "BOUNDARY_IN",
    "api.outbound": "BOUNDARY_OUT",
    "api.serialization": "BOUNDARY_OUT",
    "inbound._config_schema_base": "BOUNDARY_IN",
    "inbound._config_schema_indexing": "BOUNDARY_IN",
    "inbound._config_schema_operations": "BOUNDARY_IN",
    "inbound._config_schema_persistence": "BOUNDARY_IN",
    "inbound._config_schema_querying": "BOUNDARY_IN",
    "inbound._config_schema_root": "BOUNDARY_IN",
    "inbound.config_mapper": "BOUNDARY_IN",
    "inbound.config_schema": "BOUNDARY_IN",
    "indexing.scip_runner": "INFRA",
    "main.cli": "TRANSPORT_FACADE",
    "main.config": "CONFIG_LOADER",
    "main.fastcode": "COMPOSITION_ROOT",
    "scip.indexers": "INFRA",
    "scip.loader": "INFRA",
    "scip.pb2": "INFRA",
    "scip.transform": "INFRA",
    "scip.models": "KERNEL",
    "ports": "PORTS",
    "store.infrastructure": "INFRA",
    "utils.archive": "UTILS",
    "utils.filesystem": "UTILS",
    "utils.hashing": "UTILS",
    "utils.ids": "UTILS",
    "utils.clock": "UTILS",
    "utils.json": "UTILS",
    "utils.materialization": "UTILS",
    "utils.path_utils": "UTILS",
    "utils.paths": "UTILS",
    "utils.text": "UTILS",
}

TOP_LEVEL_LAYERS = {
    "api": "TRANSPORT_FACADE",
    "foundation": "FOUNDATION",
    "kernel": "KERNEL",
    "mcp": "TRANSPORT_FACADE",
    "main": "COMPOSITION_ROOT",
    "indexing": "APP_RUNTIME",
    "query": "APP_RUNTIME",
    "runtime_support": "RUNTIME_SUPPORT",
    "store": "APP_RUNTIME",
    "runtime": "RUNTIME_SUPPORT",
    "inbound": "BOUNDARY_IN",
    "retrieval": "DOMAIN",
    "graph": "DOMAIN",
    "scip": "DOMAIN",
    "semantic": "DOMAIN",
    "ir": "KERNEL",
    "utils": "UTILS",
}

ALLOWED_LAYER_IMPORTS = {
    # Transport facades adapt protocol/CLI/MCP requests to the composition root
    # and boundary DTOs. They do not wire concrete infrastructure directly.
    "TRANSPORT_FACADE": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "DOMAIN",
        "KERNEL",
        "RUNTIME_SUPPORT",
        "UTILS",
        "FOUNDATION",
    },
    # The composition root is the only layer that may see both app-runtime
    # use cases and concrete infrastructure adapters.
    "COMPOSITION_ROOT": {
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "RUNTIME_SUPPORT",
        "DOMAIN",
        "KERNEL",
        "UTILS",
        "FOUNDATION",
    },
    # Config loading is external inbound mechanics, kept separate from runtime
    # contracts and from app/domain packages.
    "CONFIG_LOADER": {
        "CONFIG_LOADER",
        "BOUNDARY_IN",
        "RUNTIME_SUPPORT",
        "FOUNDATION",
    },
    # App runtime/use-case shell and infra are sibling shell roles. App runtime
    # coordinates workflows through ports and pure/domain packages; it must not
    # reach sideways into concrete infra adapters.
    "APP_RUNTIME": {
        "APP_RUNTIME",
        "PORTS",
        "DOMAIN",
        "KERNEL",
        "RUNTIME_SUPPORT",
        "UTILS",
        "FOUNDATION",
    },
    # Infra implements effects behind ports. It must not import app-runtime
    # orchestration, facades, config loaders, or inbound/outbound DTOs.
    "INFRA": {"INFRA", "PORTS", "KERNEL", "RUNTIME_SUPPORT", "UTILS", "FOUNDATION"},
    # Ports are internal capability contracts. They may name existing domain or
    # common contract types, but they never construct or import shell code.
    "PORTS": {"PORTS", "DOMAIN", "KERNEL", "FOUNDATION"},
    "BOUNDARY_IN": {"BOUNDARY_IN", "RUNTIME_SUPPORT", "FOUNDATION"},
    "BOUNDARY_OUT": {"BOUNDARY_OUT", "DOMAIN", "KERNEL", "FOUNDATION"},
    "RUNTIME_SUPPORT": {"RUNTIME_SUPPORT", "FOUNDATION"},
    "DOMAIN": {"DOMAIN", "KERNEL", "UTILS", "FOUNDATION"},
    "KERNEL": {"KERNEL", "UTILS", "FOUNDATION"},
    "UTILS": {"UTILS", "FOUNDATION"},
    "FOUNDATION": {"FOUNDATION"},
}

DENY = {
    "DOMAIN": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "RUNTIME_SUPPORT",
    },
    "BOUNDARY_IN": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "DOMAIN",
        "BOUNDARY_OUT",
        "KERNEL",
        "UTILS",
    },
    "BOUNDARY_OUT": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "BOUNDARY_IN",
        "RUNTIME_SUPPORT",
        "UTILS",
    },
    "CONFIG_LOADER": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "DOMAIN",
        "KERNEL",
        "UTILS",
        "BOUNDARY_OUT",
    },
    "RUNTIME_SUPPORT": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
        "PORTS",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "DOMAIN",
        "KERNEL",
        "UTILS",
    },
    "PORTS": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "RUNTIME_SUPPORT",
        "UTILS",
    },
    "INFRA": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
    },
    "KERNEL": {
        "RUNTIME_SUPPORT",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "DOMAIN",
        "PORTS",
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
    },
    "UTILS": {
        "RUNTIME_SUPPORT",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "DOMAIN",
        "KERNEL",
        "PORTS",
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
    },
    "FOUNDATION": {
        "RUNTIME_SUPPORT",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
        "DOMAIN",
        "KERNEL",
        "UTILS",
        "PORTS",
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "APP_RUNTIME",
        "INFRA",
    },
    "APP_RUNTIME": {
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
        "INFRA",
        "BOUNDARY_IN",
        "BOUNDARY_OUT",
    },
}

DOMAIN_PACKAGES = ("graph", "retrieval", "scip", "semantic")
DOMAIN_PUBLIC_API_MODULES = {
    "graph.contracts",
    "retrieval.contracts",
    "scip.contracts",
    "semantic.contracts",
}
KERNEL_DOMAIN_TYPE_MODULES = {
    "ir.element",
    "ir.projection",
    "ir.types",
    "kernel.identifiers",
    "scip.models",
}
PORT_IMPLEMENTATIONS = (
    ("store.infrastructure.runtime.DBRuntime", "ports.storage.StoreDatabaseRuntime"),
    (
        "store.infrastructure.graph_runtime.LadybugGraphRuntime",
        "ports.storage.DocumentGraphRuntime",
    ),
    (
        "store.infrastructure.execution.SubprocessSemanticHelperRuntime",
        "ports.execution.SemanticHelperRuntime",
    ),
    (
        "indexing.scip_runner.SubprocessScipIndexerRuntime",
        "ports.execution.ScipIndexerRuntime",
    ),
    ("indexing.embedder.CodeEmbedder", "ports.embedding.EmbeddingProvider"),
    ("indexing.terminus.TerminusPublisher", "ports.publishing.LineagePublisher"),
    ("store.file_artifacts.FileArtifactStore", "ports.artifacts.FileArtifactStore"),
    ("store.unit_artifacts.UnitArtifactStore", "ports.artifacts.UnitArtifactStore"),
    ("store.snapshot.SnapshotStore", "ports.jobs.RedoJobQueue"),
    ("store.snapshot.SnapshotStore", "ports.publishing.EventSink"),
    ("store.index_run.IndexRunStore", "ports.jobs.PublishRetryQueue"),
    ("store.index_run.IndexRunStore", "ports.jobs.IndexRunStore"),
)
BANNED_DOMAIN_IMPORTS = {"pydantic", "sqlite3", "subprocess", "urllib"}
BANNED_PORT_IMPORTS = {
    "anthropic",
    "dotenv",
    "faiss",
    "httpx",
    "numpy",
    "openai",
    "pydantic",
    "requests",
    "sqlite3",
    "subprocess",
    "urllib",
}
LEGACY_RAW_PORT_SIGNATURES = {
    # Storage/runtime ports intentionally expose backend handles.
    ("ports/storage.py", "StoreDatabaseRuntime", "connect", "return"),
    ("ports/storage.py", "StoreDatabaseRuntime", "execute", "conn"),
    ("ports/storage.py", "StoreDatabaseRuntime", "execute", "params"),
    ("ports/storage.py", "StoreDatabaseRuntime", "execute", "return"),
    ("ports/storage.py", "StoreDatabaseRuntime", "executemany", "conn"),
    ("ports/storage.py", "StoreDatabaseRuntime", "executemany", "params_seq"),
    ("ports/storage.py", "StoreDatabaseRuntime", "executemany", "return"),
    ("ports/storage.py", "StoreDatabaseRuntime", "begin_write", "conn"),
    # Existing compatibility payload methods. New port methods should use typed
    # records/views or domain/common dataclasses instead of extending this list.
    ("ports/storage.py", "DocumentGraphRuntime", "sync_docs", "chunks"),
    ("ports/storage.py", "DocumentGraphRuntime", "sync_docs", "mentions"),
    ("ports/artifacts.py", "UnitArtifactStore", "refresh_units", "elements"),
    ("ports/artifacts.py", "UnitArtifactStore", "replace_snapshot_units", "elements"),
    (
        "ports/artifacts.py",
        "UnitArtifactStore",
        "publish_snapshot_units_delta",
        "elements",
    ),
    (
        "ports/artifacts.py",
        "UnitArtifactStore",
        "publish_snapshot_units_delta",
        "return",
    ),
    (
        "ports/artifacts.py",
        "UnitArtifactStore",
        "replace_snapshot_file_ir_shards",
        "shards",
    ),
    (
        "ports/artifacts.py",
        "UnitArtifactStore",
        "replace_snapshot_file_ir_shards",
        "return",
    ),
    (
        "ports/artifacts.py",
        "UnitArtifactStore",
        "publish_snapshot_file_ir_shards_delta",
        "shards",
    ),
    (
        "ports/artifacts.py",
        "UnitArtifactStore",
        "publish_snapshot_file_ir_shards_delta",
        "reused_shards",
    ),
    (
        "ports/artifacts.py",
        "UnitArtifactStore",
        "publish_snapshot_file_ir_shards_delta",
        "return",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "list_file_ir_records_for_file_infos",
        "file_infos",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "list_parsed_element_records_for_file_infos",
        "file_infos",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "list_embedding_ref_records_for_file_infos",
        "file_infos",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "list_semantic_fact_records_for_file_infos",
        "file_infos",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "file_ir_payload_from_record",
        "return",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "parsed_elements_payload_from_record",
        "return",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "embedding_refs_payload_from_record",
        "return",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "semantic_facts_payload_from_record",
        "return",
    ),
    ("ports/artifacts.py", "FileArtifactStore", "upsert_file_ir_shards", "shards"),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "upsert_file_ir_shards",
        "file_infos",
    ),
    ("ports/artifacts.py", "FileArtifactStore", "upsert_file_ir_shards", "return"),
    ("ports/artifacts.py", "FileArtifactStore", "upsert_parsed_elements", "elements"),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "upsert_parsed_elements",
        "file_infos",
    ),
    ("ports/artifacts.py", "FileArtifactStore", "upsert_parsed_elements", "return"),
    ("ports/artifacts.py", "FileArtifactStore", "upsert_embedding_refs", "rows"),
    ("ports/artifacts.py", "FileArtifactStore", "upsert_embedding_refs", "file_infos"),
    ("ports/artifacts.py", "FileArtifactStore", "upsert_embedding_refs", "return"),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "upsert_semantic_fact_shards",
        "shards",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "upsert_semantic_fact_shards",
        "file_infos",
    ),
    (
        "ports/artifacts.py",
        "FileArtifactStore",
        "upsert_semantic_fact_shards",
        "return",
    ),
    ("ports/embedding.py", "EmbeddingProvider", "fingerprint", "return"),
    ("ports/embedding.py", "EmbeddingProvider", "embed_many", "return"),
    ("ports/embedding.py", "EmbeddingProvider", "embed_elements", "reuse_index"),
    ("ports/execution.py", "SemanticHelperRuntime", "run", "return"),
    ("ports/jobs.py", "RedoJobQueue", "enqueue_redo_task", "payload"),
    ("ports/jobs.py", "RedoJobQueue", "claim_redo_task", "return"),
    ("ports/publishing.py", "EventSink", "claim_outbox_event", "return"),
    (
        "ports/publishing.py",
        "LineagePublisher",
        "publish_snapshot_lineage_for_snapshot",
        "manifest",
    ),
    (
        "ports/publishing.py",
        "LineagePublisher",
        "publish_snapshot_lineage_for_snapshot",
        "git_meta",
    ),
    (
        "ports/publishing.py",
        "LineagePublisher",
        "publish_snapshot_lineage_for_snapshot",
        "previous_snapshot_symbols",
    ),
    ("ports/publishing.py", "LineagePublisher", "flush_outbox", "return"),
    ("ports/retrieval.py", "VectorSearchStore", "search", "query_vector"),
    (
        "ports/retrieval.py",
        "VectorSearchStore",
        "search",
        "query_embedding_fingerprint",
    ),
    ("ports/retrieval.py", "VectorSearchStore", "load_repo_overviews", "return"),
    (
        "ports/retrieval.py",
        "VectorSearchStore",
        "search_repository_overviews",
        "query_vector",
    ),
    (
        "ports/retrieval.py",
        "VectorSearchStore",
        "search_repository_overviews",
        "query_embedding_fingerprint",
    ),
    (
        "ports/retrieval.py",
        "VectorSearchStore",
        "search_repository_overviews",
        "return",
    ),
    (
        "ports/retrieval.py",
        "HybridRetrievalStore",
        "semantic_search",
        "query_embedding",
    ),
    (
        "ports/retrieval.py",
        "HybridRetrievalStore",
        "semantic_search",
        "query_embedding_fingerprint",
    ),
    ("ports/retrieval.py", "HybridRetrievalStore", "semantic_search", "return"),
    ("ports/retrieval.py", "HybridRetrievalStore", "keyword_search", "return"),
}
CAPABILITY_PORT_NAME_PARTS = (
    "Adapter",
    "Clock",
    "Generator",
    "Provider",
    "Queue",
    "Repository",
    "Runtime",
    "Sink",
    "Store",
)
PORTS_DIR = PACKAGE_ROOT / "ports"
STDLIB_IMPORTS = set(sys.stdlib_module_names) | {"__future__"}


def _layer_for_module(module_path: str) -> str | None:
    for prefix, layer in sorted(
        PREFIX_LAYERS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if module_path == prefix or module_path.startswith(f"{prefix}."):
            return layer
    top_level = module_path.split(".", 1)[0]
    return TOP_LEVEL_LAYERS.get(top_level)


def _importer_module_for_file(filepath: Path) -> str | None:
    rel = filepath.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts) if parts else None


def _target_module_for_import_from(filepath: Path, node: ast.ImportFrom) -> str | None:
    if node.level:
        rel_parts = filepath.relative_to(PACKAGE_ROOT).with_suffix("").parts
        package_parts = list(rel_parts[:-1])
        base_parts = package_parts[: len(package_parts) - (node.level - 1)]
        if node.module:
            target_parts = [*base_parts, *node.module.split(".")]
        elif node.names:
            target_parts = [*base_parts, *node.names[0].name.split(".")]
        else:
            return None
        return ".".join(target_parts) if target_parts else None

    if not node.module:
        return None
    if node.module == "fastcode" and node.names:
        return node.names[0].name
    if node.module.startswith("fastcode."):
        return node.module.removeprefix("fastcode.")
    return None


def _tree(filepath: Path) -> ast.Module:
    return ast.parse(filepath.read_text())


def _iter_python_files() -> list[Path]:
    ignored_parts = {"__pycache__", "_stubs"}
    return [
        path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if not ignored_parts.intersection(path.relative_to(PACKAGE_ROOT).parts)
    ]


def _get_fastcode_imports(filepath: Path) -> list[tuple[str, int, bool]]:
    imports: list[tuple[str, int, bool]] = []
    for node in ast.walk(_tree(filepath)):
        if isinstance(node, ast.ImportFrom):
            target = _target_module_for_import_from(filepath, node)
            is_absolute_fastcode = bool(
                node.level == 0 and node.module and node.module.startswith("fastcode")
            )
            if target:
                imports.append((target, node.lineno, is_absolute_fastcode))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fastcode."):
                    imports.append(
                        (alias.name.removeprefix("fastcode."), node.lineno, True)
                    )
    return imports


def _is_contract_or_type_module(module_path: str) -> bool:
    return any(
        module_path == allowed or module_path.startswith(f"{allowed}.")
        for allowed in (*DOMAIN_PUBLIC_API_MODULES, *KERNEL_DOMAIN_TYPE_MODULES)
    )


def _domain_package_for_module(module_path: str) -> str | None:
    root = module_path.split(".", 1)[0]
    return root if root in DOMAIN_PACKAGES else None


def _is_domain_public_api_module(module_path: str) -> bool:
    return any(
        module_path == allowed or module_path.startswith(f"{allowed}.")
        for allowed in DOMAIN_PUBLIC_API_MODULES
    )


def _expr_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    return None


def _is_protocol_class(node: ast.ClassDef) -> bool:
    return any(_expr_name(base) == "Protocol" for base in node.bases)


def _annotation_text(annotation: ast.expr | None) -> str:
    return ast.unparse(annotation) if annotation is not None else "<missing>"


def _is_raw_payload_annotation(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    text = ast.unparse(annotation)
    return (
        text == "Any"
        or "Any" in text
        or text == "dict"
        or text.startswith("dict[")
        or "dict[" in text
        or text == "Mapping"
        or text.startswith("Mapping[")
        or "Mapping[" in text
    )


def _iter_protocol_methods(
    filepath: Path,
) -> list[tuple[ast.ClassDef, ast.FunctionDef]]:
    methods: list[tuple[ast.ClassDef, ast.FunctionDef]] = []
    for node in ast.walk(_tree(filepath)):
        if not isinstance(node, ast.ClassDef) or not _is_protocol_class(node):
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append((node, child))
    return methods


def _iter_port_files() -> list[Path]:
    if not PORTS_DIR.is_dir():
        return []
    return [
        path
        for path in sorted(PORTS_DIR.rglob("*.py"))
        if "__pycache__" not in path.relative_to(PACKAGE_ROOT).parts
    ]


def _defines_protocol_class(filepath: Path) -> bool:
    return any(
        isinstance(node, ast.ClassDef) and _is_protocol_class(node)
        for node in ast.walk(_tree(filepath))
    )


def _has_capability_port_name(node: ast.ClassDef) -> bool:
    return any(part in node.name for part in CAPABILITY_PORT_NAME_PARTS)


def _split_qualified_class(qualified_name: str) -> tuple[str, str]:
    module_name, _, class_name = qualified_name.rpartition(".")
    return module_name, class_name


def _module_file(module_name: str) -> Path:
    return PACKAGE_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")


def _class_def(qualified_name: str) -> ast.ClassDef:
    module_name, class_name = _split_qualified_class(qualified_name)
    module_path = _module_file(module_name)
    for node in ast.walk(_tree(module_path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{qualified_name} is missing")


def _method_signature_shape(
    node: ast.FunctionDef,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    positional = tuple(arg.arg for arg in [*node.args.posonlyargs, *node.args.args])
    keyword_only = tuple(arg.arg for arg in node.args.kwonlyargs)
    return positional, keyword_only


def _protocol_methods(
    node: ast.ClassDef,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    methods: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
            methods[child.name] = _method_signature_shape(child)
    return methods


def _protocol_attributes(node: ast.ClassDef) -> set[str]:
    attributes: set[str] = set()
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            attributes.add(child.target.id)
    return attributes


def _implementation_methods(
    node: ast.ClassDef,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    methods: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            methods.setdefault(child.name, _method_signature_shape(child))
    return methods


def _implementation_attributes(node: ast.ClassDef) -> set[str]:
    attributes: set[str] = set()
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            attributes.add(child.target.id)
        elif isinstance(child, ast.Assign):
            attributes.update(
                target.id for target in child.targets if isinstance(target, ast.Name)
            )

    for child in node.body:
        if not isinstance(child, ast.FunctionDef) or child.name != "__init__":
            continue
        for assign in ast.walk(child):
            targets: list[ast.expr] = []
            if isinstance(assign, ast.AnnAssign):
                targets.append(assign.target)
            elif isinstance(assign, ast.Assign):
                targets.extend(assign.targets)
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attributes.add(target.attr)
    return attributes


def _imported_roots(filepath: Path) -> list[tuple[str, int]]:
    roots: list[tuple[str, int]] = []
    for node in ast.walk(_tree(filepath)):
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.append((node.module.split(".", 1)[0], node.lineno))
        elif isinstance(node, ast.Import):
            roots.extend(
                (alias.name.split(".", 1)[0], node.lineno) for alias in node.names
            )
    return roots


def test_all_runtime_modules_are_classified() -> None:
    """Every runtime Python module must live in one of the explicit layers."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if importer_module is None:
            continue
        if _layer_for_module(importer_module) is None:
            violations.append(str(py_file.relative_to(PACKAGE_ROOT)))
    assert not violations, "Unclassified runtime modules:\n" + "\n".join(violations)


def test_strict_layer_dag_imports() -> None:
    """Packages may only import within their layer or downward."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if importer_module is None:
            continue
        importer_layer = _layer_for_module(importer_module)
        if importer_layer is None:
            continue
        for target, line, is_absolute_fastcode in _get_fastcode_imports(py_file):
            target_layer = _layer_for_module(target)
            if target_layer is None:
                continue
            if importer_layer == "FOUNDATION" and is_absolute_fastcode:
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                    f"FOUNDATION imports absolute fastcode.{target}"
                )
            if target_layer in DENY.get(importer_layer, set()):
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                    f"{importer_module} ({importer_layer}) imports "
                    f"{target} ({target_layer})"
                )
    assert not violations, "Layer DAG violations:\n" + "\n".join(violations)


def test_fcis_dependency_flow_is_allowlisted() -> None:
    """Every compile-time import must follow the explicit FCIS dependency graph."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if importer_module is None:
            continue
        importer_layer = _layer_for_module(importer_module)
        if importer_layer is None:
            continue
        allowed_targets = ALLOWED_LAYER_IMPORTS[importer_layer]
        for target, line, _ in _get_fastcode_imports(py_file):
            target_layer = _layer_for_module(target)
            if target_layer is None or target_layer in allowed_targets:
                continue
            violations.append(
                f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                f"{importer_module} ({importer_layer}) imports "
                f"{target} ({target_layer}); allowed targets are "
                f"{', '.join(sorted(allowed_targets))}"
            )

    assert not violations, "FCIS dependency-flow violations:\n" + "\n".join(violations)


def test_fcis_shell_subroles_are_explicitly_classified() -> None:
    """The shell is split into app-runtime, infra, and capability-port roles."""
    expected = {
        "api.routes": "TRANSPORT_FACADE",
        "api.web": "TRANSPORT_FACADE",
        "api.inbound": "BOUNDARY_IN",
        "api.outbound": "BOUNDARY_OUT",
        "main.cli": "TRANSPORT_FACADE",
        "main.config": "CONFIG_LOADER",
        "main.fastcode": "COMPOSITION_ROOT",
        "foundation.byte_count": "FOUNDATION",
        "foundation.non_empty_string": "FOUNDATION",
        "foundation.positive_int": "FOUNDATION",
        "kernel.identifiers": "KERNEL",
        "indexing.pipeline": "APP_RUNTIME",
        "indexing.scip_runner": "INFRA",
        "query.retriever": "APP_RUNTIME",
        "runtime_support.health": "RUNTIME_SUPPORT",
        "runtime_support.retry": "RUNTIME_SUPPORT",
        "store.snapshot": "APP_RUNTIME",
        "store.infrastructure.db": "INFRA",
        "runtime.config": "RUNTIME_SUPPORT",
        "ir.types": "KERNEL",
        "utils.clock": "UTILS",
        "utils.filesystem": "UTILS",
        "ports.storage": "PORTS",
        "ports.artifacts": "PORTS",
        "ports.embedding": "PORTS",
        "ports.publishing": "PORTS",
        "ports.jobs": "PORTS",
    }

    violations = [
        f"{module}: expected {layer}, got {_layer_for_module(module)}"
        for module, layer in expected.items()
        if _layer_for_module(module) != layer
    ]
    assert not violations, "FCIS shell role classification drift:\n" + "\n".join(
        violations
    )


def test_capability_ports_are_shared_compile_time_contracts() -> None:
    """App-runtime and infra may import ports; ports may not wire either side."""
    violations: list[str] = []

    if "PORTS" in DENY.get("APP_RUNTIME", set()):
        violations.append("APP_RUNTIME must be allowed to import PORTS")
    if "PORTS" in DENY.get("INFRA", set()):
        violations.append("INFRA must be allowed to import PORTS")

    port_denials = DENY.get("PORTS", set())
    for forbidden_layer in (
        "APP_RUNTIME",
        "INFRA",
        "TRANSPORT_FACADE",
        "COMPOSITION_ROOT",
        "CONFIG_LOADER",
    ):
        if forbidden_layer not in port_denials:
            violations.append(f"PORTS must not import {forbidden_layer}")

    assert not violations, "Capability port direction drift:\n" + "\n".join(violations)


def test_app_runtime_and_infra_are_sibling_shell_roles() -> None:
    """Use-case shell and infra adapters meet at ports or the composition root."""
    violations: list[str] = []
    for importer_layer, forbidden_layer in (
        ("APP_RUNTIME", "INFRA"),
        ("INFRA", "APP_RUNTIME"),
    ):
        if forbidden_layer not in DENY.get(importer_layer, set()):
            violations.append(f"{importer_layer} must not import {forbidden_layer}")
        if forbidden_layer in ALLOWED_LAYER_IMPORTS[importer_layer]:
            violations.append(
                f"{importer_layer} allowlist includes sibling shell {forbidden_layer}"
            )
    for shell_layer in ("APP_RUNTIME", "INFRA"):
        if "PORTS" not in ALLOWED_LAYER_IMPORTS[shell_layer]:
            violations.append(f"{shell_layer} must depend on PORTS for effects")
    assert not violations, "FCIS sibling shell role drift:\n" + "\n".join(violations)


def test_capability_ports_are_contract_only() -> None:
    """Ports define external capabilities, not infra, app logic, or DTOs."""
    violations: list[str] = []
    for py_file in _iter_port_files():
        for imported_root, line in _imported_roots(py_file):
            if imported_root in BANNED_PORT_IMPORTS:
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                    f"port imports {imported_root}"
                )
    assert not violations, "Capability port import violations:\n" + "\n".join(
        violations
    )


def test_capability_ports_live_only_in_global_ports_package() -> None:
    """Capability traits live in fastcode.ports, not local package ports files."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        module = _importer_module_for_file(py_file)
        if module is None or module == "ports" or module.startswith("ports."):
            continue
        rel = py_file.relative_to(PACKAGE_ROOT)
        if py_file.name == "ports.py":
            violations.append(f"{rel}: local ports.py capability module")
            continue
        if py_file.name != "contracts.py" or not _defines_protocol_class(py_file):
            continue
        protocol_names = [
            node.name
            for node in ast.walk(_tree(py_file))
            if isinstance(node, ast.ClassDef)
            and _is_protocol_class(node)
            and _has_capability_port_name(node)
        ]
        if protocol_names:
            violations.append(
                f"{rel}: local capability protocol(s) {', '.join(protocol_names)}"
            )

    assert not violations, (
        "External capability ports must live under fastcode.ports:\n"
        + "\n".join(violations)
    )


def test_ports_use_domain_or_common_types_without_owning_models() -> None:
    """Ports describe capabilities using existing domain/common types."""
    violations: list[str] = []
    for py_file in _iter_port_files():
        rel = py_file.relative_to(PACKAGE_ROOT)
        tree = _tree(py_file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if _is_protocol_class(node):
                continue
            violations.append(
                f"{rel}:{node.lineno}: ports may not define model class {node.name}"
            )

        for target, line, _ in _get_fastcode_imports(py_file):
            if _is_contract_or_type_module(target):
                continue
            violations.append(
                f"{rel}:{line}: ports import fastcode.{target}; use domain/common "
                "contract or type modules only"
            )

    assert not violations, (
        "Ports should define capability protocols over existing domain/common "
        "types, not own payload models or import app/domain logic:\n"
        + "\n".join(violations)
    )


def test_new_port_methods_do_not_expose_raw_payload_types() -> None:
    """Ports speak typed internal records/views; raw payloads need explicit legacy buy-in."""
    violations: list[str] = []
    for py_file in _iter_port_files():
        rel = py_file.relative_to(PACKAGE_ROOT).as_posix()
        for protocol, method in _iter_protocol_methods(py_file):
            args = [
                *method.args.posonlyargs,
                *method.args.args,
                *method.args.kwonlyargs,
            ]
            for arg in args:
                if arg.arg in {"self", "cls"}:
                    continue
                key = (rel, protocol.name, method.name, arg.arg)
                if key in LEGACY_RAW_PORT_SIGNATURES:
                    continue
                if _is_raw_payload_annotation(arg.annotation):
                    violations.append(
                        f"{rel}:{method.lineno}: {protocol.name}.{method.name}() "
                        f"argument {arg.arg}: {_annotation_text(arg.annotation)}"
                    )
            key = (rel, protocol.name, method.name, "return")
            if key in LEGACY_RAW_PORT_SIGNATURES:
                continue
            if _is_raw_payload_annotation(method.returns):
                violations.append(
                    f"{rel}:{method.lineno}: {protocol.name}.{method.name}() "
                    f"return: {_annotation_text(method.returns)}"
                )

    assert not violations, (
        "New or changed port signatures expose raw dict/Mapping/Any payloads. "
        "Use typed records/views or add a deliberate legacy exception:\n"
        + "\n".join(violations)
    )


def test_port_implementations_structurally_match_declared_ports() -> None:
    """Known concrete adapters must align with their global port protocols."""
    violations: list[str] = []
    for implementation, protocol in PORT_IMPLEMENTATIONS:
        implementation_class = _class_def(implementation)
        protocol_class = _class_def(protocol)

        implementation_methods = _implementation_methods(implementation_class)
        implementation_attributes = _implementation_attributes(implementation_class)
        for attr in sorted(_protocol_attributes(protocol_class)):
            if attr not in implementation_attributes:
                violations.append(f"{implementation} missing port attribute {attr}")

        for method, signature in sorted(_protocol_methods(protocol_class).items()):
            implementation_signature = implementation_methods.get(method)
            if implementation_signature is None:
                violations.append(f"{implementation} missing port method {method}()")
                continue
            if implementation_signature != signature:
                violations.append(
                    f"{implementation}.{method}{implementation_signature} does not "
                    f"match {protocol}.{method}{signature}"
                )

    assert not violations, "Implementation/port alignment drift:\n" + "\n".join(
        violations
    )


def test_store_database_clients_use_database_runtime_port() -> None:
    """Store clients depend on the DB port, not the concrete DB adapter."""
    violations: list[str] = []

    for py_file in sorted((PACKAGE_ROOT / "store").glob("*.py")):
        for target, line, _ in _get_fastcode_imports(py_file):
            if target == "store.infrastructure.runtime" or target.startswith(
                "store.infrastructure.runtime."
            ):
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: imports {target}"
                )

    assert not violations, (
        "Store modules should receive StoreDatabaseRuntime instead of importing "
        "the concrete DB runtime:\n" + "\n".join(violations)
    )


def test_indexing_publish_clients_use_lineage_publisher_port() -> None:
    """Indexing services depend on the publishing port, not the concrete adapter."""
    checked_files = [
        PACKAGE_ROOT / "indexing" / "pipeline.py",
        PACKAGE_ROOT / "indexing" / "publishing.py",
    ]
    violations: list[str] = []

    for py_file in checked_files:
        for target, line, _ in _get_fastcode_imports(py_file):
            if target == "indexing.terminus" or target.startswith("indexing.terminus."):
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: imports {target}"
                )

    assert not violations, (
        "Indexing publish clients should receive LineagePublisher instead of "
        "importing the concrete Terminus adapter:\n" + "\n".join(violations)
    )


def test_query_retriever_uses_embedding_provider_port() -> None:
    """Query retrieval depends on the embedding port, not the concrete embedder."""
    retriever_file = PACKAGE_ROOT / "query" / "retriever.py"
    violations = [
        f"{retriever_file.relative_to(PACKAGE_ROOT)}:{line}: imports {target}"
        for target, line, _ in _get_fastcode_imports(retriever_file)
        if target == "indexing.embedder" or target.startswith("indexing.embedder.")
    ]

    assert not violations, (
        "Query retriever should receive EmbeddingProvider instead of importing "
        "the concrete CodeEmbedder:\n" + "\n".join(violations)
    )


def test_storage_runtime_ports_do_not_own_artifact_store_contracts() -> None:
    """Low-level storage runtime ports stay separate from artifact-store ports."""
    storage_file = PACKAGE_ROOT / "ports" / "storage.py"
    storage_class_names = {
        node.name
        for node in ast.walk(_tree(storage_file))
        if isinstance(node, ast.ClassDef)
    }
    forbidden_names = {
        "FileArtifactRecordView",
        "FileArtifactStore",
        "UnitArtifactStore",
    }
    violations = sorted(storage_class_names & forbidden_names)

    assert not violations, (
        "ports.storage should contain low-level runtime contracts only; "
        "artifact-store contracts belong in ports.artifacts:\n" + "\n".join(violations)
    )


def test_domain_packages_use_other_domain_public_apis_only() -> None:
    """Domain internals stay local; cross-domain use goes through public APIs."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if importer_module is None:
            continue
        importer_domain = _domain_package_for_module(importer_module)
        if importer_domain is None:
            continue
        for target, line, _ in _get_fastcode_imports(py_file):
            target_domain = _domain_package_for_module(target)
            if target_domain is None or target_domain == importer_domain:
                continue
            if _is_domain_public_api_module(target):
                continue
            violations.append(
                f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                f"domain module {importer_module} imports non-public "
                f"domain API fastcode.{target}"
            )

    assert not violations, (
        "Cross-domain imports must target public domain API modules:\n"
        + "\n".join(violations)
    )


def test_app_runtime_imports_domain_public_api_not_private_modules() -> None:
    """App-runtime code should not reach into domain private modules."""
    violations: list[str] = []
    for py_file in _iter_python_files():
        importer_module = _importer_module_for_file(py_file)
        if (
            importer_module is None
            or _layer_for_module(importer_module) != "APP_RUNTIME"
        ):
            continue
        for target, line, _ in _get_fastcode_imports(py_file):
            if _layer_for_module(target) != "DOMAIN":
                continue
            parts = target.split(".")
            if len(parts) > 1 and any(part.startswith("_") for part in parts[1:]):
                violations.append(
                    f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                    f"imports private domain module fastcode.{target}"
                )

    assert not violations, (
        "App-runtime code should import domain public APIs, not private "
        "implementation modules:\n" + "\n".join(violations)
    )


def test_store_database_runtime_port_excludes_generic_row_materialization() -> None:
    """The DB port must not expose row_to_dict as a cross-layer capability."""
    contracts_file = PACKAGE_ROOT / "ports" / "storage.py"
    violations: list[str] = []
    for node in ast.walk(_tree(contracts_file)):
        if not isinstance(node, ast.ClassDef) or node.name != "StoreDatabaseRuntime":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "row_to_dict":
                violations.append(
                    f"{contracts_file.relative_to(PACKAGE_ROOT)}:{child.lineno}: "
                    "StoreDatabaseRuntime exposes row_to_dict()"
                )

    assert not violations, (
        "StoreDatabaseRuntime should expose typed SQL execution capabilities, not "
        "generic row materialization:\n" + "\n".join(violations)
    )


def test_domain_contracts_use_frozen_dataclasses_not_pydantic() -> None:
    """Domain contracts are local frozen dataclasses and never Pydantic models."""
    violations: list[str] = []
    for package in DOMAIN_PACKAGES:
        contracts_file = PACKAGE_ROOT / package / "contracts.py"
        if not contracts_file.exists():
            violations.append(f"{package}/contracts.py is missing")
            continue
        tree = _tree(contracts_file)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "pydantic" in node.module
            ):
                violations.append(
                    f"{contracts_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: pydantic import"
                )
            if not isinstance(node, ast.ClassDef):
                continue
            dataclass_decorator = next(
                (
                    decorator
                    for decorator in node.decorator_list
                    if (
                        isinstance(decorator, ast.Call)
                        and getattr(decorator.func, "id", None) == "dataclass"
                    )
                    or getattr(decorator, "id", None) == "dataclass"
                ),
                None,
            )
            if dataclass_decorator is None:
                continue
            if not isinstance(dataclass_decorator, ast.Call):
                violations.append(
                    f"{contracts_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                    f"{node.name} dataclass must set frozen=True"
                )
                continue
            frozen = any(
                keyword.arg == "frozen"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in dataclass_decorator.keywords
            )
            if not frozen:
                violations.append(
                    f"{contracts_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                    f"{node.name} dataclass must set frozen=True"
                )
    assert not violations, "Domain contract violations:\n" + "\n".join(violations)


def test_domain_modules_contain_no_shell_io_imports() -> None:
    """Domain modules must not pull in obvious shell I/O libraries."""
    violations: list[str] = []
    for package in DOMAIN_PACKAGES:
        for py_file in (PACKAGE_ROOT / package).rglob("*.py"):
            for node in ast.walk(_tree(py_file)):
                imported_roots: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.append(node.module.split(".", 1)[0])
                elif isinstance(node, ast.Import):
                    imported_roots.extend(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                for imported_root in imported_roots:
                    if imported_root in BANNED_DOMAIN_IMPORTS:
                        violations.append(
                            f"{py_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                            f"imports {imported_root}"
                        )
    assert not violations, "Banned imports in domain modules:\n" + "\n".join(violations)


def test_foundation_uses_only_stdlib_imports() -> None:
    """Foundation stays a small stdlib-only leaf layer."""
    violations: list[str] = []
    foundation_dir = PACKAGE_ROOT / "foundation"
    if foundation_dir.is_dir():
        for py_file in foundation_dir.rglob("*.py"):
            for node in ast.walk(_tree(py_file)):
                imported_roots: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.level == 0:
                        imported_roots.append(node.module.split(".", 1)[0])
                elif isinstance(node, ast.Import):
                    imported_roots.extend(
                        alias.name.split(".", 1)[0] for alias in node.names
                    )
                for imported_root in imported_roots:
                    if imported_root not in STDLIB_IMPORTS:
                        violations.append(
                            f"{py_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                            f"imports non-stdlib module {imported_root}"
                        )

    assert not violations, "Non-stdlib imports in Foundation layer:\n" + "\n".join(
        violations
    )


def test_utils_use_only_stdlib_and_foundation_imports() -> None:
    """Utils stay generic and may depend only on stdlib plus foundation."""
    violations: list[str] = []
    for py_file in (PACKAGE_ROOT / "utils").rglob("*.py"):
        for node in ast.walk(_tree(py_file)):
            imported_roots: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 0:
                    imported_roots.append(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Import):
                imported_roots.extend(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            for imported_root in imported_roots:
                if imported_root not in STDLIB_IMPORTS | {"fastcode"}:
                    violations.append(
                        f"{py_file.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                        f"imports non-stdlib module {imported_root}"
                    )
        for target, line, _ in _get_fastcode_imports(py_file):
            if _layer_for_module(target) == "FOUNDATION":
                continue
            if target.startswith("utils.") or target == "utils":
                continue
            violations.append(
                f"{py_file.relative_to(PACKAGE_ROOT)}:{line}: "
                f"utils imports fastcode.{target}; only foundation or utils is allowed"
            )
    assert not violations, "Invalid imports in Utils layer:\n" + "\n".join(violations)
