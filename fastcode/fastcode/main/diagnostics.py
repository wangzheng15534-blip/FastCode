"""Diagnostic bundle builder extracted from FastCode."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Mapping
from importlib import util as importlib_util
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from fastcode.app.store.runs.index_run_contracts import IndexRunRecord
from fastcode.app.store.snapshots.manifest import ManifestStore
from fastcode.app.store.snapshots.snapshot import SnapshotStore
from fastcode.app.store.vectors.vector import VectorStore
from fastcode.utils.clock import utc_now

from fastcode.runtime_support.runtime_state import RuntimeState

# ---------------------------------------------------------------------------
# Module-level constants (moved from fastcode.py)
# ---------------------------------------------------------------------------

_DIAGNOSTIC_PYTHON_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("core", "numpy", "numpy"),
    ("core", "pydantic", "pydantic"),
    ("core", "tree_sitter", "tree_sitter"),
    ("core", "gitpython", "git"),
    ("retrieval", "faiss", "faiss"),
    ("retrieval", "rank_bm25", "rank_bm25"),
    ("retrieval", "networkx", "networkx"),
    ("retrieval", "igraph", "igraph"),
    ("llm", "openai", "openai"),
    ("llm", "anthropic", "anthropic"),
    ("llm", "tiktoken", "tiktoken"),
    ("api", "fastapi", "fastapi"),
    ("api", "uvicorn", "uvicorn"),
    ("api", "flask", "flask"),
    ("postgres", "psycopg", "psycopg"),
    ("postgres", "psycopg_pool", "psycopg_pool"),
    ("postgres", "pgvector", "pgvector"),
    ("cache", "redis", "redis"),
    ("docs", "chonkie", "chonkie"),
    ("embeddings", "sentence_transformers", "sentence_transformers"),
    ("mcp", "mcp", "mcp"),
    ("scip", "protobuf", "google.protobuf"),
    ("ladybug", "real_ladybug", "real_ladybug"),
)

_DIAGNOSTIC_EXTERNAL_TOOLS: tuple[tuple[str, str], ...] = (
    ("git", "git"),
    ("scip", "scip"),
    ("node", "node"),
    ("go", "go"),
    ("cargo", "cargo"),
    ("javac", "javac"),
)

_DIAGNOSTIC_REDACTION = "[redacted]"
_DIAGNOSTIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(api[_-]?key|token|password|passwd|secret)"
    r"(\s*[:=]\s*)"
    r"[^,\s;]+",
    re.IGNORECASE,
)
_DIAGNOSTIC_AUTH_HEADER_RE = re.compile(
    r"\b(authorization)(\s*[:=]\s*)(?:(Bearer)\s+)?[^,\s;]+",
    re.IGNORECASE,
)
_DIAGNOSTIC_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_DIAGNOSTIC_URL_USERINFO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://)([^/\s@]+@)")
_DIAGNOSTIC_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "auth_token",
    "authorization",
    "password",
    "passwd",
    "secret",
    "credential",
    "private_key",
    "postgres_dsn",
    "dsn",
    "cookie",
)

# ---------------------------------------------------------------------------
# Module-level pure helpers (no state, no self)
# ---------------------------------------------------------------------------


def diagnostic_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def diagnostic_list(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return list(value)


def diagnostic_key_is_sensitive(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(marker in normalized for marker in _DIAGNOSTIC_SECRET_KEY_MARKERS)


def redact_diagnostic_string(value: str) -> str:
    redacted = value
    try:
        parts = urlsplit(value)
    except ValueError:
        parts = None
    if parts is not None and parts.scheme and parts.netloc:
        host = parts.hostname or ""
        if host:
            netloc = host
            try:
                port = parts.port
            except ValueError:
                port = None
            if port is not None:
                netloc = f"{netloc}:{port}"
            if parts.username or parts.password:
                netloc = f"{_DIAGNOSTIC_REDACTION}@{netloc}"
            redacted = urlunsplit(
                (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
            )
    redacted = _DIAGNOSTIC_URL_USERINFO_RE.sub(rf"\1{_DIAGNOSTIC_REDACTION}@", redacted)

    def _redact_auth_header(match: re.Match[str]) -> str:
        scheme = f"{match.group(3)} " if match.group(3) else ""
        return f"{match.group(1)}{match.group(2)}{scheme}{_DIAGNOSTIC_REDACTION}"

    redacted = _DIAGNOSTIC_AUTH_HEADER_RE.sub(_redact_auth_header, redacted)
    redacted = _DIAGNOSTIC_BEARER_RE.sub(f"Bearer {_DIAGNOSTIC_REDACTION}", redacted)
    return _DIAGNOSTIC_SECRET_ASSIGNMENT_RE.sub(
        rf"\1\2{_DIAGNOSTIC_REDACTION}", redacted
    )


def redact_diagnostic_value(key: str, value: Any) -> Any:
    if diagnostic_key_is_sensitive(key):
        return bool(value) if key.endswith("_configured") else _DIAGNOSTIC_REDACTION
    if isinstance(value, str):
        return redact_diagnostic_string(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_diagnostic_value(str(item_key), item_value)
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_diagnostic_value(key, item) for item in value]
    return value


def _configured(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def diagnostic_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return diagnostic_mapping(parsed)


def diagnostic_json_string_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item is not None]


def diagnostic_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    storage = diagnostic_mapping(config.get("storage"))
    repository = diagnostic_mapping(config.get("repository"))
    embedding = diagnostic_mapping(config.get("embedding"))
    indexing = diagnostic_mapping(config.get("indexing"))
    vector_store = diagnostic_mapping(config.get("vector_store"))
    retrieval = diagnostic_mapping(config.get("retrieval"))
    generation = diagnostic_mapping(config.get("generation"))
    evaluation = diagnostic_mapping(config.get("evaluation"))
    cache = diagnostic_mapping(config.get("cache"))
    terminus = diagnostic_mapping(config.get("terminus"))
    projection = diagnostic_mapping(config.get("projection"))
    supported_extensions = diagnostic_list(repository.get("supported_extensions"))
    ignore_patterns = diagnostic_list(repository.get("ignore_patterns"))
    return {
        "storage": {
            "backend": str(storage.get("backend") or "sqlite"),
            "postgres_dsn_configured": _configured(storage.get("postgres_dsn")),
            "pool_min": storage.get("pool_min"),
            "pool_max": storage.get("pool_max"),
        },
        "repository": {
            "repo_root": redact_diagnostic_value("repo_root", config.get("repo_root")),
            "max_file_size_mb": repository.get("max_file_size_mb"),
            "exclude_site_packages": bool(
                repository.get("exclude_site_packages", False)
            ),
            "ignore_pattern_count": len(ignore_patterns),
            "supported_extension_count": len(supported_extensions),
        },
        "embedding": {
            "provider": embedding.get("provider"),
            "model": embedding.get("model"),
            "device": embedding.get("device"),
            "batch_size": embedding.get("batch_size"),
            "ollama_url_configured": _configured(embedding.get("ollama_url")),
        },
        "indexing": {
            "levels": [str(item) for item in diagnostic_list(indexing.get("levels"))],
            "allow_direct_index": bool(indexing.get("allow_direct_index", False)),
        },
        "vector_store": {
            "persist_directory": redact_diagnostic_value(
                "persist_directory", vector_store.get("persist_directory")
            ),
            "distance_metric": vector_store.get("distance_metric"),
            "shard_storage": vector_store.get("shard_storage"),
        },
        "retrieval": {
            "retrieval_backend": retrieval.get("retrieval_backend"),
            "graph_expansion_backend": retrieval.get("graph_expansion_backend"),
            "max_results": retrieval.get("max_results"),
            "semantic_weight": retrieval.get("semantic_weight"),
            "keyword_weight": retrieval.get("keyword_weight"),
            "graph_weight": retrieval.get("graph_weight"),
        },
        "generation": {
            "provider": generation.get("provider"),
            "model": generation.get("model"),
            "temperature": generation.get("temperature"),
            "max_tokens": generation.get("max_tokens"),
        },
        "evaluation": {
            "enabled": bool(evaluation.get("enabled", False)),
            "in_memory_index": bool(evaluation.get("in_memory_index", False)),
            "disable_cache": bool(evaluation.get("disable_cache", False)),
            "disable_persistence": bool(evaluation.get("disable_persistence", False)),
            "force_reindex": bool(evaluation.get("force_reindex", False)),
        },
        "cache": {
            "enabled": bool(cache.get("enabled", True)),
            "backend": cache.get("backend"),
            "cache_directory": redact_diagnostic_value(
                "cache_directory", cache.get("cache_directory")
            ),
            "cache_queries": bool(cache.get("cache_queries", False)),
        },
        "terminus": {
            "endpoint_configured": _configured(terminus.get("endpoint")),
            "api_key_configured": _configured(terminus.get("api_key")),
        },
        "projection": {
            "postgres_dsn_configured": _configured(projection.get("postgres_dsn")),
            "enable_leiden": bool(projection.get("enable_leiden", False)),
            "llm_enabled": bool(projection.get("llm_enabled", False)),
        },
    }


def _dependency_available(import_name: str) -> bool:
    try:
        return importlib_util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def diagnostic_dependency_summary() -> dict[str, Any]:
    python_dependencies = [
        {
            "group": group,
            "name": name,
            "import_name": import_name,
            "available": _dependency_available(import_name),
        }
        for group, name, import_name in _DIAGNOSTIC_PYTHON_DEPENDENCIES
    ]
    external_tools = []
    for name, executable in _DIAGNOSTIC_EXTERNAL_TOOLS:
        resolved = shutil.which(executable)
        external_tools.append(
            {
                "name": name,
                "executable": executable,
                "available": resolved is not None,
                "path": redact_diagnostic_value("path", resolved),
            }
        )
    return {
        "python": python_dependencies,
        "external_tools": external_tools,
    }


# ---------------------------------------------------------------------------
# DiagnosticBuilder — wraps instance-state-dependent methods
# ---------------------------------------------------------------------------


class DiagnosticBuilder:
    """Builds support-safe runtime diagnostic bundles.

    Receives the external dependencies it needs rather than reaching through
    a FastCode instance.
    """

    def __init__(
        self,
        config: dict[str, Any],
        vector_store: VectorStore,
        snapshot_store: SnapshotStore,
        manifest_store: ManifestStore,
        state: RuntimeState,
        logger: logging.Logger,
        eval_config: dict[str, Any],
        index_run_store: Any,
        projection_store: Any,
        cache_manager: Any,
        loader: Any,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._snapshot_store = snapshot_store
        self._manifest_store = manifest_store
        self._state = state
        self._logger = logger
        self._eval_config = eval_config
        self._index_run_store = index_run_store
        self._projection_store = projection_store
        self._cache_manager = cache_manager
        self._loader = loader

    # -- instance methods that read injected state --------------------------

    def _diagnostic_storage_summary(self) -> dict[str, Any]:
        snapshot_store = self._snapshot_store
        db_runtime = getattr(snapshot_store, "db_runtime", None)
        vector_store = self._vector_store
        projection_store = self._projection_store
        cache_manager = self._cache_manager
        config = diagnostic_mapping(self._config)
        cache_config = diagnostic_mapping(config.get("cache"))
        return {
            "backend": str(getattr(db_runtime, "backend", "unknown")),
            "sqlite_path": redact_diagnostic_value(
                "sqlite_path", getattr(db_runtime, "sqlite_path", None)
            ),
            "postgres_dsn_configured": _configured(
                getattr(db_runtime, "postgres_dsn", None)
            ),
            "pool_min": getattr(db_runtime, "pool_min", None),
            "pool_max": getattr(db_runtime, "pool_max", None),
            "pool_configured": getattr(db_runtime, "pool", None) is not None,
            "vector_persist_dir": redact_diagnostic_value(
                "vector_persist_dir", getattr(vector_store, "persist_dir", None)
            ),
            "vector_in_memory": bool(getattr(vector_store, "in_memory", False)),
            "cache_backend": cache_config.get("backend"),
            "cache_enabled": bool(cache_config.get("enabled", True)),
            "cache_manager_kind": type(cache_manager).__name__
            if cache_manager is not None
            else None,
            "projection_store_enabled": bool(
                getattr(projection_store, "enabled", False)
            ),
        }

    def _latest_index_run_diagnostic_payload(self) -> dict[str, Any] | None:
        latest_run_record: IndexRunRecord | None = None
        index_run_store = self._index_run_store
        get_latest_run_record = getattr(index_run_store, "get_latest_run_record", None)
        if callable(get_latest_run_record):
            latest_run_record = cast(IndexRunRecord | None, get_latest_run_record())
        if latest_run_record is None:
            return None

        payload: dict[str, Any] = {
            "run_id": latest_run_record.run_id,
            "repo_name": latest_run_record.repo_name,
            "snapshot_id": latest_run_record.snapshot_id,
            "branch": latest_run_record.branch,
            "commit_id": latest_run_record.commit_id,
            "status": latest_run_record.status,
            "error_message": redact_diagnostic_value(
                "error_message", latest_run_record.error_message
            ),
            "warnings": redact_diagnostic_value(
                "warnings",
                diagnostic_json_string_list(latest_run_record.warnings_json),
            ),
            "created_at": latest_run_record.created_at,
            "started_at": latest_run_record.started_at,
            "completed_at": latest_run_record.completed_at,
        }
        snapshot_record = None
        snapshot_store = self._snapshot_store
        get_snapshot_record = getattr(snapshot_store, "get_snapshot_record", None)
        if callable(get_snapshot_record):
            snapshot_record = get_snapshot_record(latest_run_record.snapshot_id)
        if snapshot_record is None:
            return payload

        metadata = diagnostic_json_object(
            getattr(snapshot_record, "metadata_json", None)
        )
        payload["snapshot"] = {
            "snapshot_id": getattr(snapshot_record, "snapshot_id", None),
            "repo_name": getattr(snapshot_record, "repo_name", None),
            "branch": getattr(snapshot_record, "branch", None),
            "commit_id": getattr(snapshot_record, "commit_id", None),
            "tree_id": getattr(snapshot_record, "tree_id", None),
            "artifact_key": getattr(snapshot_record, "artifact_key", None),
            "created_at": getattr(snapshot_record, "created_at", None),
            "warnings": redact_diagnostic_value(
                "warnings",
                [
                    str(item)
                    for item in diagnostic_list(metadata.get("warnings"))
                    if item is not None
                ],
            ),
            "pipeline_layers": redact_diagnostic_value(
                "pipeline_layers",
                [
                    diagnostic_mapping(item)
                    for item in diagnostic_list(metadata.get("pipeline_layers"))
                    if isinstance(item, Mapping)
                ],
            ),
            "pipeline_metrics": redact_diagnostic_value(
                "pipeline_metrics",
                diagnostic_mapping(metadata.get("pipeline_metrics")),
            ),
        }
        return payload

    def build_diagnostic_bundle(self) -> dict[str, Any]:
        """Build a support-safe runtime diagnostic bundle."""
        config_payload = diagnostic_mapping(self._config)
        repo_info = diagnostic_mapping(getattr(self._state, "repo_info", {}))
        return {
            "schema_version": "fastcode.diagnostic_bundle.v1",
            "generated_at": utc_now(),
            "runtime": {
                "repo_loaded": bool(getattr(self._state, "repo_loaded", False)),
                "repo_indexed": bool(getattr(self._state, "repo_indexed", False)),
                "multi_repo_mode": bool(getattr(self._state, "multi_repo_mode", False)),
                "repo_info": {
                    "name": repo_info.get("name"),
                    "url": redact_diagnostic_value("url", repo_info.get("url")),
                    "file_count": repo_info.get("file_count"),
                    "total_size_mb": repo_info.get("total_size_mb"),
                    "branch": repo_info.get("branch"),
                    "commit": repo_info.get("commit"),
                },
                "loaded_repository_count": len(
                    getattr(self._state, "loaded_repositories", {}) or {}
                ),
                "loader_repo_path": redact_diagnostic_value(
                    "loader_repo_path",
                    getattr(self._loader, "repo_path", None),
                ),
            },
            "config_summary": diagnostic_config_summary(config_payload),
            "storage": self._diagnostic_storage_summary(),
            "dependencies": diagnostic_dependency_summary(),
            "latest_index_run": self._latest_index_run_diagnostic_payload(),
        }
