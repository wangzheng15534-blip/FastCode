"""Filesystem/tool/runtime operations for semantic helper execution.

This is an effect_tool module: it owns concrete external-mechanism adapters
(subprocess invocation, Go-binary compilation, filesystem JSON cache, file
identity hashing). It was relocated out of the meaning_core ``semantic``
package so the semantic layer stays pure (no io/serde_json/subprocess).

Dependency inversion: this module must NOT import meaning_core. It defines its
own local value types (``SemanticHelperSpec``, ``SemanticHelperCacheEntry``,
``HelperRunResult``) and operates on the spec via plain attribute access. The
pure ``SemanticHelperOps`` Protocol and the ``SemanticHelperInvocation``/
``ToolDiagnostic`` mapping live in meaning_core
(``fastcode.semantic.resolvers.engine.helper_contract``); this class satisfies
that Protocol structurally without importing it. meaning_core wraps the raw
``HelperRunResult`` into ``SemanticHelperInvocation`` + ``ToolDiagnostic``.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import sys
import tempfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from fastcode.utils.filesystem import (
    compute_file_sha256,
    file_content_identity,
    file_stat_identity,
    resolve_absolute_root,
)
from fastcode.utils.json import load_json_object, write_json_object_atomic

logger = logging.getLogger(__name__)


def _empty_payload() -> dict[str, Any]:
    return {}


def _empty_stats() -> dict[str, Any]:
    return {}


def _empty_warnings() -> list[str]:
    return []


def _empty_raw_diagnostics() -> list[dict[str, str]]:
    return []


def _normalize_path(path: str) -> str:
    # Local pure copy of fastcode.semantic.resolvers.engine.resolver_support's
    # _normalize_path. Duplicated here to avoid an effect_tool -> meaning_core
    # import edge; both copies are deterministic string ops.
    normalized = path.replace("\\", "/")
    normalized = normalized.removeprefix("./")
    return posixpath.normpath(normalized)


def validate_helper_paths(
    paths: list[str], repo_root: str
) -> tuple[list[str], list[str]]:
    """Validate helper file paths are regular files within repo root.

    Returns (safe_paths, rejected_paths). Rejects symlinks, missing files,
    and files outside the repo root. This performs filesystem I/O and so
    belongs in effect_tool.
    """
    repo = Path(repo_root).resolve()
    safe: list[str] = []
    rejected: list[str] = []
    for p in paths:
        original = Path(p)
        resolved = original.resolve()
        if original.is_symlink():
            rejected.append(p)
            continue
        if not resolved.is_file():
            rejected.append(p)
            continue
        try:
            resolved.relative_to(repo)
        except ValueError:
            rejected.append(p)
            continue
        safe.append(p)
    return safe, rejected


@dataclass(frozen=True)
class SemanticHelperSpec:
    """Execution/cache identity for one helper-backed resolver.

    Local effect_tool copy structurally compatible with the meaning_core
    ``fastcode.semantic.resolvers.engine.helper_contract.SemanticHelperSpec``
    (same fields). meaning_core constructs the spec and passes it in; this
    module reads attributes off it without importing the meaning_core type.
    """

    cache_version: str
    language: str
    source_name: str
    frontend_kind: str
    extractor_name: str
    required_tools: tuple[str, ...]
    helper_filename: str
    helper_runtime: str
    helper_timeout_seconds: int
    file_extensions: tuple[str, ...]


@dataclass(frozen=True)
class SemanticHelperCacheEntry:
    """Concrete cache entry for a helper invocation identity (effect_tool)."""

    key: str
    path: str
    identity: dict[str, Any]


@dataclass(frozen=True)
class HelperRunResult:
    """Raw effect_tool result of invoking a helper command.

    ``raw_diagnostics`` are plain dicts with the keys
    ``{language, tool, code, message}``; the meaning_core resolver wraps each
    into a ``ToolDiagnostic``.
    """

    payload: dict[str, Any] = field(default_factory=_empty_payload)
    stats: dict[str, Any] = field(default_factory=_empty_stats)
    warnings: list[str] = field(default_factory=_empty_warnings)
    raw_diagnostics: list[dict[str, str]] = field(default_factory=_empty_raw_diagnostics)


class SemanticHelperOperations:
    """Filesystem/tool/runtime operations for semantic helper execution."""

    def __init__(self, helper_runtime: Any | None = None) -> None:
        self._helper_runtime = helper_runtime

    def set_runtime(self, helper_runtime: Any | None) -> None:
        self._helper_runtime = helper_runtime

    @staticmethod
    def resolve_repo_root(repo_root: str | None) -> str:
        return resolve_absolute_root(repo_root)

    def find_tool(self, tool: str) -> str | None:
        find_executable = getattr(self._helper_runtime, "find_executable", None)
        if not callable(find_executable):
            return None
        return cast(str | None, find_executable(tool))

    def has_tools(self, spec: SemanticHelperSpec) -> bool:
        return all(self.find_tool(tool) is not None for tool in spec.required_tools)

    def missing_tool_diagnostics(
        self, spec: SemanticHelperSpec
    ) -> list[dict[str, str]]:
        return [
            {
                "language": spec.language,
                "tool": tool,
                "code": "required_tool_missing",
                "message": (
                    f"'{tool}' not found in PATH; {spec.language} resolution is "
                    "structural-only"
                ),
            }
            for tool in spec.required_tools
            if self.find_tool(tool) is None
        ]

    def target_files(
        self,
        target_paths: set[str],
        *,
        repo_root: str,
        spec: SemanticHelperSpec,
    ) -> list[str]:
        raw: list[str] = []
        for path in sorted(target_paths):
            normalized = path if os.path.isabs(path) else os.path.join(repo_root, path)
            if spec.file_extensions and not normalized.endswith(spec.file_extensions):
                continue
            raw.append(os.path.abspath(normalized))
        safe, rejected = validate_helper_paths(raw, repo_root)
        if rejected:
            logger.warning(
                "Rejected %d helper file paths (symlinks, missing, or outside repo)",
                len(rejected),
            )
        return safe

    @staticmethod
    def helper_path(spec: SemanticHelperSpec) -> Path:
        # This module lives at fastcode/fastcode/infrastructure/execution/; the
        # helper scripts live under the semantic package two parents up
        # (fastcode/fastcode/semantic/resolvers/helpers/). Resolving from
        # __file__ keeps the path stable regardless of the process CWD.
        return (
            Path(__file__).resolve().parents[2]
            / "semantic"
            / "resolvers"
            / "helpers"
            / spec.helper_filename
        )

    def helper_command(
        self, spec: SemanticHelperSpec, helper_files: list[str]
    ) -> list[str]:
        helper_path = str(self.helper_path(spec))
        if spec.helper_runtime == "node":
            node_path = self.find_tool("node") or "node"
            return [node_path, helper_path, *helper_files]
        if spec.helper_runtime == "go":
            go_path = self.find_tool("go") or "go"
            compiled = self.compiled_go_helper_command(spec, go_path, Path(helper_path))
            if compiled:
                return [compiled, *helper_files]
            return [go_path, "run", helper_path, "--", *helper_files]
        return [sys.executable, helper_path, *helper_files]

    def compiled_go_helper_command(
        self, spec: SemanticHelperSpec, go_path: str, helper_path: Path
    ) -> str | None:
        if not helper_path.exists():
            return None
        binary_path = self.go_helper_binary_path(helper_path)
        try:
            if binary_path.exists():
                return str(binary_path)
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = binary_path.with_suffix(binary_path.suffix + ".tmp")
            result = self.run_command(
                spec,
                [go_path, "build", "-o", str(tmp_path), str(helper_path)],
                cwd=str(helper_path.parent),
            )
            if result.returncode != 0:
                return None
            os.replace(tmp_path, binary_path)
            binary_path.chmod(0o755)
            return str(binary_path)
        except (OSError, RuntimeError, TimeoutError):
            return None

    @staticmethod
    def go_helper_binary_path(helper_path: Path) -> Path:
        digest = compute_file_sha256(helper_path)
        if digest is None:
            digest = "unreadable"
        digest = digest[:16]
        suffix = ".exe" if os.name == "nt" else ""
        return (
            Path(tempfile.gettempdir())
            / "fastcode-helper-cache"
            / f"{helper_path.stem}-{digest}{suffix}"
        )

    def cache_entry(
        self,
        spec: SemanticHelperSpec,
        helper_files: list[str],
        *,
        repo_root: str,
    ) -> SemanticHelperCacheEntry:
        identity = {
            "cache_version": spec.cache_version,
            "language": spec.language,
            "source": spec.source_name,
            "frontend_kind": spec.frontend_kind,
            "extractor": spec.extractor_name,
            "repo_root": os.path.realpath(repo_root),
            "helper": self.helper_identity(spec),
            "tools": self.required_tool_identities(spec),
            "targets": [
                fingerprint
                for helper_file in sorted(helper_files)
                if (
                    fingerprint := self.target_file_fingerprint(
                        helper_file,
                        repo_root=repo_root,
                    )
                )
                is not None
            ],
        }
        key = sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_dir = self.helper_cache_dir(repo_root)
        return SemanticHelperCacheEntry(
            key=key,
            path=str(cache_dir / f"{key}.json"),
            identity=identity,
        )

    @staticmethod
    def helper_cache_dir(repo_root: str) -> Path:
        cache_dir = Path(repo_root) / ".fastcode" / "semantic_helper_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def helper_identity(self, spec: SemanticHelperSpec) -> dict[str, Any]:
        helper_path = self.helper_path(spec)
        payload: dict[str, Any] = {
            "filename": spec.helper_filename,
            "runtime": spec.helper_runtime,
            "path": str(helper_path),
            "exists": helper_path.exists(),
        }
        if helper_path.exists():
            identity = file_content_identity(helper_path)
            if identity is None:
                payload["unreadable"] = True
            else:
                payload.update(identity)
        return payload

    def required_tool_identities(
        self, spec: SemanticHelperSpec
    ) -> list[dict[str, Any]]:
        identities: list[dict[str, Any]] = []
        for tool in sorted(spec.required_tools):
            tool_path = self.find_tool(tool)
            payload: dict[str, Any] = {
                "tool": tool,
                "available": tool_path is not None,
                "path": tool_path,
            }
            if tool_path:
                identity = file_stat_identity(tool_path)
                if identity is None:
                    payload["unreadable"] = True
                else:
                    payload.update(identity)
            identities.append(payload)
        return identities

    @staticmethod
    def target_file_fingerprint(
        helper_file: str,
        *,
        repo_root: str,
    ) -> dict[str, Any] | None:
        identity = file_content_identity(helper_file)
        if identity is None:
            return None
        rel_path = _normalize_path(os.path.relpath(helper_file, repo_root))
        return {"path": rel_path, **identity}

    @staticmethod
    def load_cache(entry: SemanticHelperCacheEntry) -> dict[str, Any] | None:
        if not os.path.exists(entry.path):
            return None
        payload = load_json_object(entry.path)
        if payload is None:
            return None
        if payload.get("key") != entry.key or payload.get("identity") != entry.identity:
            return None
        helper_payload = payload.get("helper_payload")
        return (
            cast(dict[str, Any], helper_payload)
            if isinstance(helper_payload, dict)
            else None
        )

    @staticmethod
    def save_cache(
        entry: SemanticHelperCacheEntry,
        helper_payload: dict[str, Any],
    ) -> None:
        payload = {
            "key": entry.key,
            "identity": entry.identity,
            "helper_payload": helper_payload,
        }
        write_json_object_atomic(entry.path, payload)

    def run_helper(
        self,
        spec: SemanticHelperSpec,
        helper_files: list[str],
        *,
        repo_root: str,
    ) -> HelperRunResult:
        command = self.helper_command(spec, helper_files)
        stats: dict[str, Any] = {"helper_command": command}
        try:
            result = self.run_command(
                spec,
                command,
                cwd=repo_root,
            )
        except (TimeoutError, FileNotFoundError, OSError, RuntimeError) as exc:
            return HelperRunResult(
                stats=stats,
                warnings=[f"{spec.source_name}_helper_failed: {exc}"],
                raw_diagnostics=[
                    {
                        "language": spec.language,
                        "tool": spec.helper_filename,
                        "code": "tool_invocation_failed",
                        "message": str(exc),
                    }
                ],
            )

        stats["helper_exit_code"] = result.returncode
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            warnings = [f"{spec.source_name}_helper_error: {stderr}"] if stderr else []
            return HelperRunResult(
                stats=stats,
                warnings=warnings,
                raw_diagnostics=[
                    {
                        "language": spec.language,
                        "tool": spec.helper_filename,
                        "code": "helper_nonzero_exit",
                        "message": stderr
                        or f"helper exited with code {result.returncode}",
                    }
                ],
            )

        if not result.stdout.strip():
            return HelperRunResult(stats=stats)

        try:
            return HelperRunResult(
                payload=json.loads(result.stdout),
                stats=stats,
            )
        except json.JSONDecodeError as exc:
            return HelperRunResult(
                stats=stats,
                warnings=[f"{spec.source_name}_helper_invalid_json: {exc}"],
                raw_diagnostics=[
                    {
                        "language": spec.language,
                        "tool": spec.helper_filename,
                        "code": "invalid_helper_json",
                        "message": str(exc),
                    }
                ],
            )

    def run_command(
        self,
        spec: SemanticHelperSpec,
        command: list[str],
        *,
        cwd: str,
    ) -> Any:
        if self._helper_runtime is None:
            msg = "semantic helper runtime is not configured"
            raise RuntimeError(msg)
        return self._helper_runtime.run(
            command,
            cwd=cwd,
            timeout=spec.helper_timeout_seconds,
        )
