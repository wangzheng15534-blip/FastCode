"""Filesystem persistence for legacy CodeGraphBuilder artifacts."""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, cast

from fastcode.graph.build import CodeGraphBuilder
from fastcode.ir.element import CodeElement, serialize_code_element
from fastcode.utils.filesystem import ensure_dir
from fastcode.utils.path_utils import file_path_to_module_path

from .graph_payloads import (
    _GRAPH_SHARD_STORAGE_VERSION,
    _copy_or_link_file,
    _deserialize_elements,
    _edge_sort_key,
    _empty_graph_payloads,
    _empty_shard_payload,
    _graph_shard_bytes,
    _graph_shard_digest,
    _graph_shard_filename,
    _imports_path_key,
    _node_counts,
    _path_key_from_element,
    _sort_graph_shard_payload,
)


class GraphArtifactStore:
    """Persist and restore CodeGraphBuilder state at the storage boundary."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.persist_dir = str(
            config.get("vector_store", {}).get(
                "persist_directory", "./data/vector_store"
            )
        )
        ensure_dir(self.persist_dir)
        self.logger = logging.getLogger(__name__)

    def save(self, builder: CodeGraphBuilder, name: str = "index") -> bool:
        """Save graph data to disk."""
        try:
            self._write_graph_bundle(builder, name)
            self.logger.info("Saved graph data to %s", self.persist_dir)
            return True
        except Exception as exc:
            self.logger.error("Failed to save graph data: %s", exc)
            return False

    def save_incremental(
        self,
        builder: CodeGraphBuilder,
        name: str,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
    ) -> dict[str, int]:
        """Save graph data while reusing unchanged path shards."""
        try:
            stats = self._write_graph_bundle_incremental(
                builder,
                name,
                previous_name=previous_name,
                reusable_path_keys=reusable_path_keys,
            )
            self.logger.info(
                "Saved graph data to %s with %d reused shards",
                self.persist_dir,
                stats["graph_shards_reused"],
            )
            return stats
        except Exception as exc:
            self.logger.error("Failed to save graph data incrementally: %s", exc)
            return {"graph_shards_reused": 0, "graph_shards_written": 0}

    def publish_delta(
        self,
        builder: CodeGraphBuilder,
        name: str,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
    ) -> dict[str, int | str | None]:
        """Publish changed graph shards plus reusable previous path shards."""
        try:
            stats = self._write_graph_bundle_incremental(
                builder,
                name,
                previous_name=previous_name,
                reusable_path_keys=reusable_path_keys,
            )
            return {
                **stats,
                "fallback_reason": None,
            }
        except Exception as exc:
            self.logger.error("Failed to publish graph delta: %s", exc)
            return {
                "graph_shards_reused": 0,
                "graph_shards_written": 0,
                "graph_shards_removed": 0,
                "graph_rows_reused": 0,
                "graph_rows_written": 0,
                "fallback_reason": str(exc),
            }

    def load(self, builder: CodeGraphBuilder, name: str = "index") -> bool:
        """Load graph data from disk into a builder."""
        try:
            payload = self._load_graph_payload(name)
            if payload is None:
                self.logger.warning("Graph data not found: %s", self.legacy_path(name))
                return False
            self._restore_builder(builder, payload)
            dep_nodes, inheritance_nodes, call_nodes = _node_counts(payload)
            self.logger.info(
                "Loaded graph data with %d dependency nodes, %d inheritance nodes, "
                "%d call nodes",
                dep_nodes,
                inheritance_nodes,
                call_nodes,
            )
            return True
        except Exception as exc:
            self.logger.error("Failed to load graph data: %s", exc)
            return False

    def merge(self, builder: CodeGraphBuilder, name: str) -> bool:
        """Load and merge graph data from another repository artifact."""
        try:
            payload = self._load_graph_payload(name)
            if payload is None:
                self.logger.warning(
                    "Graph data not found for merging: %s", self.legacy_path(name)
                )
                return False
            elements = _deserialize_elements(payload)
            imports_by_file = cast(
                dict[str, list[dict[str, Any]]], payload["imports_by_file"]
            )
            builder.merge_graph_state(
                imports_by_file=imports_by_file,
                elements=elements,
                graph_payloads=cast(
                    dict[str, dict[str, list[Any]]], payload["graph_payloads"]
                ),
            )
            self.logger.info("Merged graph data from %s", name)
            return True
        except Exception as exc:
            self.logger.error("Failed to merge graph data from %s: %s", name, exc)
            return False

    def has_saved_graph(self, name: str) -> bool:
        return os.path.exists(self.manifest_path(name)) or os.path.exists(
            self.legacy_path(name)
        )

    def graph_artifact_paths(self, name: str) -> list[str]:
        paths = [
            self.legacy_path(name),
            self.manifest_path(name),
            self.shards_dir(name),
        ]
        return [path for path in paths if os.path.exists(path)]

    def legacy_path(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_graphs.pkl")

    def manifest_path(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_graph_manifest.json")

    def shards_dir(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}_graph_shards")

    def load_resolver_index(
        self,
        name: str,
        *,
        excluded_path_keys: set[str] | None = None,
        repo_root: str = "",
    ) -> dict[str, Any] | None:
        """Load compact module/export resolver maps from graph shards.

        This avoids rehydrating full ``CodeElement`` objects when an incremental
        graph rebuild only needs previous file/module/symbol lookup tables.
        """

        manifest = self._load_graph_manifest(name)
        if not self._graph_manifest_supports_reuse(manifest):
            return None
        excluded = excluded_path_keys or set()
        file_map: dict[str, str] = {}
        module_map: dict[str, str] = {}
        export_map: dict[str, dict[str, str]] = {}
        file_path_to_module: dict[str, str] = {}
        symbol_payloads: list[dict[str, Any]] = []
        for path_key, payload in self._iter_graph_shard_payloads(name, manifest):
            if path_key in excluded:
                continue
            element_rows = payload.get("elements", [])
            if not isinstance(element_rows, list):
                continue
            for row in element_rows:
                if not isinstance(row, dict):
                    continue
                element_payload = row.get("payload")
                if not isinstance(element_payload, dict):
                    continue
                element = cast(dict[str, Any], element_payload)
                element_type = str(element.get("type") or "")
                file_path = str(element.get("file_path") or "")
                element_id = str(element.get("id") or "")
                if not file_path or not element_id:
                    continue
                if element_type == "file":
                    abs_path = os.path.abspath(file_path)
                    module_path = file_path_to_module_path(file_path, repo_root)
                    file_map[abs_path] = element_id
                    if module_path:
                        module_map[module_path] = element_id
                        file_path_to_module[file_path] = module_path
                elif element_type in {"class", "function"}:
                    symbol_payloads.append(element)

        for element in symbol_payloads:
            file_path = str(element.get("file_path") or "")
            module_path = file_path_to_module.get(file_path)
            if not module_path:
                module_path = file_path_to_module_path(file_path, repo_root)
            if not module_path:
                continue
            symbol_name = str(element.get("name") or "")
            element_id = str(element.get("id") or "")
            if not symbol_name or not element_id:
                continue
            exports = export_map.setdefault(module_path, {})
            exports[symbol_name] = element_id
            metadata = element.get("metadata")
            class_name = (
                str(metadata.get("class_name") or "")
                if isinstance(metadata, dict)
                else ""
            )
            if class_name:
                exports[f"{class_name}.{symbol_name}"] = element_id

        return {
            "file_map": file_map,
            "module_map": module_map,
            "export_map": export_map,
            "stats": {
                "files_processed": len(file_map),
                "modules_created": len(module_map),
                "symbols_exported": sum(
                    len(symbols) for symbols in export_map.values()
                ),
                "errors": 0,
            },
        }

    def affected_path_keys_for_delta(
        self,
        name: str,
        *,
        changed_path_keys: set[str],
        removed_path_keys: set[str],
    ) -> set[str]:
        """Return path shards whose graph payload touches changed/removed paths."""

        manifest = self._load_graph_manifest(name)
        if not self._graph_manifest_supports_reuse(manifest):
            return set(changed_path_keys) | set(removed_path_keys)
        target_path_keys = set(changed_path_keys) | set(removed_path_keys)
        target_node_ids: set[str] = set()
        loaded_payloads = list(self._iter_graph_shard_payloads(name, manifest))
        for path_key, payload in loaded_payloads:
            if path_key not in target_path_keys:
                continue
            element_rows = payload.get("elements", [])
            if isinstance(element_rows, list):
                for row in element_rows:
                    if not isinstance(row, dict):
                        continue
                    element_payload = row.get("payload")
                    if isinstance(element_payload, dict) and element_payload.get("id"):
                        target_node_ids.add(str(element_payload["id"]))
            graphs = payload.get("graphs")
            if isinstance(graphs, dict):
                for graph_payload in graphs.values():
                    if not isinstance(graph_payload, dict):
                        continue
                    for node_id in graph_payload.get("nodes", []):
                        if node_id:
                            target_node_ids.add(str(node_id))

        affected = set(target_path_keys)
        if not target_node_ids:
            return affected
        for path_key, payload in loaded_payloads:
            graphs = payload.get("graphs")
            if not isinstance(graphs, dict):
                continue
            for graph_payload in graphs.values():
                if not isinstance(graph_payload, dict):
                    continue
                for edge in graph_payload.get("edges", []):
                    if not isinstance(edge, dict):
                        continue
                    source = edge.get("source")
                    target = edge.get("target")
                    source_touches_target = (
                        source is not None and str(source) in target_node_ids
                    )
                    target_touches_target = (
                        target is not None and str(target) in target_node_ids
                    )
                    if source_touches_target or target_touches_target:
                        affected.add(path_key)
                        break
                if path_key in affected:
                    break
        return affected

    def _path_key_for_node_id(
        self,
        node_id: str,
        *,
        element_by_id: dict[str, CodeElement],
    ) -> str:
        element = element_by_id.get(node_id)
        if element is not None:
            return _path_key_from_element(element, 0)
        return f"__pathless_node__:{node_id}"

    def _load_graph_manifest(self, name: str) -> dict[str, Any] | None:
        manifest_path = self.manifest_path(name)
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else None

    def _iter_graph_shard_payloads(
        self,
        name: str,
        manifest: dict[str, Any] | None,
    ) -> list[tuple[str, dict[str, Any]]]:
        if manifest is None:
            return []
        shard_dir = self.shards_dir(name)
        payloads: list[tuple[str, dict[str, Any]]] = []
        for entry in manifest.get("shards", []):
            if not isinstance(entry, dict) or not entry.get("shard_file"):
                continue
            path_key = str(entry.get("path_key") or "")
            if not path_key:
                continue
            shard_path = os.path.join(shard_dir, str(entry["shard_file"]))
            try:
                with open(shard_path, "rb") as handle:
                    payload = pickle.load(handle)
            except Exception as exc:
                self.logger.warning(
                    "Failed to read previous graph shard %s: %s",
                    shard_path,
                    exc,
                )
                continue
            if isinstance(payload, dict):
                payloads.append((path_key, cast(dict[str, Any], payload)))
        return payloads

    def _load_graph_sequences_by_path(
        self,
        name: str,
    ) -> tuple[dict[str, list[int]], int]:
        manifest = self._load_graph_manifest(name)
        if manifest is None:
            return {}, -1
        sequences_by_path: dict[str, list[int]] = {}
        max_sequence_no = -1
        for path_key, payload in self._iter_graph_shard_payloads(name, manifest):
            element_rows = payload.get("elements", [])
            if not isinstance(element_rows, list):
                continue
            sequence_nos = [
                int(row["sequence_no"])
                for row in element_rows
                if isinstance(row, dict) and isinstance(row.get("sequence_no"), int)
            ]
            if sequence_nos:
                max_sequence_no = max(max_sequence_no, *sequence_nos)
            sequences_by_path[path_key] = sequence_nos
        return sequences_by_path, max_sequence_no

    def _build_incremental_graph_sequence_plan(
        self,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
        grouped_counts: dict[str, int],
    ) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
        previous_manifest = self._load_graph_manifest(previous_name)
        if not self._graph_manifest_supports_reuse(previous_manifest):
            return self._fresh_incremental_graph_sequence_plan(grouped_counts), {}
        previous_entries = {
            str(entry.get("path_key")): cast(dict[str, Any], entry)
            for entry in (
                previous_manifest.get("shards", []) if previous_manifest else []
            )
            if isinstance(entry, dict) and entry.get("path_key")
        }
        previous_sequences, max_previous_sequence_no = (
            self._load_graph_sequences_by_path(previous_name)
        )
        sequences_by_path: dict[str, list[int]] = {}
        reusable_entries: dict[str, dict[str, Any]] = {}
        used_sequences: set[int] = set()
        for path_key in sorted(reusable_path_keys):
            if path_key.startswith("__pathless__"):
                continue
            previous_entry = previous_entries.get(path_key)
            previous_sequence_nos = previous_sequences.get(path_key)
            if not previous_entry or previous_sequence_nos is None:
                continue
            if any(
                sequence_no in used_sequences for sequence_no in previous_sequence_nos
            ):
                continue
            count = grouped_counts.get(path_key)
            if count is not None and len(previous_sequence_nos) != count:
                continue
            sequences_by_path[path_key] = list(previous_sequence_nos)
            reusable_entries[path_key] = previous_entry
            used_sequences.update(previous_sequence_nos)

        next_sequence_no = max(max_previous_sequence_no + 1, 0)
        for path_key, count in grouped_counts.items():
            if path_key in sequences_by_path:
                continue
            while next_sequence_no in used_sequences:
                next_sequence_no += 1
            assigned = list(range(next_sequence_no, next_sequence_no + count))
            sequences_by_path[path_key] = assigned
            used_sequences.update(assigned)
            next_sequence_no += count
        return sequences_by_path, reusable_entries

    @staticmethod
    def _graph_manifest_supports_reuse(manifest: dict[str, Any] | None) -> bool:
        if manifest is None:
            return False
        return int(manifest.get("version") or 0) == _GRAPH_SHARD_STORAGE_VERSION

    @staticmethod
    def _fresh_incremental_graph_sequence_plan(
        grouped_counts: dict[str, int],
    ) -> dict[str, list[int]]:
        sequences_by_path: dict[str, list[int]] = {}
        next_sequence_no = 0
        for path_key, count in grouped_counts.items():
            sequences_by_path[path_key] = list(
                range(next_sequence_no, next_sequence_no + count)
            )
            next_sequence_no += count
        return sequences_by_path

    def _grouped_payloads(
        self,
        builder: CodeGraphBuilder,
        *,
        sequences_by_path: dict[str, list[int]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        persisted_elements = builder.persisted_elements()
        element_by_id = {element.id: element for element in persisted_elements}
        file_path_to_path_key: dict[str, str] = {}
        grouped_payloads: dict[str, dict[str, Any]] = {}
        grouped_seen: dict[str, int] = {}

        for row_index, element in enumerate(persisted_elements):
            path_key = _path_key_from_element(element, row_index)
            if sequences_by_path is None:
                sequence_no = row_index
            else:
                sequence_nos = sequences_by_path.get(path_key)
                seen_count = grouped_seen.get(path_key, 0)
                if sequence_nos is None or len(sequence_nos) <= seen_count:
                    msg = f"Missing incremental graph sequence for path: {path_key}"
                    raise RuntimeError(msg)
                grouped_seen[path_key] = seen_count + 1
                sequence_no = sequence_nos[seen_count]

            grouped = grouped_payloads.setdefault(path_key, _empty_shard_payload())
            grouped["elements"].append(
                {
                    "sequence_no": sequence_no,
                    "payload": serialize_code_element(element),
                }
            )
            if element.file_path:
                file_path_to_path_key[str(element.file_path)] = path_key

        for import_index, (file_path, imports) in enumerate(
            builder.imports_by_file.items()
        ):
            path_key = file_path_to_path_key.get(
                str(file_path), _imports_path_key(str(file_path), import_index)
            )
            grouped = grouped_payloads.setdefault(path_key, _empty_shard_payload())
            grouped["imports"].append(
                {"file_path": str(file_path), "imports": list(imports)}
            )

        for graph_name, graph_payload in builder.graph_payloads().items():
            for node_id in graph_payload.get("nodes", []):
                path_key = self._path_key_for_node_id(
                    str(node_id), element_by_id=element_by_id
                )
                grouped = grouped_payloads.setdefault(path_key, _empty_shard_payload())
                grouped["graphs"][graph_name]["nodes"].append(str(node_id))
            for edge in graph_payload.get("edges", []):
                if not isinstance(edge, dict):
                    continue
                source = edge.get("source")
                target = edge.get("target")
                if source is None or target is None:
                    continue
                attrs = edge.get("attrs")
                path_key = self._path_key_for_node_id(
                    str(source), element_by_id=element_by_id
                )
                grouped = grouped_payloads.setdefault(path_key, _empty_shard_payload())
                grouped["graphs"][graph_name]["edges"].append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "attrs": dict(attrs) if isinstance(attrs, dict) else {},
                    }
                )

        return grouped_payloads

    def _write_graph_bundle(self, builder: CodeGraphBuilder, name: str) -> None:
        shard_dir = self.shards_dir(name)
        ensure_dir(shard_dir)
        existing_manifest = self._load_graph_manifest(name)
        existing_by_path = {
            str(entry.get("path_key")): cast(dict[str, Any], entry)
            for entry in (
                existing_manifest.get("shards", []) if existing_manifest else []
            )
            if isinstance(entry, dict) and entry.get("path_key")
        }
        grouped_payloads = self._grouped_payloads(builder)
        manifest: dict[str, Any] = {
            "version": _GRAPH_SHARD_STORAGE_VERSION,
            "shards": [],
        }
        active_files: set[str] = set()
        for path_key, payload in grouped_payloads.items():
            _sort_graph_shard_payload(payload)
            shard_bytes = _graph_shard_bytes(payload)
            digest = _graph_shard_digest(shard_bytes)
            existing = existing_by_path.get(path_key)
            shard_file = (
                str(existing.get("shard_file"))
                if existing and existing.get("shard_file")
                else _graph_shard_filename(path_key)
            )
            shard_path = os.path.join(shard_dir, shard_file)
            active_files.add(shard_file)
            if not (
                existing
                and existing.get("digest") == digest
                and os.path.exists(shard_path)
            ):
                tmp_path = f"{shard_path}.tmp"
                with open(tmp_path, "wb") as handle:
                    handle.write(shard_bytes)
                os.replace(tmp_path, shard_path)
            manifest["shards"].append(
                {
                    "path_key": path_key,
                    "shard_file": shard_file,
                    "digest": digest,
                    "element_count": len(payload["elements"]),
                    "import_count": len(payload["imports"]),
                }
            )

        if existing_manifest is not None:
            for entry in existing_manifest.get("shards", []):
                if not isinstance(entry, dict):
                    continue
                shard_file = entry.get("shard_file")
                if not shard_file or shard_file in active_files:
                    continue
                stale_path = os.path.join(shard_dir, str(shard_file))
                if os.path.exists(stale_path):
                    os.remove(stale_path)

        self._write_manifest_and_cleanup_legacy(name, manifest)

    def _write_graph_bundle_incremental(
        self,
        builder: CodeGraphBuilder,
        name: str,
        *,
        previous_name: str,
        reusable_path_keys: set[str],
    ) -> dict[str, int]:
        shard_dir = self.shards_dir(name)
        ensure_dir(shard_dir)
        previous_shard_dir = self.shards_dir(previous_name)
        previous_manifest = self._load_graph_manifest(previous_name)
        previous_entries = {
            str(entry.get("path_key")): cast(dict[str, Any], entry)
            for entry in (
                previous_manifest.get("shards", []) if previous_manifest else []
            )
            if isinstance(entry, dict) and entry.get("path_key")
        }

        grouped_counts: dict[str, int] = {}
        for sequence_no, element in enumerate(builder.persisted_elements()):
            path_key = _path_key_from_element(element, sequence_no)
            grouped_counts[path_key] = grouped_counts.get(path_key, 0) + 1
        sequences_by_path, reusable_entries = (
            self._build_incremental_graph_sequence_plan(
                previous_name=previous_name,
                reusable_path_keys=reusable_path_keys,
                grouped_counts=grouped_counts,
            )
        )
        grouped_payloads = self._grouped_payloads(
            builder, sequences_by_path=sequences_by_path
        )

        manifest: dict[str, Any] = {
            "version": _GRAPH_SHARD_STORAGE_VERSION,
            "shards": [],
        }
        active_files: set[str] = set()
        active_path_keys: set[str] = set()
        reused = 0
        written = 0
        rows_reused = 0
        copied_reusable_path_keys: set[str] = set()

        def _copy_reusable_entry(
            path_key: str,
            reusable: dict[str, Any],
        ) -> bool:
            nonlocal reused, rows_reused
            shard_file = str(reusable.get("shard_file") or "")
            source_path = os.path.join(previous_shard_dir, shard_file)
            target_path = os.path.join(shard_dir, shard_file)
            if not shard_file or not os.path.exists(source_path):
                return False
            _copy_or_link_file(source_path, target_path)
            active_files.add(shard_file)
            active_path_keys.add(path_key)
            element_count = int(reusable.get("element_count") or 0)
            manifest["shards"].append(
                {
                    "path_key": path_key,
                    "shard_file": shard_file,
                    "digest": str(reusable.get("digest") or ""),
                    "element_count": element_count,
                    "import_count": int(reusable.get("import_count") or 0),
                }
            )
            reused += 1
            rows_reused += element_count
            copied_reusable_path_keys.add(path_key)
            return True

        for path_key, payload in grouped_payloads.items():
            reusable = reusable_entries.get(path_key)
            if reusable is not None and _copy_reusable_entry(path_key, reusable):
                continue

            _sort_graph_shard_payload(payload)
            shard_bytes = _graph_shard_bytes(payload)
            digest = _graph_shard_digest(shard_bytes)
            shard_file = _graph_shard_filename(path_key)
            shard_path = os.path.join(shard_dir, shard_file)
            active_files.add(shard_file)
            tmp_path = f"{shard_path}.tmp"
            with open(tmp_path, "wb") as handle:
                handle.write(shard_bytes)
            os.replace(tmp_path, shard_path)
            manifest["shards"].append(
                {
                    "path_key": path_key,
                    "shard_file": shard_file,
                    "digest": digest,
                    "element_count": len(payload["elements"]),
                    "import_count": len(payload["imports"]),
                }
            )
            active_path_keys.add(path_key)
            written += 1

        for path_key, reusable in reusable_entries.items():
            if path_key in copied_reusable_path_keys or path_key in grouped_payloads:
                continue
            _copy_reusable_entry(path_key, reusable)

        for entry_name in os.listdir(shard_dir):
            if not entry_name.endswith(".pkl") or entry_name in active_files:
                continue
            os.remove(os.path.join(shard_dir, entry_name))

        manifest["shards"].sort(key=lambda entry: str(entry.get("path_key") or ""))
        self._write_manifest_and_cleanup_legacy(name, manifest)
        removed = len(set(previous_entries) - active_path_keys)
        return {
            "graph_shards_reused": reused,
            "graph_shards_written": written,
            "graph_shards_removed": removed,
            "graph_rows_reused": rows_reused,
            "graph_rows_written": len(builder.persisted_elements()),
        }

    def _write_manifest_and_cleanup_legacy(
        self, name: str, manifest: dict[str, Any]
    ) -> None:
        manifest_path = self.manifest_path(name)
        tmp_manifest = f"{manifest_path}.tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_manifest, manifest_path)

        legacy_graph_path = self.legacy_path(name)
        if os.path.exists(legacy_graph_path):
            os.remove(legacy_graph_path)

    def _load_graph_payload(self, name: str) -> dict[str, Any] | None:
        manifest = self._load_graph_manifest(name)
        if manifest is not None:
            try:
                graph_payloads = _empty_graph_payloads()
                imports_by_file: dict[str, list[dict[str, Any]]] = {}
                ordered_elements: list[tuple[int, dict[str, Any]]] = []
                shard_dir = self.shards_dir(name)
                for entry in manifest.get("shards", []):
                    if not isinstance(entry, dict) or not entry.get("shard_file"):
                        continue
                    shard_path = os.path.join(shard_dir, str(entry["shard_file"]))
                    with open(shard_path, "rb") as handle:
                        payload = pickle.load(handle)
                    if not isinstance(payload, dict):
                        continue

                    element_rows = payload.get("elements", [])
                    if isinstance(element_rows, list):
                        for row in element_rows:
                            if not isinstance(row, dict):
                                continue
                            sequence_no = row.get("sequence_no")
                            element_payload = row.get("payload")
                            if not isinstance(sequence_no, int) or not isinstance(
                                element_payload, dict
                            ):
                                continue
                            ordered_elements.append(
                                (sequence_no, cast(dict[str, Any], element_payload))
                            )

                    import_rows = payload.get("imports", [])
                    if isinstance(import_rows, list):
                        for row in import_rows:
                            if not isinstance(row, dict):
                                continue
                            file_path = row.get("file_path")
                            imports = row.get("imports")
                            if not isinstance(file_path, str) or not isinstance(
                                imports, list
                            ):
                                continue
                            imports_by_file[file_path] = cast(
                                list[dict[str, Any]], imports
                            )

                    graphs_payload = payload.get("graphs", {})
                    if not isinstance(graphs_payload, dict):
                        continue
                    for graph_name, graph_payload in graphs_payload.items():
                        target_payload = graph_payloads.get(str(graph_name))
                        if target_payload is None or not isinstance(
                            graph_payload, dict
                        ):
                            continue
                        target_payload["nodes"].extend(
                            str(node_id)
                            for node_id in cast(
                                list[Any], graph_payload.get("nodes", [])
                            )
                        )
                        for edge in cast(list[Any], graph_payload.get("edges", [])):
                            if not isinstance(edge, dict):
                                continue
                            source = edge.get("source")
                            target = edge.get("target")
                            if source is None or target is None:
                                continue
                            attrs = edge.get("attrs")
                            target_payload["edges"].append(
                                {
                                    "source": str(source),
                                    "target": str(target),
                                    "attrs": (
                                        dict(cast(dict[str, Any], attrs))
                                        if isinstance(attrs, dict)
                                        else {}
                                    ),
                                }
                            )

                ordered_elements.sort(key=lambda item: item[0])
                for graph_payload in graph_payloads.values():
                    graph_payload["nodes"] = sorted(set(graph_payload["nodes"]))
                    graph_payload["edges"].sort(key=_edge_sort_key)
                return {
                    "graph_payloads": graph_payloads,
                    "imports_by_file": imports_by_file,
                    "element_payloads": [payload for _, payload in ordered_elements],
                }
            except Exception as exc:
                self.logger.warning(
                    "Failed to load sharded graph data for %s: %s", name, exc
                )

        graph_path = self.legacy_path(name)
        if not os.path.exists(graph_path):
            return None
        try:
            with open(graph_path, "rb") as handle:
                data = pickle.load(handle)
            if not isinstance(data, dict):
                return None
            if "element_by_id" in data and isinstance(data["element_by_id"], dict):
                element_payloads = list(
                    cast(dict[str, dict[str, Any]], data["element_by_id"]).values()
                )
            elif "element_by_name" in data and isinstance(
                data["element_by_name"], dict
            ):
                self.logger.warning(
                    "Legacy cache detected: restoring from element_by_name "
                    "(some duplicate functions may be lost)."
                )
                element_payloads = list(
                    cast(dict[str, dict[str, Any]], data["element_by_name"]).values()
                )
            else:
                element_payloads = []
            return {
                "graph_payloads": {
                    "call": CodeGraphBuilder.graph_payload_from_object(
                        data.get("call_graph")
                    ),
                    "dependency": CodeGraphBuilder.graph_payload_from_object(
                        data.get("dependency_graph")
                    ),
                    "inheritance": CodeGraphBuilder.graph_payload_from_object(
                        data.get("inheritance_graph")
                    ),
                },
                "imports_by_file": data.get("imports_by_file", {}),
                "element_payloads": element_payloads,
            }
        except Exception as exc:
            self.logger.warning(
                "Failed to load legacy graph data for %s: %s", name, exc
            )
            return None

    def _restore_builder(
        self, builder: CodeGraphBuilder, payload: dict[str, Any]
    ) -> None:
        elements = _deserialize_elements(payload)
        imports_by_file = cast(
            dict[str, list[dict[str, Any]]], payload["imports_by_file"]
        )
        builder.restore_graph_state(
            imports_by_file=imports_by_file,
            elements=elements,
            graph_payloads=cast(
                dict[str, dict[str, list[Any]]], payload["graph_payloads"]
            ),
        )
