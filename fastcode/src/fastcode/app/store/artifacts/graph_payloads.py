"""Private payload helpers for graph artifact storage."""
# pyright: reportUnusedFunction=false

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from typing import Any, cast

from fastcode.ir.element import CodeElement, deserialize_code_element

_GRAPH_SHARD_STORAGE_VERSION = 1


def _empty_graph_payloads() -> dict[str, dict[str, list[Any]]]:
    return {
        "call": {"nodes": [], "edges": []},
        "dependency": {"nodes": [], "edges": []},
        "inheritance": {"nodes": [], "edges": []},
    }


def _empty_shard_payload() -> dict[str, Any]:
    return {
        "elements": [],
        "imports": [],
        "graphs": _empty_graph_payloads(),
    }


def _graph_shard_filename(path_key: str) -> str:
    digest = hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:20]
    return f"{digest}.pkl"


def _graph_shard_bytes(payload: dict[str, Any]) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _graph_shard_digest(shard_bytes: bytes) -> str:
    return hashlib.sha256(shard_bytes).hexdigest()


def _copy_or_link_file(source_path: str, target_path: str) -> None:
    if os.path.abspath(source_path) == os.path.abspath(target_path):
        return
    if os.path.exists(target_path):
        os.remove(target_path)
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _path_key_from_element(element: CodeElement, sequence_no: int) -> str:
    if element.relative_path:
        return str(element.relative_path)
    if element.file_path:
        return str(element.file_path)
    return f"__pathless__:{element.id or sequence_no}"


def _imports_path_key(file_path: str, fallback_index: int) -> str:
    if file_path:
        return str(file_path)
    return f"__pathless_imports__:{fallback_index}"


def _edge_sort_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    attrs = edge.get("attrs", {})
    try:
        attrs_json = json.dumps(attrs, sort_keys=True, default=repr)
    except TypeError:
        attrs_json = repr(attrs)
    return (
        str(edge.get("source") or ""),
        str(edge.get("target") or ""),
        attrs_json,
    )


def _sort_graph_shard_payload(payload: dict[str, Any]) -> None:
    elements = cast(list[dict[str, Any]], payload["elements"])
    imports = cast(list[dict[str, Any]], payload["imports"])
    graphs = cast(dict[str, dict[str, Any]], payload["graphs"])
    elements.sort(key=lambda row: int(row.get("sequence_no", 0)))
    imports.sort(key=lambda row: str(row.get("file_path") or ""))
    for graph_payload in graphs.values():
        graph_payload["nodes"] = sorted(
            {str(node_id) for node_id in graph_payload.get("nodes", [])}
        )
        edges = cast(list[dict[str, Any]], graph_payload["edges"])
        edges.sort(key=_edge_sort_key)


def _deserialize_elements(payload: dict[str, Any]) -> list[CodeElement]:
    element_payloads = cast(list[dict[str, Any]], payload["element_payloads"])
    return [
        deserialize_code_element(element_payload)
        for element_payload in element_payloads
    ]


def _node_counts(payload: dict[str, Any]) -> tuple[int, int, int]:
    graph_payloads = cast(dict[str, dict[str, list[Any]]], payload["graph_payloads"])
    return (
        len(graph_payloads["dependency"]["nodes"]),
        len(graph_payloads["inheritance"]["nodes"]),
        len(graph_payloads["call"]["nodes"]),
    )
