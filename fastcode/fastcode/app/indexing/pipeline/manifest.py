"""File manifest construction, persistence, and change detection.

Moved from main/fastcode.py (assembly_root) to use_flow (app/indexing)
because manifest operations are indexing workflow logic, not composition
root wiring.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, cast

from fastcode.ir.element import CodeElement

logger = logging.getLogger(__name__)


def build_file_manifest(
    elements: list[CodeElement],
    repo_root: str,
    *,
    repo_name: str = "",
) -> dict[str, Any]:
    """Build a file manifest mapping files to their mtime/size and element IDs."""
    manifest: dict[str, Any] = {
        "repo_name": repo_name,
        "created_at": datetime.now().isoformat(),
        "files": {},
    }

    for elem in elements:
        rel_path = elem.relative_path
        if rel_path not in manifest["files"]:
            abs_path = os.path.join(repo_root, rel_path)
            try:
                stat = os.stat(abs_path)
                manifest["files"][rel_path] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "element_ids": [],
                }
            except OSError:
                manifest["files"][rel_path] = {
                    "mtime": 0.0,
                    "size": 0,
                    "element_ids": [],
                }
        manifest["files"][rel_path]["element_ids"].append(elem.id)

    return manifest


def save_file_manifest(
    manifest: dict[str, Any],
    repo_name: str,
    persist_dir: str,
) -> None:
    """Save file manifest to disk as JSON."""
    manifest_path = os.path.join(persist_dir, f"{repo_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved file manifest: %s", manifest_path)


def load_file_manifest(
    repo_name: str,
    persist_dir: str,
) -> dict[str, Any] | None:
    """Load file manifest from disk. Returns None if missing."""
    manifest_path = os.path.join(persist_dir, f"{repo_name}_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load manifest for '%s': %s", repo_name, e)
        return None


def load_existing_metadata(
    repo_name: str,
    vector_store: Any,
    persist_dir: str,
) -> list[dict[str, Any]]:
    """Load existing vector store metadata for a repo directly from disk."""
    try:
        load_metadata_payload = getattr(vector_store, "load_metadata_payload", None)
        if callable(load_metadata_payload):
            data = load_metadata_payload(repo_name)
            metadata = data.get("metadata", []) if isinstance(data, dict) else []
            if isinstance(metadata, list):
                return cast(list[dict[str, Any]], metadata)
    except Exception as e:
        logger.warning("Failed to load metadata for '%s': %s", repo_name, e)
    meta_path = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        metadata = data.get("metadata", []) if isinstance(data, dict) else []
        return (
            cast(list[dict[str, Any]], metadata) if isinstance(metadata, list) else []
        )
    except Exception as e:
        logger.warning("Failed to load metadata for '%s': %s", repo_name, e)
        return []


def detect_file_changes(
    repo_name: str,
    current_files: list[dict[str, Any]],
    persist_dir: str,
) -> dict[str, Any] | None:
    """Compare current files against saved manifest to detect changes.

    Returns dict with added/modified/deleted/unchanged lists, or None
    if no manifest exists.
    """
    manifest = load_file_manifest(repo_name, persist_dir)
    if manifest is None:
        return None

    manifest_files = manifest.get("files", {})

    # Build lookup of current files with stat info
    current_lookup: dict[str, dict[str, Any]] = {}
    for file_info in current_files:
        rel_path = file_info["relative_path"]
        abs_path = file_info["path"]
        try:
            stat = os.stat(abs_path)
            current_lookup[rel_path] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "file_info": file_info,
            }
        except OSError:
            continue

    added, modified, deleted, unchanged = [], [], [], []

    for rel_path, info in current_lookup.items():
        if rel_path not in manifest_files:
            added.append(rel_path)
        else:
            saved = manifest_files[rel_path]
            if info["mtime"] != saved["mtime"] or info["size"] != saved["size"]:
                modified.append(rel_path)
            else:
                unchanged.append(rel_path)

    deleted.extend(
        rel_path for rel_path in manifest_files if rel_path not in current_lookup
    )

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "unchanged": unchanged,
        "manifest": manifest,
        "current_lookup": current_lookup,
    }


def collect_unchanged_elements(
    manifest: dict[str, Any],
    unchanged_files: list[str],
    existing_metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect element dicts and IDs for unchanged files from existing metadata."""
    unchanged_element_ids: set[str] = set()
    for rel_path in unchanged_files:
        file_entry: dict[str, Any] = manifest.get("files", {}).get(rel_path, {})
        for elem_id in file_entry.get("element_ids", []):
            unchanged_element_ids.add(elem_id)

    unchanged_elements = [
        meta for meta in existing_metadata if meta.get("id") in unchanged_element_ids
    ]

    return unchanged_elements, list(unchanged_element_ids)
