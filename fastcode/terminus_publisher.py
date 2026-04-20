"""
TerminusDB lineage publisher.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict


class TerminusPublisher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        terminus_cfg = config.get("terminus", {})
        self.endpoint = terminus_cfg.get("endpoint")
        self.api_key = terminus_cfg.get("api_key")
        self.timeout = int(terminus_cfg.get("timeout_seconds", 15))

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def publish_snapshot_lineage(
        self,
        snapshot: Dict[str, Any],
        manifest: Dict[str, Any],
        git_meta: Dict[str, Any],
        previous_snapshot_symbols: Dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        if not self.endpoint:
            raise RuntimeError("Terminus endpoint is not configured")

        payload = self.build_lineage_payload(
            snapshot=snapshot,
            manifest=manifest,
            git_meta=git_meta,
            previous_snapshot_symbols=previous_snapshot_symbols,
        )
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"X-Idempotency-Key": idempotency_key} if idempotency_key else {}),
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 300:
                    raise RuntimeError(f"Terminus publish failed: HTTP {resp.status}")
                self.logger.info("Published snapshot lineage to Terminus")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Terminus publish error: {e}") from e

    def build_lineage_payload(
        self,
        snapshot: Dict[str, Any],
        manifest: Dict[str, Any],
        git_meta: Dict[str, Any],
        previous_snapshot_symbols: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        repo_name = snapshot.get("repo_name") or git_meta.get("repo_name")
        snapshot_id = snapshot.get("snapshot_id")
        if not snapshot_id:
            raise ValueError("snapshot_id is required for lineage payload")
        if not repo_name:
            raise ValueError("repo_name is required for lineage payload")
        branch = snapshot.get("branch") or git_meta.get("branch")
        commit_id = snapshot.get("commit_id") or git_meta.get("commit_id")

        repo_node_id = f"repo:{repo_name}"
        branch_node_id = f"branch:{repo_name}:{branch}" if branch else None
        commit_node_id = f"commit:{repo_name}:{commit_id}" if commit_id else None
        snapshot_node_id = f"snapshot:{snapshot_id}"
        manifest_id = manifest.get("manifest_id")
        manifest_node_id = f"manifest:{manifest_id}" if manifest_id else None
        run_id = manifest.get("index_run_id")
        run_node_id = f"index_run:{run_id}" if run_id else None

        nodes = [
            {"id": repo_node_id, "type": "Repository", "props": {"repo_name": repo_name}},
            {
                "id": snapshot_node_id,
                "type": "Snapshot",
                "props": {
                    "snapshot_id": snapshot_id,
                    "repo_name": repo_name,
                    "branch": branch,
                    "commit_id": commit_id,
                },
            },
        ]
        if branch_node_id:
            nodes.append({"id": branch_node_id, "type": "Branch", "props": {"repo_name": repo_name, "name": branch}})
        if commit_node_id:
            nodes.append({"id": commit_node_id, "type": "Commit", "props": {"repo_name": repo_name, "commit_id": commit_id}})
        if run_node_id:
            nodes.append({"id": run_node_id, "type": "IndexRun", "props": {"run_id": run_id}})
        if manifest_node_id:
            nodes.append(
                {
                    "id": manifest_node_id,
                    "type": "Manifest",
                    "props": {
                        "manifest_id": manifest_id,
                        "ref_name": manifest.get("ref_name"),
                        "status": manifest.get("status"),
                        "published_at": manifest.get("published_at"),
                    },
                }
            )

        edges = [{"type": "repo_snapshot", "src": repo_node_id, "dst": snapshot_node_id}]
        if branch_node_id:
            edges.append({"type": "branch_head", "src": branch_node_id, "dst": snapshot_node_id})
        if commit_node_id:
            edges.append({"type": "commit_snapshot", "src": commit_node_id, "dst": snapshot_node_id})
            parent_ids = git_meta.get("parent_commit_ids") or []
            if not parent_ids and git_meta.get("parent_commit_id"):
                parent_ids = [git_meta.get("parent_commit_id")]
            for parent_id in parent_ids:
                if not parent_id:
                    continue
                parent_node_id = f"commit:{repo_name}:{parent_id}"
                nodes.append(
                    {
                        "id": parent_node_id,
                        "type": "Commit",
                        "props": {"repo_name": repo_name, "commit_id": parent_id},
                    }
                )
                edges.append({"type": "commit_parent", "src": commit_node_id, "dst": parent_node_id})
        if run_node_id:
            edges.append({"type": "index_run_for_snapshot", "src": run_node_id, "dst": snapshot_node_id})
        if manifest_node_id:
            edges.append({"type": "snapshot_manifest", "src": snapshot_node_id, "dst": manifest_node_id})
            prev = manifest.get("previous_manifest_id")
            if prev:
                edges.append({"type": "manifest_supersedes", "src": manifest_node_id, "dst": f"manifest:{prev}"})

        documents = snapshot.get("documents") or []
        symbols = snapshot.get("symbols") or []
        for doc in documents:
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue
            node_id = f"doc:{snapshot_id}:{doc_id}"
            nodes.append(
                {
                    "id": node_id,
                    "type": "DocumentVersion",
                    "props": {
                        "doc_id": doc_id,
                        "path": doc.get("path"),
                        "language": doc.get("language"),
                    },
                }
            )
            edges.append({"type": "snapshot_contains_document", "src": snapshot_node_id, "dst": node_id})

        for sym in symbols:
            symbol_id = sym.get("symbol_id")
            if not symbol_id:
                continue
            node_id = f"symbol:{snapshot_id}:{symbol_id}"
            nodes.append(
                {
                    "id": node_id,
                    "type": "SymbolVersion",
                    "props": {
                        "symbol_id": symbol_id,
                        "display_name": sym.get("display_name"),
                        "path": sym.get("path"),
                        "kind": sym.get("kind"),
                    },
                }
            )
            doc_path = sym.get("path")
            if doc_path:
                matching = [d for d in documents if d.get("path") == doc_path]
                if matching:
                    edges.append(
                        {
                            "type": "document_defines_symbol",
                            "src": f"doc:{snapshot_id}:{matching[0].get('doc_id')}",
                            "dst": node_id,
                        }
                    )
            ext_symbol = sym.get("external_symbol_id")
            if ext_symbol and previous_snapshot_symbols and ext_symbol in previous_snapshot_symbols:
                edges.append(
                    {
                        "type": "symbol_version_from",
                        "src": node_id,
                        "dst": previous_snapshot_symbols[ext_symbol],
                    }
                )

        return {
            "version": "v1",
            "snapshot_id": snapshot_id,
            "manifest_id": manifest_id,
            "nodes": nodes,
            "edges": edges,
            "git_meta": git_meta,
        }
