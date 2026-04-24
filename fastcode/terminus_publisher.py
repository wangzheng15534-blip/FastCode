"""
TerminusDB lineage publisher.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .semantic_ir import _resolution_to_confidence


class TerminusPublisher:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        terminus_cfg = config.get("terminus", {})
        self.endpoint = terminus_cfg.get("endpoint")
        self.api_key = terminus_cfg.get("api_key")
        self.timeout = int(terminus_cfg.get("timeout_seconds", 15))

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def _deterministic_event_id(self, snapshot_id: str, payload: str) -> str:
        """Generate a deterministic event ID from snapshot_id + payload hash."""
        h = hashlib.sha256(f"{snapshot_id}:{payload}".encode()).hexdigest()[:32]
        return f"outbox:{snapshot_id}:{h}"

    def enqueue_publish(
        self,
        snapshot_id: str,
        payload: dict[str, Any],
        snapshot_store: Any,
        idempotency_key: str | None = None,
    ) -> str | None:
        """Enqueue a lineage publish event into the outbox instead of doing direct HTTP POST.

        Returns the event_id if enqueued, None if outbox is not available (non-postgres).
        The event_id is deterministic from snapshot_id + payload for idempotency.
        """
        payload_str = json.dumps(payload, sort_keys=True)
        event_id = idempotency_key or self._deterministic_event_id(
            snapshot_id, payload_str
        )
        inserted = snapshot_store.enqueue_outbox_event(
            event_id=event_id,
            event_type="lineage_publish",
            payload=payload_str,
            snapshot_id=snapshot_id,
        )
        if inserted:
            self.logger.info(
                "Enqueued publish event %s for snapshot %s", event_id, snapshot_id
            )
            return event_id
        self.logger.info("Publish event %s already exists, skipping", event_id)
        return event_id

    def flush_outbox(self, snapshot_store: Any, limit: int = 10) -> dict[str, int]:
        """Flush pending outbox events by attempting HTTP POST for each.

        Returns {"processed": int, "succeeded": int, "failed": int}.
        """
        result = {"processed": 0, "succeeded": 0, "failed": 0}
        events = snapshot_store.claim_outbox_event(limit=limit)
        if not events:
            return result
        for event in events:
            event_id = event["event_id"]
            payload_str = event["payload"]
            try:
                payload = json.loads(payload_str)
            except (json.JSONDecodeError, ValueError):
                self.logger.error(
                    "Outbox event %s has malformed payload, marking failed", event_id
                )
                snapshot_store.mark_outbox_event_failed(
                    event_id, "malformed payload JSON"
                )
                result["processed"] += 1
                result["failed"] += 1
                continue
            try:
                self._do_post(payload)
                snapshot_store.mark_outbox_event_done(event_id)
                self.logger.info("Outbox event %s published successfully", event_id)
                result["succeeded"] += 1
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(
                    "Outbox event %s publish failed: %s", event_id, error_msg
                )
                snapshot_store.mark_outbox_event_failed(event_id, error_msg)
                result["failed"] += 1
            result["processed"] += 1
        return result

    def get_pending_count(self, snapshot_store: Any) -> int:
        """Return count of pending + retryable failed outbox events."""
        return snapshot_store.get_outbox_pending_count()

    def publish_snapshot_lineage(
        self,
        snapshot: dict[str, Any],
        manifest: dict[str, Any],
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None = None,
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
        self._do_post(payload, idempotency_key=idempotency_key)

    def _do_post(
        self,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> None:
        """Execute the HTTP POST to TerminusDB."""
        if not self.endpoint:
            raise RuntimeError("Terminus endpoint is not configured")

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
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
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                if resp.status >= 300:  # noqa: PLR2004
                    raise RuntimeError(f"Terminus publish failed: HTTP {resp.status}")
                self.logger.info("Published snapshot lineage to Terminus")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Terminus publish error: {e}") from e

    def build_code_graph_payload(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Build TerminusDB payload for code graph (symbol nodes + relation edges).

        Reads the ``units`` and ``relations`` fields from the snapshot dict
        (``ir.v2`` schema produced by ``IRSnapshot.to_dict()``).  Symbol nodes
        are created for every non-file, non-doc unit.  Relation edges carry the
        full resolution metadata so downstream consumers can apply confidence
        bands.
        """
        snapshot_id = snapshot.get("snapshot_id")
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for unit in snapshot.get("units") or []:
            kind = unit.get("kind", "")
            if kind in ("file", "doc"):
                continue
            unit_id = unit.get("unit_id")
            if not unit_id:
                continue
            node_id = f"sym:{snapshot_id}:{unit_id}"
            source_set = unit.get("source_set") or []
            nodes.append(
                {
                    "id": node_id,
                    "type": "Symbol",
                    "props": {
                        "unit_id": unit_id,
                        "display_name": unit.get("display_name"),
                        "kind": kind,
                        "path": unit.get("path"),
                        "language": unit.get("language"),
                        "start_line": unit.get("start_line"),
                        "end_line": unit.get("end_line"),
                        "qualified_name": unit.get("qualified_name"),
                        "scip_symbol": unit.get("primary_anchor_symbol_id"),
                        "source_set": source_set
                        if isinstance(source_set, list)
                        else list(source_set),
                    },
                }
            )

        for rel in snapshot.get("relations") or []:
            rel_id = rel.get("relation_id")
            if not rel_id:
                continue
            src_id = rel.get("src_unit_id")
            dst_id = rel.get("dst_unit_id")
            if not src_id or not dst_id:
                continue
            support_sources = rel.get("support_sources") or []
            edges.append(
                {
                    "id": f"rel:{snapshot_id}:{rel_id}",
                    "type": rel.get("relation_type", ""),
                    "src": f"sym:{snapshot_id}:{src_id}",
                    "dst": f"sym:{snapshot_id}:{dst_id}",
                    "confidence": _resolution_to_confidence(
                        rel.get("resolution_state", "")
                    ),
                    "resolution_state": rel.get("resolution_state", ""),
                    "source_set": support_sources
                    if isinstance(support_sources, list)
                    else list(support_sources),
                }
            )

        return {"nodes": nodes, "edges": edges}

    def load_graph_nodes(self, snapshot_id: str) -> list[dict[str, Any]]:
        """Query TerminusDB for all symbol nodes in a snapshot.

        Returns an empty list if TerminusDB is not configured or the query
        endpoint is unavailable.
        """
        if not self.endpoint:
            return []
        raise NotImplementedError(
            "load_graph_nodes requires a TerminusDB query endpoint; "
            "publish-only mode is currently supported"
        )

    def load_graph_edges(
        self, snapshot_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Query TerminusDB for edges in a snapshot, optionally filtered by type.

        Returns an empty list if TerminusDB is not configured.
        """
        if not self.endpoint:
            return []
        raise NotImplementedError(
            "load_graph_edges requires a TerminusDB query endpoint; "
            "publish-only mode is currently supported"
        )

    def build_lineage_payload(  # noqa: PLR0912, PLR0915
        self,
        snapshot: dict[str, Any],
        manifest: dict[str, Any],
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None = None,
    ) -> dict[str, Any]:
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
            {
                "id": repo_node_id,
                "type": "Repository",
                "props": {"repo_name": repo_name},
            },
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
            nodes.append(
                {
                    "id": branch_node_id,
                    "type": "Branch",
                    "props": {"repo_name": repo_name, "name": branch},
                }
            )
        if commit_node_id:
            nodes.append(
                {
                    "id": commit_node_id,
                    "type": "Commit",
                    "props": {"repo_name": repo_name, "commit_id": commit_id},
                }
            )
        if run_node_id:
            nodes.append(
                {"id": run_node_id, "type": "IndexRun", "props": {"run_id": run_id}}
            )
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

        edges = [
            {"type": "repo_snapshot", "src": repo_node_id, "dst": snapshot_node_id}
        ]
        if branch_node_id:
            edges.append(
                {"type": "branch_head", "src": branch_node_id, "dst": snapshot_node_id}
            )
        if commit_node_id:
            edges.append(
                {
                    "type": "commit_snapshot",
                    "src": commit_node_id,
                    "dst": snapshot_node_id,
                }
            )
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
                edges.append(
                    {
                        "type": "commit_parent",
                        "src": commit_node_id,
                        "dst": parent_node_id,
                    }
                )
        if run_node_id:
            edges.append(
                {
                    "type": "index_run_for_snapshot",
                    "src": run_node_id,
                    "dst": snapshot_node_id,
                }
            )
        if manifest_node_id:
            edges.append(
                {
                    "type": "snapshot_manifest",
                    "src": snapshot_node_id,
                    "dst": manifest_node_id,
                }
            )
            prev = manifest.get("previous_manifest_id")
            if prev:
                edges.append(
                    {
                        "type": "manifest_supersedes",
                        "src": manifest_node_id,
                        "dst": f"manifest:{prev}",
                    }
                )

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
            edges.append(
                {
                    "type": "snapshot_contains_document",
                    "src": snapshot_node_id,
                    "dst": node_id,
                }
            )

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
            if (
                ext_symbol
                and previous_snapshot_symbols
                and ext_symbol in previous_snapshot_symbols
            ):
                edges.append(
                    {
                        "type": "symbol_version_from",
                        "src": node_id,
                        "dst": previous_snapshot_symbols[ext_symbol],
                    }
                )

        # Merge code graph (symbol nodes + relation edges) when available
        code_graph = self.build_code_graph_payload(snapshot)
        if code_graph["nodes"] or code_graph["edges"]:
            nodes.extend(code_graph["nodes"])
            edges.extend(code_graph["edges"])

        return {
            "version": "v1",
            "snapshot_id": snapshot_id,
            "manifest_id": manifest_id,
            "nodes": nodes,
            "edges": edges,
            "git_meta": git_meta,
        }
