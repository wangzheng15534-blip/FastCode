"""
TerminusDB lineage publisher.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, cast

import fastcode.retrieval.graph.graph_build as _graph_build

from fastcode.ir.types import IRRelation, IRSnapshot
from fastcode.ports.publishing import EventSink, LineagePublisher
from fastcode.utils.hashing import deterministic_event_id


def _record_get(record: Any, field_name: str) -> Any:
    if isinstance(record, Mapping):
        return cast(Mapping[str, Any], record).get(field_name)
    return getattr(record, field_name, None)


class TerminusPublisher(LineagePublisher):
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
        return deterministic_event_id(snapshot_id, payload)

    def enqueue_publish(
        self,
        snapshot_id: str,
        payload: dict[str, Any],
        snapshot_store: EventSink,
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

    def _claim_outbox_events(
        self, snapshot_store: EventSink, limit: int
    ) -> Sequence[Any]:
        return snapshot_store.claim_outbox_event_records(limit=limit)

    def flush_outbox(
        self, snapshot_store: EventSink, limit: int = 10
    ) -> dict[str, int]:
        """Flush pending outbox events by attempting HTTP POST for each.

        Returns {"processed": int, "succeeded": int, "failed": int}.
        """
        result = {"processed": 0, "succeeded": 0, "failed": 0}
        events = self._claim_outbox_events(snapshot_store, limit)
        if not events:
            return result
        for event in events:
            event_id = str(_record_get(event, "event_id") or "")
            payload_str = str(_record_get(event, "payload") or "")
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

    def get_pending_count(self, snapshot_store: EventSink) -> int:
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

    def publish_snapshot_lineage_for_snapshot(
        self,
        snapshot: IRSnapshot,
        manifest: Any,
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        if not self.endpoint:
            raise RuntimeError("Terminus endpoint is not configured")

        payload = self.build_lineage_payload_for_snapshot(
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

    def build_code_graph_payload(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Build TerminusDB payload for code graph (symbol nodes + relation edges)."""
        return _graph_build.build_code_graph_payload(snapshot)

    def build_code_graph_payload_for_snapshot(
        self, snapshot: IRSnapshot
    ) -> dict[str, Any]:
        """Build code graph payload directly from typed IR without full dict expansion."""
        snapshot_id = snapshot.snapshot_id or ""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        known_unit_ids: set[str] = set()
        emitted_unit_ids: set[str] = set()

        for unit in snapshot.units:
            kind = unit.kind or ""
            unit_id = unit.unit_id or ""
            if unit_id:
                known_unit_ids.add(unit_id)
            if kind in ("file", "doc"):
                continue
            if not unit_id:
                continue
            emitted_unit_ids.add(unit_id)
            nodes.append(
                {
                    "id": f"sym:{snapshot_id}:{unit_id}",
                    "type": "Symbol",
                    "props": {
                        "unit_id": unit_id,
                        "display_name": unit.display_name,
                        "kind": kind,
                        "path": unit.path,
                        "language": unit.language,
                        "start_line": unit.start_line,
                        "end_line": unit.end_line,
                        "qualified_name": unit.qualified_name,
                        "scip_symbol": unit.primary_anchor_symbol_id,
                        "source_set": sorted(unit.source_set),
                    },
                }
            )

        for relation in snapshot.relations:
            edge = self._code_graph_edge_for_relation(
                snapshot_id=snapshot_id,
                relation=relation,
                known_unit_ids=known_unit_ids,
                emitted_unit_ids=emitted_unit_ids,
            )
            if edge is not None:
                edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    def _code_graph_edge_for_relation(
        self,
        *,
        snapshot_id: str,
        relation: IRRelation,
        known_unit_ids: set[str],
        emitted_unit_ids: set[str],
    ) -> dict[str, Any] | None:
        rel_id = relation.relation_id or ""
        if not rel_id:
            return None
        src_id = relation.src_unit_id or ""
        dst_id = relation.dst_unit_id or ""
        if not src_id or not dst_id:
            return None
        if (src_id in known_unit_ids and src_id not in emitted_unit_ids) or (
            dst_id in known_unit_ids and dst_id not in emitted_unit_ids
        ):
            return None
        return {
            "id": f"rel:{snapshot_id}:{rel_id}",
            "type": relation.relation_type or "",
            "src": f"sym:{snapshot_id}:{src_id}",
            "dst": f"sym:{snapshot_id}:{dst_id}",
            "confidence": relation.confidence,
            "resolution_state": relation.resolution_state or "",
            "source_set": sorted(relation.support_sources),
        }

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

    def build_lineage_payload(
        self,
        snapshot: dict[str, Any],
        manifest: dict[str, Any],
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        repo_name = snapshot.get("repo_name") or git_meta.get("repo_name")
        snapshot_id = snapshot.get("snapshot_id")
        branch = snapshot.get("branch") or git_meta.get("branch")
        commit_id = snapshot.get("commit_id") or git_meta.get("commit_id")
        documents: list[dict[str, Any]] = snapshot.get("documents") or []
        symbols: list[dict[str, Any]] = snapshot.get("symbols") or []
        code_graph = self.build_code_graph_payload(snapshot)
        return self._build_lineage_payload_from_parts(
            repo_name=repo_name,
            snapshot_id=snapshot_id,
            branch=branch,
            commit_id=commit_id,
            manifest=manifest,
            git_meta=git_meta,
            previous_snapshot_symbols=previous_snapshot_symbols,
            documents=documents,
            symbols=symbols,
            code_graph=code_graph,
        )

    def build_lineage_payload_for_snapshot(
        self,
        snapshot: IRSnapshot,
        manifest: Any,
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        repo_name = snapshot.repo_name or git_meta.get("repo_name")
        snapshot_id = snapshot.snapshot_id
        branch = snapshot.branch or git_meta.get("branch")
        commit_id = snapshot.commit_id or git_meta.get("commit_id")
        code_graph = self.build_code_graph_payload_for_snapshot(snapshot)
        return self._build_lineage_payload_from_parts(
            repo_name=repo_name,
            snapshot_id=snapshot_id,
            branch=branch,
            commit_id=commit_id,
            manifest=manifest,
            git_meta=git_meta,
            previous_snapshot_symbols=previous_snapshot_symbols,
            documents=snapshot.documents,
            symbols=snapshot.symbols,
            code_graph=code_graph,
        )

    def _build_lineage_payload_from_parts(
        self,
        *,
        repo_name: Any,
        snapshot_id: Any,
        branch: Any,
        commit_id: Any,
        manifest: Any,
        git_meta: dict[str, Any],
        previous_snapshot_symbols: dict[str, str] | None,
        documents: Sequence[Any],
        symbols: Sequence[Any],
        code_graph: dict[str, Any],
    ) -> dict[str, Any]:
        if not snapshot_id:
            raise ValueError("snapshot_id is required for lineage payload")
        if not repo_name:
            raise ValueError("repo_name is required for lineage payload")

        repo_node_id = f"repo:{repo_name}"
        branch_node_id = f"branch:{repo_name}:{branch}" if branch else None
        commit_node_id = f"commit:{repo_name}:{commit_id}" if commit_id else None
        snapshot_node_id = f"snapshot:{snapshot_id}"
        manifest_id = _record_get(manifest, "manifest_id")
        manifest_node_id = f"manifest:{manifest_id}" if manifest_id else None
        run_id = _record_get(manifest, "index_run_id")
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
                        "ref_name": _record_get(manifest, "ref_name"),
                        "status": _record_get(manifest, "status"),
                        "published_at": _record_get(manifest, "published_at"),
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
            parent_ids: list[Any] = git_meta.get("parent_commit_ids") or []
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
            prev = _record_get(manifest, "previous_manifest_id")
            if prev:
                edges.append(
                    {
                        "type": "manifest_supersedes",
                        "src": manifest_node_id,
                        "dst": f"manifest:{prev}",
                    }
                )

        for doc in documents:
            doc_id: str | None = _record_get(doc, "doc_id")
            if not doc_id:
                continue
            node_id = f"doc:{snapshot_id}:{doc_id}"
            nodes.append(
                {
                    "id": node_id,
                    "type": "DocumentVersion",
                    "props": {
                        "doc_id": doc_id,
                        "path": _record_get(doc, "path"),
                        "language": _record_get(doc, "language"),
                    },  # type: ignore[dict-item]
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
            symbol_id = _record_get(sym, "symbol_id")
            if not symbol_id:
                continue
            node_id = f"symbol:{snapshot_id}:{symbol_id}"
            nodes.append(
                {
                    "id": node_id,
                    "type": "SymbolVersion",
                    "props": {
                        "symbol_id": symbol_id,
                        "display_name": _record_get(sym, "display_name"),
                        "path": _record_get(sym, "path"),
                        "kind": _record_get(sym, "kind"),
                    },
                }
            )
            doc_path = _record_get(sym, "path")
            if doc_path:
                matching = [d for d in documents if _record_get(d, "path") == doc_path]
                if matching:
                    edges.append(
                        {
                            "type": "document_defines_symbol",
                            "src": (
                                f"doc:{snapshot_id}:"
                                f"{_record_get(matching[0], 'doc_id')}"
                            ),
                            "dst": node_id,
                        }
                    )
            ext_symbol = _record_get(sym, "external_symbol_id")
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
