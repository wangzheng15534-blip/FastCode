"""
Snapshot metadata and artifact persistence.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ..db_runtime import DBRuntime
from ..ir.types import (
    IRAttachment,
    IRCodeUnit,
    IRDocument,
    IREdge,
    IROccurrence,
    IRRelation,
    IRSnapshot,
    IRSymbol,
    IRUnitEmbedding,
    IRUnitSupport,
)
from ..utils import ensure_dir, safe_jsonable, utc_now
from ..utils.materialization import (
    BOUNDARY_GRAPH_FULL_LOAD,
    BOUNDARY_JSON_DECODE,
    BOUNDARY_JSON_ENCODE,
    BOUNDARY_NETWORKX_CONVERSION,
    BOUNDARY_SNAPSHOT_FULL_LOAD,
    increment_materialization_boundary,
)
from .records import (
    OutboxEventRecord,
    RedoTaskRecord,
    SCIPArtifactRecord,
    SnapshotRecord,
    SnapshotRefRecord,
)


class SnapshotStore:
    _SNAPSHOT_FIELDS = (
        "snapshot_id",
        "repo_name",
        "branch",
        "commit_id",
        "tree_id",
        "artifact_key",
        "ir_path",
        "ir_graphs_path",
        "created_at",
        "metadata_json",
    )
    _SNAPSHOT_REF_FIELDS = (
        "ref_id",
        "repo_name",
        "branch",
        "commit_id",
        "tree_id",
        "snapshot_id",
        "created_at",
    )
    _SCIP_ARTIFACT_FIELDS = (
        "snapshot_id",
        "indexer_name",
        "indexer_version",
        "artifact_path",
        "checksum",
        "created_at",
    )
    _SCIP_ARTIFACT_ENTRY_FIELDS = (
        "artifact_id",
        "snapshot_id",
        "sequence_no",
        "role",
        "indexer_name",
        "indexer_version",
        "artifact_path",
        "checksum",
        "created_at",
        "metadata_json",
    )
    _REDO_TASK_FIELDS = (
        "task_id",
        "task_type",
        "payload_json",
        "status",
        "attempts",
        "last_error",
        "next_attempt_at",
        "created_at",
        "updated_at",
    )
    _OUTBOX_EVENT_FIELDS = (
        "event_id",
        "event_type",
        "payload",
        "snapshot_id",
        "status",
        "attempts",
        "max_attempts",
        "created_at",
        "last_attempt_at",
        "error_message",
    )

    def __init__(
        self, persist_dir: str, storage_cfg: dict[str, Any] | None = None
    ) -> None:
        self.persist_dir = os.path.abspath(persist_dir)
        self.snapshot_root = os.path.join(self.persist_dir, "snapshots")
        ensure_dir(self.persist_dir)
        ensure_dir(self.snapshot_root)
        self.db_path = os.path.join(self.persist_dir, "lineage.db")
        self.db_runtime = DBRuntime.from_storage_config(
            sqlite_path=self.db_path, storage_cfg=storage_cfg
        )
        self._init_db()

    @staticmethod
    def _source_set_payload(values: set[str]) -> list[str]:
        return sorted(value for value in values if value)

    @classmethod
    def _document_payload(cls, doc: IRDocument) -> dict[str, Any]:
        return {
            "doc_id": doc.doc_id,
            "path": doc.path,
            "language": doc.language,
            "blob_oid": doc.blob_oid,
            "content_hash": doc.content_hash,
            "source_set": cls._source_set_payload(doc.source_set),
        }

    @classmethod
    def _symbol_payload(cls, sym: IRSymbol) -> dict[str, Any]:
        return {
            "symbol_id": sym.symbol_id,
            "external_symbol_id": sym.external_symbol_id,
            "path": sym.path,
            "display_name": sym.display_name,
            "kind": sym.kind,
            "language": sym.language,
            "qualified_name": sym.qualified_name,
            "signature": sym.signature,
            "start_line": sym.start_line,
            "start_col": sym.start_col,
            "end_line": sym.end_line,
            "end_col": sym.end_col,
            "source_priority": sym.source_priority,
            "source_set": cls._source_set_payload(sym.source_set),
            "metadata": dict(sym.metadata) if sym.metadata else {},
        }

    @staticmethod
    def _occurrence_payload(occ: IROccurrence) -> dict[str, Any]:
        return {
            "occurrence_id": occ.occurrence_id,
            "symbol_id": occ.symbol_id,
            "doc_id": occ.doc_id,
            "role": occ.role,
            "start_line": occ.start_line,
            "start_col": occ.start_col,
            "end_line": occ.end_line,
            "end_col": occ.end_col,
            "source": occ.source,
            "metadata": dict(occ.metadata) if occ.metadata else {},
        }

    @staticmethod
    def _edge_payload(edge: IREdge) -> dict[str, Any]:
        return {
            "edge_id": edge.edge_id,
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "edge_type": edge.edge_type,
            "source": edge.source,
            "confidence": edge.confidence,
            "doc_id": edge.doc_id,
            "metadata": dict(edge.metadata) if edge.metadata else {},
        }

    @staticmethod
    def _attachment_payload(attachment: IRAttachment) -> dict[str, Any]:
        return {
            "attachment_id": attachment.attachment_id,
            "target_id": attachment.target_id,
            "target_type": attachment.target_type,
            "attachment_type": attachment.attachment_type,
            "source": attachment.source,
            "confidence": attachment.confidence,
            "payload": dict(attachment.payload) if attachment.payload else {},
            "metadata": dict(attachment.metadata) if attachment.metadata else {},
        }

    @staticmethod
    def _payload_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        return {}

    @staticmethod
    def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
        value = payload.get(field_name)
        if value is None:
            raise KeyError(field_name)
        return str(value)

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float_or_default(value: Any, *, default: float = 0.0) -> float:
        if value is None or isinstance(value, bool):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _sequence_items(value: Any) -> Sequence[Any]:
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            return value
        if isinstance(value, (set, frozenset)):
            return tuple(value)
        return ()

    @classmethod
    def _string_list_payload(cls, values: Any) -> list[str]:
        return [str(value) for value in cls._sequence_items(values) if value]

    @classmethod
    def _string_set_payload(cls, values: Any) -> set[str]:
        return {str(value) for value in cls._sequence_items(values) if value}

    @staticmethod
    def _json_mapping_payload(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        normalized = {str(key): nested for key, nested in value.items()}
        jsonable = safe_jsonable(normalized)
        return jsonable if isinstance(jsonable, dict) else {}

    @staticmethod
    def _json_list_payload(value: Any) -> list[Any] | None:
        if value is None:
            return None
        jsonable = safe_jsonable(value)
        return jsonable if isinstance(jsonable, list) else None

    @classmethod
    def _code_unit_payload(cls, unit: IRCodeUnit) -> dict[str, Any]:
        return {
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "path": unit.path,
            "language": unit.language,
            "display_name": unit.display_name,
            "qualified_name": unit.qualified_name,
            "signature": unit.signature,
            "docstring": unit.docstring,
            "summary": unit.summary,
            "start_line": unit.start_line,
            "start_col": unit.start_col,
            "end_line": unit.end_line,
            "end_col": unit.end_col,
            "parent_unit_id": unit.parent_unit_id,
            "primary_anchor_symbol_id": unit.primary_anchor_symbol_id,
            "anchor_symbol_ids": cls._string_list_payload(unit.anchor_symbol_ids),
            "candidate_anchor_symbol_ids": cls._string_list_payload(
                unit.candidate_anchor_symbol_ids
            ),
            "anchor_coverage": float(unit.anchor_coverage),
            "source_set": cls._source_set_payload(unit.source_set),
            "metadata": cls._json_mapping_payload(unit.metadata),
        }

    @classmethod
    def _unit_support_payload(cls, support: IRUnitSupport) -> dict[str, Any]:
        return {
            "support_id": support.support_id,
            "unit_id": support.unit_id,
            "source": support.source,
            "support_kind": support.support_kind,
            "external_id": support.external_id,
            "role": support.role,
            "path": support.path,
            "display_name": support.display_name,
            "qualified_name": support.qualified_name,
            "signature": support.signature,
            "enclosing_external_id": support.enclosing_external_id,
            "start_line": support.start_line,
            "start_col": support.start_col,
            "end_line": support.end_line,
            "end_col": support.end_col,
            "metadata": cls._json_mapping_payload(support.metadata),
        }

    @classmethod
    def _relation_payload(cls, relation: IRRelation) -> dict[str, Any]:
        return {
            "relation_id": relation.relation_id,
            "src_unit_id": relation.src_unit_id,
            "dst_unit_id": relation.dst_unit_id,
            "relation_type": relation.relation_type,
            "resolution_state": relation.resolution_state,
            "support_sources": cls._source_set_payload(relation.support_sources),
            "support_ids": cls._string_list_payload(relation.support_ids),
            "pending_capabilities": cls._source_set_payload(
                relation.pending_capabilities
            ),
            "metadata": cls._json_mapping_payload(relation.metadata),
        }

    @classmethod
    def _embedding_payload(cls, embedding: IRUnitEmbedding) -> dict[str, Any]:
        return {
            "embedding_id": embedding.embedding_id,
            "unit_id": embedding.unit_id,
            "source": embedding.source,
            "vector": cls._json_list_payload(embedding.vector),
            "embedding_text": embedding.embedding_text,
            "model_id": embedding.model_id,
            "metadata": cls._json_mapping_payload(embedding.metadata),
        }

    @classmethod
    def _snapshot_file_payload(cls, snapshot: IRSnapshot) -> dict[str, Any]:
        return {
            "schema_version": "ir.v2",
            "repo_name": snapshot.repo_name,
            "snapshot_id": snapshot.snapshot_id,
            "branch": snapshot.branch,
            "commit_id": snapshot.commit_id,
            "tree_id": snapshot.tree_id,
            "units": [cls._code_unit_payload(unit) for unit in snapshot.units],
            "supports": [
                cls._unit_support_payload(support) for support in snapshot.supports
            ],
            "relations": [
                cls._relation_payload(relation) for relation in snapshot.relations
            ],
            "embeddings": [
                cls._embedding_payload(embedding) for embedding in snapshot.embeddings
            ],
            "metadata": cls._json_mapping_payload(snapshot.metadata),
        }

    @classmethod
    def _code_unit_from_payload(cls, data: Any) -> IRCodeUnit:
        payload = cls._payload_mapping(data)
        return IRCodeUnit(
            unit_id=cls._required_text(payload, "unit_id"),
            kind=cls._required_text(payload, "kind"),
            path=cls._required_text(payload, "path"),
            language=cls._required_text(payload, "language"),
            display_name=cls._required_text(payload, "display_name"),
            qualified_name=cls._string_or_none(payload.get("qualified_name")),
            signature=cls._string_or_none(payload.get("signature")),
            docstring=cls._string_or_none(payload.get("docstring")),
            summary=cls._string_or_none(payload.get("summary")),
            start_line=cls._int_or_none(payload.get("start_line")),
            start_col=cls._int_or_none(payload.get("start_col")),
            end_line=cls._int_or_none(payload.get("end_line")),
            end_col=cls._int_or_none(payload.get("end_col")),
            parent_unit_id=cls._string_or_none(payload.get("parent_unit_id")),
            primary_anchor_symbol_id=cls._string_or_none(
                payload.get("primary_anchor_symbol_id")
            ),
            anchor_symbol_ids=cls._string_list_payload(
                payload.get("anchor_symbol_ids")
            ),
            candidate_anchor_symbol_ids=cls._string_list_payload(
                payload.get("candidate_anchor_symbol_ids")
            ),
            anchor_coverage=cls._float_or_default(payload.get("anchor_coverage")),
            source_set=cls._string_set_payload(payload.get("source_set")),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _unit_support_from_payload(cls, data: Any) -> IRUnitSupport:
        payload = cls._payload_mapping(data)
        return IRUnitSupport(
            support_id=cls._required_text(payload, "support_id"),
            unit_id=cls._required_text(payload, "unit_id"),
            source=cls._required_text(payload, "source"),
            support_kind=cls._required_text(payload, "support_kind"),
            external_id=cls._string_or_none(payload.get("external_id")),
            role=cls._string_or_none(payload.get("role")),
            path=cls._string_or_none(payload.get("path")),
            display_name=cls._string_or_none(payload.get("display_name")),
            qualified_name=cls._string_or_none(payload.get("qualified_name")),
            signature=cls._string_or_none(payload.get("signature")),
            enclosing_external_id=cls._string_or_none(
                payload.get("enclosing_external_id")
            ),
            start_line=cls._int_or_none(payload.get("start_line")),
            start_col=cls._int_or_none(payload.get("start_col")),
            end_line=cls._int_or_none(payload.get("end_line")),
            end_col=cls._int_or_none(payload.get("end_col")),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _relation_from_payload(cls, data: Any) -> IRRelation:
        payload = cls._payload_mapping(data)
        return IRRelation(
            relation_id=cls._required_text(payload, "relation_id"),
            src_unit_id=cls._required_text(payload, "src_unit_id"),
            dst_unit_id=cls._required_text(payload, "dst_unit_id"),
            relation_type=cls._required_text(payload, "relation_type"),
            resolution_state=cls._required_text(payload, "resolution_state"),
            support_sources=cls._string_set_payload(payload.get("support_sources")),
            support_ids=cls._string_list_payload(payload.get("support_ids")),
            pending_capabilities=cls._string_set_payload(
                payload.get("pending_capabilities")
            ),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _embedding_from_payload(cls, data: Any) -> IRUnitEmbedding:
        payload = cls._payload_mapping(data)
        return IRUnitEmbedding(
            embedding_id=cls._required_text(payload, "embedding_id"),
            unit_id=cls._required_text(payload, "unit_id"),
            source=cls._required_text(payload, "source"),
            vector=cls._json_list_payload(payload.get("vector")),
            embedding_text=cls._string_or_none(payload.get("embedding_text")),
            model_id=cls._string_or_none(payload.get("model_id")),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _document_from_payload(cls, data: Any) -> IRDocument:
        payload = cls._payload_mapping(data)
        return IRDocument(
            doc_id=cls._required_text(payload, "doc_id"),
            path=cls._required_text(payload, "path"),
            language=cls._required_text(payload, "language"),
            blob_oid=cls._string_or_none(payload.get("blob_oid")),
            content_hash=cls._string_or_none(payload.get("content_hash")),
            source_set=cls._string_set_payload(payload.get("source_set")),
        )

    @classmethod
    def _symbol_from_payload(cls, data: Any) -> IRSymbol:
        payload = cls._payload_mapping(data)
        return IRSymbol(
            symbol_id=cls._required_text(payload, "symbol_id"),
            external_symbol_id=cls._string_or_none(payload.get("external_symbol_id")),
            path=cls._required_text(payload, "path"),
            display_name=cls._required_text(payload, "display_name"),
            kind=cls._required_text(payload, "kind"),
            language=cls._required_text(payload, "language"),
            qualified_name=cls._string_or_none(payload.get("qualified_name")),
            signature=cls._string_or_none(payload.get("signature")),
            start_line=cls._int_or_none(payload.get("start_line")),
            start_col=cls._int_or_none(payload.get("start_col")),
            end_line=cls._int_or_none(payload.get("end_line")),
            end_col=cls._int_or_none(payload.get("end_col")),
            source_priority=cls._int_or_none(payload.get("source_priority")) or 0,
            source_set=cls._string_set_payload(payload.get("source_set")),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _occurrence_from_payload(cls, data: Any) -> IROccurrence:
        payload = cls._payload_mapping(data)
        return IROccurrence(
            occurrence_id=cls._required_text(payload, "occurrence_id"),
            symbol_id=cls._required_text(payload, "symbol_id"),
            doc_id=cls._required_text(payload, "doc_id"),
            role=cls._required_text(payload, "role"),
            start_line=cls._int_or_none(payload.get("start_line")) or 0,
            start_col=cls._int_or_none(payload.get("start_col")) or 0,
            end_line=cls._int_or_none(payload.get("end_line")) or 0,
            end_col=cls._int_or_none(payload.get("end_col")) or 0,
            source=cls._required_text(payload, "source"),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _edge_from_payload(cls, data: Any) -> IREdge:
        payload = cls._payload_mapping(data)
        return IREdge(
            edge_id=cls._required_text(payload, "edge_id"),
            src_id=cls._required_text(payload, "src_id"),
            dst_id=cls._required_text(payload, "dst_id"),
            edge_type=cls._required_text(payload, "edge_type"),
            source=cls._required_text(payload, "source"),
            confidence=cls._required_text(payload, "confidence"),
            doc_id=cls._string_or_none(payload.get("doc_id")),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _attachment_from_payload(cls, data: Any) -> IRAttachment:
        payload = cls._payload_mapping(data)
        return IRAttachment(
            attachment_id=cls._required_text(payload, "attachment_id"),
            target_id=cls._required_text(payload, "target_id"),
            target_type=cls._required_text(payload, "target_type"),
            attachment_type=cls._required_text(payload, "attachment_type"),
            source=cls._required_text(payload, "source"),
            confidence=cls._required_text(payload, "confidence"),
            payload=cls._json_mapping_payload(payload.get("payload")),
            metadata=cls._json_mapping_payload(payload.get("metadata")),
        )

    @classmethod
    def _snapshot_from_payload(cls, data: Any) -> IRSnapshot:
        payload = cls._payload_mapping(data)
        repo_name = cls._required_text(payload, "repo_name")
        snapshot_id = cls._required_text(payload, "snapshot_id")
        branch = cls._string_or_none(payload.get("branch"))
        commit_id = cls._string_or_none(payload.get("commit_id"))
        tree_id = cls._string_or_none(payload.get("tree_id"))
        metadata = cls._json_mapping_payload(payload.get("metadata"))
        if payload.get("units") is not None or payload.get("supports") is not None:
            return IRSnapshot(
                repo_name=repo_name,
                snapshot_id=snapshot_id,
                branch=branch,
                commit_id=commit_id,
                tree_id=tree_id,
                units=[
                    cls._code_unit_from_payload(unit)
                    for unit in cls._sequence_items(payload.get("units"))
                ],
                supports=[
                    cls._unit_support_from_payload(support)
                    for support in cls._sequence_items(payload.get("supports"))
                ],
                relations=[
                    cls._relation_from_payload(relation)
                    for relation in cls._sequence_items(payload.get("relations"))
                ],
                embeddings=[
                    cls._embedding_from_payload(embedding)
                    for embedding in cls._sequence_items(payload.get("embeddings"))
                ],
                metadata=metadata,
            )
        return IRSnapshot(
            repo_name=repo_name,
            snapshot_id=snapshot_id,
            branch=branch,
            commit_id=commit_id,
            tree_id=tree_id,
            documents=[
                cls._document_from_payload(document)
                for document in cls._sequence_items(payload.get("documents"))
            ],
            symbols=[
                cls._symbol_from_payload(symbol)
                for symbol in cls._sequence_items(payload.get("symbols"))
            ],
            occurrences=[
                cls._occurrence_from_payload(occurrence)
                for occurrence in cls._sequence_items(payload.get("occurrences"))
            ],
            edges=[
                cls._edge_from_payload(edge)
                for edge in cls._sequence_items(payload.get("edges"))
            ],
            attachments=[
                cls._attachment_from_payload(attachment)
                for attachment in cls._sequence_items(payload.get("attachments"))
            ],
            metadata=metadata,
        )

    @classmethod
    def _graph_json_payload(cls, value: Any, *, _depth: int = 0) -> Any:
        if _depth > 12:
            return repr(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): cls._graph_json_payload(nested, _depth=_depth + 1)
                for key, nested in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._graph_json_payload(item, _depth=_depth + 1) for item in value]
        return repr(value)

    def _init_db(self) -> None:
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    repo_name TEXT NOT NULL,
                    branch TEXT,
                    commit_id TEXT,
                    tree_id TEXT,
                    artifact_key TEXT NOT NULL,
                    ir_path TEXT NOT NULL,
                    ir_graphs_path TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """,
            )
            if self.db_runtime.backend == "postgres":
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_refs (
                        ref_id BIGSERIAL PRIMARY KEY,
                        repo_name TEXT NOT NULL,
                        branch TEXT,
                        commit_id TEXT,
                        tree_id TEXT,
                        snapshot_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(repo_name, commit_id, snapshot_id)
                    )
                    """,
                )
            else:
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_refs (
                        ref_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        repo_name TEXT NOT NULL,
                        branch TEXT,
                        commit_id TEXT,
                        tree_id TEXT,
                        snapshot_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(repo_name, commit_id, snapshot_id)
                    )
                    """,
                )
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS scip_artifacts (
                    snapshot_id TEXT PRIMARY KEY,
                    indexer_name TEXT NOT NULL,
                    indexer_version TEXT,
                    artifact_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    component TEXT NOT NULL,
                    version TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    PRIMARY KEY (component, version)
                )
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                CREATE INDEX IF NOT EXISTS idx_snapshot_refs_repo_branch
                ON snapshot_refs (repo_name, branch, created_at DESC)
                """,
            )
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO schema_migrations (component, version, applied_at)
                VALUES (?, ?, ?)
                ON CONFLICT(component, version) DO NOTHING
                """,
                ("core_metadata", "v1", utc_now()),
            )
            if self.db_runtime.backend == "postgres":
                # Git backbone
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS repositories (
                        repo_id TEXT PRIMARY KEY,
                        repo_name TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_refs (
                        repo_id TEXT NOT NULL,
                        ref_name TEXT NOT NULL,
                        commit_id TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, ref_name)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_commits (
                        repo_id TEXT NOT NULL,
                        commit_id TEXT NOT NULL,
                        tree_id TEXT,
                        parent_commit_id TEXT,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, commit_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_trees (
                        repo_id TEXT NOT NULL,
                        tree_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, tree_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS git_blobs (
                        repo_id TEXT NOT NULL,
                        blob_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (repo_id, blob_id)
                    )
                    """,
                )
                # Relational facts
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_documents (
                        snapshot_id TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        path TEXT NOT NULL,
                        language TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, doc_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS symbols (
                        snapshot_id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        path TEXT,
                        display_name TEXT,
                        qualified_name TEXT,
                        kind TEXT,
                        language TEXT,
                        source_priority INTEGER,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, symbol_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS occurrences (
                        snapshot_id TEXT NOT NULL,
                        occurrence_id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        role TEXT,
                        start_line INTEGER,
                        start_col INTEGER,
                        end_line INTEGER,
                        end_col INTEGER,
                        source TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, occurrence_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS edges (
                        snapshot_id TEXT NOT NULL,
                        edge_id TEXT NOT NULL,
                        src_id TEXT NOT NULL,
                        dst_id TEXT NOT NULL,
                        edge_type TEXT NOT NULL,
                        source TEXT,
                        confidence TEXT,
                        doc_id TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, edge_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS attachments (
                        snapshot_id TEXT NOT NULL,
                        attachment_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        attachment_type TEXT NOT NULL,
                        source TEXT,
                        confidence TEXT,
                        payload_json TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, attachment_id)
                    )
                    """,
                )
                # Staging + hardening
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS snapshot_staging (
                        stage_id TEXT PRIMARY KEY,
                        snapshot_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL,
                        promoted_at TEXT
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS resource_locks (
                        lock_name TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        fencing_token BIGINT NOT NULL DEFAULT 0
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    ALTER TABLE resource_locks
                    ADD COLUMN IF NOT EXISTS fencing_token BIGINT NOT NULL DEFAULT 0
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS redo_tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT,
                        next_attempt_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS publish_outbox (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        snapshot_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        attempts INTEGER NOT NULL DEFAULT 0,
                        max_attempts INTEGER NOT NULL DEFAULT 5,
                        created_at TEXT NOT NULL,
                        last_attempt_at TEXT,
                        error_message TEXT
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE INDEX IF NOT EXISTS idx_outbox_status
                    ON publish_outbox(status)
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS design_documents (
                        snapshot_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        repo_name TEXT NOT NULL,
                        path TEXT NOT NULL,
                        title TEXT,
                        heading TEXT,
                        doc_type TEXT,
                        content TEXT NOT NULL,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, chunk_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS design_doc_mentions (
                        snapshot_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        symbol_id TEXT NOT NULL,
                        symbol_name TEXT,
                        confidence TEXT,
                        metadata_json TEXT,
                        PRIMARY KEY (snapshot_id, chunk_id, symbol_id)
                    )
                    """,
                )
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO schema_migrations (component, version, applied_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(component, version) DO NOTHING
                    """,
                    ("pg_full_spec_alignment", "v1", utc_now()),
                )
            conn.commit()

    def artifact_key_for_snapshot(self, snapshot_id: str) -> str:
        return f"snap_{hashlib.md5(snapshot_id.encode('utf-8')).hexdigest()[:20]}"

    def snapshot_dir(self, snapshot_id: str) -> str:
        safe = self.artifact_key_for_snapshot(snapshot_id)
        path = os.path.join(self.snapshot_root, safe)
        ensure_dir(path)
        return path

    def save_snapshot(
        self, snapshot: IRSnapshot, metadata: dict[str, Any] | None = None
    ) -> SnapshotRecord:
        snap_dir = self.snapshot_dir(snapshot.snapshot_id)
        ir_path = os.path.join(snap_dir, "ir_snapshot.json")
        tmp_ir_path = f"{ir_path}.tmp"
        with open(tmp_ir_path, "w", encoding="utf-8") as f:
            snapshot_payload = self._snapshot_file_payload(snapshot)
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=(
                    len(snapshot.units)
                    + len(snapshot.supports)
                    + len(snapshot.relations)
                    + len(snapshot.embeddings)
                ),
            )
            json.dump(
                snapshot_payload,
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_ir_path, ir_path)

        artifact_key = self.artifact_key_for_snapshot(snapshot.snapshot_id)
        increment_materialization_boundary(BOUNDARY_JSON_ENCODE)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO snapshots (
                    snapshot_id, repo_name, branch, commit_id, tree_id, artifact_key,
                    ir_path, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    repo_name=excluded.repo_name,
                    branch=excluded.branch,
                    commit_id=excluded.commit_id,
                    tree_id=excluded.tree_id,
                    artifact_key=excluded.artifact_key,
                    ir_path=excluded.ir_path,
                    metadata_json=excluded.metadata_json
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.repo_name,
                    snapshot.branch,
                    snapshot.commit_id,
                    snapshot.tree_id,
                    artifact_key,
                    ir_path,
                    utc_now(),
                    metadata_json,
                ),
            )
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO snapshot_refs (
                    repo_name, branch, commit_id, tree_id, snapshot_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_name, commit_id, snapshot_id) DO NOTHING
                """,
                (
                    snapshot.repo_name,
                    snapshot.branch,
                    snapshot.commit_id,
                    snapshot.tree_id,
                    snapshot.snapshot_id,
                    utc_now(),
                ),
            )
            conn.commit()

        return SnapshotRecord(
            snapshot_id=snapshot.snapshot_id,
            repo_name=snapshot.repo_name,
            branch=snapshot.branch,
            commit_id=snapshot.commit_id,
            tree_id=snapshot.tree_id,
            artifact_key=artifact_key,
            ir_path=ir_path,
            ir_graphs_path=None,
            created_at=utc_now(),
            metadata_json=metadata_json,
        )

    def update_snapshot_metadata(
        self, snapshot_id: str, metadata: dict[str, Any]
    ) -> None:
        increment_materialization_boundary(BOUNDARY_JSON_ENCODE)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "UPDATE snapshots SET metadata_json=? WHERE snapshot_id=?",
                (metadata_json, snapshot_id),
            )
            conn.commit()

    @staticmethod
    def _ir_graph_items(ir_graphs: Any) -> dict[str, Any]:
        from ..ir.graph import IRGraphs

        if isinstance(ir_graphs, IRGraphs):
            return {
                "dependency_graph": ir_graphs.dependency_graph,
                "call_graph": ir_graphs.call_graph,
                "inheritance_graph": ir_graphs.inheritance_graph,
                "reference_graph": ir_graphs.reference_graph,
                "containment_graph": ir_graphs.containment_graph,
            }
        if isinstance(ir_graphs, dict):
            return ir_graphs
        return {}

    def save_ir_graphs(self, snapshot_id: str, ir_graphs: Any) -> str:
        snap_dir = self.snapshot_dir(snapshot_id)
        path = os.path.join(snap_dir, "ir_graphs.json")
        import networkx as nx

        serializable: dict[str, Any] = {}
        for name, graph in self._ir_graph_items(ir_graphs).items():
            to_payload = getattr(graph, "to_payload", None)
            if callable(to_payload):
                serializable[name] = to_payload()
            elif isinstance(graph, nx.Graph):
                increment_materialization_boundary(
                    BOUNDARY_NETWORKX_CONVERSION,
                    items=graph.number_of_nodes() + graph.number_of_edges(),
                )
                serializable[name] = nx.node_link_data(graph)
            else:
                serializable[name] = self._graph_json_payload(graph)
        with open(path, "w", encoding="utf-8") as f:
            increment_materialization_boundary(
                BOUNDARY_JSON_ENCODE,
                items=len(serializable),
            )
            json.dump({"graphs": serializable}, f, ensure_ascii=False)
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "UPDATE snapshots SET ir_graphs_path=? WHERE snapshot_id=?",
                (path, snapshot_id),
            )
            conn.commit()
        return path

    def load_ir_graphs(self, snapshot_id: str) -> Any | None:
        record = self.get_snapshot_record(snapshot_id)
        if not record:
            return None
        path = record.ir_graphs_path
        if not path or not os.path.exists(path):
            return None
        import networkx as nx

        from ..ir.graph import IRGraphView

        try:
            with open(path, encoding="utf-8") as f:
                increment_materialization_boundary(BOUNDARY_GRAPH_FULL_LOAD)
                increment_materialization_boundary(BOUNDARY_JSON_DECODE)
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logging.getLogger(__name__).warning(
                "ir_graphs at %s is not JSON (legacy pickle format?), skipping", path
            )
            return None
        graphs_data = data.get("graphs", {})
        result: dict[str, Any] = {}
        for name, graph_data in graphs_data.items():
            if (
                isinstance(graph_data, dict)
                and graph_data.get("storage_version") == IRGraphView.STORAGE_VERSION
            ):
                result[name] = IRGraphView.from_payload(graph_data)
            elif (
                isinstance(graph_data, dict)
                and "nodes" in graph_data
                and ("links" in graph_data or "edges" in graph_data)
                and isinstance(graph_data["nodes"], list)
                and (
                    not graph_data["nodes"] or isinstance(graph_data["nodes"][0], dict)
                )
            ):
                increment_materialization_boundary(
                    BOUNDARY_NETWORKX_CONVERSION,
                    items=len(graph_data["nodes"]),
                )
                result[name] = nx.node_link_graph(graph_data, directed=True)
            else:
                result[name] = graph_data
        graph_names = {
            "dependency_graph",
            "call_graph",
            "inheritance_graph",
            "reference_graph",
            "containment_graph",
        }
        if graph_names.issubset(result) and all(
            isinstance(result[name], (nx.Graph, IRGraphView)) for name in graph_names
        ):
            from ..ir.graph import IRGraphs

            return IRGraphs(
                dependency_graph=result["dependency_graph"],
                call_graph=result["call_graph"],
                inheritance_graph=result["inheritance_graph"],
                reference_graph=result["reference_graph"],
                containment_graph=result["containment_graph"],
            )
        return result

    def load_snapshot(self, snapshot_id: str) -> IRSnapshot | None:
        record = self.get_snapshot_record(snapshot_id)
        if not record:
            return None
        ir_path = record.ir_path
        if not ir_path or not os.path.exists(ir_path):
            return None
        with open(ir_path, encoding="utf-8") as f:
            increment_materialization_boundary(BOUNDARY_SNAPSHOT_FULL_LOAD)
            increment_materialization_boundary(BOUNDARY_JSON_DECODE)
            data = json.load(f)
        return self._snapshot_from_payload(data)

    def get_snapshot_record(self, snapshot_id: str) -> SnapshotRecord | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT * FROM snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        return self._row_to_snapshot_record(row)

    def find_by_repo_commit(
        self, repo_name: str, commit_id: str
    ) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM snapshots
                WHERE repo_name=? AND commit_id=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (repo_name, commit_id),
            ).fetchone()
        record = self._row_to_snapshot_record(row)
        return self._snapshot_payload(record) if record is not None else None

    def find_by_artifact_key(self, artifact_key: str) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM snapshots
                WHERE artifact_key=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (artifact_key,),
            ).fetchone()
        record = self._row_to_snapshot_record(row)
        return self._snapshot_payload(record) if record is not None else None

    def resolve_snapshot_for_ref(
        self, repo_name: str, branch: str
    ) -> dict[str, Any] | None:
        record = self.resolve_snapshot_for_ref_record(repo_name, branch)
        return self._snapshot_ref_payload(record) if record is not None else None

    def resolve_snapshot_for_ref_record(
        self, repo_name: str, branch: str
    ) -> SnapshotRefRecord | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM snapshot_refs
                WHERE repo_name=? AND branch=?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (repo_name, branch),
            ).fetchone()
        return self._row_to_snapshot_ref_record(row)

    def list_repo_ref_records(self, repo_name: str) -> list[SnapshotRefRecord]:
        with self.db_runtime.connect() as conn:
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT branch, commit_id, tree_id, snapshot_id, created_at, ref_id, repo_name
                FROM snapshot_refs
                WHERE repo_name=?
                ORDER BY created_at DESC
                """,
                (repo_name,),
            ).fetchall()
        return [
            record
            for row in rows
            if (record := self._row_to_snapshot_ref_record(row)) is not None
        ]

    def save_scip_artifact_ref(
        self,
        snapshot_id: str,
        *,
        indexer_name: str = "unknown",
        indexer_version: str | None = None,
        artifact_path: str = "",
        checksum: str = "",
    ) -> dict[str, Any]:
        artifacts = self.save_scip_artifact_refs(
            snapshot_id,
            artifacts=[
                {
                    "indexer_name": indexer_name,
                    "indexer_version": indexer_version,
                    "artifact_path": artifact_path,
                    "checksum": checksum,
                }
            ],
        )
        return artifacts[0]

    def _ensure_scip_artifact_entries_table(self, conn: Any) -> None:
        self.db_runtime.execute(
            conn,
            """
            CREATE TABLE IF NOT EXISTS scip_artifact_entries (
                artifact_id TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                sequence_no INTEGER NOT NULL,
                role TEXT NOT NULL,
                indexer_name TEXT NOT NULL,
                indexer_version TEXT,
                artifact_path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT,
                UNIQUE(snapshot_id, sequence_no)
            )
            """,
        )

    def save_scip_artifact_refs(
        self,
        snapshot_id: str,
        *,
        artifacts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not artifacts:
            return []

        created_at = utc_now()
        normalized_artifacts: list[SCIPArtifactRecord] = []
        for sequence_no, artifact in enumerate(artifacts):
            metadata = self._scip_metadata_mapping(artifact)
            if artifact.get("language"):
                metadata.setdefault("language", artifact.get("language"))
            normalized_artifacts.append(
                SCIPArtifactRecord(
                    artifact_id=f"{snapshot_id}:scip:{sequence_no}",
                    snapshot_id=snapshot_id,
                    sequence_no=sequence_no,
                    role="primary" if sequence_no == 0 else "secondary",
                    indexer_name=str(artifact.get("indexer_name") or "unknown"),
                    indexer_version=(
                        str(indexer_version)
                        if (indexer_version := artifact.get("indexer_version"))
                        is not None
                        else None
                    ),
                    artifact_path=str(artifact.get("artifact_path") or ""),
                    checksum=str(artifact.get("checksum") or ""),
                    created_at=created_at,
                    metadata_json=self._serialize_scip_metadata_json(metadata),
                )
            )

        primary_artifact = normalized_artifacts[0]
        with self.db_runtime.connect() as conn:
            self._ensure_scip_artifact_entries_table(conn)
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO scip_artifacts (
                    snapshot_id, indexer_name, indexer_version, artifact_path, checksum, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    indexer_name=excluded.indexer_name,
                    indexer_version=excluded.indexer_version,
                    artifact_path=excluded.artifact_path,
                    checksum=excluded.checksum,
                    created_at=excluded.created_at
                """,
                (
                    primary_artifact.snapshot_id,
                    primary_artifact.indexer_name,
                    primary_artifact.indexer_version,
                    primary_artifact.artifact_path,
                    primary_artifact.checksum,
                    created_at,
                ),
            )
            self.db_runtime.execute(
                conn,
                "DELETE FROM scip_artifact_entries WHERE snapshot_id=?",
                (snapshot_id,),
            )
            for artifact in normalized_artifacts:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO scip_artifact_entries (
                        artifact_id, snapshot_id, sequence_no, role, indexer_name,
                        indexer_version, artifact_path, checksum, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact.artifact_id,
                        artifact.snapshot_id,
                        artifact.sequence_no,
                        artifact.role,
                        artifact.indexer_name,
                        artifact.indexer_version,
                        artifact.artifact_path,
                        artifact.checksum,
                        artifact.created_at,
                        artifact.metadata_json,
                    ),
                )
            conn.commit()
        return [
            self._scip_artifact_entry_payload(record) for record in normalized_artifacts
        ]

    def get_scip_artifact_ref(self, snapshot_id: str) -> dict[str, Any] | None:
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT * FROM scip_artifacts WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        primary_record = self._row_to_scip_artifact_record(row)
        if primary_record is not None:
            return self._scip_artifact_payload(primary_record)
        artifacts = self.list_scip_artifact_refs(snapshot_id)
        return artifacts[0] if artifacts else None

    def list_scip_artifact_refs(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self.db_runtime.connect() as conn:
            self._ensure_scip_artifact_entries_table(conn)
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM scip_artifact_entries
                WHERE snapshot_id=?
                ORDER BY sequence_no ASC
                """,
                (snapshot_id,),
            ).fetchall()
        artifact_records = [
            record
            for row in rows
            if (record := self._row_to_scip_artifact_entry_record(row)) is not None
        ]
        if artifact_records:
            return [
                self._scip_artifact_entry_payload(record) for record in artifact_records
            ]
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT * FROM scip_artifacts WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
        primary_record = self._row_to_scip_artifact_record(row)
        return [self._scip_artifact_payload(primary_record)] if primary_record else []

    @classmethod
    def _scip_metadata_mapping(cls, artifact: dict[str, Any]) -> dict[str, Any]:
        raw_metadata = artifact.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            return {}
        return {str(key): value for key, value in raw_metadata.items()}

    @classmethod
    def _json_safe_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe_value(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [cls._json_safe_value(item) for item in value]
        if isinstance(value, set):
            return [cls._json_safe_value(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    @classmethod
    def _serialize_scip_metadata_json(cls, metadata: dict[str, Any]) -> str:
        return json.dumps(
            cls._json_safe_value(metadata),
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _deserialize_scip_metadata_json(raw_metadata: str | None) -> dict[str, Any]:
        if raw_metadata is None:
            return {}
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError):
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _row_to_scip_artifact_record(self, row: Any) -> SCIPArtifactRecord | None:
        snapshot_id = self._row_value(row, 0, "snapshot_id")
        if snapshot_id is None:
            return None
        return SCIPArtifactRecord(
            snapshot_id=str(snapshot_id),
            indexer_name=str(self._row_value(row, 1, "indexer_name") or ""),
            indexer_version=(
                str(indexer_version)
                if (indexer_version := self._row_value(row, 2, "indexer_version"))
                is not None
                else None
            ),
            artifact_path=str(self._row_value(row, 3, "artifact_path") or ""),
            checksum=str(self._row_value(row, 4, "checksum") or ""),
            created_at=str(self._row_value(row, 5, "created_at") or ""),
        )

    def _row_to_scip_artifact_entry_record(self, row: Any) -> SCIPArtifactRecord | None:
        snapshot_id = self._row_value(row, 1, "snapshot_id")
        if snapshot_id is None:
            return None
        raw_sequence_no = self._row_value(row, 2, "sequence_no")
        return SCIPArtifactRecord(
            artifact_id=(
                str(artifact_id)
                if (artifact_id := self._row_value(row, 0, "artifact_id")) is not None
                else None
            ),
            snapshot_id=str(snapshot_id),
            sequence_no=int(raw_sequence_no) if raw_sequence_no is not None else None,
            role=(
                str(role)
                if (role := self._row_value(row, 3, "role")) is not None
                else None
            ),
            indexer_name=str(self._row_value(row, 4, "indexer_name") or ""),
            indexer_version=(
                str(indexer_version)
                if (indexer_version := self._row_value(row, 5, "indexer_version"))
                is not None
                else None
            ),
            artifact_path=str(self._row_value(row, 6, "artifact_path") or ""),
            checksum=str(self._row_value(row, 7, "checksum") or ""),
            created_at=str(self._row_value(row, 8, "created_at") or ""),
            metadata_json=(
                str(metadata_json)
                if (metadata_json := self._row_value(row, 9, "metadata_json"))
                is not None
                else None
            ),
        )

    @staticmethod
    def _scip_artifact_payload(record: SCIPArtifactRecord) -> dict[str, Any]:
        return {
            "snapshot_id": record.snapshot_id,
            "indexer_name": record.indexer_name,
            "indexer_version": record.indexer_version,
            "artifact_path": record.artifact_path,
            "checksum": record.checksum,
            "created_at": record.created_at,
        }

    @classmethod
    def _scip_artifact_entry_payload(cls, record: SCIPArtifactRecord) -> dict[str, Any]:
        payload = cls._scip_artifact_payload(record)
        payload.update(
            {
                "artifact_id": record.artifact_id,
                "sequence_no": record.sequence_no,
                "role": record.role,
                "metadata": cls._deserialize_scip_metadata_json(record.metadata_json),
            }
        )
        return payload

    def _row_to_snapshot_record(self, row: Any) -> SnapshotRecord | None:
        snapshot_id = self._row_value(row, 0, "snapshot_id")
        if snapshot_id is None:
            return None
        return SnapshotRecord(
            snapshot_id=str(snapshot_id),
            repo_name=str(self._row_value(row, 1, "repo_name") or ""),
            branch=(
                str(branch)
                if (branch := self._row_value(row, 2, "branch")) is not None
                else None
            ),
            commit_id=(
                str(commit_id)
                if (commit_id := self._row_value(row, 3, "commit_id")) is not None
                else None
            ),
            tree_id=(
                str(tree_id)
                if (tree_id := self._row_value(row, 4, "tree_id")) is not None
                else None
            ),
            artifact_key=str(self._row_value(row, 5, "artifact_key") or ""),
            ir_path=str(self._row_value(row, 6, "ir_path") or ""),
            ir_graphs_path=(
                str(ir_graphs_path)
                if (ir_graphs_path := self._row_value(row, 7, "ir_graphs_path"))
                is not None
                else None
            ),
            created_at=str(self._row_value(row, 8, "created_at") or ""),
            metadata_json=(
                str(metadata_json)
                if (metadata_json := self._row_value(row, 9, "metadata_json"))
                is not None
                else None
            ),
        )

    def _row_to_snapshot_ref_record(self, row: Any) -> SnapshotRefRecord | None:
        snapshot_id = self._row_value(row, 5, "snapshot_id")
        if snapshot_id is None:
            return None
        ref_id_value = self._row_value(row, 0, "ref_id")
        return SnapshotRefRecord(
            ref_id=int(ref_id_value) if ref_id_value is not None else None,
            repo_name=str(self._row_value(row, 1, "repo_name") or ""),
            branch=(
                str(branch)
                if (branch := self._row_value(row, 2, "branch")) is not None
                else None
            ),
            commit_id=(
                str(commit_id)
                if (commit_id := self._row_value(row, 3, "commit_id")) is not None
                else None
            ),
            tree_id=(
                str(tree_id)
                if (tree_id := self._row_value(row, 4, "tree_id")) is not None
                else None
            ),
            snapshot_id=str(snapshot_id),
            created_at=str(self._row_value(row, 6, "created_at") or ""),
        )

    @staticmethod
    def _row_value(row: Any, index: int, key: str) -> Any:
        if row is None:
            return None
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except (IndexError, KeyError, TypeError):
            try:
                return row[index]
            except (IndexError, KeyError, TypeError):
                return None

    @classmethod
    def _snapshot_payload(cls, record: SnapshotRecord) -> dict[str, Any]:
        return {
            field_name: getattr(record, field_name)
            for field_name in cls._SNAPSHOT_FIELDS
        }

    @classmethod
    def _snapshot_ref_payload(cls, record: SnapshotRefRecord) -> dict[str, Any]:
        return {
            field_name: getattr(record, field_name)
            for field_name in cls._SNAPSHOT_REF_FIELDS
        }

    def import_git_backbone(
        self, snapshot: IRSnapshot, git_meta: dict[str, Any] | None = None
    ) -> None:
        if self.db_runtime.backend != "postgres":
            return
        git_meta = git_meta or {}
        repo_id = snapshot.repo_name
        now = utc_now()
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO repositories (repo_id, repo_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET repo_name=excluded.repo_name
                """,
                (repo_id, snapshot.repo_name, now),
            )
            if snapshot.commit_id:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO git_commits (repo_id, commit_id, tree_id, parent_commit_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(repo_id, commit_id) DO UPDATE SET
                        tree_id=excluded.tree_id,
                        parent_commit_id=excluded.parent_commit_id
                    """,
                    (
                        repo_id,
                        snapshot.commit_id,
                        snapshot.tree_id,
                        git_meta.get("parent_commit_id"),
                        now,
                    ),
                )
            if snapshot.tree_id:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO git_trees (repo_id, tree_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(repo_id, tree_id) DO NOTHING
                    """,
                    (repo_id, snapshot.tree_id, now),
                )
            if snapshot.branch:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO git_refs (repo_id, ref_name, commit_id, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(repo_id, ref_name) DO UPDATE SET
                        commit_id=excluded.commit_id,
                        updated_at=excluded.updated_at
                    """,
                    (repo_id, snapshot.branch, snapshot.commit_id, now),
                )
            conn.commit()

    def save_relational_facts(self, snapshot: IRSnapshot) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM snapshot_documents WHERE snapshot_id=?",
                (snapshot.snapshot_id,),
            )
            self.db_runtime.execute(
                conn, "DELETE FROM symbols WHERE snapshot_id=?", (snapshot.snapshot_id,)
            )
            self.db_runtime.execute(
                conn,
                "DELETE FROM occurrences WHERE snapshot_id=?",
                (snapshot.snapshot_id,),
            )
            self.db_runtime.execute(
                conn, "DELETE FROM edges WHERE snapshot_id=?", (snapshot.snapshot_id,)
            )
            self.db_runtime.execute(
                conn,
                "DELETE FROM attachments WHERE snapshot_id=?",
                (snapshot.snapshot_id,),
            )
            document_rows = [
                (
                    snapshot.snapshot_id,
                    doc.doc_id,
                    doc.path,
                    doc.language,
                    json.dumps(self._document_payload(doc), ensure_ascii=False),
                )
                for doc in snapshot.documents
            ]
            if document_rows:
                self.db_runtime.executemany(
                    conn,
                    """
                    INSERT INTO snapshot_documents (snapshot_id, doc_id, path, language, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    document_rows,
                )
            symbol_rows = [
                (
                    snapshot.snapshot_id,
                    sym.symbol_id,
                    sym.path,
                    sym.display_name,
                    sym.qualified_name,
                    sym.kind,
                    sym.language,
                    sym.source_priority,
                    json.dumps(self._symbol_payload(sym), ensure_ascii=False),
                )
                for sym in snapshot.symbols
            ]
            if symbol_rows:
                self.db_runtime.executemany(
                    conn,
                    """
                    INSERT INTO symbols (
                        snapshot_id, symbol_id, path, display_name, qualified_name, kind,
                        language, source_priority, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    symbol_rows,
                )
            occurrence_rows = [
                (
                    snapshot.snapshot_id,
                    occ.occurrence_id,
                    occ.symbol_id,
                    occ.doc_id,
                    occ.role,
                    occ.start_line,
                    occ.start_col,
                    occ.end_line,
                    occ.end_col,
                    occ.source,
                    json.dumps(self._occurrence_payload(occ), ensure_ascii=False),
                )
                for occ in snapshot.occurrences
            ]
            if occurrence_rows:
                self.db_runtime.executemany(
                    conn,
                    """
                    INSERT INTO occurrences (
                        snapshot_id, occurrence_id, symbol_id, doc_id, role, start_line,
                        start_col, end_line, end_col, source, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    occurrence_rows,
                )
            seen_edge_ids: set[str] = set()
            edge_rows: list[tuple[Any, ...]] = []
            for edge in snapshot.edges:
                if edge.edge_id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge.edge_id)
                edge_rows.append(
                    (
                        snapshot.snapshot_id,
                        edge.edge_id,
                        edge.src_id,
                        edge.dst_id,
                        edge.edge_type,
                        edge.source,
                        edge.confidence,
                        edge.doc_id,
                        json.dumps(self._edge_payload(edge), ensure_ascii=False),
                    )
                )
            if edge_rows:
                self.db_runtime.executemany(
                    conn,
                    """
                    INSERT INTO edges (
                        snapshot_id, edge_id, src_id, dst_id, edge_type, source, confidence,
                        doc_id, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    edge_rows,
                )
            attachment_rows: list[tuple[Any, ...]] = []
            for attachment in snapshot.attachments:
                attachment_payload = self._attachment_payload(attachment)
                attachment_rows.append(
                    (
                        snapshot.snapshot_id,
                        attachment.attachment_id,
                        attachment.target_id,
                        attachment.target_type,
                        attachment.attachment_type,
                        attachment.source,
                        attachment.confidence,
                        json.dumps(attachment_payload["payload"], ensure_ascii=False),
                        json.dumps(attachment_payload["metadata"], ensure_ascii=False),
                    )
                )
            if attachment_rows:
                self.db_runtime.executemany(
                    conn,
                    """
                    INSERT INTO attachments (
                        snapshot_id, attachment_id, target_id, target_type, attachment_type,
                        source, confidence, payload_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    attachment_rows,
                )
            conn.commit()

    def stage_snapshot(
        self, snapshot: IRSnapshot, metadata: dict[str, Any] | None = None
    ) -> str:
        stage_id = f"stage_{uuid.uuid4().hex[:16]}"
        if self.db_runtime.backend != "postgres":
            return stage_id
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO snapshot_staging (stage_id, snapshot_id, status, metadata_json, created_at)
                VALUES (?, ?, 'staged', ?, ?)
                ON CONFLICT(stage_id) DO NOTHING
                """,
                (
                    stage_id,
                    snapshot.snapshot_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )
            conn.commit()
        return stage_id

    def promote_staged_snapshot(self, snapshot_id: str, stage_id: str) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE snapshot_staging
                SET status='published', promoted_at=?
                WHERE stage_id=? AND snapshot_id=?
                """,
                (utc_now(), stage_id, snapshot_id),
            )
            conn.commit()

    def acquire_lock(
        self, lock_name: str, owner_id: str, ttl_seconds: int = 300
    ) -> int | None:
        if self.db_runtime.backend != "postgres":
            return 1
        now = datetime.now(UTC)
        expires_at = now.timestamp() + ttl_seconds
        expires_iso = datetime.fromtimestamp(expires_at, tz=UTC).isoformat()
        now_iso = now.isoformat()
        with self.db_runtime.connect() as conn:
            # Atomic compare-and-swap: INSERT or UPDATE only if expired or same owner
            row = self.db_runtime.execute(
                conn,
                """
                INSERT INTO resource_locks (lock_name, owner_id, expires_at, updated_at, fencing_token)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at,
                    fencing_token=CASE
                        WHEN resource_locks.owner_id = excluded.owner_id
                        THEN resource_locks.fencing_token
                        ELSE resource_locks.fencing_token + 1
                    END
                WHERE resource_locks.expires_at < ? OR resource_locks.owner_id = ?
                RETURNING fencing_token
                """,
                (lock_name, owner_id, expires_iso, utc_now(), now_iso, owner_id),
            ).fetchone()
            conn.commit()
        if row is None:
            return None
        return int(row["fencing_token"])

    def validate_fencing_token(self, lock_name: str, expected_token: int) -> bool:
        if self.db_runtime.backend != "postgres":
            return True
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT fencing_token FROM resource_locks WHERE lock_name=?",
                (lock_name,),
            ).fetchone()
        if not row:
            return False
        return int(row.get("fencing_token") or 0) == int(expected_token)

    def release_lock(self, lock_name: str, owner_id: str) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                "DELETE FROM resource_locks WHERE lock_name=? AND owner_id=?",
                (lock_name, owner_id),
            )
            conn.commit()

    def enqueue_redo_task(
        self, task_type: str, payload: dict[str, Any], error: str | None = None
    ) -> str:
        task_id = f"redo_{uuid.uuid4().hex[:16]}"
        if self.db_runtime.backend != "postgres":
            return task_id
        now = utc_now()
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO redo_tasks (
                    task_id, task_type, payload_json, status, attempts, last_error, next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    task_type,
                    json.dumps(payload, ensure_ascii=False),
                    error,
                    now,
                    now,
                    now,
                ),
            )
            conn.commit()
        return task_id

    def save_design_documents(
        self,
        snapshot_id: str,
        repo_name: str,
        chunks: list[dict[str, Any]],
        mentions: list[dict[str, Any]],
    ) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn, "DELETE FROM design_documents WHERE snapshot_id=?", (snapshot_id,)
            )
            self.db_runtime.execute(
                conn,
                "DELETE FROM design_doc_mentions WHERE snapshot_id=?",
                (snapshot_id,),
            )
            for chunk in chunks:
                if not chunk.get("chunk_id"):
                    logging.getLogger(__name__).warning(
                        "Skipping design document chunk without chunk_id: %s",
                        chunk.get("path", "<unknown>"),
                    )
                    continue
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO design_documents (
                        snapshot_id, chunk_id, repo_name, path, title, heading, doc_type, content, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        chunk.get("chunk_id"),
                        repo_name,
                        chunk.get("path"),
                        chunk.get("title"),
                        chunk.get("heading"),
                        chunk.get("doc_type"),
                        chunk.get("content", ""),
                        json.dumps(chunk, ensure_ascii=False),
                    ),
                )
            for mention in mentions:
                self.db_runtime.execute(
                    conn,
                    """
                    INSERT INTO design_doc_mentions (
                        snapshot_id, chunk_id, symbol_id, symbol_name, confidence, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(snapshot_id, chunk_id, symbol_id) DO UPDATE SET
                        symbol_name=excluded.symbol_name,
                        confidence=excluded.confidence,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        snapshot_id,
                        mention.get("chunk_id"),
                        mention.get("symbol_id"),
                        mention.get("symbol_name"),
                        mention.get("confidence"),
                        json.dumps(mention, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def get_doc_mentions(self, snapshot_id: str) -> list[dict[str, Any]]:
        """Return all design doc mentions for a snapshot."""
        with self.db_runtime.connect() as conn:
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT chunk_id, symbol_id, symbol_name, confidence, metadata_json
                FROM design_doc_mentions
                WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            entry = {
                "chunk_id": r["chunk_id"],
                "symbol_id": r["symbol_id"],
                "symbol_name": r["symbol_name"],
                "confidence": r["confidence"],
            }
            metadata_json = r["metadata_json"]
            if metadata_json:
                try:
                    meta = json.loads(metadata_json)
                    entry.update({k: v for k, v in meta.items() if k not in entry})
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(entry)
        return results

    def claim_redo_task(self) -> dict[str, Any] | None:
        if self.db_runtime.backend != "postgres":
            return None
        now = utc_now()
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM redo_tasks
                WHERE status='pending'
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if not row:
                conn.commit()
                return None
            task_record = self._row_to_redo_task_record(row)
            if task_record is None:
                conn.commit()
                return None
            self.db_runtime.execute(
                conn,
                """
                UPDATE redo_tasks
                SET status='running', attempts=attempts+1, updated_at=?
                WHERE task_id=?
                """,
                (now, task_record.task_id),
            )
            conn.commit()
        task = self._redo_task_payload(task_record)
        task["status"] = "running"
        task["attempts"] = task_record.attempts + 1
        task["updated_at"] = now
        return task

    def mark_redo_task_done(self, task_id: str) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE redo_tasks
                SET status='completed', updated_at=?
                WHERE task_id=?
                """,
                (utc_now(), task_id),
            )
            conn.commit()

    def mark_redo_task_failed(
        self, task_id: str, error: str, max_attempts: int = 5
    ) -> None:
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT attempts FROM redo_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            attempts = int(self._row_value(row, 0, "attempts") or 0)
            if attempts >= max_attempts:
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE redo_tasks
                    SET status='dead', last_error=?, updated_at=?
                    WHERE task_id=?
                    """,
                    (error, utc_now(), task_id),
                )
            else:
                backoff_seconds = max(1, 2**attempts)
                next_attempt_at = (
                    datetime.now(UTC) + timedelta(seconds=backoff_seconds)
                ).isoformat()
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE redo_tasks
                    SET status='pending', last_error=?, next_attempt_at=?, updated_at=?
                    WHERE task_id=?
                    """,
                    (error, next_attempt_at, utc_now(), task_id),
                )
            conn.commit()

    # --- Publish outbox methods ---

    def enqueue_outbox_event(
        self,
        event_id: str,
        event_type: str,
        payload: str,
        snapshot_id: str,
        max_attempts: int = 5,
    ) -> bool:
        """Insert a publish event into the outbox. Returns True if inserted, False if duplicate."""
        if self.db_runtime.backend != "postgres":
            return False
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                INSERT INTO publish_outbox (
                    event_id, event_type, payload, snapshot_id, status,
                    attempts, max_attempts, created_at
                ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (event_id, event_type, payload, snapshot_id, max_attempts, utc_now()),
            )
            conn.commit()
        return True

    def claim_outbox_event(self, limit: int = 10) -> list[dict[str, Any]]:
        """Claim pending or retryable failed events from the outbox."""
        if self.db_runtime.backend != "postgres":
            return []
        now = utc_now()
        with self.db_runtime.connect() as conn:
            rows = self.db_runtime.execute(
                conn,
                """
                SELECT * FROM publish_outbox
                WHERE status = 'pending'
                   OR (status = 'failed' AND attempts < max_attempts)
                ORDER BY created_at ASC
                LIMIT ?
                FOR UPDATE SKIP LOCKED
                """,
                (limit,),
            ).fetchall()
            if not rows:
                conn.commit()
                return []
            claimed: list[dict[str, Any]] = []
            for row in rows:
                event_record = self._row_to_outbox_event_record(row)
                if event_record is None:
                    continue
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE publish_outbox
                    SET status = 'in_progress', last_attempt_at = ?
                    WHERE event_id = ?
                    """,
                    (now, event_record.event_id),
                )
                event = self._outbox_event_payload(event_record)
                event["status"] = "in_progress"
                event["last_attempt_at"] = now
                claimed.append(event)
            conn.commit()
        return claimed

    def mark_outbox_event_done(self, event_id: str) -> None:
        """Mark an outbox event as published."""
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            self.db_runtime.execute(
                conn,
                """
                UPDATE publish_outbox
                SET status = 'published'
                WHERE event_id = ?
                """,
                (event_id,),
            )
            conn.commit()

    def mark_outbox_event_failed(self, event_id: str, error: str) -> None:
        """Mark an outbox event as failed, incrementing attempts."""
        if self.db_runtime.backend != "postgres":
            return
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                "SELECT attempts, max_attempts FROM publish_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            attempts = int(self._row_value(row, 0, "attempts") or 0) + 1
            max_attempts = int(self._row_value(row, 1, "max_attempts") or 5)
            if attempts >= max_attempts:
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE publish_outbox
                    SET status = 'dead', attempts = ?, error_message = ?
                    WHERE event_id = ?
                    """,
                    (attempts, error, event_id),
                )
            else:
                self.db_runtime.execute(
                    conn,
                    """
                    UPDATE publish_outbox
                    SET status = 'failed', attempts = ?, error_message = ?
                    WHERE event_id = ?
                    """,
                    (attempts, error, event_id),
                )
            conn.commit()

    def get_outbox_pending_count(self) -> int:
        """Return count of pending + retryable failed events."""
        if self.db_runtime.backend != "postgres":
            return 0
        with self.db_runtime.connect() as conn:
            row = self.db_runtime.execute(
                conn,
                """
                SELECT COUNT(*) AS cnt FROM publish_outbox
                WHERE status = 'pending'
                   OR (status = 'failed' AND attempts < max_attempts)
                """,
            ).fetchone()
        return int(self._row_value(row, 0, "cnt") or 0)

    def _row_to_redo_task_record(self, row: Any) -> RedoTaskRecord | None:
        task_id = self._row_value(row, 0, "task_id")
        if task_id is None:
            return None
        return RedoTaskRecord(
            task_id=str(task_id),
            task_type=str(self._row_value(row, 1, "task_type") or ""),
            payload_json=str(self._row_value(row, 2, "payload_json") or ""),
            status=str(self._row_value(row, 3, "status") or ""),
            attempts=int(self._row_value(row, 4, "attempts") or 0),
            last_error=(
                str(last_error)
                if (last_error := self._row_value(row, 5, "last_error")) is not None
                else None
            ),
            next_attempt_at=(
                str(next_attempt_at)
                if (next_attempt_at := self._row_value(row, 6, "next_attempt_at"))
                is not None
                else None
            ),
            created_at=str(self._row_value(row, 7, "created_at") or ""),
            updated_at=str(self._row_value(row, 8, "updated_at") or ""),
        )

    @staticmethod
    def _redo_task_payload(record: RedoTaskRecord) -> dict[str, Any]:
        return {
            "task_id": record.task_id,
            "task_type": record.task_type,
            "payload_json": record.payload_json,
            "status": record.status,
            "attempts": record.attempts,
            "last_error": record.last_error,
            "next_attempt_at": record.next_attempt_at,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def _row_to_outbox_event_record(self, row: Any) -> OutboxEventRecord | None:
        event_id = self._row_value(row, 0, "event_id")
        if event_id is None:
            return None
        return OutboxEventRecord(
            event_id=str(event_id),
            event_type=str(self._row_value(row, 1, "event_type") or ""),
            payload=str(self._row_value(row, 2, "payload") or ""),
            snapshot_id=str(self._row_value(row, 3, "snapshot_id") or ""),
            status=str(self._row_value(row, 4, "status") or ""),
            attempts=int(self._row_value(row, 5, "attempts") or 0),
            max_attempts=int(self._row_value(row, 6, "max_attempts") or 0),
            created_at=str(self._row_value(row, 7, "created_at") or ""),
            last_attempt_at=(
                str(last_attempt_at)
                if (last_attempt_at := self._row_value(row, 8, "last_attempt_at"))
                is not None
                else None
            ),
            error_message=(
                str(error_message)
                if (error_message := self._row_value(row, 9, "error_message"))
                is not None
                else None
            ),
        )

    @staticmethod
    def _outbox_event_payload(record: OutboxEventRecord) -> dict[str, Any]:
        return {
            "event_id": record.event_id,
            "event_type": record.event_type,
            "payload": record.payload,
            "snapshot_id": record.snapshot_id,
            "status": record.status,
            "attempts": record.attempts,
            "max_attempts": record.max_attempts,
            "created_at": record.created_at,
            "last_attempt_at": record.last_attempt_at,
            "error_message": record.error_message,
        }
