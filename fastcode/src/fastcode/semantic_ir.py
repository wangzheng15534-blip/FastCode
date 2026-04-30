"""
Canonical semantic IR models centered on code units and grounded supports.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import safe_jsonable

_T = dict[str, Any]

RESOLUTION_CANDIDATE = "candidate"
RESOLUTION_STRUCTURAL = "structural"
RESOLUTION_ANCHORED = "anchored"
RESOLUTION_SEMANTIC = "semantic"
RESOLUTION_SEMANTICALLY_RESOLVED = "semantically_resolved"


def resolution_rank(value: str) -> int:
    return {
        "candidate": 0,
        "structural": 1,
        "anchored": 2,
        "semantic": 3,
        "semantically_resolved": 3,
    }.get(value, 0)


def _sorted_set(values: set[str]) -> list[str]:
    return sorted(v for v in values if v)


def _normalize_set(values: Any) -> set[str]:
    if not values:
        return set()
    return {str(v) for v in values if v}


def _copy_dict(data: dict[str, Any]) -> _T:
    """Copy a dict while preserving the key type as str."""
    result: _T = {}
    for k, v in data.items():
        result[str(k)] = v
    return result


def _resolution_to_confidence(resolution_state: str) -> str:
    return {
        "anchored": "precise",
        "semantic": "precise",
        "semantically_resolved": "precise",
        "structural": "resolved",
        "candidate": "heuristic",
    }.get(resolution_state or "", "derived")


def _confidence_to_resolution(confidence: str) -> str:
    return {
        "precise": "anchored",
        "resolved": "structural",
        "derived": "structural",
        "heuristic": "candidate",
    }.get(confidence or "", "structural")


def _unit_kind_to_symbol_kind(kind: str) -> str:
    return "documentation" if kind == "doc" else kind


def _symbol_kind_to_unit_kind(kind: str) -> str:
    return "doc" if kind == "documentation" else kind


@dataclass
class IRDocument:
    doc_id: str
    path: str
    language: str
    blob_oid: str | None = None
    content_hash: str | None = None
    source_set: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_set"] = _sorted_set(self.source_set)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRDocument:
        payload = _copy_dict(data)
        payload["source_set"] = _normalize_set(payload.get("source_set"))
        return cls(**payload)


@dataclass
class IRSymbol:
    symbol_id: str
    external_symbol_id: str | None
    path: str
    display_name: str
    kind: str
    language: str
    qualified_name: str | None = None
    signature: str | None = None
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    source_priority: int = 0
    source_set: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_set"] = _sorted_set(self.source_set)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRSymbol:
        payload = _copy_dict(data)
        payload["source_set"] = _normalize_set(payload.get("source_set"))
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


@dataclass
class IROccurrence:
    occurrence_id: str
    symbol_id: str
    doc_id: str
    role: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IROccurrence:
        payload = _copy_dict(data)
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


@dataclass
class IREdge:
    edge_id: str
    src_id: str
    dst_id: str
    edge_type: str
    source: str
    confidence: str
    doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IREdge:
        payload = _copy_dict(data)
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


@dataclass
class IRAttachment:
    attachment_id: str
    target_id: str
    target_type: str
    attachment_type: str
    source: str
    confidence: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["payload"] = safe_jsonable(self.payload)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRAttachment:
        payload = _copy_dict(data)
        payload["payload"] = safe_jsonable(payload.get("payload", {}))
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


@dataclass
class IRCodeUnit:
    unit_id: str
    kind: str
    path: str
    language: str
    display_name: str
    qualified_name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    summary: str | None = None
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    parent_unit_id: str | None = None
    primary_anchor_symbol_id: str | None = None
    anchor_symbol_ids: list[str] = field(default_factory=list)
    candidate_anchor_symbol_ids: list[str] = field(default_factory=list)
    anchor_coverage: float = 0.0
    source_set: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.unit_id

    @property
    def symbol_id(self) -> str:
        return self.unit_id

    @property
    def external_symbol_id(self) -> str | None:
        return self.primary_anchor_symbol_id

    @property
    def source_priority(self) -> int:
        if "scip" in self.source_set and "fc_structure" in self.source_set:
            return 100
        if "scip" in self.source_set:
            return 100
        if "fc_structure" in self.source_set:
            return 50
        return 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_set"] = _sorted_set(self.source_set)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRCodeUnit:
        payload = _copy_dict(data)
        payload["source_set"] = _normalize_set(payload.get("source_set"))
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        payload["anchor_symbol_ids"] = [
            str(v) for v in payload.get("anchor_symbol_ids", []) if v
        ]
        payload["candidate_anchor_symbol_ids"] = [
            str(v) for v in payload.get("candidate_anchor_symbol_ids", []) if v
        ]
        return cls(**payload)


@dataclass
class IRUnitSupport:
    support_id: str
    unit_id: str
    source: str
    support_kind: str
    external_id: str | None = None
    role: str | None = None
    path: str | None = None
    display_name: str | None = None
    qualified_name: str | None = None
    signature: str | None = None
    enclosing_external_id: str | None = None
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRUnitSupport:
        payload = _copy_dict(data)
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


@dataclass
class IRRelation:
    relation_id: str
    src_unit_id: str
    dst_unit_id: str
    relation_type: str
    resolution_state: str
    support_sources: set[str] = field(default_factory=set)
    support_ids: list[str] = field(default_factory=list)
    pending_capabilities: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def edge_id(self) -> str:
        return self.relation_id

    @property
    def src_id(self) -> str:
        return self.src_unit_id

    @property
    def dst_id(self) -> str:
        return self.dst_unit_id

    @property
    def edge_type(self) -> str:
        return self.relation_type

    @property
    def source(self) -> str:
        if self.support_sources:
            return sorted(self.support_sources)[0]
        return str((self.metadata or {}).get("source") or "")

    @property
    def confidence(self) -> str:
        return _resolution_to_confidence(self.resolution_state)

    @property
    def doc_id(self) -> str | None:
        doc_id = (self.metadata or {}).get("doc_id")
        return str(doc_id) if doc_id else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["support_sources"] = _sorted_set(self.support_sources)
        data["pending_capabilities"] = _sorted_set(self.pending_capabilities)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRRelation:
        payload = _copy_dict(data)
        payload["support_sources"] = _normalize_set(payload.get("support_sources"))
        payload["support_ids"] = [str(v) for v in payload.get("support_ids", []) if v]
        payload["pending_capabilities"] = _normalize_set(
            payload.get("pending_capabilities")
        )
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


@dataclass
class IRUnitEmbedding:
    embedding_id: str
    unit_id: str
    source: str
    vector: list[Any] | None = None
    embedding_text: str | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vector"] = safe_jsonable(self.vector)
        data["metadata"] = safe_jsonable(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRUnitEmbedding:
        payload = _copy_dict(data)
        payload["vector"] = safe_jsonable(payload.get("vector"))
        payload["metadata"] = safe_jsonable(payload.get("metadata", {}))
        return cls(**payload)


class IRSnapshot:
    def __init__(
        self,
        *,
        repo_name: str,
        snapshot_id: str,
        branch: str | None = None,
        commit_id: str | None = None,
        tree_id: str | None = None,
        units: list[IRCodeUnit] | None = None,
        supports: list[IRUnitSupport] | None = None,
        relations: list[IRRelation] | None = None,
        embeddings: list[IRUnitEmbedding] | None = None,
        metadata: dict[str, Any] | None = None,
        documents: list[IRDocument] | None = None,
        symbols: list[IRSymbol] | None = None,
        occurrences: list[IROccurrence] | None = None,
        edges: list[IREdge] | None = None,
        attachments: list[IRAttachment] | None = None,
    ) -> None:
        self.repo_name = repo_name
        self.snapshot_id = snapshot_id
        self.branch = branch
        self.commit_id = commit_id
        self.tree_id = tree_id
        self.metadata = safe_jsonable(metadata or {})

        canonical_units = list(units or [])
        canonical_supports = list(supports or [])
        canonical_relations = list(relations or [])
        canonical_embeddings = list(embeddings or [])

        if documents or symbols or occurrences or edges or attachments:
            converted = self._from_legacy_payload(
                documents=documents or [],
                symbols=symbols or [],
                occurrences=occurrences or [],
                edges=edges or [],
                attachments=attachments or [],
            )
            canonical_units.extend(converted["units"])
            canonical_supports.extend(converted["supports"])
            canonical_relations.extend(converted["relations"])
            canonical_embeddings.extend(converted["embeddings"])
            self.metadata.update(converted["metadata"])

        self.units = canonical_units
        self.supports = canonical_supports
        self.relations = canonical_relations
        self.embeddings = canonical_embeddings

    def _from_legacy_payload(
        self,
        *,
        documents: list[IRDocument],
        symbols: list[IRSymbol],
        occurrences: list[IROccurrence],
        edges: list[IREdge],
        attachments: list[IRAttachment],
    ) -> dict[str, Any]:
        units: list[IRCodeUnit] = []
        supports: list[IRUnitSupport] = []
        relations: list[IRRelation] = []
        embeddings: list[IRUnitEmbedding] = []
        unit_by_id: dict[str, IRCodeUnit] = {}
        file_ids_by_path: dict[str, str] = {}
        metadata: dict[str, Any] = {}

        for doc in documents:
            unit = IRCodeUnit(
                unit_id=doc.doc_id,
                kind="file",
                path=doc.path,
                language=doc.language,
                display_name=doc.path,
                source_set=set(doc.source_set),
                metadata={
                    "blob_oid": doc.blob_oid,
                    "content_hash": doc.content_hash,
                },
            )
            units.append(unit)
            unit_by_id[unit.unit_id] = unit
            file_ids_by_path[doc.path] = doc.doc_id

        for sym in symbols:
            unit = IRCodeUnit(
                unit_id=sym.symbol_id,
                kind=_symbol_kind_to_unit_kind(sym.kind),
                path=sym.path,
                language=sym.language,
                display_name=sym.display_name,
                qualified_name=sym.qualified_name,
                signature=sym.signature,
                start_line=sym.start_line,
                start_col=sym.start_col,
                end_line=sym.end_line,
                end_col=sym.end_col,
                primary_anchor_symbol_id=sym.external_symbol_id,
                anchor_symbol_ids=[sym.external_symbol_id]
                if sym.external_symbol_id
                else [],
                anchor_coverage=1.0 if sym.external_symbol_id else 0.0,
                source_set=set(sym.source_set),
                metadata=safe_jsonable(sym.metadata),
            )
            units.append(unit)
            unit_by_id[unit.unit_id] = unit

            doc_id = file_ids_by_path.get(sym.path)
            if doc_id:
                relations.append(
                    IRRelation(
                        relation_id=f"rel:contain:{doc_id}:{sym.symbol_id}",
                        src_unit_id=doc_id,
                        dst_unit_id=sym.symbol_id,
                        relation_type="contain",
                        resolution_state="structural",
                        support_sources=set(sym.source_set),
                        metadata={
                            "source": sorted(sym.source_set)[0]
                            if sym.source_set
                            else ""
                        },
                    )
                )

        for occ in occurrences:
            supports.append(
                IRUnitSupport(
                    support_id=occ.occurrence_id,
                    unit_id=occ.symbol_id,
                    source=occ.source,
                    support_kind="occurrence",
                    role=occ.role,
                    start_line=occ.start_line,
                    start_col=occ.start_col,
                    end_line=occ.end_line,
                    end_col=occ.end_col,
                    metadata=safe_jsonable(
                        {"doc_id": occ.doc_id, **(occ.metadata or {})}
                    ),
                )
            )

        for edge in edges:
            relations.append(
                IRRelation(
                    relation_id=edge.edge_id,
                    src_unit_id=edge.src_id,
                    dst_unit_id=edge.dst_id,
                    relation_type=edge.edge_type,
                    resolution_state=_confidence_to_resolution(edge.confidence),
                    support_sources={edge.source} if edge.source else set(),
                    metadata=safe_jsonable(
                        {
                            "doc_id": edge.doc_id,
                            **(edge.metadata or {}),
                            "source": edge.source,
                        }
                    ),
                )
            )

        snapshot_annotations: list[dict[str, Any]] = []
        for attachment in attachments:
            if (
                attachment.target_type == "symbol"
                and attachment.target_id in unit_by_id
            ):
                unit = unit_by_id[attachment.target_id]
                if attachment.attachment_type == "summary":
                    unit.summary = str(
                        (attachment.payload or {}).get("text") or unit.summary or ""
                    )
                elif attachment.attachment_type == "embedding":
                    embeddings.append(
                        IRUnitEmbedding(
                            embedding_id=attachment.attachment_id,
                            unit_id=attachment.target_id,
                            source=attachment.source,
                            vector=safe_jsonable(
                                (attachment.payload or {}).get("vector")
                            ),
                            embedding_text=(attachment.payload or {}).get("text"),
                            metadata=safe_jsonable(attachment.metadata),
                        )
                    )
                else:
                    unit.metadata.setdefault("annotations", []).append(
                        attachment.to_dict()
                    )
            elif attachment.target_type == "snapshot":
                snapshot_annotations.append(attachment.to_dict())
        if snapshot_annotations:
            metadata["snapshot_annotations"] = snapshot_annotations

        return {
            "units": units,
            "supports": supports,
            "relations": relations,
            "embeddings": embeddings,
            "metadata": metadata,
        }

    @property
    def documents(self) -> list[IRDocument]:
        docs: list[IRDocument] = []
        for unit in self.units:
            if unit.kind != "file":
                continue
            docs.append(
                IRDocument(
                    doc_id=unit.unit_id,
                    path=unit.path,
                    language=unit.language,
                    blob_oid=(unit.metadata or {}).get("blob_oid"),
                    content_hash=(unit.metadata or {}).get("content_hash"),
                    source_set=set(unit.source_set),
                )
            )
        return docs

    @property
    def symbols(self) -> list[IRSymbol]:
        symbols: list[IRSymbol] = []
        for unit in self.units:
            if unit.kind in {"file", "doc"}:
                continue
            metadata: dict[str, Any] = safe_jsonable(unit.metadata)
            raw_aliases: list[Any] = metadata.get("aliases") or []
            alias_ids: list[str] = list(
                dict.fromkeys(
                    [str(a) for a in raw_aliases] + unit.candidate_anchor_symbol_ids
                )
            )
            if alias_ids:
                metadata["aliases"] = alias_ids
            symbols.append(
                IRSymbol(
                    symbol_id=unit.unit_id,
                    external_symbol_id=unit.primary_anchor_symbol_id,
                    path=unit.path,
                    display_name=unit.display_name,
                    kind=_unit_kind_to_symbol_kind(unit.kind),
                    language=unit.language,
                    qualified_name=unit.qualified_name,
                    signature=unit.signature,
                    start_line=unit.start_line,
                    start_col=unit.start_col,
                    end_line=unit.end_line,
                    end_col=unit.end_col,
                    source_priority=unit.source_priority,
                    source_set=set(unit.source_set),
                    metadata=metadata,
                )
            )
        return symbols

    @property
    def occurrences(self) -> list[IROccurrence]:
        file_units = {
            unit.path: unit.unit_id for unit in self.units if unit.kind == "file"
        }
        logical: dict[tuple[str, str, str, int, int, int, int], IROccurrence] = {}
        occurrences: list[IROccurrence] = []
        for support in self.supports:
            if support.support_kind != "occurrence":
                continue
            meta = support.metadata or {}
            doc_id = str(meta.get("doc_id") or file_units.get(support.path or "", ""))
            if not doc_id:
                unit = next(
                    (u for u in self.units if u.unit_id == support.unit_id), None
                )
                if unit:
                    doc_id = file_units.get(unit.path, "")
            if not doc_id:
                continue
            occurrence = IROccurrence(
                occurrence_id=support.support_id,
                symbol_id=support.unit_id,
                doc_id=doc_id,
                role=support.role or "reference",
                start_line=int(support.start_line or 0),
                start_col=int(support.start_col or 0),
                end_line=int(support.end_line or 0),
                end_col=int(support.end_col or 0),
                source=support.source,
                metadata=safe_jsonable(meta),
            )
            key = (
                occurrence.symbol_id,
                occurrence.doc_id,
                occurrence.role,
                occurrence.start_line,
                occurrence.start_col,
                occurrence.end_line,
                occurrence.end_col,
            )
            existing = logical.get(key)
            if existing is None or (
                existing.source != "scip" and occurrence.source == "scip"
            ):
                logical[key] = occurrence
        occurrences.extend(logical.values())
        return occurrences

    @property
    def edges(self) -> list[IREdge]:
        edges: list[IREdge] = []
        for relation in self.relations:
            edges.append(
                IREdge(
                    edge_id=relation.relation_id,
                    src_id=relation.src_unit_id,
                    dst_id=relation.dst_unit_id,
                    edge_type=relation.relation_type,
                    source=relation.source,
                    confidence=relation.confidence,
                    doc_id=relation.doc_id,
                    metadata=safe_jsonable(relation.metadata),
                )
            )
        return edges

    @property
    def attachments(self) -> list[IRAttachment]:
        attachments: list[IRAttachment] = []
        for unit in self.units:
            if unit.summary and unit.kind != "file":
                attachments.append(
                    IRAttachment(
                        attachment_id=f"att:summary:{unit.unit_id}",
                        target_id=unit.unit_id,
                        target_type="symbol",
                        attachment_type="summary",
                        source="fc_structure"
                        if "fc_structure" in unit.source_set
                        else "derived",
                        confidence="derived",
                        payload={"text": unit.summary},
                        metadata={"unit_kind": unit.kind},
                    )
                )
        for embedding in self.embeddings:
            attachments.append(
                IRAttachment(
                    attachment_id=embedding.embedding_id,
                    target_id=embedding.unit_id,
                    target_type="symbol",
                    attachment_type="embedding",
                    source=embedding.source,
                    confidence="derived",
                    payload={
                        "vector": safe_jsonable(embedding.vector),
                        "text": embedding.embedding_text,
                    },
                    metadata=safe_jsonable(embedding.metadata),
                )
            )
        for annotation in self.metadata.get("snapshot_annotations", []) or []:
            ann_data: dict[str, Any] = (
                annotation if isinstance(annotation, dict) else {}
            )
            attachments.append(IRAttachment.from_dict(ann_data))
        return attachments

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ir.v2",
            "repo_name": self.repo_name,
            "snapshot_id": self.snapshot_id,
            "branch": self.branch,
            "commit_id": self.commit_id,
            "tree_id": self.tree_id,
            "units": [unit.to_dict() for unit in self.units],
            "supports": [support.to_dict() for support in self.supports],
            "relations": [relation.to_dict() for relation in self.relations],
            "embeddings": [embedding.to_dict() for embedding in self.embeddings],
            "metadata": safe_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRSnapshot:
        if data.get("units") is not None or data.get("supports") is not None:
            return cls(
                repo_name=data["repo_name"],
                snapshot_id=data["snapshot_id"],
                branch=data.get("branch"),
                commit_id=data.get("commit_id"),
                tree_id=data.get("tree_id"),
                units=[IRCodeUnit.from_dict(v) for v in data.get("units", [])],
                supports=[IRUnitSupport.from_dict(v) for v in data.get("supports", [])],
                relations=[IRRelation.from_dict(v) for v in data.get("relations", [])],
                embeddings=[
                    IRUnitEmbedding.from_dict(v) for v in data.get("embeddings", [])
                ],
                metadata=safe_jsonable(data.get("metadata", {})),
            )
        return cls(
            repo_name=data["repo_name"],
            snapshot_id=data["snapshot_id"],
            branch=data.get("branch"),
            commit_id=data.get("commit_id"),
            tree_id=data.get("tree_id"),
            documents=[IRDocument.from_dict(v) for v in data.get("documents", [])],
            symbols=[IRSymbol.from_dict(v) for v in data.get("symbols", [])],
            occurrences=[
                IROccurrence.from_dict(v) for v in data.get("occurrences", [])
            ],
            edges=[IREdge.from_dict(v) for v in data.get("edges", [])],
            attachments=[
                IRAttachment.from_dict(v) for v in data.get("attachments", [])
            ],
            metadata=safe_jsonable(data.get("metadata", {})),
        )
