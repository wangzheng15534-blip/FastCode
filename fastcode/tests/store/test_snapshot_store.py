"""Tests for snapshot_store module."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import fastcode.store.snapshot as snapshot_module
from fastcode.ir.graph import IRGraphBuilder, IRGraphs
from fastcode.ir.types import (
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
from fastcode.scip.models import SCIPArtifactRef
from fastcode.store.infrastructure.runtime import DBRuntime
from fastcode.store.snapshot import SnapshotStore
from fastcode.store.snapshot_contracts import (
    OutboxEventRecord,
    RedoTaskRecord,
    SCIPArtifactRecord,
    SnapshotRecord,
    SnapshotRefRecord,
)

# --- Strategies ---

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

file_path_st = st.tuples(identifier, identifier).map(lambda t: f"{t[0]}/{t[1]}.py")

role_st = st.sampled_from(["definition", "reference", "import", "implementation"])

edge_type_st = st.sampled_from(
    ["dependency", "call", "inheritance", "reference", "contain"]
)

source_st = st.sampled_from(["ast", "fc_structure", "scip"])
attachment_source_st = st.sampled_from(
    ["fc_structure", "fc_embedding", "llm_annotation"]
)

kind_st = st.sampled_from(
    [
        "function",
        "method",
        "class",
        "variable",
        "module",
        "interface",
        "enum",
        "constant",
    ]
)

language_st = st.sampled_from(
    ["python", "javascript", "typescript", "go", "java", "rust", "c", "cpp"]
)

line_number_st = st.integers(min_value=1, max_value=10000)

ir_document_st = st.builds(
    IRDocument,
    doc_id=st.builds(lambda x: f"doc:{x}", identifier),
    path=file_path_st,
    language=language_st,
    blob_oid=st.none() | st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    content_hash=st.none()
    | st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    source_set=st.sets(source_st, max_size=2),
)

ir_symbol_st = st.builds(
    IRSymbol,
    symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
    external_symbol_id=st.none() | identifier,
    path=file_path_st,
    display_name=identifier,
    kind=kind_st,
    language=language_st,
    qualified_name=st.none() | st.builds(lambda x: f"pkg.{x}", identifier),
    signature=st.none() | st.just("def foo(x: int) -> str"),
    start_line=st.none() | line_number_st,
    start_col=st.none() | st.integers(min_value=0, max_value=120),
    end_line=st.none() | line_number_st,
    end_col=st.none() | st.integers(min_value=0, max_value=120),
    source_priority=st.integers(min_value=0, max_value=200),
    source_set=st.sets(source_st, max_size=2),
    metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
)

ir_occurrence_st = st.builds(
    IROccurrence,
    occurrence_id=st.builds(lambda x: f"occ:{x}", identifier),
    symbol_id=st.builds(lambda x: f"sym:{x}", identifier),
    doc_id=st.builds(lambda x: f"doc:{x}", identifier),
    role=role_st,
    start_line=line_number_st,
    start_col=st.integers(min_value=0, max_value=120),
    end_line=line_number_st,
    end_col=st.integers(min_value=0, max_value=120),
    source=source_st,
    metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
)

ir_edge_st = st.builds(
    IREdge,
    edge_id=st.builds(lambda x: f"edge:{x}", identifier),
    src_id=identifier,
    dst_id=identifier,
    edge_type=edge_type_st,
    source=source_st,
    confidence=st.sampled_from(["precise", "heuristic", "resolved", ""]),
    doc_id=st.none() | st.builds(lambda x: f"doc:{x}", identifier),
    metadata=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()),
)

ir_attachment_st = st.builds(
    IRAttachment,
    attachment_id=st.builds(lambda x: f"att:{x}", identifier),
    target_id=st.builds(lambda x: f"sym:{x}", identifier),
    target_type=st.sampled_from(["document", "symbol", "snapshot"]),
    attachment_type=st.sampled_from(["embedding", "summary", "semantic_note"]),
    source=attachment_source_st,
    confidence=st.sampled_from(["derived", "precise", "heuristic"]),
    payload=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.integers(),
            st.text(min_size=0, max_size=20),
            st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=4),
        ),
        max_size=3,
    ),
    metadata=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(st.integers(), st.text(min_size=0, max_size=20)),
        max_size=3,
    ),
)


def snapshot_st(
    max_docs: int = 3,
    max_syms: int = 5,
    max_occs: int = 8,
    max_edges: int = 4,
) -> st.SearchStrategy[IRSnapshot]:
    """Build an IRSnapshot strategy with controlled size."""
    return st.builds(
        IRSnapshot,
        repo_name=identifier,
        snapshot_id=st.builds(lambda x: f"snap:{x}", identifier),
        branch=st.none() | st.just("main"),
        commit_id=st.none()
        | st.text(alphabet="0123456789abcdef", min_size=7, max_size=40),
        tree_id=st.none() | identifier,
        documents=st.lists(ir_document_st, max_size=max_docs),
        symbols=st.lists(ir_symbol_st, max_size=max_syms),
        occurrences=st.lists(ir_occurrence_st, max_size=max_occs),
        edges=st.lists(ir_edge_st, max_size=max_edges),
        attachments=st.lists(ir_attachment_st, max_size=4),
        metadata=st.dictionaries(
            st.sampled_from(["source_modes", "version", "tool"]),
            st.one_of(
                st.just(["ast"]), st.just(["scip"]), st.just(1), st.just("fastcode")
            ),
        ),
    )


@st.composite
def connected_snapshot_st(
    draw: Any,
    n_docs: int | None = None,
    n_symbols_per_doc: int | None = None,
):
    """Generate IRSnapshot where all references are valid."""
    nd = n_docs or draw(st.integers(min_value=1, max_value=4))
    ns = n_symbols_per_doc or draw(st.integers(min_value=1, max_value=4))

    repo = draw(identifier)
    snap_id = f"snap:{draw(identifier)}"
    docs = []
    symbols = []
    occurrences = []
    edges = []

    for i in range(nd):
        path = f"dir{i % 3}/file{i}.py"
        doc_id = f"doc:{draw(identifier)}"
        docs.append(
            IRDocument(
                doc_id=doc_id,
                path=path,
                language=draw(language_st),
                source_set={"fc_structure"},
            )
        )

        for j in range(ns):
            sym_id = f"sym:{draw(identifier)}"
            symbols.append(
                IRSymbol(
                    symbol_id=sym_id,
                    external_symbol_id=None,
                    path=path,
                    display_name=f"func_{j}",
                    kind=draw(kind_st),
                    language=docs[-1].language,
                    source_priority=50,
                    source_set={"fc_structure"},
                    start_line=draw(st.integers(min_value=1, max_value=500)),
                )
            )
            occurrences.append(
                IROccurrence(
                    occurrence_id=f"occ:{draw(identifier)}",
                    symbol_id=sym_id,
                    doc_id=doc_id,
                    role="definition",
                    start_line=symbols[-1].start_line or 1,
                    start_col=0,
                    end_line=symbols[-1].start_line or 1,
                    end_col=0,
                    source="fc_structure",
                )
            )
            edges.append(
                IREdge(
                    edge_id=f"edge:{draw(identifier)}",
                    src_id=doc_id,
                    dst_id=sym_id,
                    edge_type="contain",
                    source="fc_structure",
                    confidence="resolved",
                )
            )

    return IRSnapshot(
        repo_name=repo,
        snapshot_id=snap_id,
        branch="main",
        commit_id=draw(st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)),
        documents=docs,
        symbols=symbols,
        occurrences=occurrences,
        edges=edges,
        attachments=[
            IRAttachment(
                attachment_id=f"att:{draw(identifier)}",
                target_id=snap_id,
                target_type="snapshot",
                attachment_type="summary",
                source="fc_structure",
                confidence="derived",
                payload={"text": "snapshot summary"},
                metadata={},
            )
        ],
        metadata={"source_modes": ["fc_structure"]},
    )


# --- Helpers ---


def _make_store() -> SnapshotStore:
    tmpdir = tempfile.mkdtemp(prefix="snap_prop_")
    return _make_store_for_dir(tmpdir)


def _make_store_for_dir(tmpdir: str) -> SnapshotStore:
    return SnapshotStore(
        tmpdir,
        db_runtime=DBRuntime(
            backend="sqlite",
            sqlite_path=os.path.join(os.path.abspath(tmpdir), "lineage.db"),
        ),
    )


def test_constructor_requires_injected_database_runtime() -> None:
    tmpdir = tempfile.mkdtemp(prefix="snap_di_")
    runtime = DBRuntime(
        backend="sqlite", sqlite_path=os.path.join(tmpdir, "lineage.db")
    )

    store = SnapshotStore(tmpdir, db_runtime=runtime)

    assert store.db_runtime is runtime


class _FakeCursor:
    def __init__(
        self,
        row: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
        rowcount: int = 0,
    ) -> None:
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self) -> dict[str, Any] | None:
        return self._row

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _FakePostgresQueueRuntime:
    backend = "postgres"

    def __init__(self) -> None:
        self.redo_tasks: dict[str, dict[str, Any]] = {}
        self.outbox_events: dict[str, dict[str, Any]] = {}

    def connect(self) -> _FakePostgresQueueRuntime:
        return self

    def __enter__(self) -> _FakePostgresQueueRuntime:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def commit(self) -> None:
        return None

    @staticmethod
    def row_to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def add_redo_task(
        self,
        *,
        task_id: str,
        task_type: str,
        payload_json: str,
        status: str = "pending",
        attempts: int = 0,
        last_error: str | None = None,
        next_attempt_at: str | None = None,
        created_at: str = "2026-05-05T00:00:00+00:00",
        updated_at: str = "2026-05-05T00:00:00+00:00",
    ) -> None:
        self.redo_tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "payload_json": payload_json,
            "status": status,
            "attempts": attempts,
            "last_error": last_error,
            "next_attempt_at": next_attempt_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def add_outbox_event(
        self,
        *,
        event_id: str,
        event_type: str,
        payload: str,
        snapshot_id: str,
        status: str = "pending",
        attempts: int = 0,
        max_attempts: int = 5,
        created_at: str = "2026-05-05T00:00:00+00:00",
        last_attempt_at: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.outbox_events[event_id] = {
            "event_id": event_id,
            "event_type": event_type,
            "payload": payload,
            "snapshot_id": snapshot_id,
            "status": status,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "created_at": created_at,
            "last_attempt_at": last_attempt_at,
            "error_message": error_message,
        }

    def execute(
        self, _conn: object, sql: str, params: tuple[Any, ...] = ()
    ) -> _FakeCursor:
        if "SELECT * FROM redo_tasks" in sql:
            now = str(params[0])
            candidates = [
                dict(task)
                for task in self.redo_tasks.values()
                if task["status"] == "pending"
                and (
                    task["next_attempt_at"] is None
                    or str(task["next_attempt_at"]) <= now
                )
            ]
            candidates.sort(key=lambda task: str(task["created_at"]))
            return _FakeCursor(row=candidates[0] if candidates else None)

        if "SET status='running', attempts=attempts+1, updated_at=?" in sql:
            updated_at, task_id = params
            task = self.redo_tasks[str(task_id)]
            task["status"] = "running"
            task["attempts"] = int(task["attempts"]) + 1
            task["updated_at"] = str(updated_at)
            return _FakeCursor()

        if "SELECT attempts FROM redo_tasks WHERE task_id=?" in sql:
            task = self.redo_tasks.get(str(params[0]))
            row = {"attempts": task["attempts"]} if task is not None else None
            return _FakeCursor(row=row)

        if "SET status='dead', last_error=?, updated_at=?" in sql:
            error, updated_at, task_id = params
            task = self.redo_tasks[str(task_id)]
            task["status"] = "dead"
            task["last_error"] = str(error)
            task["updated_at"] = str(updated_at)
            return _FakeCursor()

        if "SET status='pending', last_error=?, next_attempt_at=?, updated_at=?" in sql:
            error, next_attempt_at, updated_at, task_id = params
            task = self.redo_tasks[str(task_id)]
            task["status"] = "pending"
            task["last_error"] = str(error)
            task["next_attempt_at"] = str(next_attempt_at)
            task["updated_at"] = str(updated_at)
            return _FakeCursor()

        if "INSERT INTO publish_outbox" in sql:
            event_id, event_type, payload, snapshot_id, max_attempts, created_at = (
                params
            )
            if str(event_id) in self.outbox_events:
                return _FakeCursor(rowcount=0)
            self.add_outbox_event(
                event_id=str(event_id),
                event_type=str(event_type),
                payload=str(payload),
                snapshot_id=str(snapshot_id),
                max_attempts=int(max_attempts),
                created_at=str(created_at),
            )
            return _FakeCursor(rowcount=1)

        if "SELECT * FROM publish_outbox" in sql:
            limit = int(params[0])
            candidates = [
                dict(event)
                for event in self.outbox_events.values()
                if event["status"] == "pending"
                or (
                    event["status"] == "failed"
                    and int(event["attempts"]) < int(event["max_attempts"])
                )
            ]
            candidates.sort(key=lambda event: str(event["created_at"]))
            return _FakeCursor(rows=candidates[:limit])

        if "SET status = 'in_progress', last_attempt_at = ?" in sql:
            last_attempt_at, event_id = params
            event = self.outbox_events[str(event_id)]
            event["status"] = "in_progress"
            event["last_attempt_at"] = str(last_attempt_at)
            return _FakeCursor()

        if (
            "SELECT attempts, max_attempts FROM publish_outbox WHERE event_id = ?"
            in sql
        ):
            event = self.outbox_events.get(str(params[0]))
            row = (
                {
                    "attempts": event["attempts"],
                    "max_attempts": event["max_attempts"],
                }
                if event is not None
                else None
            )
            return _FakeCursor(row=row)

        if "SET status = 'dead', attempts = ?, error_message = ?" in sql:
            attempts, error, event_id = params
            event = self.outbox_events[str(event_id)]
            event["status"] = "dead"
            event["attempts"] = int(attempts)
            event["error_message"] = str(error)
            return _FakeCursor()

        if "SET status = 'failed', attempts = ?, error_message = ?" in sql:
            attempts, error, event_id = params
            event = self.outbox_events[str(event_id)]
            event["status"] = "failed"
            event["attempts"] = int(attempts)
            event["error_message"] = str(error)
            return _FakeCursor()

        if "SELECT COUNT(*) AS cnt FROM publish_outbox" in sql:
            count = sum(
                1
                for event in self.outbox_events.values()
                if event["status"] == "pending"
                or (
                    event["status"] == "failed"
                    and int(event["attempts"]) < int(event["max_attempts"])
                )
            )
            return _FakeCursor(row={"cnt": count})

        raise AssertionError(f"unexpected SQL: {sql}")


metadata_st = st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.text(min_size=0, max_size=20), st.booleans()),
    max_size=4,
)


# --- TestSnapshotSaveLoadProperties ---


class TestSnapshotSaveLoadProperties:
    @given(snap=snapshot_st())
    @settings(max_examples=30)
    def test_save_load_roundtrip_property(self, snap: IRSnapshot):
        """HAPPY: save_snapshot then load_snapshot preserves IRSnapshot fields."""
        store = _make_store()
        store.save_snapshot(snap, metadata={"test": True})
        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert loaded.repo_name == snap.repo_name
        assert loaded.snapshot_id == snap.snapshot_id
        assert loaded.branch == snap.branch
        assert loaded.commit_id == snap.commit_id
        assert loaded.tree_id == snap.tree_id
        assert len(loaded.documents) == len(snap.documents)
        assert len(loaded.symbols) == len(snap.symbols)
        assert len(loaded.occurrences) == len(snap.occurrences)
        assert len(loaded.edges) == len(snap.edges)
        assert len(loaded.attachments) == len(snap.attachments)
        assert loaded.metadata == snap.metadata

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    def test_save_returns_artifact_key_property(self, snap: IRSnapshot):
        """HAPPY: save_snapshot returns SnapshotRecord with artifact_key and snapshot_id."""
        store = _make_store()
        result = store.save_snapshot(snap)
        assert isinstance(result, SnapshotRecord)
        assert result.snapshot_id == snap.snapshot_id
        assert result.artifact_key.startswith("snap_")

    @given(snap=snapshot_st())
    @settings(max_examples=20, deadline=None)
    def test_save_idempotent_upsert_property(self, snap: IRSnapshot):
        """HAPPY: saving same snapshot twice does not raise (ON CONFLICT DO UPDATE)."""
        store = _make_store()
        r1 = store.save_snapshot(snap)
        r2 = store.save_snapshot(snap)
        assert r1.snapshot_id == r2.snapshot_id
        assert r1.artifact_key == r2.artifact_key

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    def test_get_snapshot_record_after_save_property(self, snap: IRSnapshot):
        """HAPPY: get_snapshot_record returns SnapshotRecord with all DB columns."""
        store = _make_store()
        store.save_snapshot(snap)
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        assert isinstance(record, SnapshotRecord)
        assert record.snapshot_id == snap.snapshot_id
        assert record.repo_name == snap.repo_name
        assert record.artifact_key is not None
        assert record.ir_path is not None
        assert record.created_at is not None

    @given(snap=snapshot_st())
    @settings(max_examples=20, deadline=None)
    def test_load_snapshot_roundtrip_children_property(self, snap: IRSnapshot):
        """HAPPY: load_snapshot preserves nested document/symbol/edge fields."""
        store = _make_store()
        store.save_snapshot(snap)
        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        for orig, rest in zip(snap.documents, loaded.documents, strict=True):
            assert orig.doc_id == rest.doc_id
            assert orig.path == rest.path
            assert orig.language == rest.language
            assert orig.source_set == rest.source_set
        for orig, rest in zip(snap.symbols, loaded.symbols, strict=True):
            assert orig.symbol_id == rest.symbol_id
            assert orig.display_name == rest.display_name
            assert orig.kind == rest.kind
        for orig, rest in zip(snap.attachments, loaded.attachments, strict=True):
            assert orig.attachment_id == rest.attachment_id
            assert orig.payload == rest.payload

    def test_save_load_snapshot_uses_explicit_persistence_serializers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:explicit",
            branch="main",
            commit_id="abc1234",
            tree_id="tree1234",
            units=[
                IRCodeUnit(
                    unit_id="unit:file:a.py",
                    kind="file",
                    path="a.py",
                    language="python",
                    display_name="a.py",
                    source_set={"fc_structure"},
                    metadata={"rank": 1},
                )
            ],
            supports=[
                IRUnitSupport(
                    support_id="supp:def:a",
                    unit_id="unit:file:a.py",
                    source="fc_structure",
                    support_kind="occurrence",
                    role="definition",
                    start_line=1,
                    start_col=0,
                    end_line=1,
                    end_col=4,
                    metadata={"doc_id": "unit:file:a.py"},
                )
            ],
            relations=[
                IRRelation(
                    relation_id="rel:contain:a",
                    src_unit_id="unit:file:a.py",
                    dst_unit_id="unit:file:a.py",
                    relation_type="contain",
                    resolution_state="structural",
                    support_sources={"fc_structure"},
                    metadata={"source": "fc_structure"},
                )
            ],
            embeddings=[
                IRUnitEmbedding(
                    embedding_id="emb:a",
                    unit_id="unit:file:a.py",
                    source="fc_embedding",
                    vector=[0.25, 0.5],
                    embedding_text="a.py",
                    model_id="test-model",
                    metadata={"dim": 2},
                )
            ],
            metadata={"source_modes": ["fc_structure"]},
        )

        def _boom_to_dict(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call to_dict()")

        def _boom_from_dict(
            _cls: object, _data: dict[str, Any]
        ) -> IRSnapshot | IRCodeUnit | IRUnitSupport | IRRelation | IRUnitEmbedding:
            raise AssertionError("snapshot store must not call from_dict()")

        monkeypatch.setattr(IRSnapshot, "to_dict", _boom_to_dict)
        monkeypatch.setattr(IRSnapshot, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IRCodeUnit, "to_dict", _boom_to_dict)
        monkeypatch.setattr(IRCodeUnit, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IRUnitSupport, "to_dict", _boom_to_dict)
        monkeypatch.setattr(IRUnitSupport, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IRRelation, "to_dict", _boom_to_dict)
        monkeypatch.setattr(IRRelation, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IRUnitEmbedding, "to_dict", _boom_to_dict)
        monkeypatch.setattr(IRUnitEmbedding, "from_dict", classmethod(_boom_from_dict))

        store.save_snapshot(snap)
        loaded = store.load_snapshot(snap.snapshot_id)

        assert loaded is not None
        assert loaded.snapshot_id == snap.snapshot_id
        assert loaded.units[0].metadata == {"rank": 1}
        assert loaded.supports[0].metadata == {"doc_id": "unit:file:a.py"}
        assert loaded.relations[0].support_sources == {"fc_structure"}
        assert loaded.embeddings[0].vector == [0.25, 0.5]

    def test_file_ir_shard_payloads_use_explicit_serializers(self) -> None:
        class NoDictCodeUnit(IRCodeUnit):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("file IR shards must not call unit.to_dict()")

        class NoDictUnitSupport(IRUnitSupport):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("file IR shards must not call support.to_dict()")

        class NoDictRelation(IRRelation):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("file IR shards must not call relation.to_dict()")

        class NoDictEmbedding(IRUnitEmbedding):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("file IR shards must not call embedding.to_dict()")

        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:file-ir-shards",
            units=[
                NoDictCodeUnit(
                    unit_id="unit:file:a.py",
                    kind="file",
                    path="./pkg/a.py",
                    language="python",
                    display_name="a.py",
                    source_set={"fc_structure", "scip"},
                    metadata={"content_hash": "hash-a", "rank": 1},
                )
            ],
            supports=[
                NoDictUnitSupport(
                    support_id="supp:def:a",
                    unit_id="unit:file:a.py",
                    source="fc_structure",
                    support_kind="occurrence",
                    role="definition",
                    start_line=1,
                    start_col=0,
                    end_line=1,
                    end_col=4,
                    metadata={"doc_id": "unit:file:a.py"},
                )
            ],
            relations=[
                NoDictRelation(
                    relation_id="rel:contain:a",
                    src_unit_id="unit:file:a.py",
                    dst_unit_id="unit:file:a.py",
                    relation_type="contain",
                    resolution_state="structural",
                    support_sources={"fc_structure"},
                    support_ids=["supp:def:a"],
                    metadata={"source": "fc_structure"},
                )
            ],
            embeddings=[
                NoDictEmbedding(
                    embedding_id="emb:a",
                    unit_id="unit:file:a.py",
                    source="fc_embedding",
                    vector=[0.25, 0.5],
                    embedding_text="a.py",
                    model_id="test-model",
                    metadata={"dim": 2},
                )
            ],
        )

        shards = SnapshotStore.file_ir_shard_payloads(snap)

        assert len(shards) == 1
        shard = shards[0]
        assert shard["schema_version"] == "fastcode.file_ir_shard.v1"
        assert shard["snapshot_id"] == snap.snapshot_id
        assert shard["repo_name"] == "repo"
        assert shard["relative_path"] == "pkg/a.py"
        assert shard["content_hash"] == "hash-a"
        assert shard["units"][0]["unit_id"] == "unit:file:a.py"
        assert shard["units"][0]["source_set"] == ["fc_structure", "scip"]
        assert shard["units"][0]["metadata"] == {"content_hash": "hash-a", "rank": 1}
        assert shard["supports"][0]["metadata"] == {"doc_id": "unit:file:a.py"}
        assert shard["relations"][0]["support_sources"] == ["fc_structure"]
        assert shard["relations"][0]["support_ids"] == ["supp:def:a"]
        assert shard["embeddings"][0]["vector"] is None
        assert shard["embeddings"][0]["metadata"] == {"dim": 2}
        assert [
            shard[k][0]["_sequence_no"]
            for k in ("units", "supports", "relations", "embeddings")
        ] == [0, 0, 0, 0]

    def test_file_ir_shard_paths_include_relation_owner_shards(self) -> None:
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:file-ir-relation-shards",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="file",
                    path="pkg/a.py",
                    language="python",
                    display_name="a.py",
                    metadata={"content_hash": "hash-a"},
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="file",
                    path="pkg/b.py",
                    language="python",
                    display_name="b.py",
                    metadata={"content_hash": "hash-b"},
                ),
                IRCodeUnit(
                    unit_id="unit:c",
                    kind="file",
                    path="pkg/c.py",
                    language="python",
                    display_name="c.py",
                    metadata={"content_hash": "hash-c"},
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:a:b",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b",
                    relation_type="call",
                    resolution_state="structural",
                )
            ],
        )

        rewrite_paths = SnapshotStore.file_ir_shard_paths_for_paths(
            snap,
            ["pkg/b.py"],
        )
        shards = SnapshotStore.file_ir_shard_payloads(snap, paths=rewrite_paths)
        shards_by_path = {shard["relative_path"]: shard for shard in shards}

        assert rewrite_paths == ["pkg/a.py", "pkg/b.py"]
        assert set(shards_by_path) == {"pkg/a.py", "pkg/b.py"}
        assert shards_by_path["pkg/a.py"]["relations"][0]["relation_id"] == "rel:a:b"
        assert shards_by_path["pkg/b.py"]["units"][0]["unit_id"] == "unit:b"

    def test_file_ir_shard_paths_widen_on_removed_paths(self) -> None:
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:file-ir-removal",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="file",
                    path="pkg/a.py",
                    language="python",
                    display_name="a.py",
                ),
                IRCodeUnit(
                    unit_id="unit:c",
                    kind="file",
                    path="pkg/c.py",
                    language="python",
                    display_name="c.py",
                ),
            ],
        )

        assert SnapshotStore.file_ir_shard_paths_for_paths(
            snap,
            [],
            removed_paths=["pkg/b.py"],
        ) == ["pkg/a.py", "pkg/c.py"]

    def test_save_snapshot_writes_sharded_manifest_and_lazy_path_readers(self):
        """REGRESSION: snapshot persistence uses manifest + path shards."""
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:sharded",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                    source_set={"fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b",
                    source_set={"fc_structure"},
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:a:b",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b",
                    relation_type="call",
                    resolution_state="structural",
                )
            ],
            supports=[
                IRUnitSupport(
                    support_id="sup:a",
                    unit_id="unit:a",
                    source="fc_structure",
                    support_kind="occurrence",
                    path="pkg/a.py",
                )
            ],
            embeddings=[
                IRUnitEmbedding(
                    embedding_id="emb:a",
                    unit_id="unit:a",
                    source="fc_embedding",
                    vector=[0.25, 0.5],
                )
            ],
        )

        record = store.save_snapshot(snap)

        assert record.ir_path.endswith("ir_snapshot_manifest.json")
        snap_dir = os.path.dirname(record.ir_path)
        assert os.path.isdir(os.path.join(snap_dir, "units"))
        assert os.path.isdir(os.path.join(snap_dir, "embedding_vectors"))
        metadata = store.load_snapshot_metadata(snap.snapshot_id)
        assert metadata is not None
        assert metadata["counts"]["units"] == 2
        units = store.load_snapshot_units_for_paths(snap.snapshot_id, {"pkg/a.py"})
        assert [unit.unit_id for unit in units] == ["unit:a"]
        relations = store.load_snapshot_relations_for_paths(
            snap.snapshot_id,
            {"pkg/a.py"},
        )
        assert [relation.relation_id for relation in relations] == ["rel:a:b"]
        supports = store.load_snapshot_supports_for_paths(
            snap.snapshot_id,
            {"pkg/a.py"},
        )
        assert [support.support_id for support in supports] == ["sup:a"]
        embeddings = store.load_snapshot_embeddings_for_paths(
            snap.snapshot_id,
            {"pkg/a.py"},
        )
        assert [embedding.embedding_id for embedding in embeddings] == ["emb:a"]
        assert embeddings[0].vector == pytest.approx([0.25, 0.5])

        loaded = store.load_snapshot(snap.snapshot_id)
        assert loaded is not None
        assert loaded.embeddings[0].vector == pytest.approx([0.25, 0.5])

    def test_save_snapshot_delta_reuses_unchanged_path_shards(self) -> None:
        store = _make_store()
        previous = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:delta-prev",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b",
                ),
            ],
        )
        current = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:delta-current",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b2",
                ),
            ],
        )

        previous_record = store.save_snapshot(previous)
        current_record = store.save_snapshot_delta(
            current,
            previous_snapshot_id=previous.snapshot_id,
            changed_paths=["pkg/b.py"],
            removed_paths=[],
            metadata={"run": "delta"},
        )

        with open(previous_record.ir_path, encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        with open(current_record.ir_path, encoding="utf-8") as handle:
            current_manifest = json.load(handle)
        previous_units = {
            entry["path_key"]: entry for entry in previous_manifest["units"]
        }
        current_units = {
            entry["path_key"]: entry for entry in current_manifest["units"]
        }
        assert current_manifest["delta"]["reused_shards"] >= 1
        assert (
            current_units["pkg/a.py"]["digest"] == previous_units["pkg/a.py"]["digest"]
        )
        assert (
            current_units["pkg/b.py"]["digest"] != previous_units["pkg/b.py"]["digest"]
        )
        loaded = store.load_snapshot(current.snapshot_id)
        assert loaded is not None
        assert [unit.display_name for unit in loaded.units] == ["a", "b2"]

    def test_save_snapshot_delta_does_not_serialize_unchanged_unit_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = _make_store()
        previous = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:delta-no-regroup-prev",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b",
                ),
            ],
        )
        current = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:delta-no-regroup-current",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b2",
                ),
            ],
        )
        store.save_snapshot(previous)
        original_payload = SnapshotStore._code_unit_payload

        def _guarded_payload(
            cls: type[SnapshotStore], unit: IRCodeUnit
        ) -> dict[str, Any]:
            del cls
            if unit.path == "pkg/a.py":
                raise AssertionError("unchanged units should reuse previous shards")
            return original_payload(unit)

        monkeypatch.setattr(
            SnapshotStore,
            "_code_unit_payload",
            classmethod(_guarded_payload),
        )

        current_record = store.save_snapshot_delta(
            current,
            previous_snapshot_id=previous.snapshot_id,
            changed_paths=["pkg/b.py"],
            removed_paths=[],
        )

        with open(current_record.ir_path, encoding="utf-8") as handle:
            current_manifest = json.load(handle)
        current_units = {
            entry["path_key"]: entry for entry in current_manifest["units"]
        }
        assert set(current_units) == {"pkg/a.py", "pkg/b.py"}
        assert current_manifest["delta"]["reused_shards"] >= 1

    def test_save_snapshot_delta_rewrites_relations_touching_changed_destination(
        self,
    ) -> None:
        store = _make_store()
        previous = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:delta-rel-prev",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                ),
                IRCodeUnit(
                    unit_id="unit:b-old",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b",
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:a:b",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b-old",
                    relation_type="call",
                    resolution_state="structural",
                )
            ],
        )
        current = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:delta-rel-current",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                ),
                IRCodeUnit(
                    unit_id="unit:b-new",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b2",
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:a:b",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b-new",
                    relation_type="call",
                    resolution_state="structural",
                )
            ],
        )

        previous_record = store.save_snapshot(previous)
        current_record = store.save_snapshot_delta(
            current,
            previous_snapshot_id=previous.snapshot_id,
            changed_paths=["pkg/b.py"],
            removed_paths=[],
        )

        with open(previous_record.ir_path, encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        with open(current_record.ir_path, encoding="utf-8") as handle:
            current_manifest = json.load(handle)
        previous_relations = {
            entry["path_key"]: entry for entry in previous_manifest["relations"]
        }
        current_relations = {
            entry["path_key"]: entry for entry in current_manifest["relations"]
        }
        assert (
            current_relations["pkg/a.py"]["digest"]
            != previous_relations["pkg/a.py"]["digest"]
        )
        loaded_relations = store.load_snapshot_relations_for_paths(
            current.snapshot_id,
            {"pkg/a.py"},
        )
        assert [relation.dst_unit_id for relation in loaded_relations] == ["unit:b-new"]

    def test_save_snapshot_writes_compact_symbol_index_payload(self) -> None:
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:symbol-index",
            symbols=[
                IRSymbol(
                    symbol_id="sym:auth",
                    external_symbol_id="scip:auth",
                    path="src/auth.py",
                    display_name="AuthService",
                    kind="class",
                    language="python",
                    qualified_name="pkg.auth.AuthService",
                    source_priority=100,
                    source_set={"scip"},
                    metadata={"aliases": ["ast:auth"]},
                )
            ],
        )

        store.save_snapshot(snap)
        payload = store.load_snapshot_symbol_index_payload(snap.snapshot_id)

        assert payload is not None
        assert payload["snapshot_id"] == snap.snapshot_id
        assert payload["symbols"] == [
            {
                "canonical": "sym:auth",
                "aliases": ["ast:auth", "scip:auth", "sym:auth"],
                "names": ["AuthService", "pkg.auth.AuthService"],
                "display_name": "AuthService",
                "qualified_name": "pkg.auth.AuthService",
                "kind": "class",
                "path": "src/auth.py",
                "start_line": None,
                "language": "python",
            }
        ]
        assert store.load_snapshot_symbol_record(
            snap.snapshot_id,
            "sym:auth",
        ) == {
            "symbol_id": "sym:auth",
            "external_symbol_id": "scip:auth",
            "path": "src/auth.py",
            "display_name": "AuthService",
            "kind": "class",
            "language": "python",
            "qualified_name": "pkg.auth.AuthService",
            "signature": None,
            "start_line": None,
            "start_col": None,
            "end_line": None,
            "end_col": None,
            "source_priority": 100,
            "source_set": ["scip"],
            "metadata": {"aliases": ["ast:auth"]},
        }

    def test_load_snapshot_symbol_index_backfills_legacy_missing_sidecar(
        self,
    ) -> None:
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:legacy-symbol-index",
            symbols=[
                IRSymbol(
                    symbol_id="sym:legacy",
                    external_symbol_id="scip:legacy",
                    path="src/legacy.py",
                    display_name="LegacyService",
                    kind="class",
                    language="python",
                    qualified_name="pkg.legacy.LegacyService",
                    source_set={"scip"},
                    metadata={"aliases": ["ast:legacy"]},
                )
            ],
        )
        store.save_snapshot(snap)
        symbol_index_path = store.snapshot_symbol_index_path(snap.snapshot_id)
        os.remove(symbol_index_path)

        payload = store.load_snapshot_symbol_index_payload(snap.snapshot_id)

        assert payload is not None
        assert os.path.exists(symbol_index_path)
        assert payload["symbols"][0]["canonical"] == "sym:legacy"
        assert payload["symbols"][0]["aliases"] == [
            "ast:legacy",
            "scip:legacy",
            "sym:legacy",
        ]
        symbol_record = store.load_snapshot_symbol_record(
            snap.snapshot_id, "sym:legacy"
        )
        assert symbol_record is not None
        assert symbol_record["display_name"] == "LegacyService"

    def test_load_snapshot_symbol_index_prefers_relational_fact_backfill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store()
        snap = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:rel-symbol-index")
        store.save_snapshot(snap)
        symbol_index_path = store.snapshot_symbol_index_path(snap.snapshot_id)
        os.remove(symbol_index_path)
        record = {
            "symbol_id": "sym:rel",
            "external_symbol_id": "scip:rel",
            "path": "src/rel.py",
            "display_name": "RelService",
            "kind": "class",
            "language": "python",
            "qualified_name": "pkg.rel.RelService",
            "metadata": {"aliases": ["ast:rel"]},
        }
        with store.db_runtime.connect() as conn:
            store.db_runtime.execute(
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
            store.db_runtime.execute(
                conn,
                """
                INSERT INTO symbols (
                    snapshot_id, symbol_id, path, display_name, qualified_name,
                    kind, language, source_priority, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap.snapshot_id,
                    "sym:rel",
                    "src/rel.py",
                    "RelService",
                    "pkg.rel.RelService",
                    "class",
                    "python",
                    100,
                    json.dumps(record),
                ),
            )
            conn.commit()

        monkeypatch.setattr(
            store,
            "load_snapshot",
            lambda _snapshot_id: (_ for _ in ()).throw(
                AssertionError("relational facts should avoid full snapshot load")
            ),
        )

        payload = store.load_snapshot_symbol_index_payload(snap.snapshot_id)

        assert payload is not None
        assert payload["symbols"][0]["canonical"] == "sym:rel"
        assert payload["symbols"][0]["aliases"] == [
            "ast:rel",
            "scip:rel",
            "sym:rel",
        ]
        assert os.path.exists(symbol_index_path)

    def test_load_snapshot_legacy_payload_uses_explicit_deserializers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store()
        snapshot_id = "snap:repo:legacy-explicit"
        store.save_snapshot(IRSnapshot(repo_name="repo", snapshot_id=snapshot_id))
        record = store.get_snapshot_record(snapshot_id)
        assert record is not None

        legacy_payload = {
            "repo_name": "repo",
            "snapshot_id": snapshot_id,
            "branch": "main",
            "commit_id": "abc1234",
            "tree_id": "tree1234",
            "documents": [
                {
                    "doc_id": "doc:a",
                    "path": "a.py",
                    "language": "python",
                    "source_set": ["fc_structure"],
                }
            ],
            "symbols": [
                {
                    "symbol_id": "sym:a",
                    "external_symbol_id": None,
                    "path": "a.py",
                    "display_name": "run",
                    "kind": "function",
                    "language": "python",
                    "start_line": 1,
                    "start_col": 0,
                    "end_line": 1,
                    "end_col": 3,
                    "source_priority": 50,
                    "source_set": ["fc_structure"],
                    "metadata": {"rank": 1},
                }
            ],
            "occurrences": [
                {
                    "occurrence_id": "occ:a",
                    "symbol_id": "sym:a",
                    "doc_id": "doc:a",
                    "role": "definition",
                    "start_line": 1,
                    "start_col": 0,
                    "end_line": 1,
                    "end_col": 3,
                    "source": "fc_structure",
                    "metadata": {"doc_id": "doc:a"},
                }
            ],
            "edges": [
                {
                    "edge_id": "edge:a",
                    "src_id": "doc:a",
                    "dst_id": "sym:a",
                    "edge_type": "contain",
                    "source": "fc_structure",
                    "confidence": "resolved",
                    "metadata": {"source": "fc_structure"},
                }
            ],
            "attachments": [],
            "metadata": {"source_modes": ["fc_structure"]},
        }
        with open(record.ir_path, "w", encoding="utf-8") as handle:
            json.dump(legacy_payload, handle, ensure_ascii=False)

        def _boom_from_dict(
            _cls: object, _data: dict[str, Any]
        ) -> IRSnapshot | IRDocument | IRSymbol | IROccurrence | IREdge:
            raise AssertionError("snapshot store must not call from_dict()")

        monkeypatch.setattr(IRSnapshot, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IRDocument, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IRSymbol, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IROccurrence, "from_dict", classmethod(_boom_from_dict))
        monkeypatch.setattr(IREdge, "from_dict", classmethod(_boom_from_dict))

        loaded = store.load_snapshot(snapshot_id)

        assert loaded is not None
        assert loaded.branch == "main"
        assert loaded.documents[0].doc_id == "doc:a"
        assert loaded.symbols[0].metadata == {"rank": 1}
        assert loaded.occurrences[0].doc_id == "doc:a"
        assert loaded.edges[0].edge_type == "contain"

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_load_snapshot_missing_returns_none_property(self, snapshot_id: str):
        """EDGE: load_snapshot for non-existent ID returns None."""
        store = _make_store()
        assert store.load_snapshot(snapshot_id) is None

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_get_snapshot_record_missing_returns_none_property(self, snapshot_id: str):
        """EDGE: get_snapshot_record for non-existent ID returns None."""
        store = _make_store()
        assert store.get_snapshot_record(snapshot_id) is None

    @given(snap=connected_snapshot_st())
    @settings(max_examples=20)
    def test_resolve_snapshot_for_ref_property(self, snap: IRSnapshot):
        """HAPPY: resolve_snapshot_for_ref finds snapshot by repo+branch after save."""
        assume(snap.branch is not None)
        store = _make_store()
        store.save_snapshot(snap)
        result = store.resolve_snapshot_for_ref(snap.repo_name, snap.branch)
        assert result is not None
        assert result["snapshot_id"] == snap.snapshot_id
        assert result["repo_name"] == snap.repo_name
        assert result["branch"] == snap.branch

    @given(repo=identifier, branch=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_resolve_snapshot_for_ref_missing_returns_none_property(
        self, repo: str, branch: str
    ):
        """EDGE: resolve_snapshot_for_ref returns None for unknown ref."""
        store = _make_store()
        assert store.resolve_snapshot_for_ref(repo, branch) is None

    def test_resolve_snapshot_for_ref_record_returns_typed_record(self):
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:typed",
            branch="main",
            commit_id="abc1234",
            tree_id="tree123",
        )
        store.save_snapshot(snap)

        result = store.resolve_snapshot_for_ref_record("repo", "main")

        assert isinstance(result, SnapshotRefRecord)
        assert result.snapshot_id == "snap:repo:typed"
        with pytest.raises(AttributeError):
            result.snapshot_id = "snap:repo:other"  # type: ignore[misc]

    @given(snap=snapshot_st(), metadata=metadata_st)
    @settings(max_examples=20)
    def test_save_with_metadata_property(
        self, snap: IRSnapshot, metadata: dict[str, Any]
    ):
        """HAPPY: save_snapshot stores metadata_json and get_snapshot_record returns it."""
        store = _make_store()
        store.save_snapshot(snap, metadata=metadata)
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None
        assert isinstance(record, SnapshotRecord)
        assert record.metadata_json is not None


# --- TestSnapshotStoreQueries ---


class TestSnapshotStoreQueries:
    @given(snap=snapshot_st())
    @settings(max_examples=20)
    def test_find_by_repo_commit_property(self, snap: IRSnapshot):
        """HAPPY: find_by_repo_commit returns record after save (requires commit_id)."""
        assume(snap.commit_id is not None)
        store = _make_store()
        store.save_snapshot(snap)
        result = store.find_by_repo_commit(snap.repo_name, snap.commit_id)
        assert result is not None
        assert result["snapshot_id"] == snap.snapshot_id
        assert result["repo_name"] == snap.repo_name

    @given(repo=identifier, commit=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_find_by_repo_commit_missing_returns_none_property(
        self, repo: str, commit: str
    ):
        """EDGE: find_by_repo_commit returns None for unknown repo/commit."""
        store = _make_store()
        assert store.find_by_repo_commit(repo, commit) is None

    @given(snap=snapshot_st())
    @settings(max_examples=20)
    def test_find_by_artifact_key_property(self, snap: IRSnapshot):
        """HAPPY: find_by_artifact_key returns record after save."""
        store = _make_store()
        result = store.save_snapshot(snap)
        found_record = store.find_by_artifact_key_record(result.artifact_key)
        found = store.find_by_artifact_key(result.artifact_key)
        assert found_record is not None
        assert found_record.snapshot_id == snap.snapshot_id
        assert found_record.artifact_key == result.artifact_key
        assert found is not None
        assert found["snapshot_id"] == snap.snapshot_id

    @given(key=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_find_by_artifact_key_missing_returns_none_property(self, key: str):
        """EDGE: find_by_artifact_key returns None for unknown key."""
        store = _make_store()
        assert store.find_by_artifact_key_record(key) is None
        assert store.find_by_artifact_key(key) is None

    def test_get_snapshot_record_avoids_generic_row_to_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:typed",
            branch="main",
            commit_id="abc123",
            tree_id="tree123",
        )
        store.save_snapshot(snap)

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        monkeypatch.setattr(store.db_runtime, "row_to_dict", _boom)

        record = store.get_snapshot_record(snap.snapshot_id)

        assert record is not None
        assert record.snapshot_id == snap.snapshot_id
        assert record.branch == "main"

    def test_snapshot_query_helpers_use_explicit_serializers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:compat",
            branch="main",
            commit_id="abc999",
            tree_id="tree999",
        )
        saved = store.save_snapshot(snap)

        def _boom_row(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        def _boom_snapshot(_: SnapshotRecord) -> dict[str, Any]:
            raise AssertionError(
                "snapshot store must not call SnapshotRecord.to_dict()"
            )

        def _boom_ref(_: SnapshotRefRecord) -> dict[str, Any]:
            raise AssertionError(
                "snapshot store must not call SnapshotRefRecord.to_dict()"
            )

        monkeypatch.setattr(store.db_runtime, "row_to_dict", _boom_row)
        monkeypatch.setattr(SnapshotRecord, "to_dict", _boom_snapshot)
        monkeypatch.setattr(SnapshotRefRecord, "to_dict", _boom_ref)

        by_artifact_record = store.find_by_artifact_key_record(saved.artifact_key)
        by_artifact = store.find_by_artifact_key(saved.artifact_key)
        by_commit = store.find_by_repo_commit("repo", "abc999")
        by_ref = store.resolve_snapshot_for_ref("repo", "main")

        assert by_artifact_record is not None
        assert by_artifact_record.snapshot_id == snap.snapshot_id
        assert by_artifact is not None
        assert by_artifact["snapshot_id"] == snap.snapshot_id
        assert by_commit is not None
        assert by_commit["artifact_key"] == saved.artifact_key
        assert by_ref is not None
        assert by_ref["snapshot_id"] == snap.snapshot_id

    @given(
        snap1=connected_snapshot_st(n_docs=1, n_symbols_per_doc=1),
        snap2=connected_snapshot_st(n_docs=1, n_symbols_per_doc=1),
    )
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_different_repos_independent_property(
        self, snap1: IRSnapshot, snap2: IRSnapshot
    ):
        """EDGE: snapshots from different repos queried independently."""
        from hypothesis import assume

        assume(snap1.snapshot_id != snap2.snapshot_id)
        # Force different repo names to test isolation
        snap2 = IRSnapshot(
            repo_name=snap2.repo_name
            if snap2.repo_name != snap1.repo_name
            else snap2.repo_name + "_b",
            snapshot_id=snap2.snapshot_id,
            branch=snap2.branch,
            commit_id=snap2.commit_id,
            tree_id=snap2.tree_id,
            documents=snap2.documents,
            symbols=snap2.symbols,
            occurrences=snap2.occurrences,
            edges=snap2.edges,
            metadata=snap2.metadata,
        )
        store = _make_store()
        store.save_snapshot(snap1)
        store.save_snapshot(snap2)
        r1 = store.find_by_repo_commit(snap1.repo_name, snap1.commit_id)
        r2 = store.find_by_repo_commit(snap2.repo_name, snap2.commit_id)
        assert r1 is not None
        assert r2 is not None
        assert r1["snapshot_id"] == snap1.snapshot_id
        assert r2["snapshot_id"] == snap2.snapshot_id

    @given(
        snap=snapshot_st(),
        metadata1=metadata_st,
        metadata2=metadata_st,
    )
    @settings(max_examples=15, deadline=None)
    def test_update_snapshot_metadata_property(
        self, snap: IRSnapshot, metadata1: dict[str, Any], metadata2: dict[str, Any]
    ):
        """HAPPY: update_snapshot_metadata overwrites stored metadata."""
        store = _make_store()
        store.save_snapshot(snap, metadata=metadata1)
        store.update_snapshot_metadata(snap.snapshot_id, metadata2)
        record = store.get_snapshot_record(snap.snapshot_id)
        assert record is not None

    @given(snap=snapshot_st())
    @settings(max_examples=15)
    def test_artifact_key_deterministic_property(self, snap: IRSnapshot):
        """HAPPY: artifact_key_for_snapshot is deterministic for same snapshot_id."""
        store = _make_store()
        key1 = store.artifact_key_for_snapshot(snap.snapshot_id)
        key2 = store.artifact_key_for_snapshot(snap.snapshot_id)
        assert key1 == key2

    @given(
        sid1=st.builds(lambda x: f"snap:{x}", identifier),
        sid2=st.builds(lambda x: f"snap:{x}", identifier),
    )
    @settings(max_examples=20)
    @pytest.mark.edge
    def test_artifact_key_differs_for_different_ids_property(
        self, sid1: str, sid2: str
    ):
        """EDGE: different snapshot_ids produce different artifact keys (probabilistic)."""
        store = _make_store()
        key1 = store.artifact_key_for_snapshot(sid1)
        key2 = store.artifact_key_for_snapshot(sid2)
        if sid1 != sid2:
            assert key1 != key2


# --- TestScipArtifactRefProperties ---


class TestScipArtifactRefProperties:
    @given(
        snapshot_id=identifier,
        indexer_name=identifier,
        indexer_version=st.none() | st.just("1.0.0"),
        artifact_path=st.just("/tmp/scip.dump"),
        checksum=st.just("abc123"),
    )
    @settings(max_examples=20)
    def test_save_and_get_scip_artifact_ref_property(
        self,
        snapshot_id: str,
        indexer_name: str,
        indexer_version: Any,
        artifact_path: str,
        checksum: str,
    ):
        """HAPPY: save_scip_artifact_ref then get_scip_artifact_ref roundtrip."""
        store = _make_store()
        result = store.save_scip_artifact_ref(
            snapshot_id,
            indexer_name=indexer_name,
            indexer_version=indexer_version,
            artifact_path=artifact_path,
            checksum=checksum,
        )
        assert result["snapshot_id"] == snapshot_id
        assert result["indexer_name"] == indexer_name
        assert result["artifact_path"] == artifact_path
        assert result["checksum"] == checksum
        assert result["created_at"] is not None

        loaded = store.get_scip_artifact_ref(snapshot_id)
        loaded_record = store.get_scip_artifact_ref_record(snapshot_id)
        assert loaded is not None
        assert loaded_record is not None
        assert loaded_record.snapshot_id == snapshot_id
        assert loaded_record.indexer_name == indexer_name
        assert loaded["snapshot_id"] == snapshot_id
        assert loaded["indexer_name"] == indexer_name

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_get_scip_artifact_ref_missing_returns_none_property(
        self, snapshot_id: str
    ):
        """EDGE: get_scip_artifact_ref returns None when not saved."""
        store = _make_store()
        assert store.get_scip_artifact_ref_record(snapshot_id) is None
        assert store.get_scip_artifact_ref(snapshot_id) is None
        assert store.list_scip_artifact_ref_records(snapshot_id) == []

    @given(snapshot_id=identifier)
    @settings(max_examples=15)
    def test_save_scip_artifact_ref_defaults_property(self, snapshot_id: str):
        """HAPPY: save_scip_artifact_ref with defaults uses 'unknown' indexer_name."""
        store = _make_store()
        result = store.save_scip_artifact_ref(snapshot_id)
        assert result["indexer_name"] == "unknown"
        assert result["indexer_version"] is None
        assert result["artifact_path"] == ""
        assert result["checksum"] == ""

    @given(snapshot_id=identifier)
    @settings(max_examples=15, deadline=None)
    def test_save_scip_artifact_ref_upsert_property(self, snapshot_id: str):
        """HAPPY: saving SCIP artifact ref twice updates the record."""
        store = _make_store()
        store.save_scip_artifact_ref(snapshot_id, indexer_name="v1")
        r2 = store.save_scip_artifact_ref(snapshot_id, indexer_name="v2")
        assert r2["indexer_name"] == "v2"
        loaded = store.get_scip_artifact_ref(snapshot_id)
        assert loaded["indexer_name"] == "v2"

    def test_save_scip_artifact_refs_preserves_primary_and_lineage_property(self):
        """HAPPY: multi-artifact save keeps primary accessor and ordered lineage."""
        store = _make_store()
        artifacts = store.save_scip_artifact_refs(
            "snap:repo:multi",
            artifacts=[
                {
                    "indexer_name": "scip-ts",
                    "artifact_path": "/tmp/ts.scip",
                    "checksum": "111",
                    "language": "typescript",
                },
                {
                    "indexer_name": "scip-rust",
                    "artifact_path": "/tmp/rust.scip",
                    "checksum": "222",
                    "language": "rust",
                },
            ],
        )

        assert len(artifacts) == 2
        assert artifacts[0]["role"] == "primary"
        assert artifacts[1]["metadata"]["language"] == "rust"
        assert (
            store.get_scip_artifact_ref("snap:repo:multi")["artifact_path"]
            == "/tmp/ts.scip"
        )
        records = store.list_scip_artifact_ref_records("snap:repo:multi")
        assert len(records) == 2
        assert records[0].role == "primary"
        assert records[1].metadata_json is not None
        assert len(store.list_scip_artifact_refs("snap:repo:multi")) == 2

    def test_scip_artifact_paths_use_explicit_serializers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _make_store()

        class _OpaqueMetadata:
            pass

        def _boom_row(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        def _boom_to_dict(_: SCIPArtifactRef) -> dict[str, Any]:
            raise AssertionError(
                "snapshot store must not call SCIPArtifactRef.to_dict()"
            )

        def _boom_record_to_dict(_: SCIPArtifactRecord) -> dict[str, Any]:
            raise AssertionError(
                "snapshot store must not call SCIPArtifactRecord.to_dict()"
            )

        monkeypatch.setattr(store.db_runtime, "row_to_dict", _boom_row)
        monkeypatch.setattr(SCIPArtifactRef, "to_dict", _boom_to_dict)
        monkeypatch.setattr(SCIPArtifactRecord, "to_dict", _boom_record_to_dict)

        saved = store.save_scip_artifact_refs(
            "snap:repo:typed",
            artifacts=[
                {
                    "indexer_name": "scip-ts",
                    "artifact_path": "/tmp/ts.scip",
                    "checksum": "111",
                    "metadata": {"opaque": _OpaqueMetadata()},
                },
                {
                    "indexer_name": "scip-rust",
                    "artifact_path": "/tmp/rust.scip",
                    "checksum": "222",
                    "language": "rust",
                },
            ],
        )
        loaded_record = store.get_scip_artifact_ref_record("snap:repo:typed")
        listed_records = store.list_scip_artifact_ref_records("snap:repo:typed")
        loaded = store.get_scip_artifact_ref("snap:repo:typed")
        listed = store.list_scip_artifact_refs("snap:repo:typed")

        assert saved[0]["artifact_id"] == "snap:repo:typed:scip:0"
        assert loaded_record is not None
        assert loaded_record.artifact_path == "/tmp/ts.scip"
        assert listed_records[0].metadata_json is not None
        assert loaded is not None
        assert loaded["artifact_path"] == "/tmp/ts.scip"
        assert listed[0]["metadata"]["opaque"].startswith("<")
        assert listed[1]["metadata"]["language"] == "rust"


# --- TestSnapshotStoreRelationalFacts ---


class TestSnapshotStoreRelationalFacts:
    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_save_relational_facts_sqlite_noop_property(self, snap: IRSnapshot):
        """EDGE: save_relational_facts is no-op on SQLite (returns early)."""
        store = _make_store()
        # Should not raise
        store.save_relational_facts(snap)

    def test_relational_fact_payloads_do_not_call_record_to_dict_double(self) -> None:
        class NoDictDocument(IRDocument):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("document payload should stay field-explicit")

        class NoDictSymbol(IRSymbol):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("symbol payload should stay field-explicit")

        class NoDictOccurrence(IROccurrence):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("occurrence payload should stay field-explicit")

        class NoDictEdge(IREdge):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("edge payload should stay field-explicit")

        class NoDictAttachment(IRAttachment):
            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("attachment payload should stay field-explicit")

        assert SnapshotStore._document_payload(
            NoDictDocument(
                doc_id="doc:a",
                path="pkg/a.py",
                language="python",
                source_set={"scip", "fc_structure"},
            )
        )["source_set"] == ["fc_structure", "scip"]
        assert SnapshotStore._symbol_payload(
            NoDictSymbol(
                symbol_id="sym:a",
                external_symbol_id="scip python a",
                path="pkg/a.py",
                display_name="a",
                kind="function",
                language="python",
                metadata={"rank": 1},
            )
        )["metadata"] == {"rank": 1}
        assert (
            SnapshotStore._occurrence_payload(
                NoDictOccurrence(
                    occurrence_id="occ:a",
                    symbol_id="sym:a",
                    doc_id="doc:a",
                    role="definition",
                    start_line=1,
                    start_col=0,
                    end_line=1,
                    end_col=5,
                    source="scip",
                )
            )["source"]
            == "scip"
        )
        assert (
            SnapshotStore._edge_payload(
                NoDictEdge(
                    edge_id="edge:a",
                    src_id="sym:a",
                    dst_id="sym:b",
                    edge_type="call",
                    source="scip",
                    confidence="precise",
                )
            )["edge_type"]
            == "call"
        )
        assert SnapshotStore._attachment_payload(
            NoDictAttachment(
                attachment_id="att:a",
                target_id="sym:a",
                target_type="symbol",
                attachment_type="summary",
                source="llm_annotation",
                confidence="derived",
                payload={"text": "summary"},
            )
        )["payload"] == {"text": "summary"}

    def test_save_relational_facts_batches_postgres_inserts_double(self) -> None:
        class _FakePostgresRuntime:
            backend = "postgres"

            def __init__(self) -> None:
                self.executes: list[tuple[str, tuple[Any, ...]]] = []
                self.batches: list[tuple[str, list[tuple[Any, ...]]]] = []
                self.commits = 0

            def connect(self) -> _FakePostgresRuntime:
                return self

            def __enter__(self) -> _FakePostgresRuntime:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def execute(
                self, _conn: object, sql: str, params: tuple[Any, ...] = ()
            ) -> _FakeCursor:
                self.executes.append((sql, params))
                return _FakeCursor()

            def executemany(
                self, _conn: object, sql: str, params_seq: list[tuple[Any, ...]]
            ) -> _FakeCursor:
                self.batches.append((sql, list(params_seq)))
                return _FakeCursor()

            def commit(self) -> None:
                self.commits += 1

        store = _make_store()
        runtime = _FakePostgresRuntime()
        store.db_runtime = runtime  # type: ignore[assignment]
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:batch",
            documents=[
                IRDocument(doc_id="doc:a", path="a.py", language="python"),
                IRDocument(doc_id="doc:b", path="b.py", language="python"),
            ],
            symbols=[
                IRSymbol(
                    symbol_id="sym:a",
                    external_symbol_id=None,
                    path="a.py",
                    display_name="a",
                    kind="function",
                    language="python",
                )
            ],
            occurrences=[
                IROccurrence(
                    occurrence_id="occ:a",
                    symbol_id="sym:a",
                    doc_id="doc:a",
                    role="definition",
                    start_line=1,
                    start_col=0,
                    end_line=1,
                    end_col=1,
                    source="ast",
                )
            ],
            edges=[
                IREdge(
                    edge_id="edge:a",
                    src_id="sym:a",
                    dst_id="sym:b",
                    edge_type="call",
                    source="ast",
                    confidence="resolved",
                ),
                IREdge(
                    edge_id="edge:a",
                    src_id="sym:a",
                    dst_id="sym:b",
                    edge_type="call",
                    source="ast",
                    confidence="resolved",
                ),
            ],
            attachments=[
                IRAttachment(
                    attachment_id="att:a",
                    target_id="sym:a",
                    target_type="symbol",
                    attachment_type="summary",
                    source="fc_structure",
                    confidence="derived",
                    payload={"text": "summary"},
                )
            ],
        )

        store.save_relational_facts(snapshot)

        assert len(runtime.executes) == 5
        assert all("DELETE FROM" in sql for sql, _params in runtime.executes)
        assert len(runtime.batches) == 5
        row_counts = {
            "snapshot_documents": 2,
            "symbols": 1,
            "occurrences": 1,
            "edges": 2,
            "attachments": 1,
        }
        for table_name, expected_count in row_counts.items():
            matching = [
                params
                for sql, params in runtime.batches
                if f"INSERT INTO {table_name}" in sql
            ]
            assert matching
            assert len(matching[0]) == expected_count
        assert runtime.commits == 1

    def test_save_relational_facts_delta_copies_previous_and_upserts_changed_paths_double(
        self,
    ) -> None:
        class _FakePostgresRuntime:
            backend = "postgres"

            def __init__(self) -> None:
                self.executes: list[tuple[str, tuple[Any, ...]]] = []
                self.batches: list[tuple[str, list[tuple[Any, ...]]]] = []
                self.commits = 0

            def connect(self) -> _FakePostgresRuntime:
                return self

            def __enter__(self) -> _FakePostgresRuntime:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def execute(
                self, _conn: object, sql: str, params: tuple[Any, ...] = ()
            ) -> _FakeCursor:
                self.executes.append((sql, params))
                return _FakeCursor()

            def executemany(
                self, _conn: object, sql: str, params_seq: list[tuple[Any, ...]]
            ) -> _FakeCursor:
                self.batches.append((sql, list(params_seq)))
                return _FakeCursor()

            def commit(self) -> None:
                self.commits += 1

        store = _make_store()
        runtime = _FakePostgresRuntime()
        store.db_runtime = runtime  # type: ignore[assignment]
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:delta",
            documents=[
                IRDocument(doc_id="doc:a", path="a.py", language="python"),
                IRDocument(doc_id="doc:b", path="b.py", language="python"),
            ],
            symbols=[
                IRSymbol(
                    symbol_id="sym:a",
                    external_symbol_id=None,
                    path="a.py",
                    display_name="a",
                    kind="function",
                    language="python",
                ),
                IRSymbol(
                    symbol_id="sym:b",
                    external_symbol_id=None,
                    path="b.py",
                    display_name="b",
                    kind="function",
                    language="python",
                ),
            ],
            occurrences=[
                IROccurrence(
                    occurrence_id="occ:a",
                    symbol_id="sym:a",
                    doc_id="doc:a",
                    role="definition",
                    start_line=1,
                    start_col=0,
                    end_line=1,
                    end_col=1,
                    source="ast",
                ),
                IROccurrence(
                    occurrence_id="occ:b",
                    symbol_id="sym:b",
                    doc_id="doc:b",
                    role="definition",
                    start_line=1,
                    start_col=0,
                    end_line=1,
                    end_col=1,
                    source="ast",
                ),
            ],
            edges=[
                IREdge(
                    edge_id="edge:a",
                    src_id="sym:a",
                    dst_id="sym:b",
                    edge_type="call",
                    source="ast",
                    confidence="resolved",
                    doc_id="doc:a",
                ),
                IREdge(
                    edge_id="edge:b",
                    src_id="sym:b",
                    dst_id="sym:b",
                    edge_type="call",
                    source="ast",
                    confidence="resolved",
                    doc_id="doc:b",
                ),
            ],
            attachments=[
                IRAttachment(
                    attachment_id="att:a",
                    target_id="sym:a",
                    target_type="symbol",
                    attachment_type="summary",
                    source="fc_structure",
                    confidence="derived",
                    payload={"text": "changed"},
                ),
                IRAttachment(
                    attachment_id="att:b",
                    target_id="sym:b",
                    target_type="symbol",
                    attachment_type="summary",
                    source="fc_structure",
                    confidence="derived",
                    payload={"text": "unchanged"},
                ),
                IRAttachment(
                    attachment_id="att:snapshot",
                    target_id="snap:repo:delta",
                    target_type="snapshot",
                    attachment_type="summary",
                    source="fc_structure",
                    confidence="derived",
                    payload={"text": "current snapshot"},
                ),
            ],
        )

        saved = store.save_relational_facts_delta(
            snapshot,
            previous_snapshot_id="snap:repo:prev",
            changed_paths=["a.py"],
            removed_paths=["old.py"],
        )

        assert saved is True
        assert len(runtime.executes) == 5
        assert not any("DELETE FROM" in sql for sql, _params in runtime.executes)
        assert all(
            params[:2] == ("snap:repo:delta", "snap:repo:prev")
            for _sql, params in runtime.executes
        )
        assert any(
            "SELECT ?, doc_id, path, language, metadata_json" in sql
            and "path NOT IN (?, ?)" in sql
            for sql, _params in runtime.executes
        )
        attachment_copy_sql = next(
            sql for sql, _params in runtime.executes if "FROM attachments a" in sql
        )
        assert "a.target_type='symbol'" in attachment_copy_sql
        assert "a.target_type IN ('document', 'doc', 'file')" in attachment_copy_sql

        batches_by_table = {
            table_name: rows
            for table_name in (
                "snapshot_documents",
                "symbols",
                "occurrences",
                "edges",
                "attachments",
            )
            for sql, rows in runtime.batches
            if f"INSERT INTO {table_name}" in sql
        }
        assert {row[2] for row in batches_by_table["snapshot_documents"]} == {"a.py"}
        assert {row[2] for row in batches_by_table["symbols"]} == {"a.py"}
        assert {row[3] for row in batches_by_table["occurrences"]} == {"doc:a"}
        edge_ids = {row[1] for row in batches_by_table["edges"]}
        assert "edge:a" in edge_ids
        assert "edge:b" not in edge_ids
        assert {row[1] for row in batches_by_table["attachments"]} == {
            "att:summary:sym:a",
            "att:snapshot",
        }
        assert runtime.commits == 1

    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_import_git_backbone_sqlite_noop_property(self, snap: IRSnapshot):
        """EDGE: import_git_backbone is no-op on SQLite (returns early)."""
        store = _make_store()
        store.import_git_backbone(snap, git_meta={"parent_commit_id": "deadbeef"})


# --- TestSnapshotStoreStaging ---


class TestSnapshotStoreStaging:
    @given(snap=snapshot_st())
    @settings(max_examples=15)
    def test_stage_snapshot_returns_stage_id_property(self, snap: IRSnapshot):
        """HAPPY: stage_snapshot returns a stage_id starting with 'stage_'."""
        store = _make_store()
        stage_id = store.stage_snapshot(snap)
        assert stage_id.startswith("stage_")
        assert len(stage_id) > len("stage_")

    @given(snap=snapshot_st())
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_stage_snapshot_unique_ids_property(self, snap: IRSnapshot):
        """EDGE: stage_snapshot returns unique stage_ids each call."""
        store = _make_store()
        s1 = store.stage_snapshot(snap)
        s2 = store.stage_snapshot(snap)
        assert s1 != s2

    @given(snap=snapshot_st())
    @settings(max_examples=15, deadline=None)
    @pytest.mark.edge
    def test_promote_staged_snapshot_sqlite_noop_property(self, snap: IRSnapshot):
        """EDGE: promote_staged_snapshot is no-op on SQLite (returns early)."""
        store = _make_store()
        stage_id = store.stage_snapshot(snap)
        store.promote_staged_snapshot(snap.snapshot_id, stage_id)
        # No error means success for SQLite no-op path


# --- TestSnapshotStoreLockProperties ---


class TestSnapshotStoreLockProperties:
    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15, deadline=None)
    def test_acquire_lock_returns_one_property(self, lock_name: str, owner_id: str):
        """HAPPY: acquire_lock returns 1 on SQLite backend."""
        store = _make_store()
        result = store.acquire_lock(lock_name, owner_id)
        assert result == 1

    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15)
    def test_acquire_lock_with_custom_ttl_property(self, lock_name: str, owner_id: str):
        """HAPPY: acquire_lock with custom TTL still returns 1 on SQLite."""
        store = _make_store()
        result = store.acquire_lock(lock_name, owner_id, ttl_seconds=600)
        assert result == 1

    @given(lock_name=identifier, token=st.integers(min_value=0, max_value=1000))
    @settings(max_examples=15, deadline=None)
    def test_validate_fencing_token_returns_true_property(
        self, lock_name: str, token: int
    ):
        """HAPPY: validate_fencing_token returns True on SQLite backend."""
        store = _make_store()
        assert store.validate_fencing_token(lock_name, token) is True

    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15)
    def test_release_lock_noop_property(self, lock_name: str, owner_id: str):
        """HAPPY: release_lock is no-op on SQLite (returns None, no error)."""
        store = _make_store()
        store.release_lock(lock_name, owner_id)

    @given(lock_name=identifier, owner_id=identifier)
    @settings(max_examples=15)
    @pytest.mark.edge
    def test_lock_acquire_validate_release_sequence_property(
        self, lock_name: str, owner_id: str
    ):
        """EDGE: acquire, validate, release sequence completes without error on SQLite."""
        store = _make_store()
        token = store.acquire_lock(lock_name, owner_id)
        assert token == 1
        assert store.validate_fencing_token(lock_name, token) is True
        store.release_lock(lock_name, owner_id)


# --- TestSnapshotStoreRedoProperties ---


class TestSnapshotStoreRedoProperties:
    @given(
        task_type=identifier,
        payload=st.dictionaries(identifier, st.integers(), max_size=3),
    )
    @settings(max_examples=15)
    def test_enqueue_redo_task_returns_redo_id_property(
        self, task_type: str, payload: dict[str, Any]
    ):
        """HAPPY: enqueue_redo_task returns ID starting with 'redo_'."""
        store = _make_store()
        task_id = store.enqueue_redo_task(task_type, payload)
        assert task_id.startswith("redo_")

    @given(
        task_type=identifier,
        payload=st.dictionaries(identifier, st.integers(), max_size=2),
        error=st.none() | st.just("test error"),
    )
    @settings(max_examples=15)
    def test_enqueue_redo_task_with_error_property(
        self, task_type: str, payload: dict[str, Any], error: Exception
    ):
        """HAPPY: enqueue_redo_task with optional error still returns redo_ ID."""
        store = _make_store()
        task_id = store.enqueue_redo_task(task_type, payload, error=error)
        assert task_id.startswith("redo_")

    @given(
        task_type=identifier,
        payload=st.dictionaries(identifier, st.integers(), max_size=2),
    )
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_enqueue_redo_task_unique_ids_property(
        self, task_type: str, payload: dict[str, Any]
    ):
        """EDGE: each enqueue call returns a unique task_id."""
        store = _make_store()
        t1 = store.enqueue_redo_task(task_type, payload)
        t2 = store.enqueue_redo_task(task_type, payload)
        assert t1 != t2

    @pytest.mark.edge
    def test_claim_redo_task_sqlite_noop_property(self):
        """EDGE: claim_redo_task returns None on SQLite backend."""
        store = _make_store()
        result = store.claim_redo_task()
        assert result is None

    @pytest.mark.edge
    def test_mark_redo_task_done_sqlite_noop_property(self):
        """EDGE: mark_redo_task_done is no-op on SQLite (returns None, no error)."""
        store = _make_store()
        store.mark_redo_task_done("redo_test123")

    @pytest.mark.edge
    def test_mark_redo_task_failed_sqlite_noop_property(self):
        """EDGE: mark_redo_task_failed is no-op on SQLite (returns None, no error)."""
        store = _make_store()
        store.mark_redo_task_failed(
            task_id="redo_test123", error="fail", max_attempts=3
        )

    def test_claim_redo_task_returns_running_payload_after_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SnapshotStore.__new__(SnapshotStore)
        runtime = _FakePostgresQueueRuntime()
        runtime.add_redo_task(
            task_id="redo_1",
            task_type="index_run_recovery",
            payload_json='{"run_id":"run1"}',
        )
        store.db_runtime = runtime

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        def _boom_task(_: RedoTaskRecord) -> dict[str, Any]:
            raise AssertionError(
                "snapshot store must not call RedoTaskRecord.to_dict()"
            )

        monkeypatch.setattr(
            snapshot_module, "utc_now", lambda: "2026-05-05T00:00:05+00:00"
        )
        monkeypatch.setattr(runtime, "row_to_dict", _boom)
        monkeypatch.setattr(RedoTaskRecord, "to_dict", _boom_task)

        task = store.claim_redo_task()

        assert task is not None
        assert task["task_id"] == "redo_1"
        assert task["status"] == "running"
        assert task["attempts"] == 1
        assert task["updated_at"] == "2026-05-05T00:00:05+00:00"
        assert runtime.redo_tasks["redo_1"]["status"] == "running"
        assert runtime.redo_tasks["redo_1"]["attempts"] == 1
        assert runtime.redo_tasks["redo_1"]["updated_at"] == "2026-05-05T00:00:05+00:00"
        assert store.claim_redo_task() is None

    def test_claim_redo_task_record_returns_typed_running_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SnapshotStore.__new__(SnapshotStore)
        runtime = _FakePostgresQueueRuntime()
        runtime.add_redo_task(
            task_id="redo_record",
            task_type="index_run_recovery",
            payload_json='{"run_id":"run-record"}',
        )
        store.db_runtime = runtime

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        def _boom_task(_: RedoTaskRecord) -> dict[str, Any]:
            raise AssertionError(
                "typed redo claim must not call RedoTaskRecord.to_dict()"
            )

        monkeypatch.setattr(
            snapshot_module, "utc_now", lambda: "2026-05-05T00:00:07+00:00"
        )
        monkeypatch.setattr(runtime, "row_to_dict", _boom)
        monkeypatch.setattr(RedoTaskRecord, "to_dict", _boom_task)

        task = store.claim_redo_task_record()

        assert task == RedoTaskRecord(
            task_id="redo_record",
            task_type="index_run_recovery",
            payload_json='{"run_id":"run-record"}',
            status="running",
            attempts=1,
            last_error=None,
            next_attempt_at=None,
            created_at="2026-05-05T00:00:00+00:00",
            updated_at="2026-05-05T00:00:07+00:00",
        )
        assert runtime.redo_tasks["redo_record"]["status"] == "running"
        assert runtime.redo_tasks["redo_record"]["attempts"] == 1
        assert store.claim_redo_task_record() is None

    def test_mark_redo_task_failed_uses_explicit_attempt_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SnapshotStore.__new__(SnapshotStore)
        runtime = _FakePostgresQueueRuntime()
        runtime.add_redo_task(
            task_id="redo_2",
            task_type="index_run_recovery",
            payload_json='{"run_id":"run2"}',
            status="running",
            attempts=5,
        )
        store.db_runtime = runtime

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        monkeypatch.setattr(
            snapshot_module, "utc_now", lambda: "2026-05-05T00:00:10+00:00"
        )
        monkeypatch.setattr(runtime, "row_to_dict", _boom)

        store.mark_redo_task_failed("redo_2", "boom", max_attempts=5)

        assert runtime.redo_tasks["redo_2"]["status"] == "dead"
        assert runtime.redo_tasks["redo_2"]["last_error"] == "boom"
        assert runtime.redo_tasks["redo_2"]["updated_at"] == "2026-05-05T00:00:10+00:00"


class TestSnapshotStoreOutboxPostgresProperties:
    def test_enqueue_outbox_event_returns_false_for_duplicate_event_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SnapshotStore.__new__(SnapshotStore)
        runtime = _FakePostgresQueueRuntime()
        store.db_runtime = runtime
        monkeypatch.setattr(
            snapshot_module, "utc_now", lambda: "2026-05-05T00:00:06+00:00"
        )

        inserted = store.enqueue_outbox_event(
            "evt1",
            "lineage_publish",
            '{"snapshot":"snap:1"}',
            "snap:1",
            max_attempts=2,
        )
        duplicate = store.enqueue_outbox_event(
            "evt1",
            "lineage_publish",
            '{"snapshot":"snap:duplicate"}',
            "snap:duplicate",
            max_attempts=5,
        )

        assert inserted is True
        assert duplicate is False
        assert runtime.outbox_events["evt1"]["payload"] == '{"snapshot":"snap:1"}'
        assert runtime.outbox_events["evt1"]["snapshot_id"] == "snap:1"
        assert runtime.outbox_events["evt1"]["max_attempts"] == 2

    def test_claim_outbox_event_returns_in_progress_payload_after_claim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SnapshotStore.__new__(SnapshotStore)
        runtime = _FakePostgresQueueRuntime()
        runtime.add_outbox_event(
            event_id="evt1",
            event_type="lineage_publish",
            payload='{"snapshot":"snap:1"}',
            snapshot_id="snap:1",
            created_at="2026-05-05T00:00:00+00:00",
        )
        runtime.add_outbox_event(
            event_id="evt2",
            event_type="lineage_publish",
            payload='{"snapshot":"snap:2"}',
            snapshot_id="snap:2",
            status="failed",
            attempts=1,
            max_attempts=3,
            created_at="2026-05-05T00:00:01+00:00",
        )
        runtime.add_outbox_event(
            event_id="evt3",
            event_type="lineage_publish",
            payload='{"snapshot":"snap:3"}',
            snapshot_id="snap:3",
            status="failed",
            attempts=3,
            max_attempts=3,
            created_at="2026-05-05T00:00:02+00:00",
        )
        store.db_runtime = runtime

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        def _boom_event(_: OutboxEventRecord) -> dict[str, Any]:
            raise AssertionError(
                "snapshot store must not call OutboxEventRecord.to_dict()"
            )

        monkeypatch.setattr(
            snapshot_module, "utc_now", lambda: "2026-05-05T00:00:06+00:00"
        )
        monkeypatch.setattr(runtime, "row_to_dict", _boom)
        monkeypatch.setattr(OutboxEventRecord, "to_dict", _boom_event)

        events = store.claim_outbox_event(limit=10)

        assert [event["event_id"] for event in events] == ["evt1", "evt2"]
        assert all(event["status"] == "in_progress" for event in events)
        assert all(
            event["last_attempt_at"] == "2026-05-05T00:00:06+00:00" for event in events
        )
        assert runtime.outbox_events["evt1"]["status"] == "in_progress"
        assert runtime.outbox_events["evt2"]["status"] == "in_progress"
        assert runtime.outbox_events["evt3"]["status"] == "failed"
        assert store.get_outbox_pending_count() == 0

    def test_claim_outbox_event_records_returns_typed_records(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SnapshotStore.__new__(SnapshotStore)
        runtime = _FakePostgresQueueRuntime()
        runtime.add_outbox_event(
            event_id="evt-record-1",
            event_type="lineage_publish",
            payload='{"snapshot":"snap:record:1"}',
            snapshot_id="snap:record:1",
            created_at="2026-05-05T00:00:00+00:00",
        )
        runtime.add_outbox_event(
            event_id="evt-record-2",
            event_type="lineage_publish",
            payload='{"snapshot":"snap:record:2"}',
            snapshot_id="snap:record:2",
            status="failed",
            attempts=1,
            max_attempts=3,
            error_message="retry",
            created_at="2026-05-05T00:00:01+00:00",
        )
        store.db_runtime = runtime

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        def _boom_event(_: OutboxEventRecord) -> dict[str, Any]:
            raise AssertionError(
                "typed outbox claim must not call OutboxEventRecord.to_dict()"
            )

        monkeypatch.setattr(
            snapshot_module, "utc_now", lambda: "2026-05-05T00:00:08+00:00"
        )
        monkeypatch.setattr(runtime, "row_to_dict", _boom)
        monkeypatch.setattr(OutboxEventRecord, "to_dict", _boom_event)

        events = store.claim_outbox_event_records(limit=10)

        assert events == [
            OutboxEventRecord(
                event_id="evt-record-1",
                event_type="lineage_publish",
                payload='{"snapshot":"snap:record:1"}',
                snapshot_id="snap:record:1",
                status="in_progress",
                attempts=0,
                max_attempts=5,
                created_at="2026-05-05T00:00:00+00:00",
                last_attempt_at="2026-05-05T00:00:08+00:00",
                error_message=None,
            ),
            OutboxEventRecord(
                event_id="evt-record-2",
                event_type="lineage_publish",
                payload='{"snapshot":"snap:record:2"}',
                snapshot_id="snap:record:2",
                status="in_progress",
                attempts=1,
                max_attempts=3,
                created_at="2026-05-05T00:00:01+00:00",
                last_attempt_at="2026-05-05T00:00:08+00:00",
                error_message="retry",
            ),
        ]
        assert runtime.outbox_events["evt-record-1"]["status"] == "in_progress"
        assert runtime.outbox_events["evt-record-2"]["status"] == "in_progress"
        assert store.claim_outbox_event_records(limit=10) == []

    def test_mark_outbox_event_failed_and_count_use_explicit_row_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SnapshotStore.__new__(SnapshotStore)
        runtime = _FakePostgresQueueRuntime()
        runtime.add_outbox_event(
            event_id="evt4",
            event_type="lineage_publish",
            payload='{"snapshot":"snap:4"}',
            snapshot_id="snap:4",
            status="failed",
            attempts=4,
            max_attempts=5,
        )
        store.db_runtime = runtime

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("snapshot store must not call row_to_dict()")

        monkeypatch.setattr(runtime, "row_to_dict", _boom)

        assert store.get_outbox_pending_count() == 1

        store.mark_outbox_event_failed("evt4", "publish still failing")

        assert runtime.outbox_events["evt4"]["status"] == "dead"
        assert runtime.outbox_events["evt4"]["attempts"] == 5
        assert runtime.outbox_events["evt4"]["error_message"] == "publish still failing"
        assert store.get_outbox_pending_count() == 0


# --- TestIRGraphsRoundtrip ---


class TestIRGraphsRoundtrip:
    @given(
        snap=snapshot_st(),
        graph_data=st.dictionaries(
            identifier,
            st.lists(st.integers(min_value=0, max_value=100), max_size=5),
            max_size=3,
        ),
    )
    @settings(max_examples=15)
    def test_save_load_ir_graphs_roundtrip_property(
        self, snap: IRSnapshot, graph_data: dict[str, Any]
    ):
        """HAPPY: save_ir_graphs then load_ir_graphs roundtrip via JSON."""
        store = _make_store()
        store.save_snapshot(snap)
        path = store.save_ir_graphs(snap.snapshot_id, graph_data)
        assert path.endswith(".json")
        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == graph_data

    @given(snapshot_id=identifier)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_load_ir_graphs_missing_returns_none_property(self, snapshot_id: str):
        """EDGE: load_ir_graphs returns None when snapshot not found."""
        store = _make_store()
        assert store.load_ir_graphs(snapshot_id) is None

    @given(snap=snapshot_st())
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_load_ir_graphs_none_when_not_saved_property(self, snap: IRSnapshot):
        """EDGE: load_ir_graphs returns None when snapshot exists but graphs not saved."""
        store = _make_store()
        store.save_snapshot(snap)
        assert store.load_ir_graphs(snap.snapshot_id) is None

    @given(snap=snapshot_st())
    @settings(max_examples=10)
    def test_save_ir_graphs_returns_json_path_property(self, snap: IRSnapshot):
        """HAPPY: save_ir_graphs returns path ending with ir_graphs.json."""
        store = _make_store()
        store.save_snapshot(snap)
        path = store.save_ir_graphs(snap.snapshot_id, {"nodes": [1, 2, 3]})
        assert "ir_graphs.json" in path

    def test_save_ir_graphs_avoids_object_to_dict_on_opaque_payload(
        self,
    ) -> None:
        class _OpaqueGraphValue:
            def __repr__(self) -> str:
                return "<opaque-graph-value>"

            def to_dict(self) -> dict[str, Any]:
                raise AssertionError("graph serialization must stay field-explicit")

        store = _make_store()
        snap = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:opaque-graph")
        store.save_snapshot(snap)

        store.save_ir_graphs(
            snap.snapshot_id,
            {"opaque": _OpaqueGraphValue(), "edges": [(1, 2), (2, 3)]},
        )
        loaded = store.load_ir_graphs(snap.snapshot_id)

        assert loaded == {
            "opaque": "<opaque-graph-value>",
            "edges": [[1, 2], [2, 3]],
        }

    def test_save_ir_graphs_dataclass_writes_typed_array_manifest(self) -> None:
        store = _make_store()
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:graph-shards",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                    source_set={"fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b",
                    source_set={"fc_structure"},
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:a:b",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b",
                    relation_type="call",
                    resolution_state="structural",
                )
            ],
        )
        store.save_snapshot(snap)
        graphs = IRGraphBuilder().build_graphs(snap)

        path = store.save_ir_graphs(snap.snapshot_id, graphs)
        loaded = store.load_ir_graphs(snap.snapshot_id)

        assert path.endswith("ir_graphs_manifest.json")
        assert os.path.isdir(os.path.join(os.path.dirname(path), "ir_graph_edges"))
        assert isinstance(loaded, IRGraphs)
        assert loaded.call_graph.number_of_edges() == 1

    def test_save_ir_graphs_delta_reuses_unchanged_graph_shards(self) -> None:
        store = _make_store()
        previous = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:graph-delta-prev",
            units=[
                IRCodeUnit(
                    unit_id="unit:a",
                    kind="function",
                    path="pkg/a.py",
                    language="python",
                    display_name="a",
                    source_set={"fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="unit:b",
                    kind="function",
                    path="pkg/b.py",
                    language="python",
                    display_name="b",
                    source_set={"fc_structure"},
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:import",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b",
                    relation_type="import",
                    resolution_state="structural",
                ),
                IRRelation(
                    relation_id="rel:call",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b",
                    relation_type="call",
                    resolution_state="structural",
                ),
            ],
        )
        current = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:graph-delta-current",
            units=previous.units,
            relations=[
                IRRelation(
                    relation_id="rel:import",
                    src_unit_id="unit:a",
                    dst_unit_id="unit:b",
                    relation_type="import",
                    resolution_state="structural",
                )
            ],
        )
        builder = IRGraphBuilder()
        store.save_snapshot(previous)
        previous_graphs = builder.build_graphs(previous)
        previous_path = store.save_ir_graphs(previous.snapshot_id, previous_graphs)
        store.save_snapshot(current)
        current_graphs, delta_stats = builder.build_graph_delta(
            current,
            previous_graphs=previous_graphs,
            changed_paths=["pkg/a.py"],
        )

        current_path, save_stats = store.save_ir_graphs_delta(
            current.snapshot_id,
            current_graphs,
            previous_snapshot_id=previous.snapshot_id,
            reusable_graphs=delta_stats["reusable_graphs"],
        )
        loaded = store.load_ir_graphs(current.snapshot_id)

        with open(previous_path, encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        with open(current_path, encoding="utf-8") as handle:
            current_manifest = json.load(handle)
        assert save_stats["ir_graph_shards_reused"] >= 1
        assert save_stats["ir_graph_shards_written"] >= 1
        assert (
            current_manifest["graphs"]["reference_graph"]
            == previous_manifest["graphs"]["reference_graph"]
        )
        assert isinstance(loaded, IRGraphs)
        assert loaded.call_graph.number_of_edges() == 0

    @given(snap=snapshot_st())
    @settings(max_examples=10)
    def test_save_load_ir_graphs_dataclass_roundtrip_property(self, snap: IRSnapshot):
        """REGRESSION: snapshot pipeline passes the IRGraphs dataclass directly."""
        store = _make_store()
        store.save_snapshot(snap)
        graphs = IRGraphBuilder().build_graphs(snap)
        store.save_ir_graphs(snap.snapshot_id, graphs)

        loaded = store.load_ir_graphs(snap.snapshot_id)

        assert isinstance(loaded, IRGraphs)
        assert (
            loaded.dependency_graph.number_of_nodes()
            == graphs.dependency_graph.number_of_nodes()
        )
        assert (
            loaded.call_graph.number_of_edges() == graphs.call_graph.number_of_edges()
        )
        assert (
            loaded.containment_graph.number_of_edges()
            == graphs.containment_graph.number_of_edges()
        )

    @given(
        snap=snapshot_st(),
        graph_obj=st.builds(
            lambda a, b: {a: [b, b + 1]},
            identifier,
            st.integers(min_value=0, max_value=50),
        ),
    )
    @settings(max_examples=15)
    def test_ir_graphs_json_preserves_types_property(
        self, snap: IRSnapshot, graph_obj: dict[str, Any]
    ):
        """HAPPY: JSON serialization preserves dict structure in graphs."""
        store = _make_store()
        store.save_snapshot(snap)
        store.save_ir_graphs(snap.snapshot_id, graph_obj)
        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert type(loaded) is dict
        for key, value in graph_obj.items():
            assert key in loaded
            assert loaded[key] == value

    @given(snap=snapshot_st())
    @settings(max_examples=10)
    def test_save_ir_graphs_overwrite_property(self, snap: IRSnapshot):
        """HAPPY: saving IR graphs twice overwrites previous data."""
        store = _make_store()
        store.save_snapshot(snap)
        store.save_ir_graphs(snap.snapshot_id, {"v": 1})
        store.save_ir_graphs(snap.snapshot_id, {"v": 2})
        loaded = store.load_ir_graphs(snap.snapshot_id)
        assert loaded == {"v": 2}

    def test_save_load_rich_dataclass_roundtrip(self):
        """REGRESSION: deterministic multi-file snapshot with nested units roundtrips."""
        snap = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:rich",
            units=[
                IRCodeUnit(
                    unit_id="doc:snap:repo:rich:a.py",
                    kind="file",
                    path="a.py",
                    language="python",
                    display_name="a.py",
                    source_set={"fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="doc:snap:repo:rich:b.py",
                    kind="file",
                    path="b.py",
                    language="python",
                    display_name="b.py",
                    source_set={"fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="ast:class:Config",
                    kind="class",
                    path="a.py",
                    language="python",
                    display_name="Config",
                    qualified_name="Config",
                    start_line=1,
                    end_line=20,
                    parent_unit_id="doc:snap:repo:rich:a.py",
                    source_set={"fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="ast:method:Config.load",
                    kind="method",
                    path="a.py",
                    language="python",
                    display_name="load",
                    qualified_name="Config.load",
                    start_line=5,
                    end_line=10,
                    parent_unit_id="ast:class:Config",
                    source_set={"fc_structure"},
                ),
                IRCodeUnit(
                    unit_id="ast:method:run",
                    kind="method",
                    path="b.py",
                    language="python",
                    display_name="run",
                    qualified_name="run",
                    start_line=1,
                    end_line=15,
                    parent_unit_id="doc:snap:repo:rich:b.py",
                    source_set={"fc_structure"},
                ),
            ],
            relations=[
                IRRelation(
                    relation_id="rel:import:b.py:a.py",
                    src_unit_id="doc:snap:repo:rich:b.py",
                    dst_unit_id="doc:snap:repo:rich:a.py",
                    relation_type="import",
                    resolution_state="structural",
                    support_sources={"fc_structure"},
                ),
                IRRelation(
                    relation_id="rel:call:run:Config.load",
                    src_unit_id="ast:method:run",
                    dst_unit_id="ast:method:Config.load",
                    relation_type="call",
                    resolution_state="structural",
                    support_sources={"fc_structure"},
                ),
                IRRelation(
                    relation_id="rel:contain:a.py:Config",
                    src_unit_id="doc:snap:repo:rich:a.py",
                    dst_unit_id="ast:class:Config",
                    relation_type="contain",
                    resolution_state="structural",
                    support_sources={"fc_structure"},
                ),
                IRRelation(
                    relation_id="rel:contain:Config:load",
                    src_unit_id="ast:class:Config",
                    dst_unit_id="ast:method:Config.load",
                    relation_type="contain",
                    resolution_state="structural",
                    support_sources={"fc_structure"},
                ),
            ],
        )
        store = _make_store()
        store.save_snapshot(snap)
        graphs = IRGraphBuilder().build_graphs(snap)
        store.save_ir_graphs(snap.snapshot_id, graphs)

        loaded = store.load_ir_graphs(snap.snapshot_id)

        assert isinstance(loaded, IRGraphs)
        assert loaded.dependency_graph.number_of_nodes() > 0
        assert loaded.call_graph.number_of_edges() > 0
        assert loaded.containment_graph.number_of_edges() > 0
        assert set(loaded.dependency_graph.nodes()) == set(
            graphs.dependency_graph.nodes()
        )
        assert set(loaded.call_graph.edges()) == set(graphs.call_graph.edges())
        assert set(loaded.containment_graph.edges()) == set(
            graphs.containment_graph.edges()
        )
