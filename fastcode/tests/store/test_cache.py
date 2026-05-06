from __future__ import annotations

from pathlib import Path

import pytest

from fastcode.store.cache import (
    _CACHE_EMBEDDING_KIND,
    _CACHE_JSON_KIND,
    _CACHE_RECORD_MAGIC,
    CacheManager,
)
from fastcode.store.records import (
    DialogueSessionRecord,
    DialogueTurnRecord,
    HandoffArtifactRecord,
    TurnJournalRecord,
    WorkingMemoryRecord,
)


def _cache_config(
    tmp_path: Path, *, cache_queries: bool = False
) -> dict[str, dict[str, object]]:
    return {
        "cache": {
            "enabled": True,
            "backend": "disk",
            "cache_directory": str(tmp_path),
            "cache_queries": cache_queries,
        }
    }


def test_save_dialogue_turn_uses_json_cache_envelope(tmp_path: Path) -> None:
    manager = CacheManager(_cache_config(tmp_path))

    try:
        saved = manager.save_dialogue_turn(
            session_id="session-1",
            turn_number=1,
            query="Where is config loaded?",
            answer="src/config.py",
            summary="Located config load path",
            retrieved_elements=[
                {
                    "file": "src/config.py",
                    "repo": "repo",
                    "type": "file",
                    "name": "config.py",
                    "start_line": 1,
                    "end_line": 20,
                }
            ],
            metadata={"intent": "where", "keywords": ["config"], "multi_turn": True},
        )

        assert saved is True

        raw_turn = manager.cache.get("dialogue_session-1_turn_1")
        raw_index = manager.cache.get("dialogue_session_session-1_index")

        assert isinstance(raw_turn, bytes)
        assert isinstance(raw_index, bytes)
        assert raw_turn.startswith(_CACHE_RECORD_MAGIC + _CACHE_JSON_KIND)
        assert raw_index.startswith(_CACHE_RECORD_MAGIC + _CACHE_JSON_KIND)

        turn = manager.get_dialogue_turn("session-1", 1)
        assert turn is not None
        assert turn["summary"] == "Located config load path"
        assert turn["retrieved_elements"][0]["file"] == "src/config.py"

        session_index = manager._get_session_index("session-1")
        assert session_index is not None
        assert session_index["session_id"] == "session-1"
        assert session_index["total_turns"] == 1
        assert session_index["multi_turn"] is True
        assert isinstance(session_index["created_at"], float)
        assert isinstance(session_index["last_updated"], float)
    finally:
        manager.cache.close()


def test_set_cached_embedding_payload_uses_buffer_cache_envelope(
    tmp_path: Path,
) -> None:
    manager = CacheManager(_cache_config(tmp_path))

    try:
        payload = {
            "embedding_format": "ndarray.float32.v1",
            "embedding_dtype": "float32",
            "embedding_shape": (2,),
            "embedding_bytes": b"\x00\x00\x80?\x00\x00\x00@",
            "provider": "sentence_transformers",
            "model": "test-model",
            "dimension": 2,
            "max_seq_length": 512,
            "normalize": True,
            "text_sha256": "abc123",
        }

        saved = manager.set_cached_embedding_payload("embedding_v2_abc", payload)

        assert saved is True

        raw_value = manager.cache.get("embedding_v2_abc")
        assert isinstance(raw_value, bytes)
        assert raw_value.startswith(_CACHE_RECORD_MAGIC + _CACHE_EMBEDDING_KIND)

        restored = manager.get_cached_embedding_payload("embedding_v2_abc")
        assert restored is not None
        assert restored["embedding_format"] == "ndarray.float32.v1"
        assert restored["embedding_shape"] == [2]
        assert restored["embedding_bytes"] == payload["embedding_bytes"]
    finally:
        manager.cache.close()


def test_query_result_cache_uses_json_cache_envelope(tmp_path: Path) -> None:
    manager = CacheManager(_cache_config(tmp_path, cache_queries=True))

    try:
        result = {
            "answer": "Use src/config.py",
            "sources": [
                {
                    "file": "src/config.py",
                    "repo": "repo",
                    "type": "file",
                    "name": "config.py",
                    "start_line": 1,
                    "end_line": 20,
                }
            ],
            "summary": "Config lives in src/config.py",
        }

        saved = manager.set_query_result("Where is config loaded?", "repo-hash", result)

        assert saved is True

        cache_key = manager._generate_key(
            "query", "Where is config loaded?", "repo-hash"
        )
        raw_value = manager.cache.get(cache_key)
        assert isinstance(raw_value, bytes)
        assert raw_value.startswith(_CACHE_RECORD_MAGIC + _CACHE_JSON_KIND)
        assert (
            manager.get_query_result("Where is config loaded?", "repo-hash") == result
        )
    finally:
        manager.cache.close()


def test_dialogue_record_accessors_use_explicit_serializers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = CacheManager(_cache_config(tmp_path))

    def _boom_turn(_: DialogueTurnRecord) -> dict[str, object]:
        raise AssertionError("cache manager must not call DialogueTurnRecord.to_dict()")

    def _boom_session(_: DialogueSessionRecord) -> dict[str, object]:
        raise AssertionError(
            "cache manager must not call DialogueSessionRecord.to_dict()"
        )

    monkeypatch.setattr(DialogueTurnRecord, "to_dict", _boom_turn)
    monkeypatch.setattr(DialogueSessionRecord, "to_dict", _boom_session)

    try:
        assert manager.save_dialogue_turn(
            session_id="session-typed",
            turn_number=1,
            query="Where is config loaded?",
            answer="src/config.py",
            summary="Located config load path",
            retrieved_elements=[{"file": "src/config.py", "type": "file"}],
            metadata={"multi_turn": True},
        )

        turn_record = manager.get_dialogue_turn_record("session-typed", 1)
        session_record = manager.get_session_index_record("session-typed")
        history = manager.get_dialogue_history("session-typed")
        sessions = manager.list_sessions()

        assert turn_record is not None
        assert turn_record.query == "Where is config loaded?"
        assert session_record is not None
        assert session_record.total_turns == 1
        assert session_record.multi_turn is True
        assert history[0]["summary"] == "Located config load path"
        assert sessions[0]["session_id"] == "session-typed"
    finally:
        manager.cache.close()


def test_dialogue_history_records_and_delete_session_use_typed_session_index(
    tmp_path: Path,
) -> None:
    manager = CacheManager(_cache_config(tmp_path))

    try:
        assert manager.save_dialogue_turn(
            session_id="session-history",
            turn_number=1,
            query="Q1",
            answer="A1",
            summary="S1",
            metadata={},
        )
        assert manager.save_dialogue_turn(
            session_id="session-history",
            turn_number=2,
            query="Q2",
            answer="A2",
            summary="S2",
            metadata={"multi_turn": True},
        )

        history_records = manager.get_dialogue_history_records(
            "session-history", max_turns=2
        )
        summaries = manager.get_recent_summaries("session-history", 2)
        session_records = manager.list_session_records()

        assert [record.turn_number for record in history_records] == [1, 2]
        assert summaries == [
            {"turn_number": 1, "query": "Q1", "summary": "S1"},
            {"turn_number": 2, "query": "Q2", "summary": "S2"},
        ]
        assert len(session_records) == 1
        assert session_records[0].session_id == "session-history"
        assert session_records[0].total_turns == 2
        assert session_records[0].multi_turn is True

        assert manager.delete_session("session-history") is True
        assert manager.get_dialogue_turn_record("session-history", 1) is None
        assert manager.get_session_index_record("session-history") is None
    finally:
        manager.cache.close()


def test_agent_context_records_roundtrip_and_delete_with_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = CacheManager(_cache_config(tmp_path))

    def _boom_working_memory(_: WorkingMemoryRecord) -> dict[str, object]:
        raise AssertionError(
            "cache manager must not call WorkingMemoryRecord.to_dict()"
        )

    def _boom_journal(_: TurnJournalRecord) -> dict[str, object]:
        raise AssertionError("cache manager must not call TurnJournalRecord.to_dict()")

    def _boom_handoff(_: HandoffArtifactRecord) -> dict[str, object]:
        raise AssertionError(
            "cache manager must not call HandoffArtifactRecord.to_dict()"
        )

    monkeypatch.setattr(WorkingMemoryRecord, "to_dict", _boom_working_memory)
    monkeypatch.setattr(TurnJournalRecord, "to_dict", _boom_journal)
    monkeypatch.setattr(HandoffArtifactRecord, "to_dict", _boom_handoff)

    try:
        assert manager.save_dialogue_turn(
            session_id="session-agent",
            turn_number=1,
            query="How does auth work?",
            answer="See src/auth.py",
            summary="Auth flow grounded in src/auth.py",
            retrieved_elements=[{"file": "src/auth.py", "type": "file"}],
            metadata={"multi_turn": True},
        )

        working_memory_record = WorkingMemoryRecord(
            session_id="session-agent",
            turn_number=1,
            snapshot_id="snap:1",
            artifact_key="art:1",
            compiler_fingerprint="fcx-v1",
            payload_json='{"session_id":"session-agent","turn_number":1}',
            stable_fcx="<fcx:stable>\n@fcx mode=stable sid=session-agent turn=1 snap=snap:1 art=art:1 fp=fcx-v1\nEND refs=0\n</fcx:stable>",
            turn_fcx="<fcx:turn>\n@fcx mode=turn sid=session-agent turn=1 snap=snap:1 art=art:1 fp=fcx-v1\nEND refs=0\n</fcx:turn>",
            obs_fcx="<fcx:obs>\n@fcx mode=tool sid=session-agent turn=1 snap=snap:1 art=art:1 fp=fcx-v1\nEND refs=0\n</fcx:obs>",
            full_fcx="full-fcx",
            created_at=1234.5,
        )
        turn_journal_record = TurnJournalRecord(
            session_id="session-agent",
            turn_number=1,
            snapshot_id="snap:1",
            artifact_key="art:1",
            compiler_fingerprint="fcx-v1",
            payload_json='{"session_id":"session-agent","turn_number":1}',
            created_at=1234.5,
        )
        handoff_record = HandoffArtifactRecord(
            artifact_id="hf_123",
            session_id="session-agent",
            turn_number=1,
            snapshot_id="snap:1",
            compiler_fingerprint="fcx-v1",
            mode="delegate",
            payload_json='{"artifact_id":"hf_123","session_id":"session-agent"}',
            full_fcx="full-fcx",
            created_at=1234.5,
        )

        assert manager.save_working_memory_record(working_memory_record)
        assert manager.save_turn_journal_record(turn_journal_record)
        assert manager.save_handoff_artifact_record(handoff_record)

        restored_working_memory = manager.get_working_memory_record("session-agent", 1)
        restored_journal = manager.get_turn_journal_record("session-agent", 1)
        restored_handoff = manager.get_handoff_artifact_record("hf_123")
        handoffs = manager.list_handoff_artifact_records("session-agent")

        assert restored_working_memory is not None
        assert restored_working_memory.full_fcx == "full-fcx"
        assert manager.get_latest_working_memory_record("session-agent") is not None
        assert restored_journal is not None
        assert restored_journal.compiler_fingerprint == "fcx-v1"
        assert restored_handoff is not None
        assert restored_handoff.mode == "delegate"
        assert [record.artifact_id for record in handoffs] == ["hf_123"]

        assert manager.delete_session("session-agent") is True
        assert manager.get_working_memory_record("session-agent", 1) is None
        assert manager.get_turn_journal_record("session-agent", 1) is None
        assert manager.get_handoff_artifact_record("hf_123") is None
    finally:
        manager.cache.close()
