"""Tests for ContextFacade — session/context logic extracted from FastCode."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fastcode.app.store.context_facade import ContextFacade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_working_memory_record(
    session_id: str = "sess-1",
    turn_number: int = 1,
    payload: dict[str, Any] | None = None,
) -> MagicMock:
    payload = payload or _minimal_working_memory_payload(session_id, turn_number)
    rec = MagicMock()
    rec.session_id = session_id
    rec.turn_number = turn_number
    rec.snapshot_id = "snap-1"
    rec.artifact_key = "wm-1"
    rec.compiler_fingerprint = "fp-1"
    rec.stable_fcx = "stable-fcx-data"
    rec.turn_fcx = "turn-fcx-data"
    rec.obs_fcx = "obs-fcx-data"
    rec.full_fcx = "full-fcx-data"
    rec.payload_json = json.dumps(payload)
    return rec


def _minimal_working_memory_payload(
    session_id: str = "sess-1", turn_number: int = 1
) -> dict[str, Any]:
    return {
        "artifact_id": "wm-art-1",
        "session_id": session_id,
        "turn_number": turn_number,
        "snapshot_id": "snap-1",
        "artifact_key": "wm-1",
        "compiler_fingerprint": "fp-1",
        "intent": {"question": "what?"},
        "plan": {"steps": []},
        "working_set": {"evidence_refs": [], "tool_observations": []},
        "evidence_refs": [],
        "tool_observations": [],
        "hypotheses": [],
        "rejected_hypotheses": [],
        "accepted_facts": [],
        "risk_state": {"overall": "low", "factors": []},
        "acceptance_contract": {"threshold": 0.5, "criteria": []},
        "full_fcx": "",
    }


def _make_handoff_record(
    artifact_id: str = "ho-1",
    session_id: str = "sess-1",
) -> MagicMock:
    payload = {
        "artifact_id": artifact_id,
        "session_id": session_id,
        "turn_number": 1,
        "snapshot_id": "snap-1",
        "compiler_fingerprint": "fp-1",
        "mode": "delegate",
        "working_memory_summary": {},
        "evidence_refs": [],
        "accepted_facts": [],
        "hypotheses": [],
        "risk_state": {"overall": "low", "factors": []},
        "acceptance_contract": {"threshold": 0.5, "criteria": []},
        "full_fcx": "",
        "created_at": 1000.0,
    }
    rec = MagicMock()
    rec.payload_json = json.dumps(payload)
    return rec


def _make_bundle_record(
    session_id: str = "sess-1",
    turn_number: int = 1,
) -> MagicMock:
    wm_payload = _minimal_working_memory_payload(session_id, turn_number)
    payload = {
        "bundle_id": "bundle-1",
        "session_id": session_id,
        "turn_number": turn_number,
        "snapshot_id": "snap-1",
        "artifact_key": "wm-1",
        "compiler_fingerprint": "fp-1",
        "working_memory": wm_payload,
        "turn_journal": {
            "session_id": session_id,
            "turn_number": turn_number,
            "entries": [],
        },
        "distillation": {
            "invalidation_key": "ik-1",
            "distillation_id": "dist-1",
            "fingerprint": "fp-1",
        },
        "activation": {
            "activation_id": "act-1",
            "bundle_id": "bundle-1",
            "active_ref_ids": (),
            "active_fact_ids": (),
            "active_hypothesis_ids": (),
        },
    }
    rec = MagicMock()
    rec.payload_json = json.dumps(payload)
    return rec


def _make_session_index_record(
    session_id: str = "sess-1",
    multi_turn: bool = False,
) -> MagicMock:
    rec = MagicMock()
    rec.session_id = session_id
    rec.created_at = 1000.0
    rec.total_turns = 2
    rec.last_updated = 2000.0
    rec.multi_turn = multi_turn
    return rec


def _make_dialogue_turn_record(
    session_id: str = "sess-1",
    turn_number: int = 1,
    query: str = "Hello world",
) -> MagicMock:
    rec = MagicMock()
    rec.session_id = session_id
    rec.turn_number = turn_number
    rec.query = query
    return rec


def _make_facade() -> tuple[ContextFacade, MagicMock]:
    cache_manager = MagicMock()
    facade = ContextFacade(cache_manager)
    return facade, cache_manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetSessionHistory:
    def test_delegates_to_cache_manager(self):
        facade, cm = _make_facade()
        expected = [MagicMock()]
        cm.get_dialogue_history_records.return_value = expected
        result = facade.get_session_history("sess-1")
        assert result is expected
        cm.get_dialogue_history_records.assert_called_once_with("sess-1")


class TestGetTurnContext:
    def test_returns_fcx_format(self):
        facade, cm = _make_facade()
        record = _make_working_memory_record()
        cm.get_latest_working_memory_record.return_value = record
        result = facade.get_turn_context("sess-1")
        assert result["format"] == "fcx"
        assert result["session_id"] == "sess-1"
        assert result["stable_fcx"] == "stable-fcx-data"

    def test_returns_json_format(self):
        facade, cm = _make_facade()
        record = _make_working_memory_record()
        cm.get_latest_working_memory_record.return_value = record
        result = facade.get_turn_context("sess-1", output_format="json")
        assert result["format"] == "json"
        assert "artifact" in result

    def test_raises_when_no_record(self):
        facade, cm = _make_facade()
        cm.get_latest_working_memory_record.return_value = None
        with pytest.raises(RuntimeError, match="working memory not found"):
            facade.get_turn_context("sess-1")


class TestCreateHandoff:
    def test_builds_and_saves_handoff(self):
        facade, cm = _make_facade()
        record = _make_working_memory_record()
        cm.get_latest_working_memory_record.return_value = record
        result = facade.create_handoff("sess-1")
        assert "artifact_id" in result
        assert "mode" in result
        cm.save_handoff_artifact_record.assert_called_once()

    def test_uses_specified_turn_number(self):
        facade, cm = _make_facade()
        record = _make_working_memory_record()
        cm.get_working_memory_record.return_value = record
        facade.create_handoff("sess-1", turn_number=3)
        cm.get_working_memory_record.assert_called_once_with("sess-1", 3)


class TestGetHandoffArtifact:
    def test_returns_payload(self):
        facade, cm = _make_facade()
        record = _make_handoff_record()
        cm.get_handoff_artifact_record.return_value = record
        result = facade.get_handoff_artifact("ho-1")
        assert "artifact_id" in result

    def test_raises_when_not_found(self):
        facade, cm = _make_facade()
        cm.get_handoff_artifact_record.return_value = None
        with pytest.raises(RuntimeError, match="handoff artifact not found"):
            facade.get_handoff_artifact("ho-missing")


class TestGetContextBundle:
    def test_returns_json_format(self):
        facade, cm = _make_facade()
        cm.get_latest_working_memory_record.return_value = None
        cm.get_context_bundle_record.return_value = None
        cm.get_latest_context_bundle_record.return_value = _make_bundle_record()
        result = facade.get_context_bundle("sess-1")
        assert result["format"] == "json"
        assert "bundle" in result

    def test_returns_rendered_format(self):
        facade, cm = _make_facade()
        cm.get_latest_working_memory_record.return_value = None
        cm.get_context_bundle_record.return_value = None
        cm.get_latest_context_bundle_record.return_value = _make_bundle_record()
        result = facade.get_context_bundle("sess-1", output_format="rendered")
        assert result["format"] == "rendered"
        assert "rendered" in result


class TestGetContextBundleById:
    def test_loads_by_bundle_id(self):
        facade, cm = _make_facade()
        cm.get_context_bundle_record_by_id.return_value = _make_bundle_record()
        result = facade.get_context_bundle_by_id("bundle-1")
        assert result["bundle_id"] == "bundle-1"
        cm.get_context_bundle_record_by_id.assert_called_once_with("bundle-1")

    def test_raises_when_not_found(self):
        facade, cm = _make_facade()
        cm.get_context_bundle_record_by_id.return_value = None
        with pytest.raises(RuntimeError, match="context bundle not found"):
            facade.get_context_bundle_by_id("bundle-missing")


class TestExpandContextBundleRef:
    def test_returns_expanded_ref(self):
        facade, cm = _make_facade()
        cm.get_context_bundle_record_by_id.return_value = _make_bundle_record()
        # Patch expand_bundle_source_ref to return a value
        with patch(
            "fastcode.app.store.context_facade.expand_bundle_source_ref",
            return_value={"ref_id": "e1", "expanded": True},
        ):
            result = facade.expand_context_bundle_ref("e1", bundle_id="bundle-1")
        assert result["ref_id"] == "e1"

    def test_raises_when_ref_not_found(self):
        facade, cm = _make_facade()
        cm.get_context_bundle_record_by_id.return_value = _make_bundle_record()
        with (
            patch(
                "fastcode.app.store.context_facade.expand_bundle_source_ref",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="context bundle ref not found"),
        ):
            facade.expand_context_bundle_ref("e1", bundle_id="bundle-1")


class TestCreateContextActivation:
    def test_builds_and_saves_activation(self):
        facade, cm = _make_facade()
        cm.get_context_bundle_record_by_id.return_value = _make_bundle_record()
        result = facade.create_context_activation(bundle_id="bundle-1")
        assert "activation_id" in result
        assert "bundle_id" in result
        cm.save_context_activation_record.assert_called_once()


class TestExpandContextRef:
    def test_returns_matching_ref(self):
        facade, cm = _make_facade()
        payload = _minimal_working_memory_payload()
        payload["evidence_refs"] = [
            {
                "ref_id": "e1",
                "kind": "symbol",
                "repo_name": "repo",
                "snapshot_id": "snap-1",
                "path": "main.py",
                "symbol_id": "sym-1",
                "lines": "10-20",
                "label": "main",
                "score": 0.9,
                "source": "retrieval",
                "fresh": True,
            }
        ]
        record = _make_working_memory_record(payload=payload)
        cm.get_working_memory_record.return_value = record
        result = facade.expand_context_ref("sess-1", 1, "e1")
        assert result["ref_id"] == "e1"
        assert result["path"] == "main.py"

    def test_raises_when_ref_not_found(self):
        facade, cm = _make_facade()
        record = _make_working_memory_record()
        cm.get_working_memory_record.return_value = record
        with pytest.raises(RuntimeError, match="context ref not found"):
            facade.expand_context_ref("sess-1", 1, "e-missing")


class TestDeleteSession:
    def test_delegates_to_cache_manager(self):
        facade, cm = _make_facade()
        cm.delete_session.return_value = True
        result = facade.delete_session("sess-1")
        assert result is True
        cm.delete_session.assert_called_once_with("sess-1")


class TestListSessions:
    def test_enriches_with_title(self):
        facade, cm = _make_facade()
        session_rec = _make_session_index_record()
        turn_rec = _make_dialogue_turn_record(query="How does auth work?")
        cm.list_session_records.return_value = [session_rec]
        cm.get_dialogue_turn_record.return_value = turn_rec
        result = facade.list_sessions()
        assert len(result) == 1
        assert result[0]["title"] == "How does auth work?"
        assert result[0]["session_id"] == "sess-1"

    def test_truncates_long_title(self):
        facade, cm = _make_facade()
        session_rec = _make_session_index_record()
        long_query = "x" * 100
        turn_rec = _make_dialogue_turn_record(query=long_query)
        cm.list_session_records.return_value = [session_rec]
        cm.get_dialogue_turn_record.return_value = turn_rec
        result = facade.list_sessions()
        assert len(result[0]["title"]) <= 80

    def test_fallback_title_when_no_turn(self):
        facade, cm = _make_facade()
        session_rec = _make_session_index_record()
        cm.list_session_records.return_value = [session_rec]
        cm.get_dialogue_turn_record.return_value = None
        result = facade.list_sessions()
        assert result[0]["title"] == "Session sess-1"


class TestGetSessionMultiTurn:
    def test_returns_true_when_multi_turn(self):
        facade, cm = _make_facade()
        record = MagicMock()
        record.multi_turn = True
        cm.get_session_index_record.return_value = record
        assert facade.get_session_multi_turn("sess-1") is True

    def test_returns_false_when_not_multi_turn(self):
        facade, cm = _make_facade()
        record = MagicMock()
        record.multi_turn = False
        cm.get_session_index_record.return_value = record
        assert facade.get_session_multi_turn("sess-1") is False

    def test_returns_false_when_no_record(self):
        facade, cm = _make_facade()
        cm.get_session_index_record.return_value = None
        assert facade.get_session_multi_turn("sess-1") is False


class TestParseWorkingMemoryRecordPayload:
    def test_parses_valid_payload(self):
        payload = _minimal_working_memory_payload()
        record = MagicMock()
        record.payload_json = json.dumps(payload)
        artifact = ContextFacade._parse_working_memory_record_payload(record)
        assert artifact.session_id == "sess-1"
        assert artifact.turn_number == 1

    def test_raises_on_invalid_payload(self):
        record = MagicMock()
        record.payload_json = "[]"
        with pytest.raises(RuntimeError, match="working memory payload is invalid"):
            ContextFacade._parse_working_memory_record_payload(record)


class TestOptionalStringTuple:
    def test_none_returns_none(self):
        assert ContextFacade._optional_string_tuple(None) is None

    def test_string_returns_singleton_tuple(self):
        assert ContextFacade._optional_string_tuple("abc") == ("abc",)

    def test_list_returns_tuple(self):
        assert ContextFacade._optional_string_tuple(["a", "b"]) == ("a", "b")
