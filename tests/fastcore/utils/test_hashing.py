"""Tests for fastcode.utils.hashing."""

from __future__ import annotations

from fastcode.utils.hashing import deterministic_event_id, projection_params_hash


class TestProjectionParamsHash:
    def test_deterministic(self) -> None:
        scope = {"snapshot_id": "snap:repo:abc", "kind": "architecture"}
        h1 = projection_params_hash(scope)
        h2 = projection_params_hash(scope)
        assert h1 == h2

    def test_different_scopes_differ(self) -> None:
        h1 = projection_params_hash({"snapshot_id": "a"})
        h2 = projection_params_hash({"snapshot_id": "b"})
        assert h1 != h2


class TestDeterministicEventId:
    def test_deterministic(self) -> None:
        eid1 = deterministic_event_id("snap:repo:abc", "payload")
        eid2 = deterministic_event_id("snap:repo:abc", "payload")
        assert eid1 == eid2

    def test_different_payloads_differ(self) -> None:
        eid1 = deterministic_event_id("snap:repo:abc", "payload1")
        eid2 = deterministic_event_id("snap:repo:abc", "payload2")
        assert eid1 != eid2

    def test_format_starts_with_outbox(self) -> None:
        eid = deterministic_event_id("snap:repo:abc", "payload")
        assert eid.startswith("outbox:")
