# tests/test_core_graph_build.py
"""Tests for pure graph payload construction."""

from fastcode.core.graph_build import (
    build_code_graph_payload,
    deterministic_event_id,
)


class TestDeterministicEventId:
    def test_deterministic(self):
        result1 = deterministic_event_id("snap:test:abc", "payload1")
        result2 = deterministic_event_id("snap:test:abc", "payload1")
        assert result1 == result2

    def test_different_inputs(self):
        result1 = deterministic_event_id("snap:test:abc", "payload1")
        result2 = deterministic_event_id("snap:test:abc", "payload2")
        assert result1 != result2

    def test_format(self):
        result = deterministic_event_id("snap:test:abc", "payload")
        assert result.startswith("outbox:")
        assert "snap:test:abc" in result


class TestBuildCodeGraphPayload:
    def test_skips_file_units(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [
                {"kind": "file", "unit_id": "file1"},
                {"kind": "function", "unit_id": "sym1", "display_name": "func1"},
            ],
            "relations": [],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["nodes"]) == 1
        assert "sym1" in result["nodes"][0]["id"]

    def test_skips_doc_units(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [
                {"kind": "doc", "unit_id": "doc1"},
            ],
            "relations": [],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["nodes"]) == 0

    def test_builds_edges(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [],
            "relations": [
                {
                    "relation_id": "r1",
                    "src_unit_id": "a",
                    "dst_unit_id": "b",
                    "relation_type": "calls",
                    "resolution_state": "anchored",
                },
            ],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["edges"]) == 1
        assert result["edges"][0]["type"] == "calls"

    def test_skips_units_without_id(self):
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [{"kind": "function"}],
            "relations": [],
        }
        result = build_code_graph_payload(snapshot)
        assert len(result["nodes"]) == 0

    def test_empty_snapshot(self):
        result = build_code_graph_payload({"snapshot_id": "snap:test:abc"})
        assert result == {"nodes": [], "edges": []}

    def test_confidence_uses_resolution_state(self):
        """Verify confidence is derived from resolution_state via semantic_ir mapping."""
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [],
            "relations": [
                {
                    "relation_id": "r1",
                    "src_unit_id": "a",
                    "dst_unit_id": "b",
                    "relation_type": "calls",
                    "resolution_state": "anchored",
                },
            ],
        }
        result = build_code_graph_payload(snapshot)
        # _resolution_to_confidence("anchored") returns "precise"
        assert result["edges"][0]["confidence"] == "precise"

    def test_confidence_default(self):
        """Verify default confidence when resolution_state is empty."""
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [],
            "relations": [
                {
                    "relation_id": "r1",
                    "src_unit_id": "a",
                    "dst_unit_id": "b",
                    "relation_type": "calls",
                    "resolution_state": "",
                },
            ],
        }
        result = build_code_graph_payload(snapshot)
        # _resolution_to_confidence("") returns "derived"
        assert result["edges"][0]["confidence"] == "derived"

    def test_source_set_normalization(self):
        """Verify source_set is always a list in nodes and edges."""
        snapshot = {
            "snapshot_id": "snap:test:abc",
            "units": [
                {"kind": "function", "unit_id": "sym1", "source_set": ("scip",)},
            ],
            "relations": [
                {
                    "relation_id": "r1",
                    "src_unit_id": "a",
                    "dst_unit_id": "b",
                    "relation_type": "calls",
                    "resolution_state": "anchored",
                    "support_sources": ("scip",),
                },
            ],
        }
        result = build_code_graph_payload(snapshot)
        assert isinstance(result["nodes"][0]["props"]["source_set"], list)
        assert isinstance(result["edges"][0]["source_set"], list)
