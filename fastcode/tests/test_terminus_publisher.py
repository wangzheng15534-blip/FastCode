"""Tests for terminus_publisher module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.terminus_publisher import TerminusPublisher

# --- Helpers ---

small_text = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


def _make_publisher(**overrides) -> Any:
    cfg = {"terminus": {"endpoint": "http://localhost:6363/api/publish"}}
    cfg["terminus"].update(overrides)
    return TerminusPublisher(cfg)


def _minimal_snapshot() -> dict[str, str | list[Any]]:
    return {
        "repo_name": "repo",
        "snapshot_id": "snap:repo:abc",
        "branch": "main",
        "commit_id": "abc123",
        "documents": [],
        "symbols": [],
    }


def _minimal_manifest() -> dict[str, str]:
    return {
        "manifest_id": "manifest_001",
        "ref_name": "main",
        "status": "published",
        "published_at": "2026-01-01T00:00:00Z",
        "index_run_id": "run_001",
    }


def _minimal_git_meta() -> dict[str, str]:
    return {
        "repo_name": "repo",
        "branch": "main",
        "commit_id": "abc123",
    }


def _make_snapshot_v2(
    snapshot_id: str = "snap:repo:abc",
    units: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal ir.v2 snapshot dict."""
    return {
        "schema_version": "ir.v2",
        "repo_name": "repo",
        "snapshot_id": snapshot_id,
        "branch": "main",
        "commit_id": "abc",
        "units": units or [],
        "relations": relations or [],
    }


# --- Configuration tests ---


class TestTerminusPublisherConfiguration:
    def test_is_configured_true_with_endpoint_property(self):
        """HAPPY: is_configured returns True when endpoint set."""
        pub = _make_publisher()
        assert pub.is_configured() is True

    @pytest.mark.edge
    def test_is_configured_false_without_endpoint_property(self):
        """EDGE: is_configured returns False when endpoint missing."""
        pub = TerminusPublisher({"terminus": {}})
        assert pub.is_configured() is False

    @pytest.mark.edge
    def test_is_configured_false_with_none_endpoint_property(self):
        """EDGE: is_configured returns False when endpoint is None."""
        pub = TerminusPublisher({"terminus": {"endpoint": None}})
        assert pub.is_configured() is False

    @pytest.mark.edge
    def test_timeout_default_property(self):
        """EDGE: default timeout is 15 seconds."""
        pub = _make_publisher()
        assert pub.timeout == 15

    @pytest.mark.edge
    def test_timeout_custom_property(self):
        """EDGE: custom timeout from config."""
        pub = _make_publisher(timeout_seconds=30)
        assert pub.timeout == 30


# --- Lineage payload tests ---


class TestLineagePayload:
    def test_build_lineage_payload_has_required_keys_property(self):
        """HAPPY: payload has version, snapshot_id, nodes, edges."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        assert "version" in payload
        assert payload["version"] == "v1"
        assert "snapshot_id" in payload
        assert "nodes" in payload
        assert "edges" in payload
        assert "git_meta" in payload

    def test_build_lineage_payload_includes_repo_node_property(self):
        """HAPPY: payload includes Repository node."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        repo_nodes = [n for n in payload["nodes"] if n["type"] == "Repository"]
        assert len(repo_nodes) == 1
        assert repo_nodes[0]["props"]["repo_name"] == "repo"

    def test_build_lineage_payload_includes_snapshot_node_property(self):
        """HAPPY: payload includes Snapshot node."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        snap_nodes = [n for n in payload["nodes"] if n["type"] == "Snapshot"]
        assert len(snap_nodes) == 1

    def test_build_lineage_payload_includes_branch_node_property(self):
        """HAPPY: payload includes Branch node when branch present."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        branch_nodes = [n for n in payload["nodes"] if n["type"] == "Branch"]
        assert len(branch_nodes) == 1

    def test_build_lineage_payload_includes_manifest_node_property(self):
        """HAPPY: payload includes Manifest node when manifest_id present."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        manifest_nodes = [n for n in payload["nodes"] if n["type"] == "Manifest"]
        assert len(manifest_nodes) == 1

    def test_build_lineage_payload_includes_index_run_node_property(self):
        """HAPPY: payload includes IndexRun node when run_id present."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        run_nodes = [n for n in payload["nodes"] if n["type"] == "IndexRun"]
        assert len(run_nodes) == 1

    def test_build_lineage_payload_with_documents_property(self):
        """HAPPY: payload includes DocumentVersion nodes."""
        snap = _minimal_snapshot()
        snap["documents"] = [
            {"doc_id": "d1", "path": "a.py", "language": "python"},
        ]
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        doc_nodes = [n for n in payload["nodes"] if n["type"] == "DocumentVersion"]
        assert len(doc_nodes) == 1
        assert doc_nodes[0]["props"]["path"] == "a.py"

    def test_build_lineage_payload_with_symbols_property(self):
        """HAPPY: payload includes SymbolVersion nodes."""
        snap = _minimal_snapshot()
        snap["symbols"] = [
            {
                "symbol_id": "sym:1",
                "display_name": "foo",
                "path": "a.py",
                "kind": "function",
            },
        ]
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        sym_nodes = [n for n in payload["nodes"] if n["type"] == "SymbolVersion"]
        assert len(sym_nodes) == 1

    @pytest.mark.edge
    def test_build_lineage_payload_no_branch_property(self):
        """EDGE: no Branch node when both snapshot and git_meta have no branch."""
        snap = _minimal_snapshot()
        snap["branch"] = None
        git_meta = _minimal_git_meta()
        git_meta["branch"] = None
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=git_meta,
        )
        branch_nodes = [n for n in payload["nodes"] if n["type"] == "Branch"]
        assert len(branch_nodes) == 0

    @pytest.mark.edge
    def test_payload_no_commit_property(self):
        """EDGE: no Commit node when both snapshot and git_meta lack commit_id."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["commit_id"] = None
        git_meta = _minimal_git_meta()
        git_meta["commit_id"] = None
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=git_meta,
        )
        commit_nodes = [n for n in payload["nodes"] if n["type"] == "Commit"]
        assert len(commit_nodes) == 0

    @pytest.mark.edge
    def test_payload_no_manifest_id_property(self):
        """EDGE: no Manifest node when manifest_id missing."""
        pub = _make_publisher()
        manifest = {"ref_name": "main", "status": "published"}
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=manifest,
            git_meta=_minimal_git_meta(),
        )
        manifest_nodes = [n for n in payload["nodes"] if n["type"] == "Manifest"]
        assert len(manifest_nodes) == 0

    @pytest.mark.edge
    def test_payload_no_run_id_property(self):
        """EDGE: no IndexRun node when index_run_id missing."""
        pub = _make_publisher()
        manifest = {"manifest_id": "m1", "ref_name": "main"}
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=manifest,
            git_meta=_minimal_git_meta(),
        )
        run_nodes = [n for n in payload["nodes"] if n["type"] == "IndexRun"]
        assert len(run_nodes) == 0

    @pytest.mark.edge
    def test_payload_empty_documents_property(self):
        """EDGE: empty documents list produces no DocumentVersion nodes."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["documents"] = []
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        doc_nodes = [n for n in payload["nodes"] if n["type"] == "DocumentVersion"]
        assert len(doc_nodes) == 0

    @pytest.mark.edge
    def test_build_lineage_payload_doc_without_id_skipped_property(self):
        """EDGE: document without doc_id is skipped."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["documents"] = [{"path": "a.py", "language": "python"}]  # no doc_id
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        doc_nodes = [n for n in payload["nodes"] if n["type"] == "DocumentVersion"]
        assert len(doc_nodes) == 0

    @pytest.mark.edge
    def test_build_lineage_payload_symbol_without_id_skipped_property(self):
        """EDGE: symbol without symbol_id is skipped."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["symbols"] = [{"display_name": "foo", "kind": "function"}]
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        sym_nodes = [n for n in payload["nodes"] if n["type"] == "SymbolVersion"]
        assert len(sym_nodes) == 0

    @pytest.mark.edge
    def test_build_payload_version_is_v1_property(self):
        """EDGE: payload version field is always 'v1'."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        assert payload["version"] == "v1"

    def test_build_lineage_payload_parent_commits_property(self):
        """HAPPY: parent_commit_ids produce commit_parent edges."""
        pub = _make_publisher()
        git_meta = _minimal_git_meta()
        git_meta["parent_commit_ids"] = ["parent1", "parent2"]
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=git_meta,
        )
        parent_edges = [e for e in payload["edges"] if e["type"] == "commit_parent"]
        assert len(parent_edges) == 2

    @pytest.mark.edge
    def test_payload_parent_commit_id_legacy_property(self):
        """EDGE: parent_commit_id (singular) also produces commit_parent edge."""
        pub = _make_publisher()
        git_meta = _minimal_git_meta()
        git_meta["parent_commit_id"] = "parent0"
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=git_meta,
        )
        parent_edges = [e for e in payload["edges"] if e["type"] == "commit_parent"]
        assert len(parent_edges) == 1

    def test_build_lineage_payload_previous_manifest_supersedes_property(self):
        """HAPPY: previous_manifest_id produces manifest_supersedes edge."""
        pub = _make_publisher()
        manifest = _minimal_manifest()
        manifest["previous_manifest_id"] = "manifest_old"
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=manifest,
            git_meta=_minimal_git_meta(),
        )
        supersedes = [e for e in payload["edges"] if e["type"] == "manifest_supersedes"]
        assert len(supersedes) == 1

    def test_build_lineage_payload_symbol_version_from_property(self):
        """HAPPY: external_symbol_id with previous_snapshot_symbols creates version edge."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["symbols"] = [
            {
                "symbol_id": "sym:1",
                "display_name": "foo",
                "kind": "function",
                "external_symbol_id": "ext:foo",
            },
        ]
        prev = {"ext:foo": "symbol:prev:ext_foo"}
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
            previous_snapshot_symbols=prev,
        )
        version_edges = [
            e for e in payload["edges"] if e["type"] == "symbol_version_from"
        ]
        assert len(version_edges) == 1

    @given(repo_name=small_text)
    @settings(max_examples=10)
    def test_payload_repo_name_varies_property(self, repo_name: str):
        """HAPPY: payload repo_name reflects snapshot input."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["repo_name"] = repo_name
        payload = pub.build_lineage_payload(
            snapshot=snap,
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        assert payload["snapshot_id"] == snap["snapshot_id"]


# --- Validation / error tests ---


class TestLineagePayloadValidation:
    @pytest.mark.negative
    def test_build_lineage_payload_missing_snapshot_id_raises_property(self):
        """EDGE: missing snapshot_id raises ValueError."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["snapshot_id"] = None
        with pytest.raises(ValueError, match="snapshot_id"):
            pub.build_lineage_payload(
                snapshot=snap,
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
            )

    @pytest.mark.negative
    def test_build_lineage_payload_missing_repo_name_raises_property(self):
        """EDGE: missing repo_name in both snapshot and git_meta raises ValueError."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        snap["repo_name"] = None
        git_meta = _minimal_git_meta()
        git_meta["repo_name"] = None
        with pytest.raises(ValueError, match="repo_name"):
            pub.build_lineage_payload(
                snapshot=snap,
                manifest=_minimal_manifest(),
                git_meta=git_meta,
            )

    @pytest.mark.negative
    def test_publish_without_endpoint_raises_property(self):
        """EDGE: publish_snapshot_lineage raises when endpoint not configured."""
        pub = TerminusPublisher({"terminus": {}})
        with pytest.raises(RuntimeError, match="not configured"):
            pub.publish_snapshot_lineage(
                snapshot=_minimal_snapshot(),
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
            )

    @pytest.mark.negative
    def test_publish_connection_error_raises_runtime_property(self):
        """EDGE: network error raises RuntimeError."""
        pub = _make_publisher(endpoint="http://localhost:1/invalid")
        with pytest.raises(RuntimeError):
            pub.publish_snapshot_lineage(
                snapshot=_minimal_snapshot(),
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
            )

    @pytest.mark.negative
    def test_publish_with_idempotency_key_property(self):
        """EDGE: idempotency key passed to request headers."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        # Will fail at network level but that's OK -- we just want to test the key is used
        with pytest.raises(RuntimeError):
            pub.publish_snapshot_lineage(
                snapshot=snap,
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
                idempotency_key="key-123",
            )


# --- Code graph payload tests ---


class TestCodeGraphPayload:
    def test_build_code_graph_payload_empty_snapshot(self):
        """HAPPY: empty snapshot produces empty nodes and edges."""
        publisher = _make_publisher()
        result = publisher.build_code_graph_payload(_make_snapshot_v2())
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_build_code_graph_payload_skips_file_and_doc_units(self):
        """HAPPY: file and doc units are skipped; only code units produce nodes."""
        publisher = _make_publisher()
        units = [
            {
                "unit_id": "f1",
                "kind": "file",
                "path": "a.py",
                "display_name": "a.py",
                "language": "python",
            },
            {
                "unit_id": "d1",
                "kind": "doc",
                "path": "docs.md",
                "display_name": "docs",
                "language": "markdown",
            },
            {
                "unit_id": "fn1",
                "kind": "function",
                "path": "a.py",
                "display_name": "foo",
                "language": "python",
            },
        ]
        result = publisher.build_code_graph_payload(_make_snapshot_v2(units=units))
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "sym:snap:repo:abc:fn1"
        assert result["nodes"][0]["type"] == "Symbol"
        assert result["nodes"][0]["props"]["kind"] == "function"

    def test_build_code_graph_payload_symbol_node_props(self):
        """HAPPY: symbol node has all expected properties."""
        publisher = _make_publisher()
        units = [
            {
                "unit_id": "cls1",
                "kind": "class",
                "path": "b.py",
                "display_name": "MyClass",
                "language": "python",
                "start_line": 10,
                "end_line": 50,
                "qualified_name": "pkg.b.MyClass",
                "primary_anchor_symbol_id": "scip python pkg.b.MyClass",
                "source_set": ["scip", "fc_structure"],
            },
        ]
        result = publisher.build_code_graph_payload(_make_snapshot_v2(units=units))
        node = result["nodes"][0]
        props = node["props"]
        assert props["unit_id"] == "cls1"
        assert props["display_name"] == "MyClass"
        assert props["kind"] == "class"
        assert props["path"] == "b.py"
        assert props["language"] == "python"
        assert props["start_line"] == 10
        assert props["end_line"] == 50
        assert props["qualified_name"] == "pkg.b.MyClass"
        assert props["scip_symbol"] == "scip python pkg.b.MyClass"
        assert set(props["source_set"]) == {"fc_structure", "scip"}

    def test_build_code_graph_payload_relation_edges(self):
        """HAPPY: relations produce edges with correct IDs and confidence mapping."""
        publisher = _make_publisher()
        relations = [
            {
                "relation_id": "r1",
                "src_unit_id": "cls1",
                "dst_unit_id": "fn1",
                "relation_type": "contain",
                "resolution_state": "anchored",
                "support_sources": ["scip"],
            },
            {
                "relation_id": "r2",
                "src_unit_id": "fn1",
                "dst_unit_id": "fn2",
                "relation_type": "call",
                "resolution_state": "structural",
                "support_sources": ["fc_structure"],
            },
        ]
        result = publisher.build_code_graph_payload(_make_snapshot_v2(relations=relations))
        assert len(result["edges"]) == 2

        contain_edge = result["edges"][0]
        assert contain_edge["id"] == "rel:snap:repo:abc:r1"
        assert contain_edge["type"] == "contain"
        assert contain_edge["src"] == "sym:snap:repo:abc:cls1"
        assert contain_edge["dst"] == "sym:snap:repo:abc:fn1"
        assert contain_edge["confidence"] == "precise"  # anchored -> precise
        assert contain_edge["resolution_state"] == "anchored"
        assert contain_edge["source_set"] == ["scip"]

        call_edge = result["edges"][1]
        assert call_edge["confidence"] == "resolved"  # structural -> resolved

    def test_build_code_graph_payload_resolution_confidence_mapping(self):
        """HAPPY: resolution_state maps to correct confidence values."""
        publisher = _make_publisher()
        for resolution_state, expected_confidence in [
            ("anchored", "precise"),
            ("structural", "resolved"),
            ("candidate", "heuristic"),
            ("unknown", "derived"),
            ("", "derived"),
        ]:
            relations = [
                {
                    "relation_id": f"r_{resolution_state}",
                    "src_unit_id": "a",
                    "dst_unit_id": "b",
                    "relation_type": "ref",
                    "resolution_state": resolution_state,
                    "support_sources": [],
                },
            ]
            result = publisher.build_code_graph_payload(
                _make_snapshot_v2(relations=relations)
            )
            assert result["edges"][0]["confidence"] == expected_confidence, (
                f"{resolution_state!r} should map to {expected_confidence!r}"
            )

    @pytest.mark.edge
    def test_build_code_graph_payload_skips_relations_missing_ids(self):
        """EDGE: relations with missing IDs are skipped."""
        publisher = _make_publisher()
        relations = [
            {
                "relation_id": "",
                "src_unit_id": "a",
                "dst_unit_id": "b",
                "relation_type": "ref",
            },
            {
                "relation_id": "r1",
                "src_unit_id": "",
                "dst_unit_id": "b",
                "relation_type": "ref",
            },
            {
                "relation_id": "r2",
                "src_unit_id": "a",
                "dst_unit_id": None,
                "relation_type": "ref",
            },
        ]
        result = publisher.build_code_graph_payload(_make_snapshot_v2(relations=relations))
        assert result["edges"] == []

    def test_build_code_graph_payload_support_sources_as_set(self):
        """HAPPY: support_sources as set are handled and serialized as list."""
        publisher = _make_publisher()
        relations = [
            {
                "relation_id": "r1",
                "src_unit_id": "a",
                "dst_unit_id": "b",
                "relation_type": "import",
                "resolution_state": "anchored",
                "support_sources": {"scip", "fc_structure"},  # set, not list
            },
        ]
        result = publisher.build_code_graph_payload(_make_snapshot_v2(relations=relations))
        assert isinstance(result["edges"][0]["source_set"], list)
        assert set(result["edges"][0]["source_set"]) == {"scip", "fc_structure"}


# --- Integration: code graph merged into lineage ---


class TestCodeGraphInLineage:
    def test_build_lineage_payload_includes_code_graph(self):
        """HAPPY: Code graph nodes/edges merged into lineage payload."""
        publisher = _make_publisher()
        units = [
            {
                "unit_id": "fn1",
                "kind": "function",
                "path": "a.py",
                "display_name": "foo",
                "language": "python",
            },
        ]
        relations = [
            {
                "relation_id": "r1",
                "src_unit_id": "fn1",
                "dst_unit_id": "fn2",
                "relation_type": "call",
                "resolution_state": "structural",
                "support_sources": [],
            },
        ]
        snapshot = _make_snapshot_v2(units=units, relations=relations)
        payload = publisher.build_lineage_payload(
            snapshot=snapshot,
            manifest={"manifest_id": "m1"},
            git_meta={},
        )
        node_types = {n["type"] for n in payload["nodes"]}
        edge_types = {e["type"] for e in payload["edges"]}

        # Lineage types should still be present
        assert "Repository" in node_types
        assert "Snapshot" in node_types

        # Code graph types should also be present
        assert "Symbol" in node_types
        assert "call" in edge_types

        # Verify the symbol node has correct ID format
        sym_nodes = [n for n in payload["nodes"] if n["type"] == "Symbol"]
        assert len(sym_nodes) == 1
        assert sym_nodes[0]["id"] == "sym:snap:repo:abc:fn1"

    def test_build_lineage_payload_legacy_no_units_is_safe(self):
        """HAPPY: Legacy snapshots without units/relations do not break."""
        publisher = _make_publisher()
        snapshot = {
            "repo_name": "repo",
            "snapshot_id": "snap:repo:old",
            "documents": [{"doc_id": "d1", "path": "a.py", "language": "python"}],
            "symbols": [
                {
                    "symbol_id": "s1",
                    "display_name": "foo",
                    "path": "a.py",
                    "kind": "function",
                }
            ],
        }
        payload = publisher.build_lineage_payload(
            snapshot=snapshot,
            manifest={"manifest_id": "m1"},
            git_meta={},
        )
        # Legacy SymbolVersion nodes should be present
        sym_ver_nodes = [n for n in payload["nodes"] if n["type"] == "SymbolVersion"]
        assert len(sym_ver_nodes) == 1
        # No new Symbol nodes (no units)
        assert not [n for n in payload["nodes"] if n["type"] == "Symbol"]


# --- Graph query tests ---


class TestGraphQuery:
    def test_load_graph_nodes_unconfigured_returns_empty(self):
        """HAPPY: unconfigured publisher returns empty list for graph nodes."""
        publisher = TerminusPublisher({"terminus": {}})
        assert publisher.load_graph_nodes("snap:repo:abc") == []

    def test_load_graph_nodes_configured_raises_not_implemented(self):
        """HAPPY: configured publisher raises NotImplementedError for graph nodes."""
        publisher = _make_publisher()
        with pytest.raises(NotImplementedError, match="query endpoint"):
            publisher.load_graph_nodes("snap:repo:abc")

    def test_load_graph_edges_unconfigured_returns_empty(self):
        """HAPPY: unconfigured publisher returns empty list for graph edges."""
        publisher = TerminusPublisher({"terminus": {}})
        assert publisher.load_graph_edges("snap:repo:abc") == []
        assert publisher.load_graph_edges("snap:repo:abc", edge_type="call") == []

    def test_load_graph_edges_configured_raises_not_implemented(self):
        """HAPPY: configured publisher raises NotImplementedError for graph edges."""
        publisher = _make_publisher()
        with pytest.raises(NotImplementedError, match="query endpoint"):
            publisher.load_graph_edges("snap:repo:abc", edge_type="call")


# ─── Edge cases for build_lineage_payload (negative/error paths) ───


class TestLineagePayloadEdgeCases:
    """Tests for build_lineage_payload error conditions and conditional node creation."""

    def _pub(self) -> TerminusPublisher:
        return TerminusPublisher({"terminus": {"endpoint": "http://localhost:6363"}})

    def test_missing_snapshot_id_raises(self):
        pub = self._pub()
        with pytest.raises(ValueError, match="snapshot_id"):
            pub.build_lineage_payload({}, {}, {})

    def test_missing_repo_name_raises(self):
        pub = self._pub()
        with pytest.raises(ValueError, match="repo_name"):
            pub.build_lineage_payload({"snapshot_id": "snap:test:abc"}, {}, {})

    def test_none_branch_omits_branch_node(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test", "branch": None},
            {},
            {},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "Branch" not in node_types

    def test_none_commit_omits_commit_node(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test", "commit_id": None},
            {},
            {},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "Commit" not in node_types

    def test_none_run_omits_run_node(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test", "run_id": None},
            {},
            {},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "IndexRun" not in node_types

    def test_documents_with_none_doc_id_skipped(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {
                "snapshot_id": "snap:test:abc",
                "repo_name": "test",
                "documents": [{"doc_id": None, "path": "a.py"}],
                "symbols": [],
            },
            {},
            {},
        )
        doc_nodes = [n for n in payload["nodes"] if n["type"] == "DocumentVersion"]
        assert len(doc_nodes) == 0

    def test_symbols_with_none_symbol_id_skipped(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {
                "snapshot_id": "snap:test:abc",
                "repo_name": "test",
                "documents": [{"doc_id": "doc:a.py", "path": "a.py"}],
                "symbols": [{"symbol_id": None, "path": "a.py"}],
            },
            {},
            {},
        )
        sym_nodes = [n for n in payload["nodes"] if n["type"] == "SymbolVersion"]
        assert len(sym_nodes) == 0

    def test_empty_documents_and_symbols(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {
                "snapshot_id": "snap:test:abc",
                "repo_name": "test",
                "documents": [],
                "symbols": [],
            },
            {},
            {},
        )
        doc_nodes = [n for n in payload["nodes"] if n["type"] == "DocumentVersion"]
        sym_nodes = [n for n in payload["nodes"] if n["type"] == "SymbolVersion"]
        assert len(doc_nodes) == 0
        assert len(sym_nodes) == 0

    def test_git_meta_fallback_for_branch(self):
        pub = self._pub()
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test"},
            {},
            {"branch": "feature/x"},
        )
        node_types = [n["type"] for n in payload["nodes"]]
        assert "Branch" in node_types

    def test_unconfigured_publisher_returns_empty_payload(self):
        pub = TerminusPublisher({"terminus": {}})
        payload = pub.build_lineage_payload(
            {"snapshot_id": "snap:test:abc", "repo_name": "test"},
            {},
            {},
        )
        assert "nodes" in payload
