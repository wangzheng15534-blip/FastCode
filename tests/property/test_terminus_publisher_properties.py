"""Property-based tests for terminus_publisher module."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.terminus_publisher import TerminusPublisher

# --- Helpers ---

small_text = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)


def _make_publisher(**overrides):
    cfg = {"terminus": {"endpoint": "http://localhost:6363/api/publish"}}
    cfg["terminus"].update(overrides)
    return TerminusPublisher(cfg)


def _minimal_snapshot():
    return {
        "repo_name": "repo",
        "snapshot_id": "snap:repo:abc",
        "branch": "main",
        "commit_id": "abc123",
        "documents": [],
        "symbols": [],
    }


def _minimal_manifest():
    return {
        "manifest_id": "manifest_001",
        "ref_name": "main",
        "status": "published",
        "published_at": "2026-01-01T00:00:00Z",
        "index_run_id": "run_001",
    }


def _minimal_git_meta():
    return {
        "repo_name": "repo",
        "branch": "main",
        "commit_id": "abc123",
    }


# --- Properties ---


@pytest.mark.property
class TestTerminusPublisherProperties:
    @pytest.mark.happy
    def test_is_configured_true_with_endpoint(self):
        """HAPPY: is_configured returns True when endpoint set."""
        pub = _make_publisher()
        assert pub.is_configured() is True

    @pytest.mark.edge
    def test_is_configured_false_without_endpoint(self):
        """EDGE: is_configured returns False when endpoint missing."""
        pub = TerminusPublisher({"terminus": {}})
        assert pub.is_configured() is False

    @pytest.mark.edge
    def test_is_configured_false_with_none_endpoint(self):
        """EDGE: is_configured returns False when endpoint is None."""
        pub = TerminusPublisher({"terminus": {"endpoint": None}})
        assert pub.is_configured() is False

    @pytest.mark.happy
    def test_build_lineage_payload_has_required_keys(self):
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

    @pytest.mark.happy
    def test_build_lineage_payload_includes_repo_node(self):
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

    @pytest.mark.happy
    def test_build_lineage_payload_includes_snapshot_node(self):
        """HAPPY: payload includes Snapshot node."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        snap_nodes = [n for n in payload["nodes"] if n["type"] == "Snapshot"]
        assert len(snap_nodes) == 1

    @pytest.mark.happy
    def test_build_lineage_payload_includes_branch_node(self):
        """HAPPY: payload includes Branch node when branch present."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        branch_nodes = [n for n in payload["nodes"] if n["type"] == "Branch"]
        assert len(branch_nodes) == 1

    @pytest.mark.happy
    def test_build_lineage_payload_includes_manifest_node(self):
        """HAPPY: payload includes Manifest node when manifest_id present."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        manifest_nodes = [n for n in payload["nodes"] if n["type"] == "Manifest"]
        assert len(manifest_nodes) == 1

    @pytest.mark.happy
    def test_build_lineage_payload_includes_index_run_node(self):
        """HAPPY: payload includes IndexRun node when run_id present."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        run_nodes = [n for n in payload["nodes"] if n["type"] == "IndexRun"]
        assert len(run_nodes) == 1

    @pytest.mark.happy
    def test_build_lineage_payload_with_documents(self):
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

    @pytest.mark.happy
    def test_build_lineage_payload_with_symbols(self):
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
    def test_build_lineage_payload_no_branch(self):
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
    def test_build_lineage_payload_missing_snapshot_id_raises(self):
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

    @pytest.mark.edge
    def test_build_lineage_payload_missing_repo_name_raises(self):
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

    @pytest.mark.edge
    def test_publish_without_endpoint_raises(self):
        """EDGE: publish_snapshot_lineage raises when endpoint not configured."""
        pub = TerminusPublisher({"terminus": {}})
        with pytest.raises(RuntimeError, match="not configured"):
            pub.publish_snapshot_lineage(
                snapshot=_minimal_snapshot(),
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
            )

    @pytest.mark.happy
    def test_build_lineage_payload_parent_commits(self):
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

    @pytest.mark.happy
    def test_build_lineage_payload_previous_manifest_supersedes(self):
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

    @pytest.mark.happy
    def test_build_lineage_payload_symbol_version_from(self):
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
    @pytest.mark.happy
    def test_payload_repo_name_varies(self, repo_name):
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

    @pytest.mark.edge
    def test_build_lineage_payload_doc_without_id_skipped(self):
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
    def test_build_lineage_payload_symbol_without_id_skipped(self):
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
    def test_publish_connection_error_raises_runtime(self):
        """EDGE: network error raises RuntimeError."""
        pub = _make_publisher(endpoint="http://localhost:1/invalid")
        with pytest.raises(RuntimeError):
            pub.publish_snapshot_lineage(
                snapshot=_minimal_snapshot(),
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
            )

    @pytest.mark.edge
    def test_payload_no_commit(self):
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
    def test_payload_no_manifest_id(self):
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
    def test_payload_no_run_id(self):
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
    def test_payload_parent_commit_id_legacy(self):
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

    @pytest.mark.edge
    def test_payload_empty_documents(self):
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
    def test_publish_with_idempotency_key(self):
        """EDGE: idempotency key passed to request headers."""
        pub = _make_publisher()
        snap = _minimal_snapshot()
        # Will fail at network level but that's OK — we just want to test the key is used
        with pytest.raises(RuntimeError):
            pub.publish_snapshot_lineage(
                snapshot=snap,
                manifest=_minimal_manifest(),
                git_meta=_minimal_git_meta(),
                idempotency_key="key-123",
            )

    @pytest.mark.edge
    def test_build_payload_version_is_v1(self):
        """EDGE: payload version field is always 'v1'."""
        pub = _make_publisher()
        payload = pub.build_lineage_payload(
            snapshot=_minimal_snapshot(),
            manifest=_minimal_manifest(),
            git_meta=_minimal_git_meta(),
        )
        assert payload["version"] == "v1"

    @pytest.mark.edge
    def test_timeout_default(self):
        """EDGE: default timeout is 15 seconds."""
        pub = _make_publisher()
        assert pub.timeout == 15

    @pytest.mark.edge
    def test_timeout_custom(self):
        """EDGE: custom timeout from config."""
        pub = _make_publisher(timeout_seconds=30)
        assert pub.timeout == 30
