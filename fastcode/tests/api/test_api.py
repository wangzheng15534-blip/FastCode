"""
Behavior tests for FastCode API routes.

Uses a real SnapshotStore (SQLite in tmp_path) wired into the API via
monkeypatch so endpoints exercise actual database logic instead of
returning hardcoded mock data.
"""

from __future__ import annotations

import asyncio
import functools
import io
import threading
import time
import zipfile
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import fastcode.api.routes as api
from fastcode.ir.types import IRSnapshot
from fastcode.store.manifest import ManifestStore
from fastcode.store.snapshot import SnapshotStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_ref:
        for name, payload in entries.items():
            zip_ref.writestr(name, payload)
    return buffer.getvalue()


def _make_minimal_snapshot(
    *,
    repo_name: str = "test-repo",
    snapshot_id: str = "snap:test-repo:abc123",
    branch: str | None = "main",
    commit_id: str | None = "abc123",
) -> IRSnapshot:
    """Create a valid IRSnapshot with no documents/symbols/edges."""
    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch=branch,
        commit_id=commit_id,
        tree_id="tree_001",
    )


class _FakeFastCode:
    """Minimal stand-in for FastCode with real storage backends.

    Only the attributes and methods accessed by the endpoints under test
    are provided.  Delegates to real SnapshotStore and ManifestStore so
    that database logic (SQL, ordering, 404s) is exercised.
    """

    def __init__(self, snapshot_store: SnapshotStore) -> None:
        self.snapshot_store = snapshot_store
        self.manifest_store = ManifestStore(snapshot_store.db_runtime)

    def list_repo_refs(self, repo_name: str) -> list[dict[str, Any]]:
        with self.snapshot_store.db_runtime.connect() as conn:
            rows = self.snapshot_store.db_runtime.execute(
                conn,
                """
                SELECT branch, commit_id, tree_id, snapshot_id, created_at
                FROM snapshot_refs
                WHERE repo_name=?
                ORDER BY created_at DESC
                """,
                (repo_name,),
            ).fetchall()
        return [
            d
            for r in rows
            if r
            for d in [self.snapshot_store.db_runtime.row_to_dict(r)]
            if d is not None
        ]

    def get_scip_artifact_ref(self, snapshot_id: str) -> dict[str, Any] | None:
        return self.snapshot_store.get_scip_artifact_ref(snapshot_id)

    def list_scip_artifact_refs(self, snapshot_id: str) -> list[dict[str, Any]]:
        return self.snapshot_store.list_scip_artifact_refs(snapshot_id)

    def get_branch_manifest(
        self, repo_name: str, ref_name: str
    ) -> dict[str, Any] | None:
        return self.manifest_store.get_branch_manifest(repo_name, ref_name)

    def get_snapshot_manifest(self, snapshot_id: str) -> dict[str, Any] | None:
        return self.manifest_store.get_snapshot_manifest(snapshot_id)


class _NoDictSource:
    def __init__(self) -> None:
        self.repository = "repo"
        self.file = "src/config.py"
        self.name = "load_config"
        self.type = "function"
        self.start_line = 11
        self.end_line = 24
        self.score = 0.75

    def to_dict(self) -> dict[str, Any]:
        raise AssertionError("API source serialization must not call to_dict()")


class _NoDictTurn:
    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.turn_number = 2
        self.timestamp = 1234.5
        self.query = "Where is config loaded?"
        self.answer = "src/config.py"
        self.summary = "Config loader identified"
        self.retrieved_elements = [_NoDictSource()]
        self.metadata = {
            "intent": "where",
            "keywords": ("config", "loader"),
            "repo_filter": ("repo",),
            "multi_turn": True,
        }

    def to_dict(self) -> dict[str, Any]:
        raise AssertionError("API history serialization must not call to_dict()")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def inline_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run API thread offloads inline to keep route tests deterministic."""

    async def _run_inline(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(api.asyncio, "to_thread", _run_inline)


@pytest.fixture
def stores(tmp_path: Any) -> Any:
    """Yield real storage wired into the module-global FastCode handle."""
    store = SnapshotStore(str(tmp_path / "store"))
    fake = _FakeFastCode(store)
    original = api.fastcode_instance
    api.fastcode_instance = fake
    try:
        yield fake, store
    finally:
        api.fastcode_instance = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepoRefs:
    """GET /repos/{repo_name}/refs"""

    def test_returns_empty_list_for_unknown_repo(self, stores: Any) -> None:
        body = asyncio.run(api.get_repo_refs("unknown-repo"))
        assert body["refs"] == []
        assert body["repo_name"] == "unknown-repo"

    def test_returns_refs_after_saving_snapshot(self, stores: Any) -> None:
        _fake_fc, store = stores
        snap = _make_minimal_snapshot(
            repo_name="my-repo",
            snapshot_id="snap:my-repo:deadbeef",
            branch="develop",
            commit_id="deadbeef",
        )
        store.save_snapshot(snap)

        refs = asyncio.run(api.get_repo_refs("my-repo"))["refs"]
        assert len(refs) == 1
        assert refs[0]["branch"] == "develop"
        assert refs[0]["snapshot_id"] == "snap:my-repo:deadbeef"

    def test_multiple_refs_ordered_newest_first(self, stores: Any) -> None:
        _fake_fc, store = stores
        store.save_snapshot(
            _make_minimal_snapshot(
                repo_name="r",
                snapshot_id="snap:r:first",
                branch="main",
                commit_id="c1",
            )
        )
        store.save_snapshot(
            _make_minimal_snapshot(
                repo_name="r",
                snapshot_id="snap:r:second",
                branch="feature",
                commit_id="c2",
            )
        )
        refs = asyncio.run(api.get_repo_refs("r"))["refs"]
        assert len(refs) == 2
        assert refs[0]["snapshot_id"] == "snap:r:second"
        assert refs[1]["snapshot_id"] == "snap:r:first"


class TestScipArtifacts:
    """GET /scip/artifacts/{snapshot_id}"""

    def test_returns_404_when_not_found(self, stores: Any) -> None:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api.get_scip_artifact("snap:x:nonexistent"))
        assert exc_info.value.status_code == 404

    def test_returns_artifact_after_save(self, stores: Any) -> None:
        _fake_fc, store = stores
        snap = _make_minimal_snapshot(
            snapshot_id="snap:repo:with-scip",
        )
        store.save_snapshot(snap)
        store.save_scip_artifact_ref(
            "snap:repo:with-scip",
            indexer_name="scip-python",
            indexer_version="1.0",
            artifact_path="/tmp/scip.dump",
            checksum="abc123",
        )

        body = asyncio.run(api.get_scip_artifact("snap:repo:with-scip"))
        assert body["status"] == "success"
        artifact = cast(dict[str, Any], body["artifact"])
        artifacts = cast(list[dict[str, Any]], body["artifacts"])
        assert artifact["indexer_name"] == "scip-python"
        assert artifact["checksum"] == "abc123"
        assert artifacts[0]["artifact_path"] == "/tmp/scip.dump"

    def test_returns_full_artifact_lineage_when_multiple_saved(
        self, stores: Any
    ) -> None:
        _fake_fc, store = stores
        snap = _make_minimal_snapshot(snapshot_id="snap:repo:multi-scip")
        store.save_snapshot(snap)
        store.save_scip_artifact_refs(
            snap.snapshot_id,
            artifacts=[
                {
                    "indexer_name": "scip-python",
                    "artifact_path": "/tmp/python.scip",
                    "checksum": "111",
                    "language": "python",
                },
                {
                    "indexer_name": "scip-go",
                    "artifact_path": "/tmp/go.scip",
                    "checksum": "222",
                    "language": "go",
                },
            ],
        )

        body = asyncio.run(api.get_scip_artifact("snap:repo:multi-scip"))
        artifact = cast(dict[str, Any], body["artifact"])
        artifacts = cast(list[dict[str, Any]], body["artifacts"])
        assert artifact["artifact_path"] == "/tmp/python.scip"
        assert [item["artifact_path"] for item in artifacts] == [
            "/tmp/python.scip",
            "/tmp/go.scip",
        ]


class TestManifests:
    """GET /manifests/{repo_name}/{ref_name}"""

    def test_branch_manifest_returns_404_when_not_found(self, stores: Any) -> None:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(api.get_branch_manifest("no-repo", "main"))
        assert exc_info.value.status_code == 404

    def test_branch_manifest_returns_data_after_publish(self, stores: Any) -> None:
        fake_fc, store = stores
        snap = _make_minimal_snapshot(
            repo_name="manifest-repo",
            snapshot_id="snap:manifest-repo:m1",
            branch="main",
            commit_id="c_m1",
        )
        store.save_snapshot(snap)

        fake_fc.manifest_store.publish(
            repo_name="manifest-repo",
            ref_name="main",
            snapshot_id="snap:manifest-repo:m1",
            index_run_id="run_001",
        )

        body = asyncio.run(api.get_branch_manifest("manifest-repo", "main"))
        assert body["status"] == "success"
        assert body["manifest"]["snapshot_id"] == "snap:manifest-repo:m1"
        assert body["manifest"]["repo_name"] == "manifest-repo"

    def test_snapshot_manifest_returns_data_after_publish(self, stores: Any) -> None:
        _fake_fc, store = stores
        snap = _make_minimal_snapshot(
            repo_name="manifest-repo2",
            snapshot_id="snap:manifest-repo2:s1",
            branch="develop",
            commit_id="c_s1",
        )
        store.save_snapshot(snap)

        fake_fc = api.fastcode_instance
        assert fake_fc is not None
        fake_fc.manifest_store.publish(
            repo_name="manifest-repo2",
            ref_name="develop",
            snapshot_id="snap:manifest-repo2:s1",
            index_run_id="run_002",
        )

        body = asyncio.run(api.get_snapshot_manifest("snap:manifest-repo2:s1"))
        assert body["manifest"]["snapshot_id"] == "snap:manifest-repo2:s1"


class TestCodeStatusPack:
    """GET /code-status/{snapshot_id}."""

    def test_code_status_pack_offloads_snapshot_export(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        offloaded: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

        async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            offloaded.append((func, args, kwargs))
            return func(*args, **kwargs)

        fake_fastcode = MagicMock()
        fake_fastcode.get_code_status_pack.return_value = {
            "schema_version": "code_status_pack.v0",
            "snapshot": {"snapshot_id": "snap:repo:1"},
        }
        monkeypatch.setattr(api.asyncio, "to_thread", record_to_thread)

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            body = asyncio.run(
                api.get_code_status_pack(
                    "snap:repo:1",
                    include_graph_facts=False,
                )
            )

        assert body["status"] == "success"
        assert body["pack"]["schema_version"] == "code_status_pack.v0"
        assert offloaded == [
            (
                fake_fastcode.get_code_status_pack,
                ("snap:repo:1",),
                {"include_graph_facts": False},
            )
        ]


class TestRootAndHealth:
    """Static endpoints that do not need storage."""

    def test_root_returns_metadata(self) -> None:
        body = asyncio.run(api.root())
        assert body["name"] == "FastCode API"
        assert body["version"] == "2.0.0"
        assert body["status"] == "running"


class TestSchemaDefaults:
    """Schema default factories should be typed and instance-local."""

    def test_status_response_default_lists_are_not_shared(self) -> None:
        first = api.StatusResponse(
            status="ok",
            repo_loaded=False,
            repo_indexed=False,
            repo_info={},
        )
        second = api.StatusResponse(
            status="ok",
            repo_loaded=False,
            repo_indexed=False,
            repo_info={},
        )

        first.available_repositories.append({"name": "repo-a"})
        first.loaded_repositories.append({"name": "repo-a"})

        assert second.available_repositories == []
        assert second.loaded_repositories == []


class TestIndexMultiple:
    """POST /index-multiple explicit source mapping behavior."""

    def test_maps_sources_explicitly_without_model_dump(self) -> None:
        request = api.IndexMultipleRequest(
            sources=[
                api.LoadRepositoryRequest(
                    source="https://example.com/repo.git", is_url=True
                ),
                api.LoadRepositoryRequest(source="/tmp/local-repo", is_url=False),
            ]
        )
        fake_fastcode = MagicMock()
        fake_fastcode.get_repository_stats.return_value = {"repos": 2}

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            result = asyncio.run(api.index_multiple(request))

        fake_fastcode.load_multiple_repositories.assert_called_once_with(
            [
                {"source": "https://example.com/repo.git", "is_url": True},
                {"source": "/tmp/local-repo", "is_url": False},
            ]
        )
        fake_fastcode.vector_store.invalidate_scan_cache.assert_called_once_with()
        assert result["status"] == "success"


class TestBlockingEndpointOffloads:
    """Blocking repository operations must stay off the event loop."""

    def test_repository_mutation_endpoints_use_to_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        offloaded: list[Any] = []

        async def record_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            offloaded.append(func)
            return func(*args, **kwargs)

        fake_fastcode = MagicMock()
        fake_fastcode.repo_loaded = True
        fake_fastcode._load_multi_repo_cache.return_value = True
        fake_fastcode.list_repositories.return_value = [{"repo_name": "repo"}]
        fake_fastcode.get_repository_stats.return_value = {"repo_count": 1}
        fake_fastcode.get_repository_summary.return_value = "summary"
        fake_fastcode.remove_repository.return_value = {
            "deleted_files": [],
            "freed_mb": 0.0,
        }
        fake_fastcode.cache_manager.clear.return_value = True
        fake_fastcode.cache_manager.get_stats.return_value = {"entries": 0}
        fake_fastcode.vector_store.scan_available_indexes.return_value = []
        monkeypatch.setattr(api.asyncio, "to_thread", record_to_thread)

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            asyncio.run(
                api.load_repository(
                    api.LoadRepositoryRequest(source="/tmp/repo", is_url=False)
                )
            )
            asyncio.run(api.index_repository(force=True))
            asyncio.run(
                api.load_repositories(api.LoadRepositoriesRequest(repo_names=["repo"]))
            )
            asyncio.run(
                api.index_multiple(
                    api.IndexMultipleRequest(
                        sources=[
                            api.LoadRepositoryRequest(source="/tmp/repo", is_url=False)
                        ]
                    )
                )
            )
            asyncio.run(
                api.delete_repositories(
                    api.DeleteReposRequest(repo_names=["repo"], delete_source=False)
                )
            )
            asyncio.run(api.clear_cache())
            asyncio.run(api.get_cache_stats())
            asyncio.run(api.refresh_index_cache())

        assert offloaded == [
            fake_fastcode.load_repository,
            fake_fastcode.index_repository,
            api._call_with_service_lock,
            fake_fastcode._load_multi_repo_cache,
            fake_fastcode.load_multiple_repositories,
            api._call_with_service_lock,
            fake_fastcode.remove_repository,
            api._call_with_service_lock,
            fake_fastcode.cache_manager.get_stats,
            api._refresh_index_cache_sync,
        ]


class TestUploadSecurity:
    @pytest.mark.parametrize("endpoint", ["/upload-zip", "/upload-and-index"])
    def test_upload_endpoints_reject_path_traversal_zip(
        self,
        tmp_path: Any,
        endpoint: str,
    ) -> None:
        fake_fastcode = SimpleNamespace(
            loader=SimpleNamespace(
                safe_repo_root=str(tmp_path / "repos"),
                _backup_existing_repo=lambda _path: None,
            )
        )

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            client = TestClient(api.app)
            response = client.post(
                endpoint,
                files={
                    "file": (
                        "repo.zip",
                        _zip_bytes({"../escape.py": b"bad"}),
                        "application/zip",
                    )
                },
            )

        assert response.status_code == 400
        assert "unsafe path" in response.json()["detail"]
        assert not (tmp_path / "escape.py").exists()


class TestApiSerializationBoundaries:
    def test_query_endpoint_allows_concurrent_snapshot_reads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        concurrent_count = 0
        max_concurrent = 0
        count_lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=5)

        class _ConcurrentFastCode:
            def query_snapshot(self, **kwargs: Any) -> dict[str, Any]:
                nonlocal concurrent_count, max_concurrent
                barrier.wait(timeout=5)
                with count_lock:
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                time.sleep(0.05)
                with count_lock:
                    concurrent_count -= 1
                return {
                    "answer": "ok",
                    "query": kwargs["question"],
                    "context_elements": 0,
                    "sources": [],
                }

        async def _executor_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(func, *args, **kwargs)
            )

        monkeypatch.setattr(api.asyncio, "to_thread", _executor_to_thread)
        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=_ConcurrentFastCode(),
        ):

            async def _run_queries() -> None:
                await asyncio.wait_for(
                    asyncio.gather(
                        api.query_repository(
                            api.QueryRequest(
                                question="Where is auth?",
                                snapshot_id="snap:1",
                                repo_name=None,
                                ref_name=None,
                                filters=None,
                                repo_filter=None,
                                multi_turn=False,
                                session_id="sess-1",
                            )
                        ),
                        api.query_repository(
                            api.QueryRequest(
                                question="Where is config?",
                                snapshot_id="snap:2",
                                repo_name=None,
                                ref_name=None,
                                filters=None,
                                repo_filter=None,
                                multi_turn=False,
                                session_id="sess-2",
                            )
                        ),
                    ),
                    timeout=10,
                )

            asyncio.run(_run_queries())

        assert max_concurrent == 2

    def test_query_endpoint_serializes_sources_explicitly(self) -> None:
        fake_fastcode = MagicMock()
        fake_fastcode.query_snapshot.return_value = {
            "answer": "ok",
            "query": "Where is config loaded?",
            "context_elements": 1,
            "sources": [_NoDictSource()],
        }

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            result = asyncio.run(
                api.query_repository(
                    api.QueryRequest(
                        question="Where is config loaded?",
                        snapshot_id="snap:1",
                        repo_name=None,
                        ref_name=None,
                        filters=None,
                        repo_filter=None,
                        multi_turn=False,
                        session_id=None,
                    )
                )
            )

        assert result.sources == [
            {
                "repository": "repo",
                "repo": "repo",
                "file": "src/config.py",
                "name": "load_config",
                "type": "function",
                "lines": "11-24",
                "start_line": 11,
                "end_line": 24,
                "score": 0.75,
            }
        ]
        assert result.turn_number is None

    def test_query_endpoint_propagates_turn_number(self) -> None:
        fake_fastcode = MagicMock()
        fake_fastcode.query_snapshot.return_value = {
            "answer": "ok",
            "query": "Where is config loaded?",
            "context_elements": 1,
            "sources": [_NoDictSource()],
            "turn_number": 4,
        }

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            result = asyncio.run(
                api.query_repository(
                    api.QueryRequest(
                        question="Where is config loaded?",
                        snapshot_id="snap:1",
                        repo_name=None,
                        ref_name=None,
                        filters=None,
                        repo_filter=None,
                        multi_turn=False,
                        session_id=None,
                    )
                )
            )

        assert result.turn_number == 4

    def test_session_endpoint_serializes_history_explicitly(self) -> None:
        fake_fastcode = MagicMock()
        fake_fastcode.get_session_history.return_value = [_NoDictTurn()]

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            result = asyncio.run(api.get_session("sess-1"))

        assert result == {
            "status": "success",
            "session_id": "sess-1",
            "history": [
                {
                    "session_id": "sess-1",
                    "turn_number": 2,
                    "timestamp": 1234.5,
                    "query": "Where is config loaded?",
                    "answer": "src/config.py",
                    "summary": "Config loader identified",
                    "retrieved_elements": [
                        {
                            "repository": "repo",
                            "repo": "repo",
                            "file": "src/config.py",
                            "name": "load_config",
                            "type": "function",
                            "lines": "11-24",
                            "start_line": 11,
                            "end_line": 24,
                            "score": 0.75,
                        }
                    ],
                    "metadata": {
                        "intent": "where",
                        "keywords": ["config", "loader"],
                        "repo_filter": ["repo"],
                        "multi_turn": True,
                    },
                }
            ],
        }


class TestAgentContextRoutes:
    def test_agent_context_endpoints_return_fastcode_payloads(self) -> None:
        fake_fastcode = MagicMock()
        fake_fastcode.get_turn_context.side_effect = [
            {
                "session_id": "sess-1",
                "turn_number": 3,
                "format": "fcx",
                "full_fcx": "<fcx:turn>\n...\n</fcx:turn>",
            },
            {
                "session_id": "sess-1",
                "turn_number": 2,
                "format": "json",
                "artifact": {"turn_number": 2},
            },
        ]
        fake_fastcode.create_handoff.return_value = {
            "artifact_id": "hf_123",
            "session_id": "sess-1",
            "turn_number": 2,
        }
        fake_fastcode.get_handoff_artifact.return_value = {
            "artifact_id": "hf_123",
            "session_id": "sess-1",
        }
        fake_fastcode.expand_context_ref.return_value = {
            "ref_id": "e1",
            "path": "src/auth.py",
        }
        fake_fastcode.get_context_bundle.side_effect = [
            {
                "bundle_id": "ctxb_latest",
                "session_id": "sess-1",
                "turn_number": 3,
                "format": "rendered",
                "rendered": {"text": "bundle ctxb_latest"},
            },
            {
                "bundle_id": "ctxb_123",
                "session_id": "sess-1",
                "turn_number": 2,
                "format": "json",
                "bundle": {"turn_number": 2},
            },
        ]
        fake_fastcode.get_context_bundle_by_id.return_value = {
            "bundle_id": "ctxb_123",
            "session_id": "sess-1",
        }
        fake_fastcode.expand_context_bundle_ref.return_value = {
            "bundle_id": "ctxb_123",
            "ref_id": "e1",
            "path": "src/auth.py",
        }
        fake_fastcode.create_context_activation.return_value = {
            "activation_id": "act_123",
            "bundle_id": "ctxb_123",
            "active_ref_ids": ["e1"],
        }

        with patch(
            "fastcode.api.routes._ensure_fastcode_initialized",
            return_value=fake_fastcode,
        ):
            latest = asyncio.run(api.get_latest_turn_context("sess-1", format="fcx"))
            turn = asyncio.run(api.get_turn_context("sess-1", 2, format="json"))
            latest_bundle = asyncio.run(
                api.get_latest_context_bundle(
                    "sess-1",
                    format="rendered",
                    token_budget=32,
                )
            )
            bundle = asyncio.run(
                api.get_context_bundle(
                    "sess-1",
                    2,
                    format="json",
                    token_budget=2048,
                )
            )
            bundle_by_id = asyncio.run(
                api.get_context_bundle_by_id(
                    "ctxb_123",
                    format="json",
                    token_budget=2048,
                )
            )
            expanded_bundle = asyncio.run(
                api.expand_agent_context_bundle_ref(
                    api.ExpandContextBundleRefRequest(
                        ref_id="e1",
                        session_id=None,
                        turn_number=None,
                        bundle_id="ctxb_123",
                        depth="L2",
                    )
                )
            )
            activation = asyncio.run(
                api.create_agent_context_activation(
                    api.ContextActivationRequest(
                        session_id=None,
                        turn_number=None,
                        bundle_id="ctxb_123",
                        active_ref_ids=["e1"],
                        active_fact_ids=["f1"],
                        active_hypothesis_ids=["h1"],
                        reason="focused_answer",
                    )
                )
            )
            handoff = asyncio.run(
                api.create_agent_context_handoff(
                    api.AgentContextHandoffRequest(
                        session_id="sess-1",
                        turn_number=2,
                        mode="delegate",
                    )
                )
            )
            restored = asyncio.run(api.get_agent_context_handoff("hf_123"))
            expanded = asyncio.run(
                api.expand_agent_context_ref(
                    api.ExpandContextRefRequest(
                        session_id="sess-1",
                        turn_number=2,
                        ref_id="e1",
                        depth="L2",
                    )
                )
            )

        assert latest["result"]["turn_number"] == 3
        assert turn["result"]["artifact"]["turn_number"] == 2
        assert latest_bundle["result"]["rendered"]["text"] == "bundle ctxb_latest"
        assert bundle["result"]["bundle"]["turn_number"] == 2
        assert bundle_by_id["result"]["bundle_id"] == "ctxb_123"
        assert expanded_bundle["result"]["path"] == "src/auth.py"
        assert activation["result"]["active_ref_ids"] == ["e1"]
        assert handoff["result"]["artifact_id"] == "hf_123"
        assert restored["result"]["session_id"] == "sess-1"
        assert expanded["result"]["path"] == "src/auth.py"
        assert fake_fastcode.get_turn_context.call_args_list[0].args == (
            "sess-1",
            None,
            "fcx",
        )
        assert fake_fastcode.get_turn_context.call_args_list[1].args == (
            "sess-1",
            2,
            "json",
        )
        assert fake_fastcode.get_context_bundle.call_args_list[0].args == (
            "sess-1",
            None,
            "rendered",
            32,
        )
        assert fake_fastcode.get_context_bundle.call_args_list[1].args == (
            "sess-1",
            2,
            "json",
            2048,
        )
        assert fake_fastcode.expand_context_bundle_ref.call_args.kwargs == {
            "session_id": None,
            "turn_number": None,
            "bundle_id": "ctxb_123",
            "depth": "L2",
        }
        assert fake_fastcode.create_context_activation.call_args.kwargs == {
            "session_id": None,
            "turn_number": None,
            "bundle_id": "ctxb_123",
            "active_ref_ids": ["e1"],
            "active_fact_ids": ["f1"],
            "active_hypothesis_ids": ["h1"],
            "reason": "focused_answer",
        }

    def test_agent_context_routes_surface_not_found_as_http_404(self) -> None:
        fake_fastcode = MagicMock()
        fake_fastcode.expand_context_ref.side_effect = RuntimeError(
            "context ref not found: e9"
        )

        with (
            patch(
                "fastcode.api.routes._ensure_fastcode_initialized",
                return_value=fake_fastcode,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            asyncio.run(
                api.expand_agent_context_ref(
                    api.ExpandContextRefRequest(
                        session_id="sess-1",
                        turn_number=2,
                        ref_id="e9",
                        depth="L2",
                    )
                )
            )

        assert exc_info.value.status_code == 404
        assert "context ref not found" in str(exc_info.value.detail)
