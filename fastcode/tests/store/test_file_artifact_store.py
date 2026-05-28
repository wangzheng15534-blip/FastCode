from __future__ import annotations

import tempfile
from typing import Any

import pytest

from fastcode.app.store.artifacts.file_contracts import FileArtifactRecord
from fastcode.app.store.artifacts.file import FileArtifactStore
from fastcode.infrastructure.storage.runtime import DBRuntime


def _make_store(tmp: str) -> FileArtifactStore:
    return FileArtifactStore(
        DBRuntime(backend="sqlite", sqlite_path=f"{tmp}/file_artifacts.db")
    )


def test_file_artifact_store_upserts_and_loads_file_ir_by_content_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_file_artifacts_") as tmp:
        store = _make_store(tmp)

        summary, records = store.upsert_file_ir_shards(
            repo_name="repo",
            shards=[
                {
                    "snapshot_id": "snap:old",
                    "relative_path": "pkg/a.py",
                    "content_hash": "hash-a",
                    "units": [
                        {
                            "unit_id": "unit:file:a",
                            "kind": "file",
                            "path": "pkg/a.py",
                            "metadata": {"content_hash": "hash-a"},
                        }
                    ],
                    "supports": [{"support_id": "sup:a", "unit_id": "unit:file:a"}],
                    "relations": [
                        {
                            "relation_id": "rel:a",
                            "src_unit_id": "unit:file:a",
                            "dst_unit_id": "unit:file:a",
                        }
                    ],
                    "embeddings": [{"embedding_id": "emb:a", "unit_id": "unit:file:a"}],
                }
            ],
        )

        def _boom_row_to_dict(_: object) -> dict[str, Any]:
            raise AssertionError("file artifact store must not call row_to_dict()")

        def _boom_record_to_dict(_: FileArtifactRecord) -> dict[str, Any]:
            raise AssertionError("file artifact compatibility payload is explicit")

        monkeypatch.setattr(
            store.db_runtime,
            "row_to_dict",
            _boom_row_to_dict,
            raising=False,
        )
        monkeypatch.setattr(FileArtifactRecord, "to_dict", _boom_record_to_dict)

        fetched = store.list_file_ir_records_for_file_infos(
            repo_name="repo",
            file_infos=[
                {
                    "relative_path": "./pkg/a.py",
                    "content_hash": "hash-a",
                }
            ],
            paths=["pkg/a.py"],
        )
        payload = FileArtifactStore.file_ir_payload_from_record(
            fetched[0],
            snapshot_id="snap:new",
        )

        assert summary == {
            "mode": "content_addressed",
            "artifact_type": "file_ir",
            "written_records": 1,
        }
        assert records[0].identity_kind == "content_hash"
        assert records[0].identity_value == "hash-a"
        assert fetched[0].repo_name == "repo"
        assert fetched[0].relative_path == "pkg/a.py"
        assert payload["snapshot_id"] == "snap:new"
        assert payload["relative_path"] == "pkg/a.py"
        assert payload["identity_kind"] == "content_hash"
        assert payload["identity_value"] == "hash-a"
        assert payload["units"][0]["unit_id"] == "unit:file:a"
        assert payload["counts"] == {
            "units": 1,
            "supports": 1,
            "relations": 1,
            "embeddings": 1,
        }


def test_file_artifact_store_prefers_blob_oid_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_file_artifacts_blob_") as tmp:
        store = _make_store(tmp)
        store.upsert_file_ir_shards(
            repo_name="repo",
            shards=[
                {
                    "relative_path": "pkg/a.py",
                    "blob_oid": "blob-a",
                    "content_hash": "hash-a",
                    "units": [{"unit_id": "unit:file:a", "path": "pkg/a.py"}],
                }
            ],
        )

        fetched = store.list_file_ir_records_for_file_infos(
            repo_name="repo",
            file_infos=[
                {
                    "relative_path": "pkg/a.py",
                    "git_blob_oid": "blob-a",
                    "content_hash": "different-working-tree-hash",
                }
            ],
            paths=["pkg/a.py"],
        )

        assert len(fetched) == 1
        assert fetched[0].identity_kind == "blob_oid"
        assert fetched[0].identity_value == "blob-a"


def test_file_artifact_store_upserts_parsed_elements_without_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_parsed_elements_") as tmp:
        store = _make_store(tmp)

        summary, records = store.upsert_parsed_elements(
            repo_name="repo",
            elements=[
                {
                    "id": "function:a",
                    "type": "function",
                    "name": "a",
                    "file_path": "pkg/a.py",
                    "relative_path": "pkg/a.py",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 2,
                    "code": "def a():\n    return 1\n",
                    "signature": "def a()",
                    "docstring": None,
                    "summary": None,
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "embedding": [1.0, 2.0, 3.0],
                    },
                    "repo_name": "repo",
                    "repo_url": None,
                }
            ],
            file_infos=[
                {
                    "relative_path": "pkg/a.py",
                    "content_hash": "file-hash-a",
                }
            ],
        )

        def _boom_row_to_dict(_: object) -> dict[str, Any]:
            raise AssertionError("parsed element store must not call row_to_dict()")

        def _boom_record_to_dict(_: FileArtifactRecord) -> dict[str, Any]:
            raise AssertionError("parsed element payloads must be explicit")

        monkeypatch.setattr(
            store.db_runtime,
            "row_to_dict",
            _boom_row_to_dict,
            raising=False,
        )
        monkeypatch.setattr(FileArtifactRecord, "to_dict", _boom_record_to_dict)

        fetched = store.list_parsed_element_records_for_file_infos(
            repo_name="repo",
            file_infos=[{"relative_path": "pkg/a.py", "content_hash": "file-hash-a"}],
            paths=["pkg/a.py"],
        )
        payload = FileArtifactStore.parsed_elements_payload_from_record(fetched[0])

        assert summary == {
            "mode": "content_addressed",
            "artifact_type": "parsed_elements",
            "candidate_files": 1,
            "written_records": 1,
        }
        assert records[0].identity_kind == "content_hash"
        assert records[0].identity_value == "file-hash-a"
        assert records[0].unit_count == 1
        assert payload["counts"] == {"elements": 1}
        assert payload["elements"][0]["id"] == "function:a"
        assert "embedding" not in payload["elements"][0]
        assert "embedding" not in payload["elements"][0]["metadata"]


def test_file_artifact_store_upserts_embedding_refs_by_content_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_embedding_refs_") as tmp:
        store = _make_store(tmp)

        summary, records = store.upsert_embedding_refs(
            repo_name="repo",
            rows=[
                {
                    "type": "function",
                    "relative_path": "./pkg/a.py",
                    "content_hash": "unit-hash-a",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "embedding_text_hash": "embed-text-a",
                        "embedding_artifact_ref": "vector://repo/a",
                        "embedding_fingerprint": {
                            "provider": "test",
                            "model": "tiny",
                            "dimension": 3,
                        },
                    },
                },
                {
                    "type": "class",
                    "relative_path": "pkg/a.py",
                    "metadata": {"stable_unit_id": "unit:class:a"},
                },
            ],
            file_infos=[
                {
                    "relative_path": "pkg/a.py",
                    "content_hash": "file-hash-a",
                }
            ],
        )

        def _boom_row_to_dict(_: object) -> dict[str, Any]:
            raise AssertionError("embedding ref store must not call row_to_dict()")

        def _boom_record_to_dict(_: FileArtifactRecord) -> dict[str, Any]:
            raise AssertionError("embedding ref payloads must be explicit")

        monkeypatch.setattr(
            store.db_runtime,
            "row_to_dict",
            _boom_row_to_dict,
            raising=False,
        )
        monkeypatch.setattr(FileArtifactRecord, "to_dict", _boom_record_to_dict)

        fetched = store.list_embedding_ref_records_for_file_infos(
            repo_name="repo",
            file_infos=[{"relative_path": "pkg/a.py", "content_hash": "file-hash-a"}],
            paths=["./pkg/a.py"],
        )
        payload = FileArtifactStore.embedding_refs_payload_from_record(fetched[0])

        assert summary == {
            "mode": "content_addressed",
            "artifact_type": "embedding_refs",
            "candidate_files": 1,
            "written_records": 1,
        }
        assert records[0].identity_kind == "content_hash"
        assert records[0].identity_value == "file-hash-a"
        assert records[0].unit_count == 2
        assert records[0].embedding_count == 1
        assert payload["relative_path"] == "pkg/a.py"
        assert payload["counts"] == {"units": 2, "embeddings": 1}
        assert payload["embeddings"] == [
            {
                "stable_unit_id": "unit:function:a",
                "unit_type": "function",
                "content_hash": "unit-hash-a",
                "embedding_text_hash": "embed-text-a",
                "embedding_artifact_ref": "vector://repo/a",
                "embedding_fingerprint": {
                    "provider": "test",
                    "model": "tiny",
                    "dimension": 3,
                },
            }
        ]


def test_file_artifact_store_upserts_semantic_facts_by_content_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_semantic_facts_") as tmp:
        store = _make_store(tmp)

        summary, records = store.upsert_semantic_fact_shards(
            repo_name="repo",
            shards=[
                {
                    "relative_path": "pkg/a.py",
                    "supports": [
                        {
                            "support_id": "support:a",
                            "unit_id": "unit:function:a",
                            "source": "scip",
                            "support_kind": "definition",
                        }
                    ],
                    "relations": [
                        {
                            "relation_id": "rel:a",
                            "src_unit_id": "unit:function:a",
                            "dst_unit_id": "unit:function:b",
                            "relation_type": "calls",
                        }
                    ],
                },
                {
                    "relative_path": "pkg/b.py",
                    "supports": [],
                    "relations": [],
                },
            ],
            file_infos=[
                {
                    "relative_path": "pkg/a.py",
                    "content_hash": "file-hash-a",
                },
                {
                    "relative_path": "pkg/b.py",
                    "content_hash": "file-hash-b",
                },
            ],
        )

        def _boom_row_to_dict(_: object) -> dict[str, Any]:
            raise AssertionError("semantic fact store must not call row_to_dict()")

        def _boom_record_to_dict(_: FileArtifactRecord) -> dict[str, Any]:
            raise AssertionError("semantic fact payloads must be explicit")

        monkeypatch.setattr(
            store.db_runtime,
            "row_to_dict",
            _boom_row_to_dict,
            raising=False,
        )
        monkeypatch.setattr(FileArtifactRecord, "to_dict", _boom_record_to_dict)

        fetched = store.list_semantic_fact_records_for_file_infos(
            repo_name="repo",
            file_infos=[{"relative_path": "pkg/a.py", "content_hash": "file-hash-a"}],
            paths=["pkg/a.py"],
        )
        payload = FileArtifactStore.semantic_facts_payload_from_record(fetched[0])

        assert summary == {
            "mode": "content_addressed",
            "artifact_type": "semantic_facts",
            "candidate_shards": 2,
            "written_records": 1,
        }
        assert records[0].identity_kind == "content_hash"
        assert records[0].identity_value == "file-hash-a"
        assert records[0].support_count == 1
        assert records[0].relation_count == 1
        assert payload["relative_path"] == "pkg/a.py"
        assert payload["counts"] == {"supports": 1, "relations": 1}
        assert payload["supports"][0]["support_id"] == "support:a"
        assert payload["relations"][0]["relation_id"] == "rel:a"
