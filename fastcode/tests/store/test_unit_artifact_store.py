from __future__ import annotations

import tempfile
from typing import Any

import pytest

from fastcode.store.records import UnitArtifactRecord
from fastcode.store.unit_artifacts import UnitArtifactStore


class _OpaqueValue:
    def __repr__(self) -> str:
        return "<opaque>"


def test_unit_artifact_store_replaces_and_lists_snapshot_units() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_unit_artifacts_") as tmp:
        store = UnitArtifactStore(f"{tmp}/unit_artifacts.db")

        store.replace_snapshot_units(
            "snap:1",
            elements=[
                {
                    "id": "elem:1",
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:1",
                        "content_hash": "c1",
                        "syntax_hash": "s1",
                        "signature_hash": "sig1",
                        "edge_surface_hash": "edge1",
                        "embedding_text_hash": "emb1",
                        "api_surface_hash": "api1",
                        "embedding_artifact_ref": "embedding:1",
                        "package_root": "pkg",
                        "scoped_tool_ref": "tool:old",
                    },
                }
            ],
        )

        rows = store.list_snapshot_units("snap:1")
        assert len(rows) == 1
        assert rows[0]["stable_unit_id"] == "unit:function:1"
        assert rows[0]["relative_path"] == "pkg/a.py"
        assert rows[0]["metadata"]["api_surface_hash"] == "api1"
        assert rows[0]["embedding_artifact_ref"] == "embedding:1"
        assert rows[0]["package_root"] == "pkg"
        assert rows[0]["scoped_tool_ref"] == "tool:old"

        store.replace_snapshot_units(
            "snap:1",
            elements=[
                {
                    "id": "elem:2",
                    "type": "class",
                    "relative_path": "pkg/b.py",
                    "metadata": {
                        "stable_unit_id": "unit:class:2",
                        "content_hash": "c2",
                        "syntax_hash": "s2",
                        "signature_hash": "sig2",
                        "edge_surface_hash": "edge2",
                        "embedding_text_hash": "emb2",
                        "api_surface_hash": "api2",
                    },
                }
            ],
        )

        rows = store.list_snapshot_units("snap:1")
        assert len(rows) == 1
        assert rows[0]["stable_unit_id"] == "unit:class:2"
        assert rows[0]["relative_path"] == "pkg/b.py"


def test_unit_artifact_store_refreshes_only_requested_units() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_unit_artifacts_refresh_") as tmp:
        store = UnitArtifactStore(f"{tmp}/unit_artifacts.db")

        store.replace_snapshot_units(
            "snap:1",
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "old",
                        "embedding_artifact_ref": "embedding:a:old",
                        "package_root": "pkg",
                    },
                },
                {
                    "type": "function",
                    "relative_path": "pkg/b.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:b",
                        "signature_hash": "keep",
                        "embedding_artifact_ref": "embedding:b",
                        "package_root": "pkg",
                    },
                },
            ],
        )

        store.refresh_units(
            "snap:1",
            stable_unit_ids=["unit:function:a"],
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "new",
                        "embedding_artifact_ref": "embedding:a:new",
                        "package_root": "pkg",
                        "scoped_tool_ref": "tool:repair",
                        "repair_frontier_summary": {"target_paths": ["pkg/a.py"]},
                    },
                }
            ],
        )

        rows = store.list_snapshot_units("snap:1")
        rows_by_id = {row["stable_unit_id"]: row for row in rows}
        assert rows_by_id["unit:function:a"]["signature_hash"] == "new"
        assert (
            rows_by_id["unit:function:a"]["embedding_artifact_ref"] == "embedding:a:new"
        )
        assert rows_by_id["unit:function:a"]["scoped_tool_ref"] == "tool:repair"
        assert rows_by_id["unit:function:b"]["signature_hash"] == "keep"
        assert rows_by_id["unit:function:b"]["embedding_artifact_ref"] == "embedding:b"


def test_unit_artifact_store_list_avoids_generic_row_to_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_unit_artifacts_row_") as tmp:
        store = UnitArtifactStore(f"{tmp}/unit_artifacts.db")
        store.replace_snapshot_units(
            "snap:1",
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "sig-a",
                    },
                }
            ],
        )

        def _boom(_: object) -> dict[str, Any]:
            raise AssertionError("unit artifact store must not call row_to_dict()")

        monkeypatch.setattr(store.db_runtime, "row_to_dict", _boom)

        rows = store.list_snapshot_units("snap:1")

        assert rows[0]["stable_unit_id"] == "unit:function:a"
        assert rows[0]["metadata"]["signature_hash"] == "sig-a"


def test_unit_artifact_store_exposes_typed_records() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_unit_artifacts_record_") as tmp:
        store = UnitArtifactStore(f"{tmp}/unit_artifacts.db")
        store.replace_snapshot_units(
            "snap:1",
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "content_hash": "content-a",
                        "signature_hash": "sig-a",
                        "embedding_artifact_ref": "embedding:a",
                    },
                }
            ],
        )

        records = store.list_snapshot_unit_records("snap:1")

        assert records == [
            UnitArtifactRecord(
                snapshot_id="snap:1",
                stable_unit_id="unit:function:a",
                relative_path="pkg/a.py",
                unit_type="function",
                content_hash="content-a",
                syntax_hash=None,
                signature_hash="sig-a",
                edge_surface_hash=None,
                embedding_text_hash=None,
                api_surface_hash=None,
                embedding_artifact_ref="embedding:a",
                scoped_tool_ref=None,
                package_root=None,
                repair_frontier_summary=None,
                metadata_json=(
                    '{"content_hash": "content-a", '
                    '"embedding_artifact_ref": "embedding:a", '
                    '"signature_hash": "sig-a", '
                    '"stable_unit_id": "unit:function:a"}'
                ),
                created_at=records[0].created_at,
            )
        ]


def test_unit_artifact_legacy_payload_avoids_record_to_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_unit_artifacts_compat_") as tmp:
        store = UnitArtifactStore(f"{tmp}/unit_artifacts.db")
        store.replace_snapshot_units(
            "snap:1",
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "sig-a",
                    },
                }
            ],
        )

        def _boom(_: UnitArtifactRecord) -> dict[str, Any]:
            raise AssertionError("unit artifact compatibility shim must be explicit")

        monkeypatch.setattr(UnitArtifactRecord, "to_dict", _boom)

        rows = store.list_snapshot_units("snap:1")

        assert rows[0]["stable_unit_id"] == "unit:function:a"
        assert rows[0]["metadata"]["signature_hash"] == "sig-a"


def test_unit_artifact_delta_copies_previous_records_without_legacy_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fc_unit_artifacts_delta_") as tmp:
        store = UnitArtifactStore(f"{tmp}/unit_artifacts.db")
        store.replace_snapshot_units(
            "snap:old",
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "signature_hash": "sig-a",
                        "embedding_artifact_ref": "embedding:a",
                    },
                },
                {
                    "type": "function",
                    "relative_path": "pkg/b.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:b",
                        "signature_hash": "sig-b-old",
                    },
                },
            ],
        )

        def _boom(_: str) -> list[dict[str, Any]]:
            raise AssertionError("delta copy must use typed unit artifact records")

        monkeypatch.setattr(store, "list_snapshot_units", _boom)

        summary = store.publish_snapshot_units_delta(
            "snap:new",
            previous_snapshot_id="snap:old",
            changed_paths=["pkg/b.py"],
            removed_paths=[],
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/b.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:b",
                        "signature_hash": "sig-b-new",
                    },
                }
            ],
        )

        records = store.list_snapshot_unit_records("snap:new")
        records_by_path = {record.relative_path: record for record in records}
        assert summary["copied_rows"] == 1
        assert records_by_path["pkg/a.py"].embedding_artifact_ref == "embedding:a"
        assert records_by_path["pkg/b.py"].signature_hash == "sig-b-new"


def test_unit_artifact_store_serializes_opaque_metadata_values() -> None:
    with tempfile.TemporaryDirectory(prefix="fc_unit_artifacts_opaque_") as tmp:
        store = UnitArtifactStore(f"{tmp}/unit_artifacts.db")
        store.replace_snapshot_units(
            "snap:1",
            elements=[
                {
                    "type": "function",
                    "relative_path": "pkg/a.py",
                    "metadata": {
                        "stable_unit_id": "unit:function:a",
                        "opaque": _OpaqueValue(),
                    },
                }
            ],
        )

        rows = store.list_snapshot_units("snap:1")

        assert rows[0]["metadata"]["opaque"] == "<opaque>"
