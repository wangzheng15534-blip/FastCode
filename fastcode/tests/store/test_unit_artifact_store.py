from __future__ import annotations

import tempfile

from fastcode.store.unit_artifacts import UnitArtifactStore


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
