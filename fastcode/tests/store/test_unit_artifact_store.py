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
                    },
                }
            ],
        )

        rows = store.list_snapshot_units("snap:1")
        assert len(rows) == 1
        assert rows[0]["stable_unit_id"] == "unit:function:1"
        assert rows[0]["relative_path"] == "pkg/a.py"
        assert rows[0]["metadata"]["api_surface_hash"] == "api1"

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
