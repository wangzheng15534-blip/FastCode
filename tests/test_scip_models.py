import json
import tempfile

from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.scip_loader import load_scip_artifact
from fastcode.scip_models import SCIPArtifactRef, SCIPIndex


def test_scip_index_round_trip_preserves_fields():
    raw = {
        "indexer_name": "scip-python",
        "indexer_version": "1.2.3",
        "documents": [
            {
                "path": "app.py",
                "language": "python",
                "symbols": [{"symbol": "pkg app.py foo().", "name": "foo"}],
                "occurrences": [{"symbol": "pkg app.py foo().", "role": "definition", "range": [1, 0, 1, 3]}],
            }
        ],
        "custom_meta": {"x": 1},
    }
    index = SCIPIndex.from_dict(raw)
    out = index.to_dict()
    assert out["indexer_name"] == "scip-python"
    assert out["indexer_version"] == "1.2.3"
    assert out["documents"][0]["path"] == "app.py"
    assert out["custom_meta"] == {"x": 1}


def test_load_scip_artifact_returns_typed_model():
    payload = {
        "indexer_name": "scip-python",
        "documents": [{"path": "x.py", "symbols": [], "occurrences": []}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    loaded = load_scip_artifact(path)
    assert isinstance(loaded, SCIPIndex)
    assert loaded.documents[0].path == "x.py"


def test_scip_to_ir_uses_language_hint_when_document_language_missing():
    snap = build_ir_from_scip(
        repo_name="repo",
        snapshot_id="snap:repo:1",
        scip_index={
            "documents": [
                {
                    "path": "a.txt",
                    "symbols": [{"symbol": "ext:s:1", "name": "foo"}],
                    "occurrences": [{"symbol": "ext:s:1", "role": "definition", "range": [1, 0, 1, 3]}],
                }
            ]
        },
        language_hint="python",
    )
    assert snap.documents[0].language == "python"


def test_scip_artifact_ref_to_dict():
    ref = SCIPArtifactRef(
        snapshot_id="snap:repo:1",
        indexer_name="scip-python",
        indexer_version="1.0.0",
        artifact_path="/tmp/index.scip.json",
        checksum="abc",
        created_at="2026-01-01T00:00:00+00:00",
    )
    payload = ref.to_dict()
    assert payload["snapshot_id"] == "snap:repo:1"
    assert payload["checksum"] == "abc"
