from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

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
                "occurrences": [
                    {
                        "symbol": "pkg app.py foo().",
                        "role": "definition",
                        "range": [1, 0, 1, 3],
                    }
                ],
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


def test_load_scip_artifact_returns_typed_model(tmp_path: pathlib.Path):
    payload = {
        "indexer_name": "scip-python",
        "documents": [{"path": "x.py", "symbols": [], "occurrences": []}],
    }
    artifact = tmp_path / "index.scip.json"
    artifact.write_text(json.dumps(payload))
    loaded = load_scip_artifact(str(artifact))
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
                    "occurrences": [
                        {
                            "symbol": "ext:s:1",
                            "role": "definition",
                            "range": [1, 0, 1, 3],
                        }
                    ],
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


@pytest.mark.parametrize(
    "role",
    [
        "definition",
        "reference",
        "import",
        "implementation",
        "write_access",
        "forward_definition",
        "type_definition",
    ],
)
@pytest.mark.basic
def test_scip_occurrence_role_roundtrip(role: str):
    """HAPPY: SCIPOccurrence roundtrip preserves role for all valid roles."""
    from fastcode.scip_models import SCIPOccurrence

    occ = SCIPOccurrence(symbol="pkg foo.", role=role, range=[1, 0, 1, 5])
    data = occ.to_dict()
    restored = SCIPOccurrence.from_dict(data)
    assert restored.role == role


@pytest.mark.parametrize(
    "kind",
    [
        "function",
        "method",
        "class",
        "variable",
        "module",
        "interface",
        "enum",
        "constant",
        "macro",
    ],
)
@pytest.mark.basic
def test_scip_symbol_kind_roundtrip(kind: bool):
    """HAPPY: SCIPSymbol roundtrip preserves kind for all valid kinds."""
    from fastcode.scip_models import SCIPSymbol

    sym = SCIPSymbol(symbol="pkg foo.", name="foo", kind=kind)
    data = sym.to_dict()
    restored = SCIPSymbol.from_dict(data)
    assert restored.kind == kind


@pytest.mark.parametrize(
    "range_vals",
    [
        [1, 0, 1, 5],
        [0, 0, 0, 0],
        [100, 0, 200, 50],
        [None, None, None, None],
        [1, None, None, None],
    ],
)
@pytest.mark.edge
def test_scip_occurrence_range_variants_edge(range_vals: Any):
    """EDGE: SCIPOccurrence handles various range formats including None and zero."""
    from fastcode.scip_models import SCIPOccurrence

    occ = SCIPOccurrence(symbol="pkg foo.", range=range_vals)
    data = occ.to_dict()
    restored = SCIPOccurrence.from_dict(data)
    assert restored.range == list(range_vals)


@pytest.mark.parametrize(
    "language",
    [
        "python",
        "javascript",
        "typescript",
        "go",
        "java",
        "rust",
        "c",
        "cpp",
        "c-sharp",
        None,
    ],
)
@pytest.mark.edge
def test_scip_document_language_handling_edge(language: str):
    """EDGE: SCIPDocument handles all language values including None."""
    from fastcode.scip_models import SCIPDocument

    doc = SCIPDocument(path="test.py", language=language)
    data = doc.to_dict()
    restored = SCIPDocument.from_dict(data)
    assert restored.language == language
