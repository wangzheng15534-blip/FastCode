from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from fastcode.indexing.pipeline import IndexPipeline
from fastcode.ir.element import CodeElement
from fastcode.ir.types import IRSnapshot


def _element(
    element_id: str,
    *,
    embedding: np.ndarray | None,
) -> CodeElement:
    metadata: dict[str, object] = {"stable_unit_id": f"unit:{element_id}"}
    if embedding is not None:
        metadata["embedding"] = embedding
        metadata["embedding_text"] = f"text:{element_id}"
    return CodeElement(
        id=element_id,
        type="function",
        name=element_id,
        file_path=f"/repo/{element_id}.py",
        relative_path=f"{element_id}.py",
        language="python",
        start_line=1,
        end_line=5,
        code="pass\n",
        signature=f"def {element_id}()",
        docstring=None,
        summary=None,
        metadata=metadata,
        repo_name="repo",
        repo_url=None,
    )


def test_materialize_indexed_elements_for_storage_avoids_code_element_to_dict() -> None:
    embedded = _element(
        "with_embedding", embedding=np.asarray([1.0, 2.0], dtype=np.float32)
    )
    plain = _element("without_embedding", embedding=None)

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError(
            "pipeline materialization must not call CodeElement.to_dict()"
        ),
    ):
        vectors, metadata, all_payloads = (
            IndexPipeline._materialize_indexed_elements_for_storage(
                [embedded, plain], snapshot_id="snap:1"
            )
        )

    assert isinstance(vectors, np.ndarray)
    assert vectors.dtype == np.float32
    assert vectors.shape == (1, 2)
    assert np.array_equal(vectors[0], embedded.metadata["embedding"])
    assert len(metadata) == 1
    assert len(all_payloads) == 2
    assert all(payload.get("snapshot_id") == "snap:1" for payload in all_payloads)
    assert metadata[0] is all_payloads[0]
    assert metadata[0]["metadata"]["stable_unit_id"] == "unit:with_embedding"
    assert "snapshot_id" not in embedded.metadata
    assert "source_priority" not in embedded.metadata


def _pipeline_for_unit_artifacts(repo_root: Path) -> IndexPipeline:
    pipeline = IndexPipeline.__new__(IndexPipeline)
    pipeline.loader = SimpleNamespace(repo_path=str(repo_root))
    pipeline.embedder = SimpleNamespace(
        embedding_artifact_ref=lambda text: f"artifact:{text}"
    )
    return pipeline


def test_unit_artifact_rows_avoid_code_element_to_dict_and_metadata_mutation(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='repo'\n", encoding="utf-8"
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("pass\n", encoding="utf-8")
    pipeline = _pipeline_for_unit_artifacts(tmp_path)
    element = _element("pkg/a", embedding=None)
    element.file_path = str(tmp_path / "pkg" / "a.py")
    element.relative_path = "pkg/a.py"
    element.metadata["embedding_text"] = "text:pkg/a"

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError(
            "unit artifact materialization must not call CodeElement.to_dict()"
        ),
    ):
        rows = pipeline._unit_artifact_rows([element])

    assert len(rows) == 1
    assert rows[0]["relative_path"] == "pkg/a.py"
    assert rows[0]["embedding_artifact_ref"] == "artifact:text:pkg/a"
    assert rows[0]["package_root"] == "."
    assert rows[0]["metadata"]["stable_unit_id"] == "unit:pkg/a"
    assert rows[0]["metadata"]["embedding_artifact_ref"] == "artifact:text:pkg/a"
    assert rows[0]["metadata"]["package_root"] == "."
    assert "embedding_artifact_ref" not in element.metadata
    assert "package_root" not in element.metadata


def test_unit_artifact_rows_copy_mapping_inputs_before_enrichment(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='repo'\n", encoding="utf-8"
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("pass\n", encoding="utf-8")
    pipeline = _pipeline_for_unit_artifacts(tmp_path)
    row = {
        "type": "function",
        "file_path": str(tmp_path / "pkg" / "b.py"),
        "relative_path": "pkg/b.py",
        "embedding_text": "text:pkg/b",
        "metadata": {"stable_unit_id": "unit:pkg/b"},
    }

    materialized = pipeline._unit_artifact_rows([row])

    assert len(materialized) == 1
    assert (
        materialized[0]["metadata"]["embedding_artifact_ref"] == "artifact:text:pkg/b"
    )
    assert materialized[0]["metadata"]["package_root"] == "."
    assert "embedding_artifact_ref" not in row["metadata"]
    assert "package_root" not in row["metadata"]
    assert "package_root" not in row


def test_save_relational_facts_for_index_uses_delta_for_safe_incremental_plan() -> None:
    class _DeltaStore:
        def __init__(self) -> None:
            self.delta_call: dict[str, Any] | None = None

        def save_relational_facts_delta(
            self,
            snapshot: IRSnapshot,
            *,
            previous_snapshot_id: str,
            changed_paths: list[str],
            removed_paths: list[str],
        ) -> bool:
            self.delta_call = {
                "snapshot": snapshot,
                "previous_snapshot_id": previous_snapshot_id,
                "changed_paths": list(changed_paths),
                "removed_paths": list(removed_paths),
            }
            return True

        def save_relational_facts(self, _snapshot: IRSnapshot) -> None:
            raise AssertionError("safe incremental plans should use delta persistence")

    pipeline = IndexPipeline.__new__(IndexPipeline)
    store = _DeltaStore()
    pipeline.snapshot_store = store  # type: ignore[assignment]
    snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:new")

    result = pipeline._save_relational_facts_for_index(
        snapshot,
        {
            "previous_snapshot_id": "snap:repo:prev",
            "added_paths": ["./pkg/a.py"],
            "modified_paths": ["pkg/../pkg/b.py"],
            "removed_paths": ["old.py", ""],
            "semantic_frontier_widened": 0,
        },
    )

    assert result == {
        "mode": "delta",
        "previous_snapshot_id": "snap:repo:prev",
        "changed_path_count": 2,
        "removed_path_count": 1,
    }
    assert store.delta_call is not None
    assert store.delta_call["snapshot"] is snapshot
    assert store.delta_call["changed_paths"] == ["pkg/a.py", "pkg/b.py"]
    assert store.delta_call["removed_paths"] == ["old.py"]


def test_save_relational_facts_for_index_falls_back_when_semantic_frontier_widened() -> (
    None
):
    class _FallbackStore:
        def __init__(self) -> None:
            self.delta_called = False
            self.full_snapshot: IRSnapshot | None = None

        def save_relational_facts_delta(
            self,
            _snapshot: IRSnapshot,
            **_kwargs: object,
        ) -> bool:
            self.delta_called = True
            return True

        def save_relational_facts(self, snapshot: IRSnapshot) -> None:
            self.full_snapshot = snapshot

    pipeline = IndexPipeline.__new__(IndexPipeline)
    store = _FallbackStore()
    pipeline.snapshot_store = store  # type: ignore[assignment]
    snapshot = IRSnapshot(repo_name="repo", snapshot_id="snap:repo:new")

    result = pipeline._save_relational_facts_for_index(
        snapshot,
        {
            "previous_snapshot_id": "snap:repo:prev",
            "added_paths": ["a.py"],
            "modified_paths": [],
            "removed_paths": [],
            "semantic_frontier_widened": 1,
        },
    )

    assert result == {"mode": "full"}
    assert store.delta_called is False
    assert store.full_snapshot is snapshot
