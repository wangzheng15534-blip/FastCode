from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from fastcode.indexing.pipeline import IndexPipeline
from fastcode.ir.element import CodeElement
from fastcode.ir.graph import IRGraphBuilder
from fastcode.ir.types import IRCodeUnit, IRRelation, IRSnapshot
from fastcode.scip.ast_adapter import build_ir_from_ast


def _element(
    element_id: str,
    *,
    embedding: np.ndarray | None,
) -> CodeElement:
    metadata: dict[str, object] = {"stable_unit_id": f"unit:{element_id}"}
    if embedding is not None:
        metadata["embedding"] = embedding
        metadata["embedding_text"] = f"text:{element_id}"
        metadata["embedding_fingerprint"] = {
            "provider": "test",
            "model": "stub",
            "dimension": int(embedding.shape[0]),
        }
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
    assert metadata[0]["metadata"]["embedding_fingerprint"] == {
        "provider": "test",
        "model": "stub",
        "dimension": 2,
    }
    assert "snapshot_id" not in embedded.metadata
    assert "source_priority" not in embedded.metadata


def test_ast_ir_embeddings_preserve_embedding_fingerprint_metadata() -> None:
    fingerprint = {
        "version": 2,
        "provider": "test",
        "model": "stub",
        "dimension": 2,
        "text_schema_version": 1,
    }
    element = CodeElement(
        id="pkg/a",
        type="function",
        name="a",
        file_path="/repo/pkg/a.py",
        relative_path="pkg/a.py",
        language="python",
        start_line=1,
        end_line=2,
        code="def a(): pass",
        signature="def a()",
        docstring=None,
        summary=None,
        metadata={
            "embedding": np.asarray([1.0, 2.0], dtype=np.float32),
            "embedding_text": "text:pkg/a",
            "embedding_text_hash": "hash:pkg/a",
            "embedding_artifact_ref": "artifact:text:pkg/a",
            "embedding_fingerprint": fingerprint,
        },
        repo_name="repo",
        repo_url=None,
    )

    snapshot = build_ir_from_ast("repo", "snap:1", [element], "/repo")

    assert len(snapshot.embeddings) == 1
    embedding_metadata = snapshot.embeddings[0].metadata
    assert embedding_metadata["embedding_fingerprint"] == fingerprint
    assert embedding_metadata["embedding_artifact_ref"] == "artifact:text:pkg/a"
    assert embedding_metadata["embedding_text_hash"] == "hash:pkg/a"


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

    assert result == {
        "mode": "full",
        "fallback_reason": "semantic_frontier_widened",
        "previous_snapshot_id": "snap:repo:prev",
        "changed_path_count": 1,
        "removed_path_count": 0,
    }
    assert store.delta_called is False
    assert store.full_snapshot is snapshot


def test_save_ir_graphs_for_index_publishes_graph_delta() -> None:
    builder = IRGraphBuilder()
    previous = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:prev",
        units=[
            IRCodeUnit(
                unit_id="unit:a",
                kind="function",
                path="pkg/a.py",
                language="python",
                display_name="a",
                source_set={"fc_structure"},
            ),
            IRCodeUnit(
                unit_id="unit:b",
                kind="function",
                path="pkg/b.py",
                language="python",
                display_name="b",
                source_set={"fc_structure"},
            ),
        ],
        relations=[
            IRRelation(
                relation_id="rel:import",
                src_unit_id="unit:a",
                dst_unit_id="unit:b",
                relation_type="import",
                resolution_state="structural",
            ),
            IRRelation(
                relation_id="rel:call",
                src_unit_id="unit:a",
                dst_unit_id="unit:b",
                relation_type="call",
                resolution_state="structural",
            ),
        ],
    )
    current = IRSnapshot(
        repo_name="repo",
        snapshot_id="snap:repo:current",
        units=previous.units,
        relations=[
            IRRelation(
                relation_id="rel:import",
                src_unit_id="unit:a",
                dst_unit_id="unit:b",
                relation_type="import",
                resolution_state="structural",
            )
        ],
    )

    class _GraphDeltaStore:
        def __init__(self) -> None:
            self.previous_graphs = builder.build_graphs(previous)
            self.delta_call: dict[str, Any] | None = None

        def load_ir_graphs(self, snapshot_id: str) -> Any:
            assert snapshot_id == previous.snapshot_id
            return self.previous_graphs

        def save_ir_graphs_delta(
            self,
            snapshot_id: str,
            ir_graphs: Any,
            *,
            previous_snapshot_id: str,
            reusable_graphs: list[str],
        ) -> tuple[str, dict[str, int]]:
            self.delta_call = {
                "snapshot_id": snapshot_id,
                "ir_graphs": ir_graphs,
                "previous_snapshot_id": previous_snapshot_id,
                "reusable_graphs": reusable_graphs,
            }
            return "ir_graphs_manifest.json", {
                "ir_graph_shards_reused": len(reusable_graphs),
                "ir_graph_shards_written": 2,
                "fallback_full_rewrite": 0,
            }

        def save_ir_graphs(self, _snapshot_id: str, _ir_graphs: Any) -> None:
            raise AssertionError("safe incremental graph publication should use delta")

    pipeline = IndexPipeline.__new__(IndexPipeline)
    store = _GraphDeltaStore()
    pipeline.snapshot_store = store  # type: ignore[assignment]
    pipeline.ir_graph_builder = builder

    _graphs, result = pipeline._save_ir_graphs_for_index(
        current,
        {
            "previous_snapshot_id": previous.snapshot_id,
            "added_paths": [],
            "modified_paths": ["pkg/a.py"],
            "removed_paths": [],
        },
    )

    assert result["mode"] == "delta"
    assert result["previous_snapshot_id"] == previous.snapshot_id
    assert result["ir_graph_shards_reused"] >= 1
    assert "reference_graph" in result["reusable_graphs"]
    assert "call_graph" in result["rebuilt_graphs"]
    assert store.delta_call is not None
    assert store.delta_call["snapshot_id"] == current.snapshot_id
    assert current.metadata["pipeline_metrics"]["ir_graph_delta"] == result


def test_pipeline_profile_records_stage_allocations_loader_io_and_store_bytes(
    tmp_path: Path,
) -> None:
    pipeline = IndexPipeline.__new__(IndexPipeline)
    pipeline.config = {"indexing": {"profile_pipeline": True}}
    pipeline.loader = SimpleNamespace(
        last_load_stats={
            "copied_bytes": 12,
            "copied_files": 2,
            "linked_bytes": 34,
            "linked_files": 3,
            "copy_cache_hit": True,
            "copy_cache_key": "cache-key",
        }
    )
    pipeline._active_pipeline_profile = None

    profile = pipeline._new_pipeline_profile()
    assert profile is not None
    with pipeline._profile_stage(profile, "example"):
        allocated = bytearray(4096)
        assert allocated

    store_root = tmp_path / "snapshot-store"
    store_root.mkdir()
    with pipeline._profile_store_surface(profile, "ir", str(store_root)):
        (store_root / "snapshot.json").write_bytes(b"abc")
    pipeline._profile_record_loader_io(profile)

    metrics: dict[str, Any] = {}
    pipeline._attach_pipeline_profile(metrics, profile)
    profiling = metrics["profiling"]

    assert profiling["stages"]["example"]["calls"] == 1
    assert profiling["stages"]["example"]["allocation_peak_bytes"] > 0
    assert profiling["snapshot_store_bytes_written"]["ir"] >= 3
    assert profiling["io"]["repository_copied_bytes"] == 12
    assert profiling["io"]["repository_hard_linked_bytes"] == 34
    assert profiling["io"]["repository_copy_cache_hit"] is True


def test_copy_scope_root_reports_scoped_tool_io_and_deleted_bytes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "a.py").write_text("print('a')\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    temp_root = tmp_path / "scope-copy"

    pipeline = IndexPipeline.__new__(IndexPipeline)
    pipeline.config = {"indexing": {"profile_pipeline": True}}
    pipeline._active_pipeline_profile = pipeline._new_pipeline_profile()

    materialized = pipeline._copy_scope_root(str(repo), "pkg", str(temp_root))
    assert materialized == str(temp_root)
    assert (temp_root / "pkg" / "a.py").exists()
    assert (temp_root / "pyproject.toml").exists()

    profile = pipeline._active_pipeline_profile
    assert profile is not None
    deleted_bytes = pipeline._path_size_bytes(str(temp_root))
    pipeline._profile_add_io("deleted_bytes", deleted_bytes)

    assert profile["io"]["scoped_tool_copied_files"] == 2
    assert profile["io"]["scoped_tool_copied_bytes"] >= deleted_bytes
    assert profile["io"]["deleted_bytes"] == deleted_bytes
