"""Benchmarks for scoped SCIP cache planning."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fastcode.app.indexing.pipeline.service import IndexPipeline

pytestmark = [pytest.mark.perf]


def _pipeline(tmp_path: Any) -> IndexPipeline:
    return IndexPipeline(
        config={},
        logger=SimpleNamespace(
            info=lambda *a, **kw: None, warning=lambda *a, **kw: None
        ),
        loader=SimpleNamespace(repo_path=str(tmp_path)),
        snapshot_store=SimpleNamespace(),
        manifest_store=SimpleNamespace(),
        index_run_store=SimpleNamespace(),
        unit_artifact_store=SimpleNamespace(),
        snapshot_symbol_index=SimpleNamespace(),
        vector_store=SimpleNamespace(persist_dir=str(tmp_path)),
        embedder=SimpleNamespace(),
        indexer=SimpleNamespace(),
        retriever=SimpleNamespace(),
        graph_builder=SimpleNamespace(),
        ir_graph_builder=SimpleNamespace(),
        pg_retrieval_store=None,
        terminus_publisher=SimpleNamespace(),
        doc_ingester=SimpleNamespace(),
        semantic_resolver_registry=SimpleNamespace(),
        set_repo_indexed=lambda _value: None,
        set_repo_loaded=lambda _value: None,
        set_repo_info=lambda _value: None,
    )


def test_scoped_scip_cache_key_fingerprinting_perf(
    tmp_path: Any, benchmark: Any
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='repo'\n")
    target_paths: set[str] = set()
    for index in range(200):
        rel_path = f"pkg/mod_{index}.py"
        (tmp_path / rel_path).write_text(f"def f_{index}():\n    return {index}\n")
        target_paths.add(rel_path)

    pipeline = _pipeline(tmp_path)

    def _build_key() -> str:
        return pipeline._scoped_scip_cache_entry(
            repo_root=str(tmp_path),
            language="python",
            scope_root="pkg",
            target_paths=target_paths,
        ).key

    cache_key = benchmark(_build_key)

    assert len(cache_key) == 64
