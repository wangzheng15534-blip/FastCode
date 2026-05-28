from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import networkx as nx

from fastcode.app.store.artifacts.graph import GraphArtifactStore
from fastcode.graph.build import CodeGraphBuilder
from fastcode.ir.element import CodeElement


def _config(tmp_path: Path) -> dict[str, Any]:
    return {"vector_store": {"persist_directory": str(tmp_path)}}


def _builder(tmp_path: Path) -> CodeGraphBuilder:
    return CodeGraphBuilder(_config(tmp_path))


def _store(tmp_path: Path) -> GraphArtifactStore:
    return GraphArtifactStore(_config(tmp_path))


def _element(
    element_id: str,
    name: str,
    *,
    relative_path: str = "src/module.py",
    element_type: str = "function",
) -> CodeElement:
    return CodeElement(
        id=element_id,
        type=element_type,
        name=name,
        file_path=f"/repo/{relative_path}",
        relative_path=relative_path,
        language="python",
        start_line=1,
        end_line=5,
        code="pass\n",
        signature=None,
        docstring=None,
        summary=None,
        metadata={"stable_unit_id": f"unit:{element_id}"},
        repo_name="repo",
        repo_url=None,
    )


def _graph_payload(
    *,
    element_by_name: dict[str, dict[str, Any]] | None = None,
    element_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "call_graph": nx.DiGraph(),
        "dependency_graph": nx.DiGraph(),
        "inheritance_graph": nx.DiGraph(),
        "element_by_name": element_by_name or {},
        "element_by_id": element_by_id or {},
        "imports_by_file": {},
    }


def test_graph_artifact_save_avoids_code_element_to_dict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    builder = _builder(tmp_path)
    elem = _element("func:one", "load_config")
    builder.element_by_name = {elem.name: elem}
    builder.element_by_id = {elem.id: elem}
    (tmp_path / "repo_graphs.pkl").write_bytes(b"legacy")

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError("graph save must not call CodeElement.to_dict()"),
    ):
        assert store.save(builder, "repo") is True

    assert (tmp_path / "repo_graph_manifest.json").exists()
    assert (tmp_path / "repo_graph_shards").is_dir()
    assert not (tmp_path / "repo_graphs.pkl").exists()

    loaded = _builder(tmp_path)
    assert store.load(loaded, "repo") is True

    assert loaded.element_by_id[elem.id].id == elem.id
    assert loaded.element_by_id[elem.id].metadata == elem.metadata


def test_graph_artifact_save_reuses_unchanged_shards(tmp_path: Path) -> None:
    store = _store(tmp_path)
    builder = _builder(tmp_path)
    elem_a = _element("func:a", "load_a", relative_path="src/a.py")
    elem_b = _element("func:b", "load_b", relative_path="src/b.py")
    builder.element_by_name = {elem_a.name: elem_a, elem_b.name: elem_b}
    builder.element_by_id = {elem_a.id: elem_a, elem_b.id: elem_b}
    builder.call_graph.add_edge(elem_a.id, elem_b.id, type="calls")

    assert store.save(builder, "repo") is True
    manifest = json.loads(
        (tmp_path / "repo_graph_manifest.json").read_text(encoding="utf-8")
    )
    shards_by_path = {
        entry["path_key"]: tmp_path / "repo_graph_shards" / entry["shard_file"]
        for entry in manifest["shards"]
    }
    a_before = shards_by_path["src/a.py"].stat().st_mtime_ns
    b_before = shards_by_path["src/b.py"].stat().st_mtime_ns

    time.sleep(0.01)
    elem_b_v2 = _element("func:b", "load_b", relative_path="src/b.py")
    elem_b_v2.metadata["refreshed"] = True
    builder.element_by_name = {elem_a.name: elem_a, elem_b_v2.name: elem_b_v2}
    builder.element_by_id = {elem_a.id: elem_a, elem_b_v2.id: elem_b_v2}
    assert store.save(builder, "repo") is True

    a_after = shards_by_path["src/a.py"].stat().st_mtime_ns
    b_after = shards_by_path["src/b.py"].stat().st_mtime_ns
    assert a_after == a_before
    assert b_after > b_before


def test_graph_artifact_save_incremental_reuses_previous_shards(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    previous = _builder(tmp_path)
    elem_a = _element("func:a", "load_a", relative_path="src/a.py")
    elem_b = _element("func:b", "load_b", relative_path="src/b.py")
    previous.element_by_name = {elem_a.name: elem_a, elem_b.name: elem_b}
    previous.element_by_id = {elem_a.id: elem_a, elem_b.id: elem_b}
    previous.call_graph.add_edge(elem_a.id, elem_b.id, type="calls")

    assert store.save(previous, "prev") is True
    prev_manifest = json.loads(
        (tmp_path / "prev_graph_manifest.json").read_text(encoding="utf-8")
    )
    prev_shards = {
        entry["path_key"]: tmp_path / "prev_graph_shards" / entry["shard_file"]
        for entry in prev_manifest["shards"]
    }

    current = _builder(tmp_path)
    elem_b_v2 = _element("func:b", "load_b", relative_path="src/b.py")
    elem_b_v2.metadata["refreshed"] = True
    current.element_by_name = {elem_a.name: elem_a, elem_b_v2.name: elem_b_v2}
    current.element_by_id = {elem_a.id: elem_a, elem_b_v2.id: elem_b_v2}
    current.call_graph.add_edge(elem_a.id, elem_b_v2.id, type="calls")

    stats = store.save_incremental(
        current,
        "next",
        previous_name="prev",
        reusable_path_keys={"src/a.py"},
    )

    assert stats["graph_shards_reused"] == 1
    next_manifest = json.loads(
        (tmp_path / "next_graph_manifest.json").read_text(encoding="utf-8")
    )
    next_shards = {
        entry["path_key"]: tmp_path / "next_graph_shards" / entry["shard_file"]
        for entry in next_manifest["shards"]
    }
    assert next_shards["src/a.py"].read_bytes() == prev_shards["src/a.py"].read_bytes()
    assert next_shards["src/b.py"].read_bytes() != prev_shards["src/b.py"].read_bytes()

    loaded = _builder(tmp_path)
    assert store.load(loaded, "next") is True
    assert loaded.get_related_elements(elem_a.id, max_hops=1) == {elem_a.id, elem_b.id}


def test_graph_artifact_publish_delta_carries_reusable_shards_without_full_builder(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    previous = _builder(tmp_path)
    elem_a = _element("func:a", "load_a", relative_path="src/a.py")
    elem_b = _element("func:b", "load_b", relative_path="src/b.py")
    previous.element_by_name = {elem_a.name: elem_a, elem_b.name: elem_b}
    previous.element_by_id = {elem_a.id: elem_a, elem_b.id: elem_b}
    previous.call_graph.add_edge(elem_a.id, elem_b.id, type="calls")

    assert store.save(previous, "prev") is True
    prev_manifest = json.loads(
        (tmp_path / "prev_graph_manifest.json").read_text(encoding="utf-8")
    )
    prev_shards = {
        entry["path_key"]: tmp_path / "prev_graph_shards" / entry["shard_file"]
        for entry in prev_manifest["shards"]
    }

    current_delta = _builder(tmp_path)
    elem_b_v2 = _element("func:b", "load_b", relative_path="src/b.py")
    elem_b_v2.metadata["refreshed"] = True
    current_delta.element_by_name = {elem_b_v2.name: elem_b_v2}
    current_delta.element_by_id = {elem_b_v2.id: elem_b_v2}

    stats = store.publish_delta(
        current_delta,
        "next",
        previous_name="prev",
        reusable_path_keys={"src/a.py"},
    )

    assert stats["fallback_reason"] is None
    assert stats["graph_shards_reused"] == 1
    assert stats["graph_shards_written"] == 1
    next_manifest = json.loads(
        (tmp_path / "next_graph_manifest.json").read_text(encoding="utf-8")
    )
    next_shards = {
        entry["path_key"]: tmp_path / "next_graph_shards" / entry["shard_file"]
        for entry in next_manifest["shards"]
    }
    assert set(next_shards) == {"src/a.py", "src/b.py"}
    assert next_shards["src/a.py"].read_bytes() == prev_shards["src/a.py"].read_bytes()
    assert next_shards["src/b.py"].read_bytes() != prev_shards["src/b.py"].read_bytes()

    loaded = _builder(tmp_path)
    assert store.load(loaded, "next") is True
    assert loaded.get_related_elements(elem_a.id, max_hops=1) == {elem_a.id, elem_b.id}


def test_graph_artifact_save_incremental_refuses_incompatible_manifest(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    previous = _builder(tmp_path)
    elem_a = _element("func:a", "load_a", relative_path="src/a.py")
    previous.element_by_name = {elem_a.name: elem_a}
    previous.element_by_id = {elem_a.id: elem_a}

    assert store.save(previous, "prev") is True
    manifest_path = tmp_path / "prev_graph_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    current = _builder(tmp_path)
    elem_a_v2 = _element("func:a", "load_a", relative_path="src/a.py")
    elem_a_v2.metadata["refreshed"] = True
    current.element_by_name = {elem_a_v2.name: elem_a_v2}
    current.element_by_id = {elem_a_v2.id: elem_a_v2}

    stats = store.save_incremental(
        current,
        "next",
        previous_name="prev",
        reusable_path_keys={"src/a.py"},
    )

    assert stats["graph_shards_reused"] == 0
    assert stats["graph_shards_written"] == 1


def test_graph_artifact_sharded_load_keeps_lazy_adjacency(tmp_path: Path) -> None:
    store = _store(tmp_path)
    builder = _builder(tmp_path)
    elem_a = _element("func:a", "load_a", relative_path="src/a.py")
    elem_b = _element("func:b", "load_b", relative_path="src/b.py")
    builder.element_by_name = {elem_a.name: elem_a, elem_b.name: elem_b}
    builder.element_by_id = {elem_a.id: elem_a, elem_b.id: elem_b}
    builder.call_graph.add_edge(elem_a.id, elem_b.id, type="calls")

    assert store.save(builder, "repo") is True

    loaded = _builder(tmp_path)
    assert store.load(loaded, "repo") is True
    assert loaded.call_graph.number_of_nodes() == 0

    related = loaded.get_related_elements(elem_a.id, max_hops=1)

    assert related == {elem_a.id, elem_b.id}
    assert loaded.call_graph.number_of_nodes() == 0

    path = loaded.find_path(elem_a.id, elem_b.id, graph_type="call")

    assert path == [elem_a.id, elem_b.id]
    assert loaded.call_graph.number_of_nodes() == 2


def test_graph_artifact_load_uses_explicit_deserializer(tmp_path: Path) -> None:
    store = _store(tmp_path)
    builder = _builder(tmp_path)
    payload_a = {
        "id": "func:a",
        "type": "function",
        "name": "shared_name",
        "file_path": "/repo/src/a.py",
        "relative_path": "src/a.py",
        "language": "python",
        "start_line": 1,
        "end_line": 5,
        "code": "pass\n",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {"stable_unit_id": "unit:func:a"},
        "repo_name": "repo",
        "repo_url": None,
    }
    payload_b = {
        **payload_a,
        "id": "func:b",
        "relative_path": "src/b.py",
        "file_path": "/repo/src/b.py",
        "metadata": {"stable_unit_id": "unit:func:b"},
    }
    with open(tmp_path / "repo_graphs.pkl", "wb") as handle:
        pickle.dump(
            _graph_payload(
                element_by_name={"shared_name": payload_b},
                element_by_id={"func:a": payload_a, "func:b": payload_b},
            ),
            handle,
        )

    calls: list[dict[str, Any]] = []

    def _deserialize(payload: dict[str, Any]) -> CodeElement:
        calls.append(payload)
        return _element(
            payload["id"], payload["name"], relative_path=payload["relative_path"]
        )

    with patch(
        "fastcode.app.store.artifacts.graph_payloads.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert store.load(builder, "repo") is True

    assert mock_deserialize.call_count == 2
    assert calls == [payload_a, payload_b]
    assert set(builder.element_by_id) == {"func:a", "func:b"}
    assert builder.element_by_name["shared_name"].id == "func:b"


def test_graph_artifact_merge_uses_explicit_deserializer(tmp_path: Path) -> None:
    store = _store(tmp_path)
    builder = _builder(tmp_path)
    existing = _element("func:existing", "load_config")
    builder.element_by_name = {existing.name: existing}
    builder.element_by_id = {existing.id: existing}
    builder.call_graph.add_node(existing.id)
    builder.dependency_graph.add_node("file:existing")
    builder.inheritance_graph.add_node("class:existing")

    payload_existing = {
        "id": "func:existing",
        "type": "function",
        "name": "load_config",
        "file_path": "/repo/src/existing.py",
        "relative_path": "src/existing.py",
        "language": "python",
        "start_line": 1,
        "end_line": 5,
        "code": "pass\n",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {"stable_unit_id": "unit:func:existing"},
        "repo_name": "repo",
        "repo_url": None,
    }
    payload_new = {
        **payload_existing,
        "id": "func:new",
        "name": "refresh_cache",
        "relative_path": "src/new.py",
        "file_path": "/repo/src/new.py",
        "metadata": {"stable_unit_id": "unit:func:new"},
    }
    other_call_graph = nx.DiGraph()
    other_call_graph.add_node("func:new")
    with open(tmp_path / "other_graphs.pkl", "wb") as handle:
        pickle.dump(
            {
                **_graph_payload(
                    element_by_name={"load_config": payload_existing},
                    element_by_id={
                        "func:existing": payload_existing,
                        "func:new": payload_new,
                    },
                ),
                "call_graph": other_call_graph,
            },
            handle,
        )

    calls: list[dict[str, Any]] = []

    def _deserialize(payload: dict[str, Any]) -> CodeElement:
        calls.append(payload)
        return _element(
            payload["id"], payload["name"], relative_path=payload["relative_path"]
        )

    with patch(
        "fastcode.app.store.artifacts.graph_payloads.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert store.merge(builder, "other") is True

    assert mock_deserialize.call_count == 2
    assert calls == [payload_existing, payload_new]
    assert set(builder.element_by_id) == {"func:existing", "func:new"}
    assert builder.element_by_id["func:new"].name == "refresh_cache"
    assert "func:new" in builder.call_graph
