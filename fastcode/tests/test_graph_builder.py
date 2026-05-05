from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from unittest.mock import patch

import networkx as nx

from fastcode.graph.build import CodeGraphBuilder
from fastcode.ir.element import CodeElement


def _builder(tmp_path: Path) -> CodeGraphBuilder:
    return CodeGraphBuilder({"vector_store": {"persist_directory": str(tmp_path)}})


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


def test_graph_save_avoids_code_element_to_dict(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    elem = _element("func:one", "load_config")
    builder.element_by_name = {elem.name: elem}
    builder.element_by_id = {elem.id: elem}

    with patch.object(
        CodeElement,
        "to_dict",
        autospec=True,
        side_effect=AssertionError("graph save must not call CodeElement.to_dict()"),
    ):
        assert builder.save("repo") is True

    with open(tmp_path / "repo_graphs.pkl", "rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - test fixture written locally above

    assert payload["element_by_id"][elem.id]["id"] == elem.id
    assert payload["element_by_id"][elem.id]["metadata"] == elem.metadata


def test_graph_load_uses_explicit_deserializer(tmp_path: Path) -> None:
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
        "fastcode.graph.build.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert builder.load("repo") is True

    assert mock_deserialize.call_count == 2
    assert calls == [payload_a, payload_b]
    assert set(builder.element_by_id) == {"func:a", "func:b"}
    assert builder.element_by_name["shared_name"].id == "func:b"


def test_graph_merge_uses_explicit_deserializer(tmp_path: Path) -> None:
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
        "fastcode.graph.build.deserialize_code_element",
        side_effect=_deserialize,
    ) as mock_deserialize:
        assert builder.merge_from_file("other") is True

    assert mock_deserialize.call_count == 2
    assert calls == [payload_existing, payload_new]
    assert set(builder.element_by_id) == {"func:existing", "func:new"}
    assert builder.element_by_id["func:new"].name == "refresh_cache"
    assert "func:new" in builder.call_graph
