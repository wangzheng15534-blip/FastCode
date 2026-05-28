from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from fastcode.app.indexing.pipeline.indexer import CodeIndexer
from fastcode.app.indexing.extractors.parser import FileParseResult, FunctionInfo, ImportInfo
from fastcode.ir.element import CodeElement


class _LoaderStub:
    repo_path = None

    def __init__(self, content: str) -> None:
        self._content = content

    def read_file_content(self, _file_path: str) -> str | None:
        return self._content


class _ParserStub:
    def __init__(self, result: FileParseResult) -> None:
        self._result = result

    def parse_file(self, _file_path: str, _content: str) -> FileParseResult | None:
        return self._result


class _EmbedderStub:
    def __init__(self) -> None:
        self.prepared_batches: list[list[dict[str, Any]]] = []
        self.prepared_text_batches: list[list[str]] = []

    def embed_code_elements(
        self, _elements: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        raise AssertionError("CodeIndexer should use EmbeddingService.embed_elements")

    def embed_elements(
        self,
        elements: Sequence[dict[str, Any]],
        reuse_index: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        assert reuse_index is None
        mutable_elements = list(elements)
        self.prepared_batches.append(
            [
                {
                    **element,
                    "metadata": dict(element.get("metadata", {}) or {}),
                }
                for element in mutable_elements
            ]
        )

        for index, element in enumerate(mutable_elements):
            element["embedding"] = np.asarray([float(index), float(index + 1)])
            element["embedding_text"] = f"text:{element['id']}"
            element["embedding_artifact_ref"] = f"artifact:{element['id']}"
            metadata = dict(element.get("metadata", {}) or {})
            metadata["embedding_text_hash"] = f"hash:{element['id']}"
            metadata["embedding_fingerprint"] = {
                "provider": "test",
                "model": "stub",
                "dimension": 2,
            }
            element["metadata"] = metadata

        return mutable_elements

    def embed_text(self, _text: str) -> np.ndarray:
        raise AssertionError("CodeIndexer should use EmbeddingService.embed_many")

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        self.prepared_text_batches.append(list(texts))
        return np.asarray(
            [[1.0, 0.0] for _text in texts],
            dtype=np.float32,
        )

    def embedding_fingerprint(
        self, *, resolve_dimension: bool = False
    ) -> dict[str, Any]:
        raise AssertionError("CodeIndexer should use EmbeddingService.fingerprint")

    def fingerprint(self, *, resolve_dimension: bool = False) -> dict[str, Any]:
        return {
            "provider": "test",
            "model": "stub",
            "dimension": 2 if resolve_dimension else None,
        }


class _VectorStoreStub:
    def __init__(self) -> None:
        self.saved_overviews: list[dict[str, Any]] = []

    def save_repo_overview(
        self,
        *,
        repo_name: str,
        overview_content: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        self.saved_overviews.append(
            {
                "repo_name": repo_name,
                "overview_content": overview_content,
                "embedding": embedding,
                "metadata": metadata,
            }
        )


def test_index_files_avoids_generic_element_and_import_to_dict_calls() -> None:
    parse_result = FileParseResult(
        file_path="/repo/mod.py",
        language="python",
        classes=[],
        functions=[
            FunctionInfo(
                name="foo",
                start_line=3,
                end_line=4,
                docstring=None,
                parameters=[],
                return_type=None,
                is_async=False,
                is_method=False,
                class_name=None,
                decorators=[],
                complexity=1,
            )
        ],
        imports=[ImportInfo(module="os", names=["os"], is_from=False, line=1)],
        module_docstring=None,
        total_lines=4,
        code_lines=2,
        comment_lines=0,
    )
    loader = _LoaderStub("import os\n\n\ndef foo():\n    return 1\n")
    parser = _ParserStub(parse_result)
    embedder = _EmbedderStub()
    indexer = CodeIndexer(
        {"indexing": {"levels": ["file", "function"], "generate_repo_overview": False}},
        cast(Any, loader),
        cast(Any, parser),
        cast(Any, embedder),
        None,
    )
    file_infos = [
        {
            "path": "/repo/mod.py",
            "relative_path": "mod.py",
            "size": 32,
            "extension": ".py",
        }
    ]

    with (
        patch.object(
            CodeElement,
            "to_dict",
            autospec=True,
            side_effect=AssertionError("CodeElement.to_dict() should not run"),
        ),
        patch.object(
            ImportInfo,
            "to_dict",
            autospec=True,
            side_effect=AssertionError("ImportInfo.to_dict() should not run"),
        ),
    ):
        elements = indexer.index_files(file_infos, repo_name="repo")

    assert len(embedder.prepared_batches) == 1
    prepared_batch = embedder.prepared_batches[0]
    assert len(prepared_batch) == 2
    assert all(payload["metadata"] == {} for payload in prepared_batch)

    file_element = next(element for element in elements if element.type == "file")
    function_element = next(
        element for element in elements if element.type == "function"
    )

    assert file_element.metadata["imports"] == [
        {
            "module": "os",
            "names": ["os"],
            "is_from": False,
            "line": 1,
            "level": 0,
        }
    ]
    assert file_element.metadata["embedding_text_hash"] == f"hash:{file_element.id}"
    assert function_element.metadata["embedding_artifact_ref"] == (
        f"artifact:{function_element.id}"
    )
    assert function_element.metadata["embedding_fingerprint"] == {
        "provider": "test",
        "model": "stub",
        "dimension": 2,
    }
    assert isinstance(file_element.metadata["embedding"], np.ndarray)


def test_repository_overview_metadata_includes_embedding_fingerprint() -> None:
    embedder = _EmbedderStub()
    vector_store = _VectorStoreStub()
    indexer = CodeIndexer(
        {"indexing": {"levels": ["file"], "generate_repo_overview": False}},
        cast(Any, _LoaderStub("")),
        cast(Any, _ParserStub(cast(Any, None))),
        cast(Any, embedder),
        cast(Any, vector_store),
    )
    indexer.current_repo_name = "repo"
    indexer.current_repo_url = "https://example.invalid/repo"

    indexer._save_repository_overview(
        {
            "repo_name": "repo",
            "summary": "summary",
            "structure_text": "structure",
            "readme_content": "",
            "file_structure": {},
            "has_readme": False,
        }
    )

    assert len(vector_store.saved_overviews) == 1
    assert embedder.prepared_text_batches == [["summary"]]
    assert vector_store.saved_overviews[0]["metadata"]["embedding_fingerprint"] == {
        "provider": "test",
        "model": "stub",
        "dimension": 2,
    }


def test_extract_elements_uses_precomputed_file_inventory() -> None:
    parse_result = FileParseResult(
        file_path="/repo/mod.py",
        language="python",
        classes=[],
        functions=[],
        imports=[],
        module_docstring=None,
        total_lines=1,
        code_lines=1,
        comment_lines=0,
    )
    loader = _LoaderStub("x = 1\n")
    loader.scan_files = lambda: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("extract_elements should use provided file_infos")
    )
    indexer = CodeIndexer(
        {"indexing": {"levels": ["file"], "generate_repo_overview": False}},
        cast(Any, loader),
        cast(Any, _ParserStub(parse_result)),
        cast(Any, _EmbedderStub()),
        None,
    )

    elements = indexer.extract_elements(
        repo_name="repo",
        file_infos=[
            {
                "path": "/repo/mod.py",
                "relative_path": "mod.py",
                "size": 6,
                "extension": ".py",
            }
        ],
    )

    assert [element.relative_path for element in elements] == ["mod.py"]
