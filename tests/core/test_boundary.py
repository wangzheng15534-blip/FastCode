"""Tests for I/O import guard and boundary conversion.

Part A -- test_core_modules_have_no_io_imports:
  Scans every .py in fastcode/core/ (except __init__.py) and asserts
  that none import from forbidden I/O modules.

Part B/C -- boundary conversion tests:
  Verify explicit field mapping and Rule 3 compliance (no **kwargs,
  no from_orm, no model_dump).
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

from fastcode.core.boundary import (
    CoreQueryInput,
    hit_to_response,
    query_request_to_core,
)
from fastcode.schema.core_types import Hit

CORE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "fastcode" / "core"

IO_MODULES = frozenset(
    {
        "psycopg",
        "sqlite3",
        "openai",
        "anthropic",
        "ollama",
        "requests",
        "urllib",
        "http",
        "subprocess",
        "pathlib",
        "dotenv",
        "torch",
        "sentence_transformers",
        "tiktoken",
        "pydantic",  # Rule 1: Pydantic Stops at the Door
    }
)

# Patterns that violate Rule 3: Explicit Translation
FORBIDDEN_PATTERNS = (
    "**kwargs",
    "**request",
    "**data",
    "**obj",
    "from_orm",
    "model_dump",
)


# ---------------------------------------------------------------------------
# Part A: I/O import guard
# ---------------------------------------------------------------------------


def _collect_imports(source: str) -> list[str]:
    """Return a list of top-level module names imported in *source*."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names


class TestCoreImportGuard:
    """Ensure fastcode/core/ never touches I/O libraries."""

    def test_core_modules_have_no_io_imports(self) -> None:
        py_files = sorted(
            p
            for p in CORE_DIR.iterdir()
            if p.suffix == ".py" and p.name != "__init__.py"
        )

        violations: list[str] = []
        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            imports = _collect_imports(source)
            for imp in imports:
                if imp in IO_MODULES:
                    violations.append(f"{py_file.name}: import '{imp}' is forbidden")

        assert violations == [], "I/O imports found in core/:\n" + "\n".join(
            f"  {v}" for v in violations
        )


# ---------------------------------------------------------------------------
# Part B/C: boundary conversion tests
# ---------------------------------------------------------------------------


def _make_hit(**overrides: Any) -> Hit:
    """Factory: create a Hit with sensible defaults."""
    defaults = {
        "element_id": "sym:pkg/Mod.foo",
        "element_type": "function",
        "element_name": "foo",
        "score": 0.92,
    }
    defaults.update(overrides)
    return Hit(**defaults)


class TestHitToResponse:
    """Verify Hit -> API dict conversion."""

    def test_hit_to_response_converts_to_dict(self) -> None:
        hit = _make_hit(source="semantic")
        result = hit_to_response(hit)

        assert result == {
            "id": "sym:pkg/Mod.foo",
            "type": "function",
            "name": "foo",
            "score": 0.92,
            "source": "semantic",
        }

    def test_hit_to_response_omits_extra_fields(self) -> None:
        """Only the five mapped fields should appear, not semantic_score etc."""
        hit = _make_hit(semantic_score=0.88, keyword_score=0.7)
        result = hit_to_response(hit)

        assert "semantic_score" not in result
        assert "keyword_score" not in result
        assert set(result.keys()) == {"id", "type", "name", "score", "source"}


class TestQueryRequestToCore:
    """Verify API dict -> CoreQueryInput conversion."""

    def test_query_request_to_core_extracts_fields(self) -> None:
        request = {
            "question": "How does auth work?",
            "repo_name": "myapp",
            "branch": "main",
            "snapshot_id": "snap:myapp:abc123",
            "session_id": "sess-1",
        }
        core_input = query_request_to_core(request)

        assert isinstance(core_input, CoreQueryInput)
        assert core_input.question == "How does auth work?"
        assert core_input.repo_name == "myapp"
        assert core_input.branch == "main"
        assert core_input.snapshot_id == "snap:myapp:abc123"
        assert core_input.session_id == "sess-1"

    def test_query_request_to_core_defaults_optional_fields(self) -> None:
        request = {"question": "minimal query"}
        core_input = query_request_to_core(request)

        assert core_input.question == "minimal query"
        assert core_input.repo_name is None
        assert core_input.branch is None
        assert core_input.snapshot_id is None
        assert core_input.session_id is None

    def test_query_request_to_core_raises_on_missing_question(self) -> None:
        with pytest.raises(KeyError):
            query_request_to_core({"repo_name": "x"})


class TestDbEffectsReturnDataclasses:
    """Rule 2: effects/db.py must return frozen dataclasses, never dict."""

    def test_db_effects_return_dataclasses_not_dicts(self) -> None:
        db_effects = (
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "fastcode"
            / "effects"
            / "db.py"
        )
        if not db_effects.exists():
            pytest.skip("effects/db.py not yet created")

        tree = ast.parse(db_effects.read_text(encoding="utf-8"))
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ret = node.returns
                if ret is None:
                    continue
                ret_str = ast.dump(ret)
                if "dict" in ret_str.lower() and "Any" in ret_str:
                    violations.append(
                        f"{node.name}: return type contains dict[str, Any]"
                    )
        assert violations == [], "Rule 2 violations in effects/db.py:\n" + "\n".join(
            violations
        )


class TestBoundaryExplicitTranslation:
    """Verify Rule 3: no forbidden patterns in boundary.py."""

    def test_boundary_uses_explicit_translation(self) -> None:
        source = pathlib.Path(
            __import__("fastcode.core.boundary", fromlist=[""]).__file__
        ).read_text(encoding="utf-8")

        # Build the set of line numbers that are comments or docstrings
        tree = ast.parse(source)
        non_code_lines: set[int] = set()

        # Mark docstring lines (first line of Expr(Constant) nodes)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                for lineno in range(node.lineno, node.end_lineno + 1):
                    non_code_lines.add(lineno)

        # Mark comment lines (lines starting with #)
        for i, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                non_code_lines.add(i)

        lines = source.splitlines()
        violations: list[str] = []
        for pattern in FORBIDDEN_PATTERNS:
            for i, line in enumerate(lines, start=1):
                if i in non_code_lines:
                    continue
                if pattern in line:
                    violations.append(f"  line {i}: {line.strip()}")

        assert violations == [], (
            "Forbidden patterns found in boundary.py (Rule 3 violation):\n"
            + "\n".join(violations)
        )
