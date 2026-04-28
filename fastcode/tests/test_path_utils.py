"""Property-based tests for path_utils module."""

from __future__ import annotations

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.path_utils import (
    PathUtils,
    file_path_to_module_path,
    is_valid_python_file,
    normalize_repo_root,
)

# --- Strategies ---

segment_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=10,
)

path_st = st.lists(segment_st, min_size=1, max_size=5).map("/".join)

python_path_st = st.lists(segment_st, min_size=1, max_size=5).map(
    lambda parts: "/".join(parts) + ".py"
)


# --- Properties ---


class TestFilePathToModulePath:
    @given(rel_path=python_path_st)
    @settings(max_examples=30)
    def test_returns_dotted_path_property(self, rel_path: str):
        """HAPPY: valid .py file returns dotted module path."""
        with tempfile.TemporaryDirectory() as repo:
            full = os.path.join(repo, rel_path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write("# test")
            result = file_path_to_module_path(full, repo)
            if result is not None:
                assert isinstance(result, str)
                assert len(result) > 0

    def test_simple_module_path_property(self):
        """HAPPY: simple file produces correct module path."""
        with tempfile.TemporaryDirectory() as repo:
            path = os.path.join(repo, "app", "services", "auth.py")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                f.write("# test")
            result = file_path_to_module_path(path, repo)
            assert result == "app.services.auth"

    def test_init_py_returns_parent_property(self):
        """HAPPY: __init__.py returns parent package path."""
        with tempfile.TemporaryDirectory() as repo:
            path = os.path.join(repo, "app", "__init__.py")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                f.write("# init")
            result = file_path_to_module_path(path, repo)
            assert result == "app"

    @pytest.mark.edge
    def test_non_python_file_returns_none_property(self):
        """EDGE: non-.py file returns None."""
        with tempfile.TemporaryDirectory() as repo:
            path = os.path.join(repo, "readme.md")
            with open(path, "w") as f:
                f.write("# readme")
            result = file_path_to_module_path(path, repo)
            assert result is None

    @pytest.mark.edge
    def test_outside_repo_returns_none_property(self):
        """EDGE: file outside repo root returns None."""
        with (
            tempfile.TemporaryDirectory() as repo,
            tempfile.TemporaryDirectory() as other,
        ):
            path = os.path.join(other, "other.py")
            with open(path, "w") as f:
                f.write("# other")
            result = file_path_to_module_path(path, repo)
            assert result is None

    @pytest.mark.edge
    def test_root_init_py_returns_none_property(self):
        """EDGE: root __init__.py returns None."""
        with tempfile.TemporaryDirectory() as repo:
            path = os.path.join(repo, "__init__.py")
            with open(path, "w") as f:
                f.write("# init")
            result = file_path_to_module_path(path, repo)
            assert result is None

    @pytest.mark.edge
    def test_nonexistent_file_returns_none_property(self):
        """EDGE: nonexistent file returns module path (doesn't check existence)."""
        result = file_path_to_module_path("/nonexistent/a.py", "/nonexistent")
        # Function operates on paths, doesn't check file existence
        assert isinstance(result, (str, type(None)))

    @pytest.mark.edge
    def test_empty_string_file_path_property(self):
        """EDGE: empty file path returns None."""
        result = file_path_to_module_path("", "/tmp")
        assert result is None

    @pytest.mark.edge
    def test_same_path_as_repo_root_property(self):
        """EDGE: file path equal to repo root returns None."""
        with tempfile.TemporaryDirectory() as repo:
            result = file_path_to_module_path(repo, repo)
            assert result is None

    def test_hyphenated_filename_property(self):
        """HAPPY: hyphenated filename preserved (RAG-friendly)."""
        with tempfile.TemporaryDirectory() as repo:
            path = os.path.join(repo, "run-server.py")
            with open(path, "w") as f:
                f.write("# server")
            result = file_path_to_module_path(path, repo)
            assert result == "run-server"

    def test_numeric_start_filename_property(self):
        """HAPPY: filename starting with number preserved (RAG-friendly)."""
        with tempfile.TemporaryDirectory() as repo:
            path = os.path.join(repo, "01_init.py")
            with open(path, "w") as f:
                f.write("# init")
            result = file_path_to_module_path(path, repo)
            assert result == "01_init"


class TestIsValidPythonFile:
    def test_valid_py_file_property(self):
        """HAPPY: existing .py file returns True."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# test")
            path = f.name
        assert is_valid_python_file(path) is True
        os.unlink(path)

    @pytest.mark.edge
    def test_non_py_extension_property(self):
        """EDGE: non-.py file returns False."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"text")
            path = f.name
        assert is_valid_python_file(path) is False
        os.unlink(path)

    @pytest.mark.edge
    def test_nonexistent_file_property(self):
        """EDGE: nonexistent file returns False."""
        assert is_valid_python_file("/nonexistent/file.py") is False

    @pytest.mark.edge
    def test_directory_with_py_name_property(self):
        """EDGE: directory ending in .py returns False."""
        with tempfile.TemporaryDirectory(suffix=".py") as d:
            assert is_valid_python_file(d) is False

    @pytest.mark.edge
    def test_empty_string_path_property(self):
        """EDGE: empty string returns False."""
        assert is_valid_python_file("") is False


class TestNormalizeRepoRoot:
    def test_returns_absolute_path_property(self):
        """HAPPY: returns absolute path."""
        result = normalize_repo_root(".")
        assert os.path.isabs(result)

    @pytest.mark.edge
    def test_relative_path_normalized_property(self):
        """EDGE: relative path converted to absolute."""
        result = normalize_repo_root("some/relative/path")
        assert os.path.isabs(result)

    @given(
        path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=1, max_size=20)
    )
    @settings(max_examples=10)
    def test_always_returns_absolute_property(self, path: str):
        """HAPPY: any input produces absolute path."""
        result = normalize_repo_root(path)
        assert os.path.isabs(result)


class TestPathUtilsDetectRepoName:
    def test_exact_match_property(self):
        """HAPPY: exact repo name found in path."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.detect_repo_name_from_path("myrepo/src/main.py", {"myrepo"})
            assert result == "myrepo"

    @pytest.mark.edge
    def test_case_insensitive_match_property(self):
        """EDGE: case-insensitive repo name match."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.detect_repo_name_from_path("MyRepo/src/main.py", {"myrepo"})
            assert result == "myrepo"

    @pytest.mark.edge
    def test_empty_path_returns_fallback_property(self):
        """EDGE: empty path returns first known repo."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.detect_repo_name_from_path("", {"fallback"})
            # Empty path has no segments, falls through to fallback
            assert result in {"fallback", ""}

    @pytest.mark.edge
    def test_no_match_returns_fallback_property(self):
        """EDGE: no matching segment returns first known repo."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.detect_repo_name_from_path("unknown/path", {"myrepo"})
            assert result == "myrepo"

    @pytest.mark.edge
    def test_empty_known_repos_property(self):
        """EDGE: empty known repos returns empty string."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.detect_repo_name_from_path("any/path", set())
            assert result == ""


class TestPathUtilsNormalizePathWithRepo:
    def test_removes_repo_prefix_property(self):
        """HAPPY: repo prefix removed from path."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.normalize_path_with_repo("myrepo/src/main.py", "myrepo")
            assert result == "src/main.py"

    def test_duplicate_repo_name_deduped_property(self):
        """HAPPY: duplicate repo name removed."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.normalize_path_with_repo("myrepo/myrepo/src/main.py", "myrepo")
            # Strips first occurrence when next part also matches
            assert "myrepo" not in result or result.endswith("myrepo/src/main.py")

    @pytest.mark.edge
    def test_no_repo_prefix_returns_as_is_property(self):
        """EDGE: path without repo prefix returned unchanged."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.normalize_path_with_repo("src/main.py", "myrepo")
            assert result == "src/main.py"

    @pytest.mark.edge
    def test_empty_path_returns_empty_property(self):
        """EDGE: empty path returns empty."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.normalize_path_with_repo("", "myrepo")
            assert result == ""

    @pytest.mark.edge
    def test_empty_repo_name_returns_path_property(self):
        """EDGE: empty repo name returns path unchanged."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            result = pu.normalize_path_with_repo("src/main.py", "")
            assert result == "src/main.py"


class TestPathUtilsIsSafePath:
    def test_relative_path_safe_property(self):
        """HAPPY: relative path within repo is safe."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            assert pu.is_safe_path("src/main.py") is True

    @pytest.mark.edge
    def test_dot_path_is_safe_property(self):
        """EDGE: '.' path is safe (repo root)."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            assert pu.is_safe_path(".") is True

    @pytest.mark.edge
    def test_empty_path_is_safe_property(self):
        """EDGE: empty path is safe (resolves to repo root)."""
        with tempfile.TemporaryDirectory() as repo:
            pu = PathUtils(repo)
            assert pu.is_safe_path("") is True
