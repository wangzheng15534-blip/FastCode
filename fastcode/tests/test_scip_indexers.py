"""Tests for scip_indexers module."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

try:
    _has_protobuf = importlib.util.find_spec("google.protobuf") is not None
except ModuleNotFoundError:
    _has_protobuf = False

from fastcode.scip_indexers import (
    SUPPORTED_LANGUAGES,
    detect_scip_languages,
    get_indexer_command,
)

requires_protobuf = pytest.mark.skipif(
    not _has_protobuf, reason="protobuf not installed"
)

# --- Strategies ---

language_st = st.sampled_from(list(SUPPORTED_LANGUAGES.keys()))
unsupported_st = st.sampled_from(["brainfuck", "cobol", "pascal", "assembly"])

# --- Tests ---


def test_get_indexer_command_java():
    """Java indexer command is correct."""
    cmd = get_indexer_command("java", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-java"
    assert "--output" in cmd
    assert "/out/index.scip" in cmd


def test_get_indexer_command_go():
    """Go indexer command is correct."""
    cmd = get_indexer_command("go", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-go"
    assert "--output" in cmd
    assert "/out/index.scip" in cmd


def test_get_indexer_command_python():
    """Python indexer command is correct."""
    cmd = get_indexer_command("python", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-python"
    assert "index" in cmd
    assert "/out/index.scip" in cmd


def test_get_indexer_command_ruby():
    """Ruby indexer command is correct."""
    cmd = get_indexer_command("ruby", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-ruby"
    assert "/out/index.scip" in cmd


def test_get_indexer_command_unsupported():
    """Unsupported language returns None."""
    assert get_indexer_command("brainfuck", "/out.scip") is None


def test_get_indexer_command_new_language_frontends():
    """Zig, Fortran, and Julia expose required semantic frontend commands."""
    zig = get_indexer_command("zig", "/out/zig.scip")
    fortran = get_indexer_command("fortran", "/out/fortran.scip")
    julia = get_indexer_command("julia", "/out/julia.scip")

    assert zig is not None
    assert fortran is not None
    assert julia is not None
    assert zig[0] == "zls"
    assert fortran[0] == "fortls"
    assert julia[0] == "julia"


def test_supported_languages():
    """Check all expected languages are supported."""
    expected = {
        "java",
        "go",
        "python",
        "ruby",
        "typescript",
        "javascript",
        "cpp",
        "c",
        "csharp",
        "rust",
        "kotlin",
        "scala",
        "zig",
        "fortran",
        "julia",
    }
    assert expected.issubset(set(SUPPORTED_LANGUAGES.keys()))


def test_run_scip_indexer_success(tmp_path: pathlib.Path):
    """run_scip_indexer executes indexer and returns artifact path."""
    from fastcode.scip_indexers import run_scip_indexer

    output = tmp_path / "index.scip"
    with patch("fastcode.scip_indexers.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch(
            "fastcode.scip_indexers.shutil.which", return_value="/usr/bin/scip-python"
        ):
            result = run_scip_indexer("python", str(tmp_path), str(output))
    assert result == str(output)


def test_run_scip_indexer_not_installed(tmp_path: pathlib.Path):
    """run_scip_indexer raises when indexer not installed."""
    from fastcode.scip_indexers import run_scip_indexer

    with (
        patch("fastcode.scip_indexers.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="not found"),
    ):
        run_scip_indexer("python", str(tmp_path), str(tmp_path / "out.scip"))


def test_run_scip_indexer_failure(tmp_path: pathlib.Path):
    """run_scip_indexer raises when indexer exits non-zero."""
    from fastcode.scip_indexers import run_scip_indexer

    with patch("fastcode.scip_indexers.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
        with (
            patch(
                "fastcode.scip_indexers.shutil.which",
                return_value="/usr/bin/scip-python",
            ),
            pytest.raises(RuntimeError, match="error msg"),
        ):
            run_scip_indexer("python", str(tmp_path), str(tmp_path / "out.scip"))


def test_auto_detect_scip_languages(tmp_path: pathlib.Path):
    """detect_scip_languages identifies languages present in repo."""
    (tmp_path / "Main.java").write_text("class Main {}")
    (tmp_path / "main.go").write_text("package main")
    (tmp_path / "app.py").write_text("print('hi')")
    (tmp_path / "README.md").write_text("# readme")

    languages = detect_scip_languages(str(tmp_path))
    assert "java" in languages
    assert "go" in languages
    assert "python" in languages
    assert "markdown" not in languages  # No SCIP indexer for markdown


def test_auto_detect_scip_languages_empty(tmp_path: pathlib.Path):
    """detect_scip_languages returns empty for no matching files."""
    (tmp_path / "README.md").write_text("# readme")
    languages = detect_scip_languages(str(tmp_path))
    assert len(languages) == 0


def test_auto_detect_deduplicates():
    """detect_scip_languages deduplicates languages."""
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path

        p = Path(td)
        (p / "a.java").write_text("class A {}")
        (p / "b.java").write_text("class B {}")
        languages = detect_scip_languages(td)
        assert languages.count("java") == 1


@requires_protobuf
def test_run_scip_for_language_success(tmp_path: pathlib.Path):
    """run_scip_for_language orchestrates indexing and loading."""
    from fastcode.scip_indexers import run_scip_for_language
    from fastcode.scip_pb2 import Index

    # Create a fake repo with a Java file
    (tmp_path / "Main.java").write_text("class Main {}")
    output_dir = tmp_path / "scip_output"
    output_dir.mkdir()
    # Build a fake .scip artifact
    idx = Index()
    idx.metadata.version = 0
    idx.metadata.tool_info.name = "scip-java"
    doc = idx.documents.add()
    doc.relative_path = "Main.java"
    doc.language = "java"
    artifact_path = output_dir / "java.scip"
    artifact_path.write_bytes(idx.SerializeToString())
    with patch(
        "fastcode.scip_indexers.run_scip_indexer", return_value=str(artifact_path)
    ):
        result = run_scip_for_language("java", str(tmp_path), str(output_dir))

    assert result is not None
    assert len(result.documents) == 1
    assert result.documents[0].language == "java"


def test_run_scip_for_language_not_available(tmp_path: pathlib.Path):
    """run_scip_for_language returns None when indexer not installed."""
    from fastcode.scip_indexers import run_scip_for_language

    with patch(
        "fastcode.scip_indexers.run_scip_indexer", side_effect=RuntimeError("not found")
    ):
        result = run_scip_for_language("java", str(tmp_path), str(tmp_path))

    assert result is None


# --- Property-based tests ---


class TestGetIndexerCommand:
    @given(language=language_st)
    @settings(max_examples=15)
    def test_supported_language_returns_command_property(self, language: str):
        """HAPPY: supported language returns a command list."""
        cmd = get_indexer_command(language, "/tmp/out.scip")
        assert cmd is not None
        assert isinstance(cmd, list)
        assert len(cmd) >= 3  # binary + extra_args + output_path

    @given(language=unsupported_st)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_unsupported_language_returns_none_property(self, language: str):
        """EDGE: unsupported language returns None (line 62)."""
        cmd = get_indexer_command(language, "/tmp/out.scip")
        assert cmd is None

    def test_python_uses_scip_python_property(self):
        """HAPPY: python language uses scip-python binary."""
        cmd = get_indexer_command("python", "/tmp/out.scip")
        assert cmd[0] == "scip-python"

    def test_output_path_included_property(self):
        """HAPPY: output path is last argument."""
        path = "/custom/output.scip"
        cmd = get_indexer_command("go", path)
        assert cmd[-1] == path


class TestDetectScipLanguages:
    def test_python_files_detected_property(self):
        """HAPPY: .py files detected as python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "main.py"), "w") as f:
                f.write("print('hello')")
            langs = detect_scip_languages(tmpdir)
            assert "python" in langs

    def test_mixed_languages_property(self):
        """HAPPY: multiple file types detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.py", "b.go", "c.java", "d.rs"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("// placeholder")
            langs = detect_scip_languages(tmpdir)
            assert "python" in langs
            assert "go" in langs
            assert "java" in langs
            assert "rust" in langs

    @pytest.mark.edge
    def test_empty_dir_no_languages_property(self):
        """EDGE: empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            langs = detect_scip_languages(tmpdir)
            assert langs == []

    @pytest.mark.edge
    def test_skip_dirs_excluded_property(self):
        """EDGE: .git, node_modules, __pycache__ are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skip_dir = os.path.join(tmpdir, ".git")
            os.makedirs(skip_dir)
            with open(os.path.join(skip_dir, "hook.py"), "w") as f:
                f.write("# git hook")
            langs = detect_scip_languages(tmpdir)
            assert "python" not in langs

    @pytest.mark.edge
    def test_unknown_extensions_ignored_property(self):
        """EDGE: unknown file extensions produce no languages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["data.csv", "config.yaml", "image.png"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("content")
            langs = detect_scip_languages(tmpdir)
            assert langs == []

    def test_results_sorted_property(self):
        """HAPPY: detected languages are sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["z.rs", "a.py", "m.go"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("// x")
            langs = detect_scip_languages(tmpdir)
            assert langs == sorted(langs)

    def test_typescript_tsx_detected_property(self):
        """HAPPY: .tsx files detected as typescript."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "app.tsx"), "w") as f:
                f.write("// react")
            langs = detect_scip_languages(tmpdir)
            assert "typescript" in langs

    @pytest.mark.edge
    def test_none_output_path_property(self):
        """EDGE: None output path in get_indexer_command."""
        result = get_indexer_command("python", None)
        # Should either return None or handle gracefully
        if result is not None:
            assert isinstance(result, list)

    @pytest.mark.edge
    def test_empty_output_path_property(self):
        """EDGE: empty string output path."""
        result = get_indexer_command("python", "")
        if result is not None:
            assert isinstance(result, list)

    @pytest.mark.edge
    def test_supported_languages_is_dict_property(self):
        """EDGE: SUPPORTED_LANGUAGES is a non-empty dict."""
        assert isinstance(SUPPORTED_LANGUAGES, dict)
        assert len(SUPPORTED_LANGUAGES) > 0

    @pytest.mark.edge
    def test_detect_nonexistent_dir_property(self):
        """EDGE: nonexistent directory returns empty list."""
        langs = detect_scip_languages("/nonexistent/path/xyz")
        assert langs == []

    @pytest.mark.edge
    def test_node_modules_skipped_property(self):
        """EDGE: node_modules directory is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = os.path.join(tmpdir, "node_modules")
            os.makedirs(nm)
            with open(os.path.join(nm, "lib.ts"), "w") as f:
                f.write("// lib")
            langs = detect_scip_languages(tmpdir)
            assert "typescript" not in langs

    def test_nested_dirs_scanned_property(self):
        """HAPPY: nested directories are scanned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "src", "lib")
            os.makedirs(nested)
            with open(os.path.join(nested, "utils.py"), "w") as f:
                f.write("# utils")
            langs = detect_scip_languages(tmpdir)
            assert "python" in langs

    def test_new_language_extensions_detected_property(self):
        """HAPPY: Zig, Fortran, and Julia source files are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["main.zig", "solver.f90", "model.jl"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("\n")
            langs = detect_scip_languages(tmpdir)
            assert "zig" in langs
            assert "fortran" in langs
            assert "julia" in langs
