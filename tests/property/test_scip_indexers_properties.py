"""Property-based tests for scip_indexers module."""

from __future__ import annotations

import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.scip_indexers import (
    SUPPORTED_LANGUAGES,
    detect_scip_languages,
    get_indexer_command,
)

# --- Strategies ---

language_st = st.sampled_from(list(SUPPORTED_LANGUAGES.keys()))
unsupported_st = st.sampled_from(
    ["brainfuck", "cobol", "fortran", "pascal", "assembly"]
)


# --- Properties ---


@pytest.mark.property
class TestGetIndexerCommand:
    @given(language=language_st)
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_supported_language_returns_command(self, language):
        """HAPPY: supported language returns a command list."""
        cmd = get_indexer_command(language, "/tmp/out.scip")
        assert cmd is not None
        assert isinstance(cmd, list)
        assert len(cmd) >= 3  # binary + extra_args + output_path

    @given(language=unsupported_st)
    @settings(max_examples=10)
    @pytest.mark.edge
    def test_unsupported_language_returns_none(self, language):
        """EDGE: unsupported language returns None (line 62)."""
        cmd = get_indexer_command(language, "/tmp/out.scip")
        assert cmd is None

    @pytest.mark.happy
    def test_python_uses_scip_python(self):
        """HAPPY: python language uses scip-python binary."""
        cmd = get_indexer_command("python", "/tmp/out.scip")
        assert cmd[0] == "scip-python"

    @pytest.mark.happy
    def test_output_path_included(self):
        """HAPPY: output path is last argument."""
        path = "/custom/output.scip"
        cmd = get_indexer_command("go", path)
        assert cmd[-1] == path


@pytest.mark.property
class TestDetectScipLanguages:
    @pytest.mark.happy
    def test_python_files_detected(self):
        """HAPPY: .py files detected as python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "main.py"), "w") as f:
                f.write("print('hello')")
            langs = detect_scip_languages(tmpdir)
            assert "python" in langs

    @pytest.mark.happy
    def test_mixed_languages(self):
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
    def test_empty_dir_no_languages(self):
        """EDGE: empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            langs = detect_scip_languages(tmpdir)
            assert langs == []

    @pytest.mark.edge
    def test_skip_dirs_excluded(self):
        """EDGE: .git, node_modules, __pycache__ are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skip_dir = os.path.join(tmpdir, ".git")
            os.makedirs(skip_dir)
            with open(os.path.join(skip_dir, "hook.py"), "w") as f:
                f.write("# git hook")
            langs = detect_scip_languages(tmpdir)
            assert "python" not in langs

    @pytest.mark.edge
    def test_unknown_extensions_ignored(self):
        """EDGE: unknown file extensions produce no languages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["data.csv", "config.yaml", "image.png"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("content")
            langs = detect_scip_languages(tmpdir)
            assert langs == []

    @pytest.mark.happy
    def test_results_sorted(self):
        """HAPPY: detected languages are sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["z.rs", "a.py", "m.go"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("// x")
            langs = detect_scip_languages(tmpdir)
            assert langs == sorted(langs)

    @pytest.mark.happy
    def test_typescript_tsx_detected(self):
        """HAPPY: .tsx files detected as typescript."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "app.tsx"), "w") as f:
                f.write("// react")
            langs = detect_scip_languages(tmpdir)
            assert "typescript" in langs

    @pytest.mark.edge
    def test_none_output_path(self):
        """EDGE: None output path in get_indexer_command."""
        result = get_indexer_command("python", None)
        # Should either return None or handle gracefully
        if result is not None:
            assert isinstance(result, list)

    @pytest.mark.edge
    def test_empty_output_path(self):
        """EDGE: empty string output path."""
        result = get_indexer_command("python", "")
        if result is not None:
            assert isinstance(result, list)

    @pytest.mark.edge
    def test_supported_languages_is_dict(self):
        """EDGE: SUPPORTED_LANGUAGES is a non-empty dict."""
        assert isinstance(SUPPORTED_LANGUAGES, dict)
        assert len(SUPPORTED_LANGUAGES) > 0

    @pytest.mark.edge
    def test_detect_nonexistent_dir(self):
        """EDGE: nonexistent directory returns empty list."""
        langs = detect_scip_languages("/nonexistent/path/xyz")
        assert langs == []

    @pytest.mark.edge
    def test_node_modules_skipped(self):
        """EDGE: node_modules directory is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = os.path.join(tmpdir, "node_modules")
            os.makedirs(nm)
            with open(os.path.join(nm, "lib.ts"), "w") as f:
                f.write("// lib")
            langs = detect_scip_languages(tmpdir)
            assert "typescript" not in langs

    @pytest.mark.happy
    def test_nested_dirs_scanned(self):
        """HAPPY: nested directories are scanned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "src", "lib")
            os.makedirs(nested)
            with open(os.path.join(nested, "utils.py"), "w") as f:
                f.write("# utils")
            langs = detect_scip_languages(tmpdir)
            assert "python" in langs
