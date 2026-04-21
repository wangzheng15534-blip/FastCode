"""Tests for multi-language SCIP indexer runner."""

import pytest
from unittest.mock import patch, MagicMock

try:
    import google.protobuf  # noqa: F401
    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False

requires_protobuf = pytest.mark.skipif(not _HAS_PROTOBUF, reason="protobuf not installed")


def test_get_indexer_command_java():
    """Java indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("java", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-java"
    assert "--output" in cmd
    assert "/out/index.scip" in cmd


def test_get_indexer_command_go():
    """Go indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("go", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-go"
    assert "--output" in cmd
    assert "/out/index.scip" in cmd


def test_get_indexer_command_python():
    """Python indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("python", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-python"
    assert "index" in cmd
    assert "/out/index.scip" in cmd


def test_get_indexer_command_ruby():
    """Ruby indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("ruby", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-ruby"
    assert "/out/index.scip" in cmd


def test_get_indexer_command_unsupported():
    """Unsupported language returns None."""
    from fastcode.scip_indexers import get_indexer_command
    assert get_indexer_command("brainfuck", "/out.scip") is None


def test_supported_languages():
    """Check all expected languages are supported."""
    from fastcode.scip_indexers import SUPPORTED_LANGUAGES
    expected = {"java", "go", "python", "ruby", "typescript", "javascript",
                "cpp", "c", "csharp", "rust", "php", "kotlin", "scala", "dart"}
    assert expected.issubset(set(SUPPORTED_LANGUAGES.keys()))


def test_run_scip_indexer_success(tmp_path):
    """run_scip_indexer executes indexer and returns artifact path."""
    from fastcode.scip_indexers import run_scip_indexer
    output = tmp_path / "index.scip"
    with patch("fastcode.scip_indexers.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("fastcode.scip_indexers.shutil.which", return_value="/usr/bin/scip-python"):
            result = run_scip_indexer("python", str(tmp_path), str(output))
    assert result == str(output)


def test_run_scip_indexer_not_installed(tmp_path):
    """run_scip_indexer raises when indexer not installed."""
    from fastcode.scip_indexers import run_scip_indexer
    with patch("fastcode.scip_indexers.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="not found"):
            run_scip_indexer("python", str(tmp_path), str(tmp_path / "out.scip"))


def test_run_scip_indexer_failure(tmp_path):
    """run_scip_indexer raises when indexer exits non-zero."""
    from fastcode.scip_indexers import run_scip_indexer
    with patch("fastcode.scip_indexers.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
        with patch("fastcode.scip_indexers.shutil.which", return_value="/usr/bin/scip-python"):
            with pytest.raises(RuntimeError, match="error msg"):
                run_scip_indexer("python", str(tmp_path), str(tmp_path / "out.scip"))


def test_auto_detect_scip_languages(tmp_path):
    """detect_scip_languages identifies languages present in repo."""
    from fastcode.scip_indexers import detect_scip_languages

    # Create a repo with mixed language files
    (tmp_path / "Main.java").write_text("class Main {}")
    (tmp_path / "main.go").write_text("package main")
    (tmp_path / "app.py").write_text("print('hi')")
    (tmp_path / "README.md").write_text("# readme")

    languages = detect_scip_languages(str(tmp_path))
    assert "java" in languages
    assert "go" in languages
    assert "python" in languages
    assert "markdown" not in languages  # No SCIP indexer for markdown


def test_auto_detect_scip_languages_empty(tmp_path):
    """detect_scip_languages returns empty for no matching files."""
    from fastcode.scip_indexers import detect_scip_languages

    (tmp_path / "README.md").write_text("# readme")
    languages = detect_scip_languages(str(tmp_path))
    assert len(languages) == 0


def test_auto_detect_deduplicates():
    """detect_scip_languages deduplicates languages."""
    from fastcode.scip_indexers import detect_scip_languages
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        p = Path(td)
        (p / "a.java").write_text("class A {}")
        (p / "b.java").write_text("class B {}")
        languages = detect_scip_languages(td)
        assert languages.count("java") == 1


@requires_protobuf
def test_run_scip_for_language_success(tmp_path):
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
    with patch("fastcode.scip_indexers.run_scip_indexer", return_value=str(artifact_path)):
        result = run_scip_for_language("java", str(tmp_path), str(output_dir))

    assert result is not None
    assert len(result.documents) == 1
    assert result.documents[0].language == "java"


def test_run_scip_for_language_not_available(tmp_path):
    """run_scip_for_language returns None when indexer not installed."""
    from fastcode.scip_indexers import run_scip_for_language
    with patch("fastcode.scip_indexers.run_scip_indexer", side_effect=RuntimeError("not found")):
        result = run_scip_for_language("java", str(tmp_path), str(tmp_path))

    assert result is None
