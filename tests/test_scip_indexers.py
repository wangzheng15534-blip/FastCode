"""Tests for multi-language SCIP indexer runner."""

import pytest
from unittest.mock import patch, MagicMock


def test_get_indexer_command_java():
    """Java indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("java", "/repo", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-java"
    assert "--output" in cmd


def test_get_indexer_command_go():
    """Go indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("go", "/repo", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-go"
    assert "--output" in cmd


def test_get_indexer_command_python():
    """Python indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("python", "/repo", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-python"
    assert "index" in cmd


def test_get_indexer_command_ruby():
    """Ruby indexer command is correct."""
    from fastcode.scip_indexers import get_indexer_command
    cmd = get_indexer_command("ruby", "/repo", "/out/index.scip")
    assert cmd is not None
    assert cmd[0] == "scip-ruby"


def test_get_indexer_command_unsupported():
    """Unsupported language returns None."""
    from fastcode.scip_indexers import get_indexer_command
    assert get_indexer_command("brainfuck", "/repo", "/out.scip") is None


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
