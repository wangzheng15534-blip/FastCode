# tests/test_effects_fs.py
"""Tests for FS effects — verify thin file I/O wrappers."""
import os
import tempfile

from fastcode.effects.fs import read_file, write_file, file_exists


class TestReadWriteFile:
    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            write_file(path, "hello world")
            content = read_file(path)
            assert content == "hello world"
        finally:
            os.unlink(path)

    def test_file_exists(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            assert file_exists(path)
        finally:
            os.unlink(path)

    def test_file_not_exists(self):
        assert not file_exists("/nonexistent/path/file.txt")
