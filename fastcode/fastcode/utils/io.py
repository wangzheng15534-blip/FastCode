"""Filesystem I/O operational kit (run_kit).

Direct-call generic operational helpers for the file operations that use_flow
and meaning layers need but must not perform directly (raw ``open``/``os.*``
are forbidden outside run_kit/effect units). These are meaningful, composable
operations — atomic writes, tolerant removals, directory scans — not mechanical
re-wrappers of a single builtin.

Registered as its own ``run_kit`` unit (NOT the axisless ``utils`` role), so it
may perform I/O. Axisless callers depend on this via the normal
``use_flow/meaning_* -> run_kit`` edge.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, IO


def read_text(path: str | Path, *, errors: str = "strict") -> str:
    """Read a UTF-8 text file and return its contents."""
    with open(path, encoding="utf-8", errors=errors) as handle:
        return handle.read()


def read_bytes(path: str | Path) -> bytes:
    """Read and return the raw bytes of a file."""
    with open(path, "rb") as handle:
        return handle.read()


def write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text to ``path`` (non-atomic truncating write)."""
    with open(path, "w", encoding=encoding) as handle:
        handle.write(text)


def write_bytes(path: str | Path, data: bytes) -> None:
    """Write raw bytes to ``path`` (non-atomic truncating write)."""
    with open(path, "wb") as handle:
        handle.write(data)


def atomic_write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8"
) -> None:
    """Write ``text`` to ``path`` atomically via a temp file + ``os.replace``.

    The destination is either fully replaced or left untouched; readers never
    observe a partial write.
    """
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent or ".", prefix=".tmp_", suffix=".write")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        _remove_quietly(tmp)
        raise


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a temp file + ``os.replace``."""
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent or ".", prefix=".tmp_", suffix=".write")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        _remove_quietly(tmp)
        raise


@contextmanager
def atomic_open_write(
    path: str | Path, *, mode: str = "w", encoding: str = "utf-8"
) -> Iterator[IO[Any]]:
    """Open a temp file for writing; atomically move it onto ``path`` on success.

    Yields a writable file handle for streaming writers (``np.save``,
    ``pickle.dump``, ``json.dump`` with custom options) that cannot produce all
    bytes up front. The temp file is created beside the destination and
    ``os.replace``d into place on clean exit; on any exception the temp file is
    discarded and the destination is left untouched.
    """
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    binary = "b" in mode
    fd, tmp = tempfile.mkstemp(dir=parent or ".", prefix=".tmp_", suffix=".write")
    try:
        if binary:
            handle: IO[Any] = os.fdopen(fd, mode)
        else:
            handle = os.fdopen(fd, mode, encoding=encoding)
        try:
            yield handle
        finally:
            handle.close()
        os.replace(tmp, path)
    except BaseException:
        _remove_quietly(tmp)
        raise


def atomic_replace(src: str | Path, dst: str | Path) -> None:
    """Atomically move ``src`` onto ``dst`` (``os.replace``)."""
    os.replace(src, dst)


def read_json(path: str | Path) -> Any:
    """Read and JSON-decode a UTF-8 text file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(
    path: str | Path, value: Any, *, indent: int | None = 2
) -> None:
    """Serialize ``value`` to JSON and write it to ``path`` atomically."""
    atomic_write_text(path, json.dumps(value, indent=indent, ensure_ascii=False))


def remove_file(path: str | Path) -> bool:
    """Remove a file; return True if removed, False if it did not exist."""
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def list_dir(path: str | Path) -> list[str]:
    """Return entry names in ``path`` (not full paths, not recursive)."""
    return os.listdir(path)


def iter_dir(path: str | Path) -> Iterator[str]:
    """Yield entry names in ``path`` lazily."""
    yield from os.listdir(path)


def clear_dir(path: str | Path) -> int:
    """Remove all entries inside ``path`` (non-recursive on dirs); return count.

    Only files are removed; subdirectories are left in place. Use for shard/
    cache directories whose contents are flat files.
    """
    removed = 0
    for name in os.listdir(path):
        full = os.path.join(os.fspath(path), name)
        if os.path.isfile(full):
            os.remove(full)
            removed += 1
    return removed


def ensure_dir(path: str | Path) -> str:
    """Create ``path`` (and parents) if missing; return the absolute path."""
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def exists(path: str | Path) -> bool:
    """Return True if ``path`` exists."""
    return os.path.exists(path)


def _remove_quietly(path: str | Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass
