"""Safe archive extraction helpers."""

from __future__ import annotations

import os
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastcode.utils.atoms.byte_count import ByteCount
from fastcode.utils.atoms.non_empty_string import NonEmptyString
from fastcode.utils.atoms.positive_int import PositiveInt


class UnsafeArchiveError(ValueError):
    """Raised when an archive member violates extraction policy."""


@dataclass(frozen=True)
class ZipExtractionLimits:
    max_members: PositiveInt | int = field(default_factory=lambda: PositiveInt(20_000))
    max_total_uncompressed_bytes: ByteCount | int = field(
        default_factory=lambda: ByteCount.from_mib(512)
    )
    max_member_uncompressed_bytes: ByteCount | int = field(
        default_factory=lambda: ByteCount.from_mib(100)
    )
    max_compression_ratio: float = 1000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_members", PositiveInt.coerce(self.max_members))
        object.__setattr__(
            self,
            "max_total_uncompressed_bytes",
            ByteCount.coerce(self.max_total_uncompressed_bytes),
        )
        object.__setattr__(
            self,
            "max_member_uncompressed_bytes",
            ByteCount.coerce(self.max_member_uncompressed_bytes),
        )
        if self.max_compression_ratio <= 0:
            raise ValueError("max_compression_ratio must be > 0")


_SAFE_REPO_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_ZIP_EXTRACTION_LIMITS = ZipExtractionLimits()


def safe_repo_name_from_archive(filename: str) -> str:
    """Return a filesystem-safe repository name derived from an archive name."""
    stem = Path(filename).name.rsplit(".", 1)[0]
    for suffix in ("-main", "-master", "_main", "_master"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    normalized = _SAFE_REPO_NAME_RE.sub("_", stem).strip("._-")
    return NonEmptyString.parse(normalized or "uploaded_repo").as_str()


def _member_mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0xFFFF


def _is_special_file(info: zipfile.ZipInfo) -> bool:
    mode = _member_mode(info)
    if mode == 0:
        return False
    return bool(
        stat.S_ISLNK(mode)
        or stat.S_ISCHR(mode)
        or stat.S_ISBLK(mode)
        or stat.S_ISFIFO(mode)
        or stat.S_ISSOCK(mode)
    )


def _validate_member_path(member_name: str, destination: Path) -> Path:
    if not member_name or "\x00" in member_name:
        raise UnsafeArchiveError("ZIP member has an empty or invalid name")
    if member_name.startswith(("/", "\\")):
        raise UnsafeArchiveError(f"ZIP member uses an absolute path: {member_name}")

    posix_path = PurePosixPath(member_name)
    windows_path = PureWindowsPath(member_name)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise UnsafeArchiveError(f"ZIP member uses an absolute path: {member_name}")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise UnsafeArchiveError(f"ZIP member uses an unsafe path: {member_name}")
    if any(part in {"", ".", ".."} for part in windows_path.parts):
        raise UnsafeArchiveError(f"ZIP member uses an unsafe path: {member_name}")

    target = (destination / Path(*posix_path.parts)).resolve()
    destination_resolved = destination.resolve()
    if os.path.commonpath([str(destination_resolved), str(target)]) != str(
        destination_resolved
    ):
        raise UnsafeArchiveError(f"ZIP member escapes extraction root: {member_name}")
    return target


def validate_zip_members(
    zip_ref: zipfile.ZipFile,
    destination: Path,
    *,
    limits: ZipExtractionLimits = _DEFAULT_ZIP_EXTRACTION_LIMITS,
) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Validate all ZIP members before extraction and return destination paths."""
    infos = zip_ref.infolist()
    max_members = PositiveInt.coerce(limits.max_members).as_int()
    max_total_uncompressed_bytes = ByteCount.coerce(
        limits.max_total_uncompressed_bytes
    ).as_bytes()
    max_member_uncompressed_bytes = ByteCount.coerce(
        limits.max_member_uncompressed_bytes
    ).as_bytes()
    if len(infos) > max_members:
        raise UnsafeArchiveError(
            f"ZIP archive has too many members: {len(infos)} > {max_members}"
        )

    total_size = 0
    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    for info in infos:
        if _is_special_file(info):
            raise UnsafeArchiveError(
                f"ZIP member is a symlink or special file: {info.filename}"
            )
        target = _validate_member_path(info.filename.rstrip("/"), destination)
        if not info.is_dir():
            total_size += int(info.file_size)
            if info.file_size > max_member_uncompressed_bytes:
                raise UnsafeArchiveError(
                    f"ZIP member is too large after extraction: {info.filename}"
                )
            if total_size > max_total_uncompressed_bytes:
                raise UnsafeArchiveError("ZIP archive expands beyond configured limit")
            if info.compress_size > 0:
                ratio = float(info.file_size) / float(info.compress_size)
                if ratio > limits.max_compression_ratio:
                    raise UnsafeArchiveError(
                        f"ZIP member compression ratio is suspicious: {info.filename}"
                    )
        validated.append((info, target))
    return validated


def safe_extract_zip(
    zip_ref: zipfile.ZipFile,
    destination: str | Path,
    *,
    limits: ZipExtractionLimits = _DEFAULT_ZIP_EXTRACTION_LIMITS,
) -> None:
    """Safely extract a ZIP archive after validating every member."""
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    validated = validate_zip_members(zip_ref, destination_path, limits=limits)

    for info, target in validated:
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zip_ref.open(info, "r") as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink)
