from __future__ import annotations

import os
from pathlib import Path

import pytest
from git import Repo

from fastcode.app.indexing.file_inventory import FileFingerprint, FileInventory
from fastcode.app.indexing.loader import RepositoryLoader
from fastcode.app.indexing.pipeline.service import IndexPipeline


def test_scan_files_can_emit_fingerprints(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "a.py"
    source.write_text("print('a')\n", encoding="utf-8")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    loader.repo_path = str(repo)
    loader.repo_name = "repo"

    inventory = loader.scan_file_inventory(include_fingerprints=True)
    files = inventory.to_file_info_list()

    assert isinstance(inventory, FileInventory)
    assert len(files) == 1
    assert files[0]["relative_path"] == "a.py"
    assert files[0]["content_hash"]
    assert files[0]["blob_oid"] is None
    assert files[0]["fingerprint_source"] == "content_hash"
    assert files[0]["language"] == "python"
    assert files[0]["package_root"] == "."
    assert files[0]["supported_tool_eligible"] is True
    assert files[0]["mtime"] > 0
    assert inventory.metrics()["content_hash_count"] == 1
    assert inventory.metrics()["scanned_bytes"] == source.stat().st_size
    assert inventory.metrics()["hashed_bytes"] == source.stat().st_size


def test_scan_file_inventory_prefers_git_blob_oid_for_tracked_clean_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "a.py"
    source.write_text("print('a')\n", encoding="utf-8")
    git_repo = Repo.init(repo)
    git_repo.index.add(["a.py"])
    expected_blob_oid = git_repo.git.hash_object("a.py")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    loader.repo_path = str(repo)
    loader.repo_name = "repo"

    def _boom_hash(_path: str) -> str:
        raise AssertionError("tracked clean files should use git blob ids")

    monkeypatch.setattr(
        "fastcode.app.indexing.file_inventory.compute_file_hash", _boom_hash
    )

    files = loader.scan_files(include_fingerprints=True)

    assert len(files) == 1
    assert files[0]["relative_path"] == "a.py"
    assert files[0]["blob_oid"] == expected_blob_oid
    assert files[0]["git_blob_oid"] == expected_blob_oid
    assert files[0]["content_hash"] is None
    assert files[0]["content_identity"] == expected_blob_oid
    assert files[0]["fingerprint_source"] == "git_blob_oid"


def test_scan_file_inventory_hashes_untracked_git_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    Repo.init(repo)
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    loader.repo_path = str(repo)
    loader.repo_name = "repo"
    monkeypatch.setattr(
        "fastcode.app.indexing.file_inventory.compute_file_hash",
        lambda _path: "content-hash",
    )

    inventory = loader.scan_file_inventory(include_fingerprints=True)
    files = inventory.to_file_info_list()

    assert files[0]["content_hash"] == "content-hash"
    assert files[0]["blob_oid"] is None
    assert files[0]["content_identity"] == "content-hash"
    assert files[0]["fingerprint_source"] == "content_hash"
    assert inventory.metrics()["hashed_bytes"] == (repo / "a.py").stat().st_size


def test_pipeline_scan_files_exposes_typed_inventory_metrics() -> None:
    inventory = FileInventory(
        repo_root="/repo",
        files=(
            FileFingerprint(
                path="/repo/a.py",
                relative_path="a.py",
                size=12,
                mtime=1.0,
                extension=".py",
                language="python",
                package_root=".",
                supported_tool_eligible=True,
                git_blob_oid="blob-a",
                fingerprint_source="git_blob_oid",
            ),
        ),
    )

    class _Loader:
        def scan_file_inventory(self, *, include_fingerprints: bool) -> FileInventory:
            assert include_fingerprints is True
            return inventory

    pipeline = IndexPipeline.__new__(IndexPipeline)
    pipeline.loader = _Loader()
    pipeline._last_file_inventory_metrics = {}

    files = pipeline._scan_files_for_pipeline()

    assert files[0]["blob_oid"] == "blob-a"
    assert pipeline._file_inventory_metrics_payload(files) == {
        "file_count": 1,
        "total_size_bytes": 12,
        "scanned_bytes": 12,
        "hashed_bytes": 0,
        "git_blob_oid_count": 1,
        "content_hash_count": 0,
        "fingerprinted_file_count": 1,
        "supported_tool_eligible_count": 1,
    }


def test_load_from_path_uses_in_place_mode_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    loader = RepositoryLoader(
        {
            "repo_root": str(workspace),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    monkeypatch.setattr(
        "fastcode.app.indexing.loader.shutil.copytree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-place local loading must not copy the repository")
        ),
    )

    loaded_path = loader.load_from_path(str(repo))

    assert loaded_path == str(repo)
    assert loader.repo_path == str(repo)
    assert loader.repo_load_mode == "in_place"
    assert loader.repo_is_workspace_copy is False
    assert loader.last_load_stats["copied_bytes"] == 0
    assert loader.get_repository_info()["load_mode"] == "in_place"
    assert loader.get_repository_info()["workspace_copy"] is False
    assert not (workspace / "repo").exists()


def test_load_from_path_copy_mode_preserves_workspace_copy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {
                "supported_extensions": [".py"],
                "local_source_mode": "copy",
            },
        }
    )

    loaded_path = Path(loader.load_from_path(str(repo)))

    assert loaded_path != repo
    assert loaded_path == tmp_path / "workspace" / "repo"
    assert (loaded_path / "a.py").read_text(encoding="utf-8") == "print('a')\n"
    assert loader.repo_load_mode == "copy"
    assert loader.repo_is_workspace_copy is True
    assert loader.last_load_stats["copied_files"] == 1
    assert loader.last_load_stats["copied_bytes"] > 0
    info = loader.get_repository_info()
    assert info["load_mode"] == "copy"
    assert info["workspace_copy"] is True
    assert info["copied_files"] == 1


def test_load_from_path_copy_mode_reuses_content_addressed_cache(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "a.py"
    source.write_text("print('a')\n", encoding="utf-8")
    config = {
        "repo_root": str(tmp_path / "workspace"),
        "repository": {
            "supported_extensions": [".py"],
            "local_source_mode": "copy",
            "workspace_copy_cache_directory": str(tmp_path / "copy-cache"),
        },
    }

    first_loader = RepositoryLoader(config)
    first_loaded = Path(first_loader.load_from_path(str(repo)))
    cache_path = Path(first_loader.last_load_stats["copy_cache_path"])

    assert first_loaded == tmp_path / "workspace" / "repo"
    assert first_loader.last_load_stats["copy_cache_hit"] is False
    assert first_loader.last_load_stats["copied_files"] == 1
    assert first_loader.last_load_stats["linked_files"] == 1
    assert (cache_path / "a.py").read_text(encoding="utf-8") == "print('a')\n"

    second_loader = RepositoryLoader(config)
    second_loaded = Path(second_loader.load_from_path(str(repo)))

    assert second_loaded == first_loaded
    assert second_loader.last_load_stats["copy_cache_hit"] is True
    assert second_loader.last_load_stats["copied_files"] == 0
    assert second_loader.last_load_stats["linked_files"] == 1
    assert os.stat(cache_path / "a.py").st_ino == os.stat(second_loaded / "a.py").st_ino


def test_copy_mode_scan_uses_preloaded_source_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n", encoding="utf-8")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {
                "supported_extensions": [".py"],
                "local_source_mode": "copy",
            },
        }
    )
    loaded_path = Path(loader.load_from_path(str(repo)))

    monkeypatch.setattr(
        "fastcode.app.indexing.loader.build_file_inventory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("copy mode should reuse the source-side inventory")
        ),
    )

    inventory = loader.scan_file_inventory(include_fingerprints=True)

    assert inventory.repo_root == str(loaded_path)
    assert inventory.files[0].path == str(loaded_path / "a.py")
    assert inventory.files[0].relative_path == "a.py"
    assert inventory.files[0].content_hash
    assert loader.last_load_stats["source_inventory_file_count"] == 1


def test_load_from_path_hardlink_mode_preserves_workspace_copy_without_byte_copy(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "a.py"
    source.write_text("print('a')\n", encoding="utf-8")
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {
                "supported_extensions": [".py"],
                "local_source_mode": "hardlink",
            },
        }
    )

    loaded_path = Path(loader.load_from_path(str(repo)))
    linked = loaded_path / "a.py"

    assert loaded_path != repo
    assert linked.read_text(encoding="utf-8") == "print('a')\n"
    assert os.stat(source).st_ino == os.stat(linked).st_ino
    assert loader.repo_load_mode == "hardlink"
    assert loader.repo_is_workspace_copy is True
    assert loader.last_load_stats["linked_files"] == 1
    assert loader.last_load_stats["linked_bytes"] > 0
    assert loader.last_load_stats["copied_files"] == 0
    assert loader.last_load_stats["copied_bytes"] == 0
    info = loader.get_repository_info()
    assert info["load_mode"] == "hardlink"
    assert info["workspace_copy"] is True
    assert info["linked_files"] == 1
    assert info["copied_files"] == 0


def test_pipeline_refuses_checkout_on_in_place_local_repo() -> None:
    pipeline = IndexPipeline.__new__(IndexPipeline)
    pipeline.loader = type(
        "Loader",
        (),
        {"repo_path": "/tmp/source-repo", "repo_load_mode": "in_place"},
    )()

    with pytest.raises(RuntimeError, match="would mutate an in-place local repository"):
        pipeline._checkout_target_ref(ref="feature")


def test_get_repository_info_uses_supplied_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    loader = RepositoryLoader(
        {
            "repo_root": str(tmp_path / "workspace"),
            "repository": {"supported_extensions": [".py"]},
        }
    )
    loader.repo_path = str(repo)
    loader.repo_name = "repo"

    monkeypatch.setattr(
        loader,
        "scan_files",
        lambda: (_ for _ in ()).throw(AssertionError("must use supplied files")),
    )

    info = loader.get_repository_info(
        files=[{"relative_path": "a.py", "size": 1024 * 1024}]
    )

    assert info["file_count"] == 1
    assert info["total_size_mb"] == 1.0
