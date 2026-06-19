"""
Repository Loader - Handle git cloning, local repository loading, and ZIP file extraction
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import logging
import os
import shutil
import zipfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from git import GitCommandError, Repo

from fastcode.app.indexing.file_inventory import (
    FileFingerprint,
    FileInventory,
    build_file_inventory,
)
from fastcode.utils import io as io
from fastcode.utils.archive import safe_extract_zip
from fastcode.utils.filesystem import ensure_dir, get_repo_name_from_url


class RepositoryLoader:
    """Load repositories from URLs or local paths"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.repo_config = config.get("repository", {})
        self.logger = logging.getLogger(__name__)

        self.clone_depth = self.repo_config.get("clone_depth", 1)
        self.max_file_size_mb = self.repo_config.get("max_file_size_mb", 5)
        self.ignore_patterns = self.repo_config.get("ignore_patterns", [])
        self.supported_extensions = self.repo_config.get("supported_extensions", [])
        configured_local_source_mode = str(
            self.repo_config.get("local_source_mode") or "in_place"
        )
        self.local_source_mode = (
            configured_local_source_mode
            if configured_local_source_mode in {"copy", "hardlink"}
            else "in_place"
        )
        self.safe_repo_root = os.path.abspath(self.config.get("repo_root", "./repos"))
        ensure_dir(self.safe_repo_root)
        configured_backup_root = self.repo_config.get("backup_directory")
        if configured_backup_root:
            self.repo_backup_root = os.path.abspath(configured_backup_root)
        else:
            self.repo_backup_root = os.path.join(
                os.path.dirname(self.safe_repo_root), "repo_backup"
            )
        ensure_dir(self.repo_backup_root)
        self.workspace_copy_cache_enabled = bool(
            self.repo_config.get("workspace_copy_cache_enabled", True)
        )
        configured_copy_cache_root = self.repo_config.get(
            "workspace_copy_cache_directory"
        )
        if configured_copy_cache_root:
            self.workspace_copy_cache_root = os.path.abspath(configured_copy_cache_root)
        else:
            self.workspace_copy_cache_root = os.path.join(
                self.repo_backup_root, "workspace_copy_cache"
            )
        if self.workspace_copy_cache_enabled:
            ensure_dir(self.workspace_copy_cache_root)

        self.temp_dir = None
        self.repo_path = None
        self.repo_name = None
        self.repo_source_path = None
        self.repo_load_mode = None
        self.repo_is_workspace_copy = False
        self._preloaded_file_inventory: FileInventory | None = None
        self._preloaded_file_inventory_repo_path: str | None = None
        self.last_load_stats: dict[str, Any] = {}

    def _backup_existing_repo(self, repo_path: str) -> str | None:
        """Move existing repository directory to backup workspace."""
        if not os.path.exists(repo_path):
            return None

        repo_name = os.path.basename(os.path.normpath(repo_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_name = f"{repo_name}_{timestamp}"
        backup_path = os.path.join(self.repo_backup_root, backup_name)

        suffix = 1
        while os.path.exists(backup_path):
            backup_path = os.path.join(self.repo_backup_root, f"{backup_name}_{suffix}")
            suffix += 1

        self.logger.warning(
            f"Repository path {repo_path} already exists, moving to backup: {backup_path}"
        )
        try:
            shutil.move(repo_path, backup_path)
            return backup_path
        except Exception as e:
            msg = f"Failed to backup existing repository {repo_path}: {e}"
            raise RuntimeError(msg)

    def _prepare_repo_path(self, repo_name: str, target_dir: str | None = None) -> str:
        """Prepare destination path under repository workspace."""
        base_dir = target_dir or self.safe_repo_root
        base_dir = os.path.abspath(base_dir)
        ensure_dir(base_dir)
        repo_path = os.path.join(base_dir, repo_name)

        if os.path.exists(repo_path):
            self._backup_existing_repo(repo_path)

        return repo_path

    def invalidate_preloaded_file_inventory(self) -> None:
        """Drop source-side inventory after a checkout or other workspace mutation."""
        self._preloaded_file_inventory = None
        self._preloaded_file_inventory_repo_path = None

    def _source_file_inventory(self, source_path: str) -> FileInventory:
        """Scan a local source checkout before any workspace copy is made."""
        original_repo_path = self.repo_path
        try:
            self.repo_path = source_path
            ignore_patterns = self._effective_ignore_patterns()
        finally:
            self.repo_path = original_repo_path
        return build_file_inventory(
            repo_root=source_path,
            supported_extensions=self.supported_extensions,
            ignore_patterns=ignore_patterns,
            max_file_size_mb=self.max_file_size_mb,
            include_fingerprints=True,
            logger=self.logger,
        )

    @staticmethod
    def _copy_cache_key(inventory: FileInventory) -> str:
        digest = sha256()
        for file in sorted(inventory.files, key=lambda item: item.relative_path):
            digest.update(file.relative_path.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
            digest.update(str(file.identity or "").encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(file.size).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _workspace_copy_cache_path(self, repo_name: str, cache_key: str) -> str:
        safe_repo_name = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in repo_name
        )
        return os.path.join(self.workspace_copy_cache_root, safe_repo_name, cache_key)

    @staticmethod
    def _inventory_for_repo_path(
        inventory: FileInventory, repo_path: str
    ) -> FileInventory:
        repo_path = os.path.abspath(repo_path)
        files = tuple(
            FileFingerprint(
                path=os.path.normpath(os.path.join(repo_path, file.relative_path)),
                relative_path=file.relative_path,
                size=file.size,
                mtime=file.mtime,
                extension=file.extension,
                language=file.language,
                package_root=file.package_root,
                supported_tool_eligible=file.supported_tool_eligible,
                content_hash=file.content_hash,
                git_blob_oid=file.git_blob_oid,
                fingerprint_source=file.fingerprint_source,
            )
            for file in inventory.files
        )
        return FileInventory(repo_root=os.path.normpath(repo_path), files=files)

    def _populate_workspace_copy_cache(
        self,
        *,
        source_path: str,
        cache_path: str,
        copy_function: Any,
    ) -> None:
        parent = os.path.dirname(cache_path)
        ensure_dir(parent)
        temp_path = f"{cache_path}.tmp.{os.getpid()}"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        try:
            shutil.copytree(
                source_path,
                temp_path,
                symlinks=True,
                copy_function=copy_function,
            )
            try:
                io.atomic_replace(temp_path, cache_path)
            except FileExistsError:
                shutil.rmtree(temp_path)
        except Exception:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            raise

    def load_from_url(self, url: str, target_dir: str | None = None) -> str:
        """
        Clone repository from URL

        Args:
            url: Git repository URL
            target_dir: Target directory for cloning (optional, creates temp dir if None)

        Returns:
            Path to cloned repository
        """
        self.logger.info(f"Cloning repository from {url}")

        # Get repository name
        self.repo_name = get_repo_name_from_url(url)
        self.repo_source_path = url
        self.repo_load_mode = "clone"
        self.repo_is_workspace_copy = True
        self.last_load_stats = {
            "mode": "clone",
            "source_path": url,
            "repo_path": None,
            "copied_bytes": 0,
            "copied_files": 0,
            "linked_bytes": 0,
            "linked_files": 0,
        }

        self.repo_path = self._prepare_repo_path(self.repo_name, target_dir)

        try:
            # Clone with shallow depth for faster cloning
            if self.clone_depth > 0:
                Repo.clone_from(
                    url, self.repo_path, depth=self.clone_depth, single_branch=True
                )
            else:
                Repo.clone_from(url, self.repo_path)

            self.logger.info(f"Successfully cloned to {self.repo_path}")
            self.last_load_stats["repo_path"] = self.repo_path
            return self.repo_path

        except GitCommandError as e:
            self.logger.error(f"Failed to clone repository: {e}")
            msg = f"Failed to clone repository: {e}"
            raise RuntimeError(msg)

    def load_from_path(self, path: str, target_dir: str | None = None) -> str:
        """
        Load a local repository path.

        Args:
            path: Local path to repository
            target_dir: Optional destination root. Supplying a target directory
                requests an isolated workspace copy.

        Returns:
            Path to the loaded repository root
        """
        if not os.path.exists(path):
            msg = f"Path does not exist: {path}"
            raise ValueError(msg)

        if not os.path.isdir(path):
            msg = f"Path is not a directory: {path}"
            raise ValueError(msg)

        source_path = os.path.abspath(path)
        self.repo_name = os.path.basename(source_path)
        self.repo_source_path = source_path
        destination_root = (
            os.path.abspath(target_dir) if target_dir else self.safe_repo_root
        )
        destination_path = os.path.join(destination_root, self.repo_name)

        # If source is already in workspace destination, use it directly.
        if os.path.abspath(source_path) == os.path.abspath(destination_path):
            self.repo_path = source_path
            self.repo_load_mode = "workspace"
            self.repo_is_workspace_copy = True
            self.last_load_stats = {
                "mode": "workspace",
                "source_path": source_path,
                "repo_path": self.repo_path,
                "copied_bytes": 0,
                "copied_files": 0,
                "linked_bytes": 0,
                "linked_files": 0,
            }
            self.logger.info(f"Loaded repository from workspace path {self.repo_path}")
            return self.repo_path

        if target_dir is None and self.local_source_mode == "in_place":
            self.repo_path = source_path
            self.repo_load_mode = "in_place"
            self.repo_is_workspace_copy = False
            self.last_load_stats = {
                "mode": "in_place",
                "source_path": source_path,
                "repo_path": self.repo_path,
                "copied_bytes": 0,
                "copied_files": 0,
                "linked_bytes": 0,
                "linked_files": 0,
            }
            self.logger.info("Loaded local repository in place: %s", self.repo_path)
            return self.repo_path

        self.repo_path = self._prepare_repo_path(self.repo_name, target_dir)
        source_inventory = self._source_file_inventory(source_path)
        source_inventory_metrics = source_inventory.metrics()
        copy_cache_key: str | None = None
        copy_cache_path: str | None = None
        copy_cache_hit = False
        if self.local_source_mode == "copy" and self.workspace_copy_cache_enabled:
            copy_cache_key = self._copy_cache_key(source_inventory)
            copy_cache_path = self._workspace_copy_cache_path(
                self.repo_name, copy_cache_key
            )
            copy_cache_hit = os.path.isdir(copy_cache_path)

        # Copy entire working tree, including untracked files.
        copied_bytes = 0
        copied_files = 0
        linked_bytes = 0
        linked_files = 0

        def _copy2_counting(src: str, dst: str) -> str:
            nonlocal copied_bytes, copied_files
            try:
                copied_bytes += os.stat(src).st_size
                copied_files += 1
            except OSError:
                pass
            return shutil.copy2(src, dst)

        def _hardlink_counting(src: str, dst: str) -> str:
            nonlocal copied_bytes, copied_files, linked_bytes, linked_files
            try:
                size = os.stat(src).st_size
                os.link(src, dst)
                linked_bytes += size
                linked_files += 1
                return dst
            except OSError:
                try:
                    copied_bytes += os.stat(src).st_size
                    copied_files += 1
                except OSError:
                    pass
                return shutil.copy2(src, dst)

        if self.local_source_mode == "copy" and copy_cache_path:
            if not copy_cache_hit:
                self._populate_workspace_copy_cache(
                    source_path=source_path,
                    cache_path=copy_cache_path,
                    copy_function=_copy2_counting,
                )
            shutil.copytree(
                copy_cache_path,
                self.repo_path,
                symlinks=True,
                copy_function=_hardlink_counting,
            )
        else:
            shutil.copytree(
                source_path,
                self.repo_path,
                symlinks=True,
                copy_function=(
                    _hardlink_counting
                    if self.local_source_mode == "hardlink"
                    else _copy2_counting
                ),
            )
        self.repo_load_mode = (
            "hardlink" if self.local_source_mode == "hardlink" else "copy"
        )
        self.repo_is_workspace_copy = True
        self._preloaded_file_inventory = source_inventory
        self._preloaded_file_inventory_repo_path = self.repo_path
        self.last_load_stats = {
            "mode": self.repo_load_mode,
            "source_path": source_path,
            "repo_path": self.repo_path,
            "copied_bytes": copied_bytes,
            "copied_files": copied_files,
            "linked_bytes": linked_bytes,
            "linked_files": linked_files,
            "copy_cache_enabled": bool(copy_cache_path),
            "copy_cache_hit": copy_cache_hit,
            "copy_cache_key": copy_cache_key,
            "copy_cache_path": copy_cache_path,
            "source_inventory_file_count": source_inventory_metrics["file_count"],
            "source_inventory_total_size_bytes": source_inventory_metrics[
                "total_size_bytes"
            ],
        }

        self.logger.info(
            "Loaded repository from %s to %s with %s mode "
            "(%d copied files, %d copied bytes, %d linked files, %d linked bytes)",
            source_path,
            self.repo_path,
            self.repo_load_mode,
            copied_files,
            copied_bytes,
            linked_files,
            linked_bytes,
        )
        return self.repo_path

    def load_from_zip(self, zip_path: str, target_dir: str | None = None) -> str:
        """
        Extract and load repository from ZIP file

        Args:
            zip_path: Path to ZIP file
            target_dir: Target directory for extraction (optional, creates temp dir if None)

        Returns:
            Path to extracted repository
        """
        if not os.path.exists(zip_path):
            msg = f"ZIP file does not exist: {zip_path}"
            raise ValueError(msg)

        if not zipfile.is_zipfile(zip_path):
            msg = f"File is not a valid ZIP archive: {zip_path}"
            raise ValueError(msg)

        self.logger.info(f"Extracting repository from ZIP: {zip_path}")

        # Get repository name from ZIP filename (without .zip extension)
        zip_basename = os.path.basename(zip_path)
        self.repo_name = os.path.splitext(zip_basename)[0]
        self.repo_source_path = os.path.abspath(zip_path)
        self.repo_load_mode = "zip"
        self.repo_is_workspace_copy = True
        self.last_load_stats = {
            "mode": "zip",
            "source_path": self.repo_source_path,
            "repo_path": None,
            "copied_bytes": 0,
            "copied_files": 0,
            "linked_bytes": 0,
            "linked_files": 0,
        }

        extract_path = self._prepare_repo_path(self.repo_name, target_dir)

        try:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safe_extract_zip(zip_ref, extract_path)

            # If ZIP contains a single root directory, flatten it so repository
            # always lives directly under repo workspace root/<repo_name>.
            contents = io.list_dir(extract_path)
            if len(contents) == 1 and os.path.isdir(
                os.path.join(extract_path, contents[0])
            ):
                self.logger.info(f"Detected single root directory: {contents[0]}")
                root_dir = os.path.join(extract_path, contents[0])
                for name in io.list_dir(root_dir):
                    shutil.move(
                        os.path.join(root_dir, name), os.path.join(extract_path, name)
                    )
                shutil.rmtree(root_dir)

            self.repo_path = extract_path
            self.last_load_stats["repo_path"] = self.repo_path

            self.logger.info(f"Successfully extracted to {self.repo_path}")

            # Get file count for logging
            file_count = sum(1 for _ in Path(self.repo_path).rglob("*") if _.is_file())
            self.logger.info(f"Extracted {file_count} files")

            return self.repo_path

        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid ZIP file: {e}")
            msg = f"Invalid ZIP file: {e}"
            raise RuntimeError(msg)
        except Exception as e:
            self.logger.error(f"Failed to extract ZIP file: {e}")
            msg = f"Failed to extract ZIP file: {e}"
            raise RuntimeError(msg)

    def _load_gitignore_patterns(self) -> list[str]:
        """
        Load .gitignore patterns from the repository root.

        Returns:
            List of gitignore patterns, empty if no .gitignore found
        """
        if not self.repo_path:
            return []

        gitignore_path = os.path.join(self.repo_path, ".gitignore")
        if not os.path.isfile(gitignore_path):
            return []

        patterns = []
        try:
            for raw_line in io.read_text(gitignore_path).splitlines():
                line = raw_line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    patterns.append(line)
            self.logger.info(f"Loaded {len(patterns)} patterns from .gitignore")
        except Exception as e:
            self.logger.warning(f"Failed to read .gitignore: {e}")

        return patterns

    def _effective_ignore_patterns(self) -> list[str]:
        gitignore_patterns = self._load_gitignore_patterns()
        if gitignore_patterns:
            return list(self.ignore_patterns) + gitignore_patterns
        return list(self.ignore_patterns)

    def scan_file_inventory(
        self, *, include_fingerprints: bool = False
    ) -> FileInventory:
        """Scan repository files into a typed planner inventory."""
        if not self.repo_path:
            msg = "No repository loaded"
            raise RuntimeError(msg)

        if (
            include_fingerprints
            and self._preloaded_file_inventory is not None
            and os.path.abspath(self.repo_path)
            == os.path.abspath(self._preloaded_file_inventory_repo_path or "")
        ):
            inventory = self._inventory_for_repo_path(
                self._preloaded_file_inventory, self.repo_path
            )
            self.logger.info(
                "Using source-side file inventory for %s (%d supported files)",
                self.repo_path,
                inventory.file_count,
            )
            return inventory

        self.logger.info("Scanning files in %s", self.repo_path)
        inventory = build_file_inventory(
            repo_root=self.repo_path,
            supported_extensions=self.supported_extensions,
            ignore_patterns=self._effective_ignore_patterns(),
            max_file_size_mb=self.max_file_size_mb,
            include_fingerprints=include_fingerprints,
            logger=self.logger,
        )
        self.logger.info(
            "Found %d supported files (%.2f MB total)",
            inventory.file_count,
            inventory.total_size_bytes / 1024 / 1024,
        )

        return inventory

    def scan_files(self, *, include_fingerprints: bool = False) -> list[dict[str, Any]]:
        """
        Scan repository and collect file metadata
        Returns:
            List of file metadata dictionaries
        """
        return self.scan_file_inventory(
            include_fingerprints=include_fingerprints
        ).to_file_info_list()

    def read_file_content(self, file_path: str) -> str | None:
        """
        Read file content with error handling

        Args:
            file_path: Path to file

        Returns:
            File content or None if error
        """
        try:
            return io.read_text(file_path)
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                return io.read_bytes(file_path).decode("latin-1")
            except Exception as e:
                self.logger.error(f"Failed to read {file_path}: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return None

    def get_repository_info(
        self, files: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Get repository metadata

        Returns:
            Dictionary with repository information
        """
        if not self.repo_path:
            msg = "No repository loaded"
            raise RuntimeError(msg)

        info: dict[str, Any] = {
            "name": self.repo_name,
            "path": self.repo_path,
        }
        if self.last_load_stats:
            info.update(
                {
                    "load_mode": self.last_load_stats.get("mode"),
                    "source_path": self.last_load_stats.get("source_path"),
                    "workspace_copy": self.repo_is_workspace_copy,
                    "copied_bytes": self.last_load_stats.get("copied_bytes", 0),
                    "copied_files": self.last_load_stats.get("copied_files", 0),
                    "linked_bytes": self.last_load_stats.get("linked_bytes", 0),
                    "linked_files": self.last_load_stats.get("linked_files", 0),
                    "copy_cache_enabled": self.last_load_stats.get(
                        "copy_cache_enabled", False
                    ),
                    "copy_cache_hit": self.last_load_stats.get("copy_cache_hit", False),
                    "copy_cache_key": self.last_load_stats.get("copy_cache_key"),
                    "copy_cache_path": self.last_load_stats.get("copy_cache_path"),
                    "source_inventory_file_count": self.last_load_stats.get(
                        "source_inventory_file_count", 0
                    ),
                    "source_inventory_total_size_bytes": self.last_load_stats.get(
                        "source_inventory_total_size_bytes", 0
                    ),
                }
            )

        # Try to get git info
        try:
            repo = Repo(self.repo_path)
            info.update(
                {
                    "branch": repo.active_branch.name,
                    "commit": repo.head.commit.hexsha[:8],
                    "remote_url": repo.remotes.origin.url if repo.remotes else None,
                }
            )
        except Exception:
            self.logger.debug("Not a git repository or git info unavailable")

        # Count files
        files = files if files is not None else self.scan_files()
        info.update(
            {
                "file_count": len(files),
                "total_size_mb": sum(f["size"] for f in files) / 1024 / 1024,
            }
        )

        return info

    def cleanup(self) -> None:
        """Clean up temporary directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            self.logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __del__(self) -> None:
        """Cleanup on deletion"""
        self.cleanup()
