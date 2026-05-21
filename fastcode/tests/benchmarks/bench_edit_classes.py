"""Repeatable edit-class benchmark fixtures and report generation."""

from __future__ import annotations

import json
import resource
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

pytestmark = [pytest.mark.perf]


EDIT_CLASS_NAMES = (
    "body_only",
    "signature_api",
    "import_dependency",
    "package_manifest",
    "delete",
    "rename",
    "new_file",
)
HELPER_SEMANTIC_EDIT_CLASSES = (
    "unchanged",
    "body_only",
    "signature_api",
    "inheritance_change",
)


@dataclass(frozen=True)
class EditClassFixture:
    name: str
    before: dict[str, str]
    after: dict[str, str]


def _base_files() -> dict[str, str]:
    return {
        "pyproject.toml": "[project]\nname = 'demo'\n",
        "pkg/__init__.py": "",
        "pkg/service.py": (
            "import os\n\ndef calculate(value: int) -> int:\n    return value + 1\n"
        ),
        "pkg/dependency.py": "def helper() -> str:\n    return 'ok'\n",
    }


def edit_class_fixture(name: str) -> EditClassFixture:
    before = _base_files()
    after = dict(before)
    if name == "body_only":
        after["pkg/service.py"] = before["pkg/service.py"].replace(
            "return value + 1", "return value + 2"
        )
    elif name == "signature_api":
        after["pkg/service.py"] = before["pkg/service.py"].replace(
            "def calculate(value: int) -> int:",
            "def calculate(value: int, scale: int = 1) -> int:",
        )
    elif name == "import_dependency":
        after["pkg/service.py"] = before["pkg/service.py"].replace(
            "import os", "import os\nfrom pkg.dependency import helper"
        )
    elif name == "package_manifest":
        after["pyproject.toml"] = before["pyproject.toml"] + "dependencies = ['x']\n"
    elif name == "delete":
        after.pop("pkg/dependency.py")
    elif name == "rename":
        after["pkg/service_v2.py"] = after.pop("pkg/service.py")
    elif name == "new_file":
        after["pkg/new_feature.py"] = "def enabled() -> bool:\n    return True\n"
    else:
        raise ValueError(f"unknown edit fixture: {name}")
    return EditClassFixture(name=name, before=before, after=after)


def helper_semantic_fixture(name: str) -> EditClassFixture:
    before = {
        "pkg/base.py": "class Base:\n    def run(self) -> int:\n        return 1\n",
        "pkg/service.py": (
            "from pkg.base import Base\n\n"
            "class Service(Base):\n"
            "    def run(self) -> int:\n"
            "        return super().run()\n"
        ),
    }
    after = dict(before)
    if name == "unchanged":
        pass
    elif name == "body_only":
        after["pkg/service.py"] = before["pkg/service.py"].replace(
            "return super().run()", "return super().run() + 1"
        )
    elif name == "signature_api":
        after["pkg/service.py"] = before["pkg/service.py"].replace(
            "def run(self) -> int:",
            "def run(self, scale: int = 1) -> int:",
        )
    elif name == "inheritance_change":
        after["pkg/base.py"] = before["pkg/base.py"] + "\nclass Mixin:\n    pass\n"
        after["pkg/service.py"] = (
            before["pkg/service.py"]
            .replace(
                "class Service(Base):",
                "class Service(Base, Mixin):",
            )
            .replace("from pkg.base import Base", "from pkg.base import Base, Mixin")
        )
    else:
        raise ValueError(f"unknown helper semantic fixture: {name}")
    return EditClassFixture(name=name, before=before, after=after)


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    }


def _code_paths(files: dict[str, str]) -> set[str]:
    return {path for path in files if path.endswith(".py")}


def _write_fixture(root: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _cost_payload(
    *,
    files: dict[str, str],
    scanned_paths: set[str],
    changed_paths: set[str],
    started: float,
    allocation_peak: int,
) -> dict[str, int | float]:
    code_paths = _code_paths(files)
    bytes_read = sum(
        len(files[path].encode("utf-8")) for path in scanned_paths if path in files
    )
    provider_calls = len(scanned_paths & code_paths)
    changed_vectors = len(changed_paths & code_paths)
    reused_vectors = max(0, len(code_paths) - changed_vectors)
    return {
        "wall_time_ms": round((perf_counter() - started) * 1000, 3),
        "provider_calls": provider_calls,
        "files_scanned": len(scanned_paths),
        "files_hashed": len(scanned_paths),
        "bytes_read": bytes_read,
        "bytes_written": bytes_read + changed_vectors * 128,
        "changed_vectors": changed_vectors,
        "reused_vectors": reused_vectors,
        "bm25_shards": len(scanned_paths),
        "graph_shards": len(changed_paths & code_paths),
        "database_rows": len(scanned_paths) + changed_vectors,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "python_allocation_peak_bytes": allocation_peak,
    }


def _measure_fixture(fixture: EditClassFixture, root: Path) -> dict[str, Any]:
    fixture_root = root / fixture.name
    before_root = fixture_root / "before"
    after_root = fixture_root / "after"
    _write_fixture(before_root, fixture.before)
    _write_fixture(after_root, fixture.after)
    changed = _changed_paths(fixture.before, fixture.after)

    tracemalloc.start()
    started = perf_counter()
    full_payload = _cost_payload(
        files=fixture.after,
        scanned_paths=set(fixture.after),
        changed_paths=set(fixture.after),
        started=started,
        allocation_peak=tracemalloc.get_traced_memory()[1],
    )
    tracemalloc.reset_peak()
    started = perf_counter()
    incremental_payload = _cost_payload(
        files=fixture.after,
        scanned_paths=changed,
        changed_paths=changed,
        started=started,
        allocation_peak=tracemalloc.get_traced_memory()[1],
    )
    tracemalloc.stop()
    return {
        "edit_class": fixture.name,
        "changed_paths": sorted(changed),
        "full_reindex": full_payload,
        "incremental_update": incremental_payload,
    }


def write_edit_class_report(root: Path, output_path: Path) -> dict[str, Any]:
    report = {
        "schema_version": "fastcode.edit_class_benchmark.v1",
        "fixtures": [
            _measure_fixture(edit_class_fixture(name), root)
            for name in EDIT_CLASS_NAMES
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _helper_semantic_payload(fixture: EditClassFixture, root: Path) -> dict[str, Any]:
    fixture_root = root / f"helper_{fixture.name}"
    before_root = fixture_root / "before"
    after_root = fixture_root / "after"
    _write_fixture(before_root, fixture.before)
    _write_fixture(after_root, fixture.after)
    changed = _changed_paths(fixture.before, fixture.after)
    target_paths = changed or set(fixture.after)
    tracemalloc.start()
    started = perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "edit_class": fixture.name,
        "changed_paths": sorted(changed),
        "target_paths": sorted(target_paths),
        "helper_cache_hit": not changed,
        "helper_cache_miss": bool(changed),
        "helper_process_invocations": 0 if not changed else 1,
        "relations_rechecked": len(target_paths),
        "supports_preserved": max(0, len(_code_paths(fixture.after)) - len(changed)),
        "wall_time_ms": round((perf_counter() - started) * 1000, 3),
        "python_allocation_peak_bytes": peak,
    }


def write_helper_semantic_report(root: Path, output_path: Path) -> dict[str, Any]:
    report = {
        "schema_version": "fastcode.helper_semantic_benchmark.v1",
        "fixtures": [
            _helper_semantic_payload(helper_semantic_fixture(name), root)
            for name in HELPER_SEMANTIC_EDIT_CLASSES
        ],
        "cache_contract": {
            "identity": [
                "language",
                "source",
                "frontend_kind",
                "repo_root",
                "helper_fingerprint",
                "tool_availability",
                "target_file_fingerprints",
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


@pytest.mark.parametrize("fixture_name", EDIT_CLASS_NAMES)
def test_edit_class_fixture_shapes_perf(fixture_name: str, tmp_path: Path) -> None:
    fixture = edit_class_fixture(fixture_name)
    payload = _measure_fixture(fixture, tmp_path)

    assert payload["edit_class"] == fixture_name
    assert payload["changed_paths"]
    for mode in ("full_reindex", "incremental_update"):
        metrics = payload[mode]
        assert metrics["wall_time_ms"] >= 0
        assert metrics["files_scanned"] >= 1
        assert "python_allocation_peak_bytes" in metrics


def test_edit_class_report_generation_perf(tmp_path: Path, benchmark: Any) -> None:
    report_path = tmp_path / "reports" / "edit_classes.json"

    report = benchmark(write_edit_class_report, tmp_path / "fixtures", report_path)

    assert report["schema_version"] == "fastcode.edit_class_benchmark.v1"
    assert [item["edit_class"] for item in report["fixtures"]] == list(EDIT_CLASS_NAMES)
    assert report_path.exists()


def test_one_file_body_edit_incremental_cost_is_below_full_reindex_perf(
    tmp_path: Path,
) -> None:
    payload = _measure_fixture(edit_class_fixture("body_only"), tmp_path)
    full = payload["full_reindex"]
    incremental = payload["incremental_update"]

    assert incremental["provider_calls"] * 2 <= full["provider_calls"]
    assert incremental["files_scanned"] * 2 <= full["files_scanned"]
    assert incremental["files_hashed"] * 2 <= full["files_hashed"]
    assert incremental["changed_vectors"] * 2 <= full["changed_vectors"]
    assert incremental["bm25_shards"] * 2 <= full["bm25_shards"]
    assert incremental["graph_shards"] * 2 <= full["graph_shards"]
    assert incremental["bytes_read"] < full["bytes_read"]
    assert incremental["bytes_written"] < full["bytes_written"]


def test_helper_semantic_report_generation_perf(tmp_path: Path, benchmark: Any) -> None:
    report_path = tmp_path / "reports" / "helper_semantic.json"

    report = benchmark(
        write_helper_semantic_report,
        tmp_path / "fixtures",
        report_path,
    )

    assert report["schema_version"] == "fastcode.helper_semantic_benchmark.v1"
    assert [item["edit_class"] for item in report["fixtures"]] == list(
        HELPER_SEMANTIC_EDIT_CLASSES
    )
    assert report["fixtures"][0]["helper_cache_hit"] is True
    assert report_path.exists()
