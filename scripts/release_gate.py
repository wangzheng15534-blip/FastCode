#!/usr/bin/env python3
"""Build and smoke-test release artifacts from built distributions.

The gate intentionally avoids editable installs:
- build wheel and sdist artifacts for the workspace
- install built artifacts into fresh virtualenvs with ``pip``
- smoke minimal installed CLI paths without optional heavy extras
- smoke service extras separately so dependency boundaries are explicit
- smoke a tiny installed-wheel index/query flow
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.server
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTCODE_PROJECT_DIR = PROJECT_ROOT / "fastcode"
HELPER_ASSET_SUFFIXES = [
    "fastcode/semantic/resolvers/ts_semantic_helper.js",
    "fastcode/semantic/resolvers/go_semantic_helper.go",
    "fastcode/semantic/resolvers/java_semantic_helper.py",
    "fastcode/semantic/resolvers/rust_semantic_helper.py",
    "fastcode/semantic/resolvers/csharp_semantic_helper.py",
    "fastcode/semantic/resolvers/zig_semantic_helper.py",
    "fastcode/semantic/resolvers/fortran_semantic_helper.py",
    "fastcode/semantic/resolvers/julia_semantic_helper.py",
]
CORE_ENTRYPOINTS = ["fastcode"]
SERVICE_ENTRYPOINTS = ["fastcode-api", "fastcode-mcp", "fastcode-web"]
SERVICE_EXTRAS = ("api", "mcp", "postgres", "redis")
HEAVY_EXTRAS = ("docs", "local-embeddings", "nanobot", "scip")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603


def _run_output(cmd: list[str], *, cwd: Path | None = None) -> str:
    print("+", " ".join(cmd))
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout + result.stderr


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_script(venv_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _build_artifacts(out_dir: Path) -> list[Path]:
    _run(
        [
            "uv",
            "build",
            "--all-packages",
            "--clear",
            "--out-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
    )
    artifacts = sorted(out_dir.iterdir())
    if not artifacts:
        raise RuntimeError("uv build produced no artifacts")
    return artifacts


def _archive_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    if path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(path, "r:gz") as tf:
            return tf.getnames()
    raise ValueError(f"Unsupported artifact type: {path}")


def _assert_helper_assets_present(artifacts: list[Path]) -> None:
    fastcode_artifacts = [
        path for path in artifacts if path.name.startswith("fastcode-")
    ]
    if not fastcode_artifacts:
        raise RuntimeError("fastcode build artifacts missing")
    for artifact in fastcode_artifacts:
        names = _archive_members(artifact)
        for suffix in HELPER_ASSET_SUFFIXES:
            if not any(name.endswith(suffix) for name in names):
                raise RuntimeError(
                    f"{artifact.name} is missing required helper asset {suffix}"
                )


def _write_release_config(
    config_path: Path,
    *,
    repo_root: Path,
    vector_dir: Path,
    cache_dir: Path,
    ollama_url: str,
    openai_url: str,
) -> None:
    payload = {
        "repo_root": str(repo_root),
        "storage": {"backend": "sqlite"},
        "repository": {
            "backup_directory": str(repo_root / "repo_backup"),
            "exclude_site_packages": False,
            "ignore_patterns": ["*.pyc", "__pycache__", "node_modules", ".git"],
            "supported_extensions": [".py"],
        },
        "embedding": {
            "provider": "ollama",
            "model": "release-gate",
            "ollama_url": ollama_url,
            "device": "cpu",
            "batch_size": 4,
        },
        "generation": {
            "provider": "openai",
            "model": "release-gate",
            "base_url": openai_url,
            "openai_api_key": "release-gate",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "query": {
            "use_llm_enhancement": False,
        },
        "cache": {
            "enabled": True,
            "cache_directory": str(cache_dir),
            "cache_queries": False,
            "cache_embeddings": True,
        },
        "vector_store": {
            "persist_directory": str(vector_dir),
            "distance_metric": "cosine",
        },
        "evaluation": {
            "enabled": False,
            "in_memory_index": False,
            "disable_cache": False,
            "disable_persistence": False,
            "force_reindex": False,
        },
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_smoke_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'release_smoke'\nversion = '0.0.1'\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "a.py").write_text(
        "def answer():\n    return 'release smoke'\n",
        encoding="utf-8",
    )
    return repo


class _FakeAIHandler(http.server.BaseHTTPRequestHandler):
    server_version = "ReleaseGateAI/1.0"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if self.path.endswith("/chat/completions"):
            response = self._chat_response(payload)
        else:
            response = self._embedding_response(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _embedding_response(self, payload: dict[str, object]) -> bytes:
        prompt = str(payload.get("prompt") or payload.get("input") or "")
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        embedding = [
            round(((digest[index % len(digest)] / 255.0) * 2.0) - 1.0, 6)
            for index in range(8)
        ]
        return json.dumps({"embedding": embedding}).encode("utf-8")

    def _chat_response(self, payload: dict[str, object]) -> bytes:
        prompt = ""
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                prompt = str(last_message.get("content") or "")
        if "pkg/a.py" in prompt or "answer()" in prompt:
            answer = "pkg/a.py defines answer()."
        else:
            answer = "release smoke answer."
        return json.dumps(
            {
                "id": "chatcmpl-release-gate",
                "object": "chat.completion",
                "created": 0,
                "model": str(payload.get("model") or "release-gate"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        ).encode("utf-8")

    def log_message(self, *_args: object) -> None:
        return None


@contextlib.contextmanager
def _fake_ai_server() -> Iterator[tuple[str, str]]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()
    server = http.server.ThreadingHTTPServer((host, port), _FakeAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}/api/embeddings", f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _artifact_requirement(path: Path, fastcode_extras: tuple[str, ...]) -> str:
    if path.name.startswith("fastcode-") and fastcode_extras:
        return f"{path}[{','.join(fastcode_extras)}]"
    return str(path)


def _install_artifacts(
    venv_python: Path,
    artifacts: list[Path],
    *,
    fastcode_extras: tuple[str, ...] = (),
) -> None:
    requirements = [_artifact_requirement(path, fastcode_extras) for path in artifacts]
    _run([str(venv_python), "-m", "pip", "install", *requirements])


def _smoke_core_imports(venv_python: Path) -> None:
    _run(
        [
            str(venv_python),
            "-c",
            (
                "import fastcode; "
                "import fastcode.client.cli; "
                "import fastcode.app.indexing.embedder; "
                "print(fastcode.__name__)"
            ),
        ]
    )


def _smoke_service_imports(venv_python: Path) -> None:
    _run(
        [
            str(venv_python),
            "-c",
            (
                "import fastcode; "
                "import fastcode.api.routes; "
                "import fastcode.api.web; "
                "import fastcode.mcp.server; "
                "print(fastcode.__name__)"
            ),
        ]
    )


def _smoke_entrypoints(venv_dir: Path, entrypoints: list[str]) -> None:
    for entrypoint in entrypoints:
        script = _venv_script(venv_dir, entrypoint)
        _run([str(script), "--help"])


def _smoke_index_and_query(
    venv_dir: Path,
    *,
    config_path: Path,
    repo_path: Path,
) -> None:
    fastcode = _venv_script(venv_dir, "fastcode")
    _run(
        [
            str(fastcode),
            "index",
            "--repo-path",
            str(repo_path),
            "--config",
            str(config_path),
        ]
    )
    output = _run_output(
        [
            str(fastcode),
            "query",
            "--repo-path",
            str(repo_path),
            "--query",
            "What does pkg/a.py define?",
            "--config",
            str(config_path),
        ]
    )
    if "pkg/a.py defines answer()." not in output:
        raise RuntimeError("installed wheel query smoke did not return fake AI answer")
    if "Error: LLM provider not configured" in output:
        raise RuntimeError("installed wheel query smoke used an unconfigured LLM")


def _smoke_distribution(
    dist_dir: Path,
    *,
    artifact_pattern: str,
    fastcode_extras: tuple[str, ...] = (),
    smoke_query: bool = False,
    smoke_services: bool = False,
) -> None:
    with (
        tempfile.TemporaryDirectory(prefix="fastcode-release-env-") as env_root_str,
        tempfile.TemporaryDirectory(prefix="fastcode-release-work-") as work_root_str,
    ):
        env_root = Path(env_root_str)
        work_root = Path(work_root_str)
        venv_dir = env_root / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)])
        venv_python = _venv_python(venv_dir)

        install_nanobot_artifact = "nanobot" in fastcode_extras
        artifacts = sorted(
            path
            for path in dist_dir.glob(artifact_pattern)
            if path.name.startswith("fastcode-")
            or (install_nanobot_artifact and path.name.startswith("nanobot_ai-"))
        )
        if not artifacts:
            raise RuntimeError(f"No matching artifacts found in {dist_dir}")
        _install_artifacts(
            venv_python,
            artifacts,
            fastcode_extras=fastcode_extras,
        )
        _smoke_core_imports(venv_python)
        _smoke_entrypoints(venv_dir, CORE_ENTRYPOINTS)
        if smoke_services:
            _smoke_service_imports(venv_python)
            _smoke_entrypoints(venv_dir, SERVICE_ENTRYPOINTS)

        if smoke_query:
            repo_root = work_root / "repo_root"
            vector_dir = work_root / "vector_store"
            cache_dir = work_root / "cache"
            repo_root.mkdir(parents=True, exist_ok=True)
            vector_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            repo_path = _create_smoke_repo(work_root)
            with _fake_ai_server() as (ollama_url, openai_url):
                config_path = work_root / "release_config.json"
                _write_release_config(
                    config_path,
                    repo_root=repo_root,
                    vector_dir=vector_dir,
                    cache_dir=cache_dir,
                    ollama_url=ollama_url,
                    openai_url=openai_url,
                )
                _smoke_index_and_query(
                    venv_dir,
                    config_path=config_path,
                    repo_path=repo_path,
                )


def _run_release_smokes(build_dir: Path, *, include_heavy_extras: bool) -> None:
    service_extras = SERVICE_EXTRAS + (HEAVY_EXTRAS if include_heavy_extras else ())
    _smoke_distribution(
        build_dir,
        artifact_pattern="*.tar.gz",
    )
    _smoke_distribution(
        build_dir,
        artifact_pattern="*.whl",
        smoke_query=True,
    )
    _smoke_distribution(
        build_dir,
        artifact_pattern="*.whl",
        fastcode_extras=service_extras,
        smoke_services=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Leave the build directory on disk for inspection.",
    )
    parser.add_argument(
        "--include-heavy-extras",
        action="store_true",
        help=(
            "Also install docs, local embedding, nanobot, and SCIP extras. "
            "This can pull large ML/CUDA dependencies."
        ),
    )
    args = parser.parse_args(argv)

    if args.keep_artifacts:
        build_root = PROJECT_ROOT / "dist" / "_release_gate"
        build_root.mkdir(parents=True, exist_ok=True)
        artifacts = _build_artifacts(build_root)
        build_dir = build_root
    else:
        with tempfile.TemporaryDirectory(
            prefix="fastcode-release-build-"
        ) as build_root:
            build_dir = Path(build_root)
            artifacts = _build_artifacts(build_dir)
            _assert_helper_assets_present(artifacts)
            _run_release_smokes(
                build_dir,
                include_heavy_extras=args.include_heavy_extras,
            )
            return 0

    _assert_helper_assets_present(artifacts)
    _run_release_smokes(build_dir, include_heavy_extras=args.include_heavy_extras)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
