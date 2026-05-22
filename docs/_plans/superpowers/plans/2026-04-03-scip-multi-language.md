# SCIP Multi-Language Indexer Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-written tree-sitter parsers with standard SCIP indexers for multi-language support — download the SCIP protobuf schema, generate Python bindings, and integrate language-specific indexers into the pipeline.

**Architecture:** FastCode already has SCIP integration (`scip_loader.py` → `scip_to_ir.py` → `ir_merge.py`). We extend it with: (1) protobuf-based binary .scip parsing (eliminating the external CLI dependency), (2) a general `run_scip_indexer()` that runs the appropriate indexer per language, and (3) pipeline integration that auto-detects languages and runs indexers. Tree-sitter parsers remain as fallback when SCIP indexers aren't installed.

**Tech Stack:** protobuf (already installed v6.33.6), scip.proto (from sourcegraph/scip), subprocess to run external indexers, existing `scip_models.py` + `scip_to_ir.py` adapters.

---

## Files Changed

### New Files
```
fastcode/scip_pb2.py              # Auto-generated from scip.proto (DO NOT EDIT)
fastcode/scip_indexers.py         # run_scip_indexer(language, repo_path, output_path)
tests/test_scip_indexers.py       # Tests for indexer detection and execution
tests/test_scip_binary.py         # Tests for binary .scip protobuf parsing
```

### Modified Files
```
fastcode/scip_loader.py           # Add _load_binary_scip() using protobuf
fastcode/scip_models.py           # Add from_protobuf() class methods
pyproject.toml                    # Add grpcio-tools to dev deps (for protoc generation)
fastcode/main.py                  # Auto-detect languages and run SCIP indexers
```

---

## Task 1: Download scip.proto and Generate Python Bindings

**Files:**
- Create: `fastcode/scip_pb2.py` (auto-generated)
- Modify: `pyproject.toml` (add grpcio-tools to dev deps)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scip_binary.py
"""Tests for binary SCIP protobuf parsing."""

import pytest


def test_scip_pb2_module_importable():
    """Protobuf bindings module must be importable."""
    from fastcode.scip_pb2 import Index
    idx = Index()
    assert idx.metadata.tool_info.name == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scip_binary.py::test_scip_pb2_module_importable -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fastcode.scip_pb2'`

- [ ] **Step 3: Download scip.proto and generate bindings**

```bash
# Download the protobuf schema
curl -sSL https://raw.githubusercontent.com/sourcegraph/scip/main/scip.proto -o /tmp/scip.proto

# Generate Python bindings
pip install grpcio-tools
python -m grpc_tools.protoc \
  --proto_path=/tmp \
  --python_out=fastcode \
  /tmp/scip.proto

# Fix relative import in generated file (grpc_tools generates absolute import)
sed -i 's/^import scip_pb2/from fastcode import scip_pb2/' fastcode/scip_pb2.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scip_binary.py::test_scip_pb2_module_importable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastcode/scip_pb2.py
git commit -m "feat: add SCIP protobuf bindings generated from scip.proto"
```

---

## Task 2: Add Binary .scip Parsing to scip_loader.py

**Files:**
- Modify: `fastcode/scip_loader.py:19-51` (extend `load_scip_artifact`)
- Test: `tests/test_scip_binary.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scip_binary.py — append to existing file

def test_load_binary_scip_artifact(tmp_path):
    """Binary .scip files parse without external CLI."""
    from fastcode.scip_pb2 import Index, Document, Occurrence, SymbolInformation, Metadata, ToolInfo

    # Build a minimal binary SCIP index
    idx = Index()
    idx.metadata.version = 0
    idx.metadata.tool_info.name = "test-indexer"
    idx.metadata.tool_info.version = "1.0.0"
    doc = idx.documents.add()
    doc.relative_path = "src/main.py"
    doc.language = "python"
    sym = doc.symbols.add()
    sym.symbol = "scip test src/main.py main()`"
    sym.display_name = "main"

    # Write binary
    scip_path = tmp_path / "index.scip"
    scip_path.write_bytes(idx.SerializeToString())

    from fastcode.scip_loader import load_scip_artifact
    result = load_scip_artifact(str(scip_path))

    assert len(result.documents) == 1
    assert result.documents[0].path == "src/main.py"
    assert result.documents[0].language == "python"
    assert len(result.documents[0].symbols) == 1
    assert result.documents[0].symbols[0].name == "main"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scip_binary.py::test_load_binary_scip_artifact -v`
Expected: FAIL — `ValueError: '.scip' provided but 'scip' CLI is not available in PATH`

- [ ] **Step 3: Implement binary parsing in scip_loader.py**

Replace the `.scip` block in `load_scip_artifact()` (lines 32-47) with protobuf-first parsing, falling back to CLI only if protobuf fails:

```python
# In load_scip_artifact(), replace the ext == ".scip" block:
if ext == ".scip":
    # Try protobuf parsing first (no external CLI needed)
    try:
        from .scip_pb2 import Index

        with open(path, "rb") as f:
            raw = f.read()
        pb_index = Index()
        pb_index.ParseFromString(raw)
        return _protobuf_to_scip_index(pb_index)
    except Exception as exc:
        logger.debug("Protobuf parsing failed, trying scip CLI: %s", exc)
        # Fallback to CLI
        scip_cli = shutil.which("scip")
        if not scip_cli:
            raise ValueError(
                f".scip artifact could not be parsed (protobuf error: {exc}) "
                "and 'scip' CLI is not available in PATH"
            ) from exc
        candidate_cmds = [
            [scip_cli, "print", "--json", path],
            [scip_cli, "dump", "--json", path],
        ]
        last_error = None
        for cmd in candidate_cmds:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout.strip():
                return SCIPIndex.from_dict(json.loads(proc.stdout))
            last_error = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"Failed to decode .scip artifact via scip CLI: {last_error}")
```

Add a helper function `_protobuf_to_scip_index()` that converts the protobuf `Index` message to `SCIPIndex`:

```python
def _protobuf_to_scip_index(pb_index) -> SCIPIndex:
    """Convert a protobuf Index message to SCIPIndex."""
    documents = []
    for doc in pb_index.documents:
        symbols = []
        for sym in doc.symbols:
            r = list(sym.documentation) if sym.documentation else []
            symbols.append(
                SCIPSymbol(
                    symbol=sym.symbol,
                    name=sym.display_name or None,
                    kind=_scip_kind_to_str(sym.kind),
                    documentation=r,
                )
            )
        occurrences = []
        for occ in doc.occurrences:
            r = list(occ.range) if occ.range else [None, None, None, None]
            roles = occ.symbol_roles
            role = "definition" if roles & 1 else "reference"
            occurrences.append(
                SCIPOccurrence(
                    symbol=occ.symbol,
                    role=role,
                    range=r,
                )
            )
        lang = doc.language if doc.language else None
        documents.append(
            SCIPDocument(
                path=doc.relative_path,
                language=lang,
                symbols=symbols,
                occurrences=occurrences,
            )
        )
    return SCIPIndex(
        documents=documents,
        indexer_name=pb_index.metadata.tool_info.name or None,
        indexer_version=pb_index.metadata.tool_info.version or None,
    )


def _scip_kind_to_str(kind_value: int) -> str:
    """Convert SCIP protobuf Kind enum to string."""
    from .scip_pb2 import SymbolInformation

    kind_map = {
        SymbolInformation.Kind.FUNCTION: "function",
        SymbolInformation.Kind.METHOD: "method",
        SymbolInformation.Kind.CLASS: "class",
        SymbolInformation.Kind.INTERFACE: "interface",
        SymbolInformation.Kind.ENUM: "enum",
        SymbolInformation.Kind.ENUM_MEMBER: "enum_member",
        SymbolInformation.Kind.VARIABLE: "variable",
        SymbolInformation.Kind.CONSTANT: "constant",
        SymbolInformation.Kind.PROPERTY: "property",
        SymbolInformation.Kind.TYPE: "type",
        SymbolInformation.Kind.MACRO: "macro",
        SymbolInformation.Kind.MODULE: "module",
        SymbolInformation.Kind.NAMESPACE: "namespace",
        SymbolInformation.Kind.PACKAGE: "package",
        SymbolInformation.Kind.PARAMETER: "parameter",
        SymbolInformation.Kind.TYPE_PARAMETER: "type_parameter",
        SymbolInformation.Kind.CONSTRUCTOR: "constructor",
        SymbolInformation.Kind.STRUCT: "struct",
    }
    return kind_map.get(kind_value, "symbol")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scip_binary.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add fastcode/scip_loader.py tests/test_scip_binary.py
git commit -m "feat: parse binary .scip files via protobuf without external CLI"
```

---

## Task 3: Add Multi-Language SCIP Indexer Runner

**Files:**
- Create: `fastcode/scip_indexers.py`
- Create: `tests/test_scip_indexers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scip_indexers.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scip_indexers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fastcode.scip_indexers'`

- [ ] **Step 3: Implement scip_indexers.py**

```python
"""
Multi-language SCIP indexer runner.

Runs the appropriate SCIP indexer (scip-java, scip-go, scip-python, etc.)
based on the target language. Each indexer produces a binary .scip artifact.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Map language name → (binary_name, extra_args)
SUPPORTED_LANGUAGES: Dict[str, Tuple[str, List[str]]] = {
    "java": ("scip-java", ["index", "--output"]),
    "kotlin": ("scip-java", ["index", "--output"]),
    "scala": ("scip-java", ["index", "--output"]),
    "go": ("scip-go", ["index", "--output"]),
    "python": ("scip-python", ["index", "--output"]),
    "ruby": ("scip-ruby", ["index", "--output"]),
    "typescript": ("scip-typescript", ["index", "--output"]),
    "javascript": ("scip-typescript", ["index", "--output"]),
    "c": ("scip-clang", ["index", "--output"]),
    "cpp": ("scip-clang", ["index", "--output"]),
    "csharp": ("scip-dotnet", ["index", "--output"]),
    "rust": ("rust-analyzer", ["scip", "--output"]),
    "php": ("scip-php", ["index", "--output"]),
    "dart": ("scip-dart", ["index", "--output"]),
}


def get_indexer_command(
    language: str,
    repo_path: str,
    output_path: str,
) -> Optional[List[str]]:
    """Build the indexer command for a language. Returns None if unsupported."""
    entry = SUPPORTED_LANGUAGES.get(language)
    if not entry:
        return None
    binary_name, extra_args = entry
    return [binary_name] + extra_args + [output_path]


def run_scip_indexer(
    language: str,
    repo_path: str,
    output_path: str,
) -> str:
    """
    Run the SCIP indexer for the given language.

    Returns the output artifact path on success.
    Raises RuntimeError if indexer is not installed or fails.
    """
    cmd = get_indexer_command(language, repo_path, output_path)
    if cmd is None:
        raise RuntimeError(f"No SCIP indexer available for language: {language}")

    binary_name = cmd[0]
    binary_path = shutil.which(binary_name)
    if not binary_path:
        raise RuntimeError(
            f"SCIP indexer '{binary_name}' not found in PATH. "
            f"Install it to enable {language} support via SCIP."
        )

    cmd[0] = binary_path
    logger.info("Running SCIP indexer: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary_name} failed ({proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scip_indexers.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 5: Commit**

```bash
git add fastcode/scip_indexers.py tests/test_scip_indexers.py
git commit -m "feat: add multi-language SCIP indexer runner"
```

---

## Task 4: Add Binary Parsing Tests for Protobuf Conversion

**Files:**
- Modify: `tests/test_scip_binary.py`
- Modify: `fastcode/scip_models.py` (add `documentation` field to SCIPSymbol if needed)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scip_binary.py — append

def test_binary_scip_with_occurrences(tmp_path):
    """Binary SCIP with occurrences converts correctly."""
    from fastcode.scip_pb2 import (
        Index, Document, Occurrence, SymbolInformation,
        Metadata, ToolInfo,
    )

    idx = Index()
    idx.metadata.version = 0
    idx.metadata.tool_info.name = "scip-java"
    idx.metadata.tool_info.version = "2.1.0"

    doc = idx.documents.add()
    doc.relative_path = "src/Main.java"
    doc.language = "java"

    sym = doc.symbols.add()
    sym.symbol = "scip java com/example Main main(java.lang.String[])`"
    sym.display_name = "main"
    sym.kind = SymbolInformation.Kind.METHOD

    # Add a definition occurrence (symbol_roles bit 0 = definition)
    occ = doc.occurrences.add()
    occ.symbol = sym.symbol
    occ.range.extend([10, 4, 10, 20])
    occ.symbol_roles = 1  # Definition

    # Add a reference occurrence
    occ2 = doc.occurrences.add()
    occ2.symbol = sym.symbol
    occ2.range.extend([20, 2, 20, 8])
    occ2.symbol_roles = 0  # Reference

    scip_path = tmp_path / "index.scip"
    scip_path.write_bytes(idx.SerializeToString())

    from fastcode.scip_loader import load_scip_artifact
    result = load_scip_artifact(str(scip_path))

    assert result.indexer_name == "scip-java"
    assert result.indexer_version == "2.1.0"
    assert len(result.documents) == 1

    doc_result = result.documents[0]
    assert doc_result.path == "src/Main.java"
    assert doc_result.language == "java"
    assert len(doc_result.symbols) == 1
    assert doc_result.symbols[0].name == "main"
    assert doc_result.symbols[0].kind == "method"

    assert len(doc_result.occurrences) == 2
    assert doc_result.occurrences[0].role == "definition"
    assert doc_result.occurrences[0].range == [10, 4, 10, 20]
    assert doc_result.occurrences[1].role == "reference"


def test_binary_scip_empty_index(tmp_path):
    """Empty SCIP index produces empty SCIPIndex."""
    from fastcode.scip_pb2 import Index

    idx = Index()
    idx.metadata.version = 0

    scip_path = tmp_path / "empty.scip"
    scip_path.write_bytes(idx.SerializeToString())

    from fastcode.scip_loader import load_scip_artifact
    result = load_scip_artifact(str(scip_path))

    assert len(result.documents) == 0
    assert result.indexer_name is None or result.indexer_name == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scip_binary.py::test_binary_scip_with_occurrences tests/test_scip_binary.py::test_binary_scip_empty_index -v`
Expected: May pass if Task 2 implementation is complete — verify behavior.

- [ ] **Step 3: Fix any issues in _protobuf_to_scip_index()**

If tests fail, fix the conversion logic in `scip_loader.py`. Common issues:
- `kind` enum value mapping
- Empty `indexer_name` vs `None` handling
- Range list length handling

- [ ] **Step 4: Run all scip_binary tests to verify they pass**

Run: `uv run pytest tests/test_scip_binary.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_scip_binary.py fastcode/scip_loader.py
git commit -m "test: add binary SCIP parsing tests with occurrences and edge cases"
```

---

## Task 5: Integrate SCIP Indexers into the Index Pipeline

**Files:**
- Modify: `fastcode/main.py:637-689` (SCIP processing block)
- Test: `tests/test_scip_indexers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scip_indexers.py — append

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scip_indexers.py::test_auto_detect_scip_languages -v`
Expected: FAIL — `ImportError: cannot import name 'detect_scip_languages'`

- [ ] **Step 3: Implement detect_scip_languages() in scip_indexers.py**

```python
# Append to fastcode/scip_indexers.py

# Map file extension → language name (only languages with SCIP indexers)
_EXTENSION_MAP: Dict[str, str] = {
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".py": "python",
    ".rb": "ruby",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rs": "rust",
    ".php": "php",
    ".dart": "dart",
}


def detect_scip_languages(repo_path: str) -> List[str]:
    """Walk the repo and return deduplicated list of languages with SCIP indexers."""
    seen: set[str] = set()
    for root, _dirs, files in os.walk(repo_path):
        # Skip hidden and common non-source directories
        dirs_to_skip = {".git", ".hg", "node_modules", "__pycache__", ".venv", "venv"}
        _dirs[:] = [d for d in _dirs if d not in dirs_to_skip]
        for fname in files:
            _, ext = os.path.splitext(fname)
            lang = _EXTENSION_MAP.get(ext)
            if lang:
                seen.add(lang)
    return sorted(seen)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scip_indexers.py -v`
Expected: PASS (all 12 tests)

- [ ] **Step 5: Commit**

```bash
git add fastcode/scip_indexers.py tests/test_scip_indexers.py
git commit -m "feat: add language auto-detection for SCIP indexer selection"
```

---

## Task 6: Update Pipeline to Auto-Run SCIP Indexers

**Files:**
- Modify: `fastcode/main.py` (update SCIP processing block)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scip_indexers.py — append

def test_run_scip_for_language_success(tmp_path):
    """run_scip_for_language orchestrates detection, indexing, and loading."""
    from fastcode.scip_indexers import run_scip_for_language
    from fastcode.scip_pb2 import Index, Document, Metadata

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scip_indexers.py::test_run_scip_for_language -v`
Expected: FAIL — `ImportError: cannot import name 'run_scip_for_language'`

- [ ] **Step 3: Implement run_scip_for_language()**

```python
# Append to fastcode/scip_indexers.py

from .scip_loader import load_scip_artifact


def run_scip_for_language(
    language: str,
    repo_path: str,
    output_dir: str,
) -> Optional["SCIPIndex"]:
    """
    Run the SCIP indexer for one language and load the result.

    Returns SCIPIndex on success, None if indexer not available.
    """
    output_path = os.path.join(output_dir, f"{language}.scip")
    try:
        artifact_path = run_scip_indexer(language, repo_path, output_path)
        return load_scip_artifact(artifact_path)
    except RuntimeError as exc:
        logger.warning("SCIP indexer for %s unavailable: %s", language, exc)
        return None
```

Note: Add `from __future__ import annotations` at top of file is already there. The `"SCIPIndex"` string annotation avoids circular import issues.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scip_indexers.py -v`
Expected: PASS (all 14 tests)

- [ ] **Step 5: Commit**

```bash
git add fastcode/scip_indexers.py tests/test_scip_indexers.py
git commit -m "feat: add run_scip_for_language orchestration function"
```

---

## Task 7: Add Binary SCIP Round-Trip Integration Test

**Files:**
- Modify: `tests/test_scip_binary.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scip_binary.py — append

def test_binary_scip_to_ir_round_trip(tmp_path):
    """Binary SCIP → SCIPIndex → IRSnapshot produces valid IR."""
    from fastcode.scip_pb2 import Index, Document, SymbolInformation, Metadata
    from fastcode.scip_loader import load_scip_artifact
    from fastcode.adapters.scip_to_ir import build_ir_from_scip

    idx = Index()
    idx.metadata.version = 0
    idx.metadata.tool_info.name = "scip-go"
    idx.metadata.tool_info.version = "0.2.0"

    doc = idx.documents.add()
    doc.relative_path = "main.go"
    doc.language = "go"

    sym = doc.symbols.add()
    sym.symbol = "scip golang example/main main()`"
    sym.display_name = "main"
    sym.kind = SymbolInformation.Kind.FUNCTION

    occ = doc.occurrences.add()
    occ.symbol = sym.symbol
    occ.range.extend([5, 0, 5, 12])
    occ.symbol_roles = 1  # Definition

    scip_path = tmp_path / "index.scip"
    scip_path.write_bytes(idx.SerializeToString())

    scip_index = load_scip_artifact(str(scip_path))
    snapshot = build_ir_from_scip(
        repo_name="example",
        snapshot_id="snap:example:abc123",
        scip_index=scip_index,
    )

    assert len(snapshot.documents) == 1
    assert snapshot.documents[0].path == "main.go"
    assert len(snapshot.symbols) == 1
    assert snapshot.symbols[0].display_name == "main"
    assert snapshot.symbols[0].kind == "function"
    assert snapshot.symbols[0].source_priority == 100
    assert len(snapshot.occurrences) == 1
    assert snapshot.occurrences[0].role == "definition"
    assert len(snapshot.edges) == 2  # containment + ref edge
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_scip_binary.py::test_binary_scip_to_ir_round_trip -v`
Expected: PASS (all infrastructure from Tasks 1-2 is in place)

- [ ] **Step 3: Commit**

```bash
git add tests/test_scip_binary.py
git commit -m "test: add binary SCIP → IR round-trip integration test"
```

---

## Task 8: Verify Full Test Suite Passes

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/test_scip_binary.py tests/test_scip_indexers.py tests/test_scip_models.py tests/test_ir_core.py -v
```

Expected: ALL PASS

- [ ] **Step 2: Run existing tests for regressions**

```bash
uv run pytest tests/ -v -k "not bench and not e2e"
```

Expected: ALL PASS, no regressions from existing tests

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any regressions from SCIP integration"
```

---

## Summary

| Task | What | Tests Added |
|------|------|-------------|
| 1 | SCIP protobuf bindings | 1 |
| 2 | Binary .scip parsing | 1 |
| 3 | Multi-language indexer runner | 9 |
| 4 | Binary parsing edge cases | 2 |
| 5 | Language auto-detection | 3 |
| 6 | Pipeline orchestration | 2 |
| 7 | IR round-trip integration | 1 |
| 8 | Regression check | 0 |
| **Total** | | **19 tests** |

The existing tree-sitter parsers (`fastcode/parsers/_*.py`) remain as fallback when SCIP indexers aren't installed. The pipeline preference is: **SCIP first** (precise semantics), **tree-sitter second** (structural fallback), merged via `ir_merge.py`.
