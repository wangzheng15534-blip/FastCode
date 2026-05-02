#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

USE_RE = re.compile(r"^\s*use\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE | re.IGNORECASE)
SUB_RE = re.compile(r"^\s*(?:subroutine|function)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE | re.IGNORECASE)
MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE | re.IGNORECASE)
CALL_RE = re.compile(r"\bcall\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)
TYPE_RE = re.compile(r"type\s*,\s*extends\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*::\s*([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def line_col(text: str, index: int) -> tuple[int, int]:
    line = text.count("\n", 0, index) + 1
    col = index - (text.rfind("\n", 0, index) + 1 if "\n" in text[:index] else 0)
    return line, col


def main() -> int:
    root = Path.cwd()
    files = [Path(arg) if Path(arg).is_absolute() else root / arg for arg in sys.argv[1:]]
    declarations: dict[str, list[dict[str, Any]]] = {}
    imports: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    inherits: list[dict[str, Any]] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        rel_path = rel(file_path, root)
        for regex in (SUB_RE, MODULE_RE):
            for match in regex.finditer(text):
                name = match.group(1)
                line, col = line_col(text, match.start(1))
                declarations.setdefault(name.lower(), []).append({"path": rel_path, "name": name, "line": line, "col": col})
        for match in TYPE_RE.finditer(text):
            child = match.group(2)
            line, col = line_col(text, match.start(2))
            declarations.setdefault(child.lower(), []).append({"path": rel_path, "name": child, "line": line, "col": col})

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        rel_path = rel(file_path, root)
        for match in USE_RE.finditer(text):
            import_name = match.group(1)
            target = declarations.get(import_name.lower(), [])
            line, col = line_col(text, match.start(1))
            imports.append(
                {
                    "source_path": rel_path,
                    "target_path": target[0]["path"] if target else "",
                    "import_name": import_name,
                    "source_line": line,
                    "source_col": col,
                }
            )
        for match in CALL_RE.finditer(text):
            name = match.group(1)
            target = declarations.get(name.lower(), [])
            if not target:
                continue
            line, col = line_col(text, match.start(1))
            calls.append(
                {
                    "source_path": rel_path,
                    "target_path": target[0]["path"],
                    "call_name": name,
                    "target_name": target[0]["name"],
                    "target_symbol": target[0]["name"],
                    "source_line": line,
                    "source_col": col,
                    "target_line": target[0]["line"],
                    "target_col": target[0]["col"],
                }
            )
        for match in TYPE_RE.finditer(text):
            base_name = match.group(1)
            source_name = match.group(2)
            source_line, _ = line_col(text, match.start(2))
            target = declarations.get(base_name.lower(), [])
            if not target:
                continue
            inherits.append(
                {
                    "source_path": rel_path,
                    "source_name": source_name,
                    "source_line": source_line,
                    "target_path": target[0]["path"],
                    "target_name": target[0]["name"],
                    "target_line": target[0]["line"],
                    "base_name": base_name,
                }
            )

    payload = {
        "imports": imports,
        "calls": calls,
        "inherits": inherits,
        "stats": {"files": len(files), "imports": len(imports), "calls": len(calls), "inherits": len(inherits)},
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
