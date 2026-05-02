#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

USE_RE = re.compile(r"^\s*use\s+([^;]+);", re.MULTILINE)
FN_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*!?\s*\(")
STRUCT_RE = re.compile(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\b")


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

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        rel_path = rel(file_path, root)
        for match in FN_RE.finditer(text):
            name = match.group(1)
            line, col = line_col(text, match.start(1))
            declarations.setdefault(name, []).append({"path": rel_path, "name": name, "line": line, "col": col})
        for match in STRUCT_RE.finditer(text):
            name = match.group(1)
            line, col = line_col(text, match.start(1))
            declarations.setdefault(name, []).append({"path": rel_path, "name": name, "line": line, "col": col})

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        rel_path = rel(file_path, root)
        for match in USE_RE.finditer(text):
            import_name = match.group(1).strip()
            simple_name = import_name.split("::")[-1]
            line, col = line_col(text, match.start(1))
            targets = declarations.get(simple_name, [])
            imports.append(
                {
                    "source_path": rel_path,
                    "target_path": targets[0]["path"] if targets else "",
                    "import_name": import_name,
                    "source_line": line,
                    "source_col": col,
                }
            )
        for match in CALL_RE.finditer(text):
            name = match.group(1)
            if name in {"if", "while", "loop", "match"}:
                continue
            targets = declarations.get(name, [])
            if not targets:
                continue
            line, col = line_col(text, match.start(1))
            target = targets[0]
            calls.append(
                {
                    "source_path": rel_path,
                    "target_path": target["path"],
                    "call_name": name,
                    "target_name": name,
                    "target_symbol": name,
                    "source_line": line,
                    "source_col": col,
                    "target_line": target["line"],
                    "target_col": target["col"],
                }
            )

    payload = {
        "imports": imports,
        "calls": calls,
        "stats": {"files": len(files), "imports": len(imports), "calls": len(calls)},
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
