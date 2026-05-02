#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

IMPORT_RE = re.compile(r"^\s*import\s+([\w\.]+)\s*;", re.MULTILINE)
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DECL_RE = re.compile(
    r"\b(?:public|private|protected|static|final|synchronized|abstract|native|default|strictfp\s+)*"
    r"(?:[A-Za-z_][\w<>\[\]\.]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
)


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def line_col(text: str, index: int) -> tuple[int, int]:
    line = text.count("\n", 0, index) + 1
    col = index - (text.rfind("\n", 0, index) + 1 if "\n" in text[:index] else 0)
    return line, col


def main() -> int:
    root = Path.cwd()
    paths = [Path(arg) for arg in sys.argv[1:]]
    files = [root / path for path in paths]
    declarations: dict[str, list[dict[str, Any]]] = {}
    imports: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        rel_path = rel(file_path, root)
        for match in DECL_RE.finditer(text):
            name = match.group(1)
            line, col = line_col(text, match.start(1))
            declarations.setdefault(name, []).append(
                {
                    "path": rel_path,
                    "name": name,
                    "line": line,
                    "col": col,
                }
            )
        for match in IMPORT_RE.finditer(text):
            import_name = match.group(1)
            simple_name = import_name.split(".")[-1]
            line, col = line_col(text, match.start(1))
            target = declarations.get(simple_name, [])
            imports.append(
                {
                    "source_path": rel_path,
                    "target_path": target[0]["path"] if target else "",
                    "import_name": import_name,
                    "source_line": line,
                    "source_col": col,
                }
            )

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        rel_path = rel(file_path, root)
        for match in CALL_RE.finditer(text):
            name = match.group(1)
            if name in {"if", "for", "while", "switch", "catch", "return", "new", "super", "this"}:
                continue
            target = declarations.get(name, [])
            if not target:
                continue
            line, col = line_col(text, match.start(1))
            calls.append(
                {
                    "source_path": rel_path,
                    "target_path": target[0]["path"],
                    "call_name": name,
                    "target_name": name,
                    "target_symbol": name,
                    "source_line": line,
                    "source_col": col,
                    "target_line": target[0]["line"],
                    "target_col": target[0]["col"],
                }
            )

    payload = {
        "imports": imports,
        "calls": calls,
        "stats": {
            "files": len(files),
            "imports": len(imports),
            "calls": len(calls),
        },
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
