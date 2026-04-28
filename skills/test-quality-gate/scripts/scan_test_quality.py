# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Test Quality Gate — Static antipattern scanner.

Detects cheaty test patterns in Python test files:
  TA001 - `or True` / `and False` tautology in assertions
  TA002 - Self-comparison (`assert x == x`, `assert x is x`)
  TA004 - No assertion in test body
  TA005 - Fake property test (`@given` with all `st.just()`/`st.none()`)
  TA007 - Shape-only assertion (sole `assert isinstance` or `assert "key" in`)
  TA008 - Trivially true condition (`assert len(x) >= 0`, `assert x is not None` as sole check)

Usage:
  uv run scripts/scan_test_quality.py tests/
  uv run scripts/scan_test_quality.py tests/test_ir_core.py
  uv run scripts/scan_test_quality.py tests/ --json
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Finding model
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    code: str
    severity: str  # CRITICAL | HIGH | MEDIUM
    file: str
    line: int
    col: int
    message: str
    snippet: str = ""
    risk: str = ""  # "P×I=N"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_test_function(node: ast.AST) -> bool:
    """Check if node is a test function or method."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name.startswith("test_")
    return False


def _get_source_line(source: str, lineno: int) -> str:
    """Get a source line by 1-based line number."""
    lines = source.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _count_asserts(body: list[ast.stmt]) -> int:
    """Count assert statements in a function body (recursive into with/for/if)."""
    count = 0
    for stmt in body:
        if isinstance(stmt, ast.Assert):
            count += 1
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            # Detect snapshot.assert_match(...) and similar assertion methods
            func = stmt.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr
                in (
                    "assert_match",
                    "assertEqual",
                    "assertIn",
                    "assertTrue",
                    "assertRaises",
                    "assert_called",
                    "assert_called_once",
                    "assert_called_with",
                    "assert_not_called",
                )
            ) or (isinstance(func, ast.Name) and func.id == "fail"):
                count += 1
        elif isinstance(stmt, ast.With):
            # Check for pytest.raises context manager
            for item in stmt.items:
                if isinstance(item.context_expr, ast.Call):
                    func = item.context_expr.func
                    if (isinstance(func, ast.Attribute) and func.attr == "raises") or (
                        isinstance(func, ast.Name) and func.id == "raises"
                    ):
                        count += 1
            count += _count_asserts(stmt.body)
        elif isinstance(stmt, (ast.For, ast.If, ast.While, ast.Try)):
            for sub_body in _stmt_bodies(stmt):
                count += _count_asserts(sub_body)
        # Don't recurse into nested functions/classes
    return count


def _stmt_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """Extract all body lists from a compound statement."""
    bodies: list[list[ast.stmt]] = []
    for field_name in ("body", "orelse", "finalbody", "handlers"):
        val = getattr(stmt, field_name, None)
        if isinstance(val, list):
            if val and isinstance(val[0], ast.AST):
                # handlers is list of ExceptHandler, each has body
                if field_name == "handlers":
                    for handler in val:
                        if hasattr(handler, "body"):
                            bodies.append(handler.body)
                else:
                    bodies.append(val)
    return bodies


def _has_given_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function has @given decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "given":
            return True
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id == "given"
        ):
            return True
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == "given"
        ):
            return True
    return False


def _given_uses_only_trivial_strategies(dec: ast.Call) -> bool:
    """Check if @given decorator uses only st.just() / st.none() for ALL params."""
    if not dec.keywords:
        return False  # positional args — can't easily check

    all_trivial = True
    has_keyword = False

    for kw in dec.keywords:
        if kw.arg is None:
            continue  # **kwargs
        has_keyword = True
        val = kw.value
        if not _is_trivial_strategy(val):
            all_trivial = False
            break

    return has_keyword and all_trivial


def _is_trivial_strategy(node: ast.expr) -> bool:
    """Check if expression is st.just(...) or st.none()."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in ("just", "none"):
                return True
            if func.attr == "none":
                return True
    return False


def _is_boolop_with_constant(node: ast.expr, op_type: type, value: object) -> bool:
    """Check if node is BoolOp containing a constant."""
    if not isinstance(node, ast.BoolOp):
        return False
    if not isinstance(node.op, op_type):
        return False
    for v in node.values:
        if isinstance(v, ast.Constant) and v.value is value:
            return True
    return False


def _name_str(node: ast.expr) -> str | None:
    """Get variable name from a plain Name node only.

    Returns None for Attribute nodes (obj.attr) to avoid false positives
    where `assert snap.path == path` would match both sides as "path".
    """
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_shape_only_assert(node: ast.Assert) -> bool:
    """Check if assertion only checks key existence or type, not values."""
    test = node.test

    # assert isinstance(x, T)
    if isinstance(test, ast.Call):
        func = test.func
        if isinstance(func, ast.Name) and func.id == "isinstance":
            return True

    # assert "key" in obj
    if isinstance(test, ast.Compare):
        if len(test.ops) == 1 and isinstance(test.ops[0], ast.In):
            return True

    return False


def _is_trivially_true(node: ast.Assert) -> bool:
    """Check if assertion is trivially true."""
    test = node.test

    # assert x is not None (as sole or main check)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        if isinstance(test.operand, ast.Compare):
            if any(isinstance(op, ast.Is) for op in test.operand.ops):
                return True

    # assert len(x) >= 0 or assert len(x) > -1
    if isinstance(test, ast.Compare):
        for op, comp in zip(test.ops, test.comparators):
            if isinstance(test.left, ast.Call):
                func = test.left.func
                if isinstance(func, ast.Name) and func.id == "len":
                    if (
                        isinstance(op, ast.GtE)
                        and isinstance(comp, ast.Constant)
                        and comp.value == 0
                    ):
                        return True
                    if isinstance(op, ast.Gt) and isinstance(comp, ast.UnaryOp):
                        if isinstance(comp.op, ast.USub):
                            return True

    # assert result == result.lower() when result is already lowercase
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        if isinstance(test.ops[0], ast.Eq):
            if isinstance(test.comparators[0], ast.Call):
                func = test.comparators[0].func
                if isinstance(func, ast.Attribute) and func.attr == "lower":
                    left_name = _name_str(test.left)
                    call_obj_name = _name_str(func.value)
                    if left_name and call_obj_name and left_name == call_obj_name:
                        return True

    return False


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------


def scan_or_true(node: ast.Assert) -> Finding | None:
    """TA001: Detect `assert ... or True`."""
    if _is_boolop_with_constant(node.test, ast.Or, True):
        return Finding(
            code="TA001",
            severity="CRITICAL",
            file="",
            line=node.lineno,
            col=node.col_offset,
            message="Tautological assertion: `or True` makes this always pass",
            risk="P=3 × I=3 = 9",
        )
    # Also check `and False`
    if _is_boolop_with_constant(node.test, ast.And, False):
        return Finding(
            code="TA001",
            severity="CRITICAL",
            file="",
            line=node.lineno,
            col=node.col_offset,
            message="Tautological assertion: `and False` makes this always fail",
            risk="P=3 × I=3 = 9",
        )
    return None


def scan_self_comparison(node: ast.Assert) -> Finding | None:
    """TA002: Detect `assert x == x`, `assert x is x`."""
    test = node.test
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return None

    op = test.ops[0]
    if not isinstance(op, (ast.Eq, ast.Is, ast.IsNot, ast.NotEq)):
        return None

    left_name = _name_str(test.left)
    right_name = _name_str(test.comparators[0])

    if left_name and right_name and left_name == right_name:
        return Finding(
            code="TA002",
            severity="CRITICAL",
            file="",
            line=node.lineno,
            col=node.col_offset,
            message=f"Self-comparison: `{left_name}` compared to itself",
            risk="P=3 × I=3 = 9",
        )
    return None


def scan_no_assertion(func: ast.FunctionDef | ast.AsyncFunctionDef) -> Finding | None:
    """TA004: Test function with no assertions."""
    if not _is_test_function(func):
        return None
    assert_count = _count_asserts(func.body)
    if assert_count == 0:
        return Finding(
            code="TA004",
            severity="CRITICAL",
            file="",
            line=func.lineno,
            col=func.col_offset,
            message=f"Test `{func.name}` has no assertions",
            risk="P=2 × I=3 = 6",
        )
    return None


def scan_fake_property_test(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Finding | None:
    """TA005: @given with all st.just()/st.none() strategies."""
    if not _has_given_decorator(func):
        return None

    for dec in func.decorator_list:
        if isinstance(dec, ast.Call) and _given_uses_only_trivial_strategies(dec):
            trivial_args = []
            for kw in dec.keywords:
                if kw.arg and _is_trivial_strategy(kw.value):
                    trivial_args.append(kw.arg)
            return Finding(
                code="TA005",
                severity="HIGH",
                file="",
                line=func.lineno,
                col=func.col_offset,
                message=(
                    f"Fake property test: @given with trivial strategies "
                    f"({', '.join(trivial_args)}). Convert to plain test."
                ),
                risk="P=3 × I=2 = 6",
            )
    return None


def scan_shape_only(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Finding]:
    """TA007: Test where ALL assertions are shape-only (isinstance, key-in).

    Only flags when every assertion in the test checks type/key existence
    with no value verification. A test with `assert isinstance(x, T)` AND
    `assert x.value == expected` is NOT flagged.
    """
    if not _is_test_function(func):
        return []

    asserts = [n for n in ast.walk(func) if isinstance(n, ast.Assert)]
    if not asserts:
        return []  # TA004 handles no-assertion case

    shape_asserts = [a for a in asserts if _is_shape_only_assert(a)]
    # Only flag if ALL assertions are shape-only
    if len(shape_asserts) == len(asserts) and shape_asserts:
        # Return one finding per shape-only assertion
        findings = []
        for a in shape_asserts:
            findings.append(
                Finding(
                    code="TA007",
                    severity="HIGH",
                    file="",
                    line=a.lineno,
                    col=a.col_offset,
                    message="Shape-only test: all assertions check type/key existence, not values",
                    risk="P=2 × I=2 = 4",
                )
            )
        return findings
    return []


def scan_trivially_true(node: ast.Assert) -> Finding | None:
    """TA008: Trivially true condition."""
    if _is_trivially_true(node):
        return Finding(
            code="TA008",
            severity="HIGH",
            file="",
            line=node.lineno,
            col=node.col_offset,
            message="Trivially true assertion: condition is always satisfied",
            risk="P=2 × I=2 = 4",
        )
    return None


# ---------------------------------------------------------------------------
# File scanner
# ---------------------------------------------------------------------------


def scan_file(path: Path) -> list[Finding]:
    """Scan a single Python test file for antipatterns."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    findings: list[Finding] = []

    for node in ast.walk(tree):
        # Check test functions for no-assertion, fake-property-test, and shape-only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_test_function(node):
                f = scan_no_assertion(node)
                if f:
                    f.file = str(path)
                    f.snippet = _get_source_line(source, f.line)
                    findings.append(f)

                f = scan_fake_property_test(node)
                if f:
                    f.file = str(path)
                    f.snippet = _get_source_line(source, f.line)
                    findings.append(f)

                # Shape-only: check at function level
                for sf in scan_shape_only(node):
                    sf.file = str(path)
                    sf.snippet = _get_source_line(source, sf.line)
                    findings.append(sf)

        # Check individual assertions
        if isinstance(node, ast.Assert):
            for scanner in (scan_or_true, scan_self_comparison, scan_trivially_true):
                f = scanner(node)
                if f:
                    f.file = str(path)
                    f.snippet = _get_source_line(source, f.line)
                    findings.append(f)

    return findings


def scan_paths(paths: Sequence[Path]) -> list[Finding]:
    """Scan multiple paths (files and directories)."""
    all_findings: list[Finding] = []

    for p in paths:
        if p.is_file() and p.suffix == ".py":
            all_findings.extend(scan_file(p))
        elif p.is_dir():
            for py_file in sorted(p.rglob("test_*.py")):
                all_findings.extend(scan_file(py_file))
            for py_file in sorted(p.rglob("*_test.py")):
                all_findings.extend(scan_file(py_file))

    return all_findings


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_report(findings: list[Finding]) -> str:
    """Format findings as markdown report."""
    if not findings:
        return "## Test Quality Gate: PASS\nNo antipatterns found.\n"

    critical = [f for f in findings if f.severity == "CRITICAL"]
    high = [f for f in findings if f.severity == "HIGH"]
    medium = [f for f in findings if f.severity == "MEDIUM"]

    lines = [
        "## Test Quality Gate Report",
        f"**Findings:** {len(findings)} ({len(critical)} CRITICAL / {len(high)} HIGH / {len(medium)} MEDIUM)",
        "",
    ]

    if critical:
        lines.append("### CRITICAL (must fix)")
        lines.append("| File:Line | Code | Message | Risk |")
        lines.append("|-----------|------|---------|------|")
        for f in critical:
            lines.append(f"| {f.file}:{f.line} | {f.code} | {f.message} | {f.risk} |")
        lines.append("")

    if high:
        lines.append("### HIGH (should fix)")
        lines.append("| File:Line | Code | Message | Risk |")
        lines.append("|-----------|------|---------|------|")
        for f in high:
            lines.append(f"| {f.file}:{f.line} | {f.code} | {f.message} | {f.risk} |")
        lines.append("")

    if medium:
        lines.append("### MEDIUM (consider fixing)")
        lines.append("| File:Line | Code | Message | Risk |")
        lines.append("|-----------|------|---------|------|")
        for f in medium:
            lines.append(f"| {f.file}:{f.line} | {f.code} | {f.message} | {f.risk} |")
        lines.append("")

    # Gate decision
    if critical:
        lines.append("**Gate: BLOCK** — CRITICAL findings give false confidence")
    elif len(high) >= 3:
        lines.append("**Gate: WARN** — 3+ HIGH findings suggest coverage gaps")
    else:
        lines.append("**Gate: PASS** — findings are low severity")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: scan_test_quality.py <path>... [--json]")
        sys.exit(1)

    use_json = "--json" in args
    paths = [Path(a) for a in args if a != "--json"]

    findings = scan_paths(paths)

    if use_json:
        data = [
            {
                "code": f.code,
                "severity": f.severity,
                "file": f.file,
                "line": f.line,
                "col": f.col,
                "message": f.message,
                "snippet": f.snippet,
                "risk": f.risk,
            }
            for f in findings
        ]
        print(json.dumps(data, indent=2))
    else:
        print(format_report(findings))

    # Exit code: 1 if CRITICAL findings, 0 otherwise
    has_critical = any(f.severity == "CRITICAL" for f in findings)
    sys.exit(1 if has_critical else 0)


if __name__ == "__main__":
    main()
