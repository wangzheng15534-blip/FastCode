"""
Demo: IR Pipeline -- AST + SCIP merge, graph building, validation.

Usage:
    cd /home/jacob/develop/FastCode
    python -m demos.demo_ir_pipeline

Shows:
    1. Building IR from synthetic AST elements
    2. Building IR from a synthetic SCIP payload
    3. Merging both snapshots (SCIP wins on overlap)
    4. Validating the merged snapshot
    5. Building all 5 graph types from merged IR
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastcode.indexer import CodeElement
from fastcode.adapters.ast_to_ir import build_ir_from_ast
from fastcode.adapters.scip_to_ir import build_ir_from_scip
from fastcode.ir_merge import merge_ir
from fastcode.ir_validators import validate_snapshot
from fastcode.ir_graph_builder import IRGraphBuilder


def main():
    # --- 1. AST elements ---
    elements = [
        CodeElement(
            id="el_1", name="AuthService", type="class",
            file_path="/repo/app/auth.py", relative_path="app/auth.py",
            language="python", start_line=10, end_line=50,
            code="class AuthService: ...", summary="Authentication service",
            signature="class AuthService",
            docstring=None,
            metadata={"imports": [{"module": "db"}], "bases": ["BaseService"]},
        ),
        CodeElement(
            id="el_2", name="login", type="function",
            file_path="/repo/app/auth.py", relative_path="app/auth.py",
            language="python", start_line=20, end_line=35,
            code="def login(): ...", summary="Login handler",
            signature="login()",
            docstring=None,
            metadata={"class_name": "AuthService"},
        ),
        CodeElement(
            id="el_3", name="BaseService", type="class",
            file_path="/repo/app/base.py", relative_path="app/base.py",
            language="python", start_line=1, end_line=20,
            code="class BaseService: ...", summary="Base service class",
            signature="class BaseService",
            docstring=None,
            metadata={},
        ),
    ]

    # --- 2. Build AST IR ---
    ast_snapshot = build_ir_from_ast(
        repo_name="demo-repo",
        snapshot_id="snap:demo:abc123",
        elements=elements,
        repo_root="/repo",
    )
    print(f"AST IR: {len(ast_snapshot.documents)} docs, {len(ast_snapshot.symbols)} symbols, "
          f"{len(ast_snapshot.occurrences)} occurrences, {len(ast_snapshot.edges)} edges")
    for s in ast_snapshot.symbols:
        print(f"  AST symbol: {s.symbol_id} ({s.source_priority})")
    for e in ast_snapshot.edges:
        print(f"  AST edge: {e.edge_type} {e.src_id[:30]}... -> {e.dst_id[:30]}... "
              f"[{e.source}/{e.confidence}]")

    # --- 3. Build SCIP IR ---
    scip_payload = {
        "indexer_name": "scip-python",
        "indexer_version": "0.5.0",
        "documents": [
            {
                "path": "app/auth.py",
                "language": "python",
                "symbols": [
                    {"symbol": "pkg app/auth.py AuthService.", "name": "AuthService",
                     "kind": "class", "range": [10, 0, 50, 0]},
                    {"symbol": "pkg app/auth.py AuthService.login().", "name": "login",
                     "kind": "method", "range": [20, 4, 35, 0]},
                ],
                "occurrences": [
                    {"symbol": "pkg app/auth.py AuthService.login().", "role": "definition",
                     "range": [20, 4, 35, 0]},
                    {"symbol": "pkg app/auth.py AuthService.login().", "role": "reference",
                     "range": [100, 0, 100, 5]},
                ],
            }
        ],
    }
    scip_snapshot = build_ir_from_scip(
        repo_name="demo-repo",
        snapshot_id="snap:demo:abc123",
        scip_index=scip_payload,
    )
    print(f"\nSCIP IR: {len(scip_snapshot.documents)} docs, {len(scip_snapshot.symbols)} symbols, "
          f"{len(scip_snapshot.occurrences)} occurrences, {len(scip_snapshot.edges)} edges")
    for s in scip_snapshot.symbols:
        print(f"  SCIP symbol: {s.symbol_id} ({s.source_priority})")
    for e in scip_snapshot.edges:
        print(f"  SCIP edge: {e.edge_type} {e.src_id[:30]}... -> {e.dst_id[:30]}... "
              f"[{e.source}/{e.confidence}]")

    # --- 4. Merge ---
    merged = merge_ir(ast_snapshot, scip_snapshot)
    print(f"\nMerged IR: {len(merged.documents)} docs, {len(merged.symbols)} symbols, "
          f"{len(merged.occurrences)} occurrences, {len(merged.edges)} edges")
    for s in merged.symbols:
        aliases = s.metadata.get("aliases", [])
        alias_str = f" (aliases: {aliases})" if aliases else ""
        print(f"  Merged symbol: {s.symbol_id} source_set={s.source_set}{alias_str}")

    # --- 5. Validate ---
    errors = validate_snapshot(merged)
    if errors:
        print(f"\nValidation FAILED ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nValidation PASSED (no errors)")

    # --- 6. Build graphs ---
    builder = IRGraphBuilder()
    graphs = builder.build_graphs(merged)
    print(f"\nGraphs built:")
    for name in [
        "dependency_graph", "call_graph", "inheritance_graph",
        "reference_graph", "containment_graph",
    ]:
        g = getattr(graphs, name)
        print(f"  {name}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    print("\nDone.")


if __name__ == "__main__":
    main()
