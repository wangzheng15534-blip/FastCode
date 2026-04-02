"""Tests for binary SCIP protobuf parsing."""

import pytest


def test_scip_pb2_module_importable():
    """Protobuf bindings module must be importable."""
    from fastcode.scip_pb2 import Index
    idx = Index()
    assert idx.metadata.tool_info.name == ""


def test_load_binary_scip_artifact(tmp_path):
    """Binary .scip files parse without external CLI."""
    from fastcode.scip_pb2 import Index

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


def test_binary_scip_with_occurrences(tmp_path):
    """Binary SCIP with occurrences converts correctly."""
    from fastcode.scip_pb2 import Index, SymbolInformation

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
    sym.kind = SymbolInformation.Kind.Method

    # Definition occurrence (symbol_roles bit 0 = definition)
    occ = doc.occurrences.add()
    occ.symbol = sym.symbol
    occ.range.extend([10, 4, 10, 20])
    occ.symbol_roles = 1  # Definition

    # Reference occurrence
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


def test_binary_scip_to_ir_round_trip(tmp_path):
    """Binary SCIP -> SCIPIndex -> IRSnapshot produces valid IR."""
    from fastcode.scip_pb2 import Index, SymbolInformation
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
    sym.kind = SymbolInformation.Kind.Function

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
