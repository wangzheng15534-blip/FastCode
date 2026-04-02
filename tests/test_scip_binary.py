"""Tests for binary SCIP protobuf parsing."""

import pytest


def test_scip_pb2_module_importable():
    """Protobuf bindings module must be importable."""
    from fastcode.scip_pb2 import Index
    idx = Index()
    assert idx.metadata.tool_info.name == ""
