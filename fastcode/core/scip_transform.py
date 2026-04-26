# fastcode/core/scip_transform.py
"""Pure SCIP transform functions — extracted from scip_loader.py."""

from __future__ import annotations


def symbol_role_to_str(roles: int) -> str:
    """Convert SCIP symbol_roles bitmask to a semantic role string."""
    if roles & 1:  # Definition
        return "definition"
    if roles & 2:  # Import
        return "import"
    if roles & 4:  # WriteAccess
        return "write_access"
    if roles & 64:  # ForwardDefinition
        return "forward_definition"
    return "reference"


def scip_kind_to_str(kind_value: int) -> str:
    """Convert SCIP protobuf Kind enum to string."""
    try:
        from fastcode.scip_pb2 import SymbolInformation
    except ImportError:
        return "symbol"

    kind_map = {
        SymbolInformation.Kind.Function: "function",
        SymbolInformation.Kind.Method: "method",
        SymbolInformation.Kind.Class: "class",
        SymbolInformation.Kind.Interface: "interface",
        SymbolInformation.Kind.Enum: "enum",
        SymbolInformation.Kind.EnumMember: "enum_member",
        SymbolInformation.Kind.Variable: "variable",
        SymbolInformation.Kind.Constant: "constant",
        SymbolInformation.Kind.Property: "property",
        SymbolInformation.Kind.Type: "type",
        SymbolInformation.Kind.Macro: "macro",
        SymbolInformation.Kind.Module: "module",
        SymbolInformation.Kind.Namespace: "namespace",
        SymbolInformation.Kind.Package: "package",
        SymbolInformation.Kind.Parameter: "parameter",
        SymbolInformation.Kind.TypeParameter: "type_parameter",
        SymbolInformation.Kind.Constructor: "constructor",
        SymbolInformation.Kind.Struct: "struct",
    }
    return kind_map.get(kind_value, "symbol")
