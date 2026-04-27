# fastcode/core/scip_transform.py
"""Pure SCIP transform functions — extracted from scip_loader.py."""

from __future__ import annotations

# Manual mapping from protobuf Kind enum int values to strings.
# Avoids importing the protobuf module (no type stubs) at type-check time.
_KIND_MAP: dict[int, str] = {
    1: "function",
    2: "method",
    3: "class",
    4: "interface",
    5: "enum",
    6: "enum_member",
    7: "variable",
    8: "constant",
    9: "property",
    10: "type",
    11: "macro",
    12: "module",
    13: "namespace",
    14: "package",
    15: "parameter",
    16: "type_parameter",
    17: "constructor",
    18: "struct",
}


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
    return _KIND_MAP.get(kind_value, "symbol")
