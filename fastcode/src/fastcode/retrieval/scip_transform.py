# fastcode/core/scip_transform.py
"""Pure SCIP transform functions — extracted from scip_loader.py."""

from __future__ import annotations

# Manual mapping from protobuf Kind enum int values to strings.
# Avoids importing the protobuf module (no type stubs) at type-check time.
_KIND_MAP: dict[int, str] = {
    7: "class",
    8: "constant",
    9: "constructor",
    11: "enum",
    12: "enum_member",
    15: "field",
    16: "file",
    17: "function",
    21: "interface",
    25: "macro",
    26: "method",
    29: "module",
    30: "namespace",
    35: "package",
    37: "parameter",
    41: "property",
    49: "struct",
    54: "type",
    58: "type_parameter",
    61: "variable",
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
