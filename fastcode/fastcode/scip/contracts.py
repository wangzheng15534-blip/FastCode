"""Frozen SCIP-domain constants and contracts."""

from __future__ import annotations

from dataclasses import dataclass


class ScipKind:
    """SCIP symbol kind constants."""

    FUNCTION = "Function"
    METHOD = "Method"
    CLASS = "Class"
    MODULE = "Module"
    INTERFACE = "Interface"
    ENUM = "Enum"
    VARIABLE = "Variable"
    CONSTANT = "Constant"
    PROPERTY = "Property"
    TYPE = "Type"
    UNKNOWN = "Unknown"


class ScipRole:
    """SCIP symbol occurrence role constants."""

    DEFINITION = "Definition"
    REFERENCE = "Reference"
    IMPORT = "Import"
    WRITE_ACCESS = "WriteAccess"
    FORWARD_DEFINITION = "ForwardDefinition"


@dataclass(frozen=True)
class ScipResolutionRequest:
    """Inputs for resolving a SCIP symbol occurrence."""

    symbol_name: str
    current_file_id: str
    current_module_path: str
