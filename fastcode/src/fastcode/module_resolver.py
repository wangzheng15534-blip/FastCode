from fastcode.global_index_builder import GlobalIndexBuilder


class ModuleResolver:
    """
    Module resolver for FastCode system.

    Given current file context and import information, resolves the target file ID.
    Handles relative imports (from ..utils import x) and absolute imports (from utils import x).
    Returns None for third-party libraries (external to current repository).
    """

    def __init__(self, index: GlobalIndexBuilder) -> None:
        """
        Initialize resolver with dependency injection.

        Args:
            index: GlobalIndexBuilder instance containing module_map and other lookup tables
        """
        self.index = index

    def resolve_import(
        self,
        current_module_path: str,
        import_name: str,
        level: int,
        is_package: bool = False,
    ) -> str | None:
        """
        Resolve import to target file ID.

        Args:
            current_module_path: ...
            import_name: ...
            level: ...
            is_package: Boolean indicating if the source file is an __init__.py (package root).
                        If True, relative imports starting with '.' stay in the current directory.
        """
        if level > 0:
            # Pass the is_package flag to the relative import handler
            return self._resolve_relative_import(
                current_module_path, import_name, level, is_package
            )
        return self._resolve_absolute_import(import_name)

    def _resolve_relative_import(
        self,
        current_module_path: str,
        import_name: str,
        level: int,
        is_package: bool = False,
    ) -> str | None:
        """
        Resolve relative import by navigating up the module hierarchy.
        """

        # Split current module path into components
        current_parts = current_module_path.split(".")

        # Determine how many levels to strip
        # __init__.py: level=1 (from .) means "current package" -> strip 0
        # regular .py: level=1 (from .) means "parent package" -> strip 1
        strip_count = level - 1 if is_package else level

        # Check bounds
        if strip_count > len(current_parts):
            return None

        # Perform the strip
        if strip_count > 0:
            parent_parts = current_parts[:-strip_count]
        else:
            parent_parts = current_parts

        # Build target module path
        if import_name:
            target_parts = [*parent_parts, import_name]
            target_module_path = ".".join(target_parts)
        else:
            target_module_path = ".".join(parent_parts) if parent_parts else None

        # Lookup in module_map
        if target_module_path and target_module_path in self.index.module_map:
            return self.index.module_map[target_module_path]

        return None

    def _resolve_absolute_import(self, import_name: str) -> str | None:
        """
        Resolve absolute import by direct lookup in module_map.

        Returns None if import_name is not found (likely third-party library).
        """
        if not import_name:
            return None

        # Direct lookup in module_map
        if import_name in self.index.module_map:
            return self.index.module_map[import_name]

        # If not found, this is likely a third-party library
        return None
