"""Resolver registry and default resolver set.

Registers compiler-backed resolvers as primary, with graph-backed structural
resolvers as internal fallbacks.  The registry provides:

- ``applicable()``: returns resolvers matching files in target_paths
- ``applicable_for_capabilities()``: returns resolvers that can fulfil
  specific pending capabilities (capability gating)
"""

from __future__ import annotations

from fastcode.ir.element import CodeElement
from fastcode.ir.types import IRSnapshot
from fastcode.semantic.resolution import SemanticResolver
from fastcode.semantic.resolvers.engine.language_graph import (
    CSharpSemanticResolver,
    FortranSemanticResolver,
    GoSemanticResolver,
    JavaScriptSemanticResolver,
    JavaSemanticResolver,
    JuliaSemanticResolver,
    RustSemanticResolver,
    TypeScriptSemanticResolver,
    ZigSemanticResolver,
)
from fastcode.semantic.resolvers.languages.c_family import (
    CppSemanticResolver,
    CSemanticResolver,
)
from fastcode.semantic.resolvers.languages.csharp import CSharpCompilerResolver
from fastcode.semantic.resolvers.languages.fortran import FortranCompilerResolver
from fastcode.semantic.resolvers.languages.go import GoCompilerResolver
from fastcode.semantic.resolvers.languages.java import JavaCompilerResolver
from fastcode.semantic.resolvers.languages.js_ts import (
    JavaScriptCompilerResolver,
    TypeScriptCompilerResolver,
)
from fastcode.semantic.resolvers.languages.julia import JuliaCompilerResolver
from fastcode.semantic.resolvers.languages.python import PythonSemanticResolver
from fastcode.semantic.resolvers.languages.rust import RustCompilerResolver
from fastcode.semantic.resolvers.languages.zig import ZigCompilerResolver


class SemanticResolverRegistry:
    """Capability-oriented resolver registry."""

    def __init__(self, resolvers: list[SemanticResolver] | None = None) -> None:
        self._resolvers = list(resolvers or [])

    def register(self, resolver: SemanticResolver) -> None:
        self._resolvers.append(resolver)

    def all(self) -> list[SemanticResolver]:
        return list(self._resolvers)

    def applicable(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
    ) -> list[SemanticResolver]:
        return [
            resolver
            for resolver in self._resolvers
            if resolver.applicable(
                snapshot=snapshot, elements=elements, target_paths=target_paths
            )
        ]

    def applicable_for_capabilities(
        self,
        *,
        snapshot: IRSnapshot,
        elements: list[CodeElement],
        target_paths: set[str],
        required_capabilities: frozenset[str],
    ) -> list[SemanticResolver]:
        """Return resolvers that are both applicable AND can fulfil at least
        one of the *required_capabilities*.

        This is the capability-gating entry point: callers pass the union of
        ``pending_capabilities`` from unresolved relations and the registry
        returns only resolvers that advertise matching capabilities.
        """
        if not required_capabilities:
            return self.applicable(
                snapshot=snapshot, elements=elements, target_paths=target_paths
            )
        return [
            resolver
            for resolver in self._resolvers
            if resolver.applicable(
                snapshot=snapshot, elements=elements, target_paths=target_paths
            )
            and resolver.capabilities & required_capabilities
        ]


def build_default_semantic_resolver_registry(
    semantic_helper_runtime: object | None = None,
) -> SemanticResolverRegistry:
    """Build the default resolver registry.

    Each compiler-backed resolver wraps its graph-backed fallback so that
    when external tools are missing, structural evidence is still emitted.
    """
    registry = SemanticResolverRegistry(
        [
            PythonSemanticResolver(),
            # JS/TS — compiler-backed with graph fallback
            JavaScriptCompilerResolver(fallback=JavaScriptSemanticResolver()),
            TypeScriptCompilerResolver(fallback=TypeScriptSemanticResolver()),
            # JVM
            JavaCompilerResolver(fallback=JavaSemanticResolver()),
            # Go
            GoCompilerResolver(fallback=GoSemanticResolver()),
            # Rust
            RustCompilerResolver(fallback=RustSemanticResolver()),
            # .NET
            CSharpCompilerResolver(fallback=CSharpSemanticResolver()),
            # C/C++ (structural, Clang upgrade staged)
            CSemanticResolver(),
            CppSemanticResolver(),
            # Emerging
            ZigCompilerResolver(fallback=ZigSemanticResolver()),
            FortranCompilerResolver(fallback=FortranSemanticResolver()),
            JuliaCompilerResolver(fallback=JuliaSemanticResolver()),
        ]
    )
    if semantic_helper_runtime is not None:
        for resolver in registry.all():
            resolver.set_tool_runtime(semantic_helper_runtime)
            setter = getattr(resolver, "set_helper_runtime", None)
            if callable(setter):
                setter(semantic_helper_runtime)
    return registry
