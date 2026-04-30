"""Resolver registry and default resolver set.

Registers compiler-backed resolvers as primary, with graph-backed structural
resolvers as internal fallbacks.  The registry provides:

- ``applicable()``: returns resolvers matching files in target_paths
- ``applicable_for_capabilities()``: returns resolvers that can fulfil
  specific pending capabilities (capability gating)
"""

from __future__ import annotations

from ..indexer import CodeElement
from ..semantic_ir import IRSnapshot
from .base import SemanticResolver
from .c_family import CppSemanticResolver, CSemanticResolver
from .csharp import CSharpCompilerResolver
from .fortran import FortranCompilerResolver
from .go import GoCompilerResolver
from .java import JavaCompilerResolver
from .js_ts import JavaScriptCompilerResolver, TypeScriptCompilerResolver
from .julia import JuliaCompilerResolver
from .language_graph import (
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
from .python import PythonSemanticResolver
from .rust import RustCompilerResolver
from .zig import ZigCompilerResolver


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


def build_default_semantic_resolver_registry() -> SemanticResolverRegistry:
    """Build the default resolver registry.

    Each compiler-backed resolver wraps its graph-backed fallback so that
    when external tools are missing, structural evidence is still emitted.
    """
    return SemanticResolverRegistry(
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
