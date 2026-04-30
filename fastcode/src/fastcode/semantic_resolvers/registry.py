"""Resolver registry and default resolver set."""

from __future__ import annotations

from ..indexer import CodeElement
from ..semantic_ir import IRSnapshot
from .base import SemanticResolver
from .c_family import CSemanticResolver, CppSemanticResolver
from .python import PythonSemanticResolver


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


def build_default_semantic_resolver_registry() -> SemanticResolverRegistry:
    return SemanticResolverRegistry(
        [
            PythonSemanticResolver(),
            CSemanticResolver(),
            CppSemanticResolver(),
        ]
    )
