"""
Snapshot-scoped canonical symbol index and alias resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .semantic_ir import IRSnapshot


@dataclass
class SnapshotSymbolMaps:
    canonical_by_alias: Dict[str, str] = field(default_factory=dict)
    aliases_by_canonical: Dict[str, Set[str]] = field(default_factory=dict)
    symbols_by_name: Dict[str, Set[str]] = field(default_factory=dict)
    symbols_by_path: Dict[str, Set[str]] = field(default_factory=dict)


class SnapshotSymbolIndex:
    def __init__(self):
        self._by_snapshot: Dict[str, SnapshotSymbolMaps] = {}

    def register_snapshot(self, snapshot: IRSnapshot) -> None:
        maps = SnapshotSymbolMaps()
        for symbol in snapshot.symbols:
            canonical = symbol.symbol_id
            maps.canonical_by_alias[canonical] = canonical
            maps.aliases_by_canonical.setdefault(canonical, set()).add(canonical)
            if symbol.display_name:
                maps.symbols_by_name.setdefault(symbol.display_name, set()).add(canonical)
            if symbol.qualified_name:
                maps.symbols_by_name.setdefault(symbol.qualified_name, set()).add(canonical)
            if symbol.path:
                maps.symbols_by_path.setdefault(symbol.path, set()).add(canonical)

            aliases = (symbol.metadata or {}).get("aliases", []) if symbol.metadata else []
            for alias in aliases:
                if not alias:
                    continue
                maps.canonical_by_alias[alias] = canonical
                maps.aliases_by_canonical.setdefault(canonical, set()).add(alias)

        self._by_snapshot[snapshot.snapshot_id] = maps

    def has_snapshot(self, snapshot_id: str) -> bool:
        return snapshot_id in self._by_snapshot

    def canonicalize_symbol(self, snapshot_id: str, symbol_id: str) -> Optional[str]:
        maps = self._by_snapshot.get(snapshot_id)
        if maps is None:
            return None
        return maps.canonical_by_alias.get(symbol_id)

    def get_aliases(self, snapshot_id: str, canonical_symbol_id: str) -> List[str]:
        maps = self._by_snapshot.get(snapshot_id)
        if maps is None:
            return []
        aliases = maps.aliases_by_canonical.get(canonical_symbol_id, set())
        return sorted(list(aliases))

    def resolve_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: Optional[str] = None,
        name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Optional[str]:
        maps = self._by_snapshot.get(snapshot_id)
        if maps is None:
            return None
        if symbol_id:
            canonical = maps.canonical_by_alias.get(symbol_id)
            if canonical:
                return canonical
        if name:
            candidates = maps.symbols_by_name.get(name)
            if candidates:
                return sorted(candidates)[0]
        if path:
            candidates = maps.symbols_by_path.get(path)
            if candidates:
                return sorted(candidates)[0]
        return None
