"""
Snapshot-scoped canonical unit index and anchor resolver.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass, field

from .semantic_ir import IRSnapshot


@dataclass
class SnapshotSymbolMaps:
    canonical_by_alias: dict[str, str] = field(default_factory=dict)
    aliases_by_canonical: dict[str, set[str]] = field(default_factory=dict)
    symbols_by_name: dict[str, set[str]] = field(default_factory=dict)
    symbols_by_path: dict[str, set[str]] = field(default_factory=dict)


class SnapshotSymbolIndex:
    def __init__(self) -> None:
        self._by_snapshot: dict[str, SnapshotSymbolMaps] = {}

    def register_snapshot(self, snapshot: IRSnapshot) -> None:
        maps = SnapshotSymbolMaps()
        for unit in snapshot.units:
            if unit.kind in {"file", "doc"}:
                continue
            canonical = unit.unit_id
            aliases = {canonical}
            if unit.primary_anchor_symbol_id:
                aliases.add(unit.primary_anchor_symbol_id)
            aliases.update(unit.anchor_symbol_ids)
            aliases.update(unit.candidate_anchor_symbol_ids)
            aliases.update((unit.metadata or {}).get("aliases", []))

            for alias in aliases:
                if not alias:
                    continue
                maps.canonical_by_alias[str(alias)] = canonical
                maps.aliases_by_canonical.setdefault(canonical, set()).add(str(alias))

            for name in [unit.display_name, unit.qualified_name]:
                if name:
                    maps.symbols_by_name.setdefault(str(name), set()).add(canonical)
            if unit.path:
                maps.symbols_by_path.setdefault(unit.path, set()).add(canonical)

        self._by_snapshot[snapshot.snapshot_id] = maps

    def has_snapshot(self, snapshot_id: str) -> bool:
        return snapshot_id in self._by_snapshot

    def canonicalize_symbol(self, snapshot_id: str, symbol_id: str) -> str | None:
        maps = self._by_snapshot.get(snapshot_id)
        if maps is None:
            return None
        return maps.canonical_by_alias.get(symbol_id)

    def get_aliases(self, snapshot_id: str, canonical_symbol_id: str) -> list[str]:
        maps = self._by_snapshot.get(snapshot_id)
        if maps is None:
            return []
        return sorted(maps.aliases_by_canonical.get(canonical_symbol_id, set()))

    def resolve_symbol(
        self,
        snapshot_id: str,
        *,
        symbol_id: str | None = None,
        name: str | None = None,
        path: str | None = None,
    ) -> str | None:
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
