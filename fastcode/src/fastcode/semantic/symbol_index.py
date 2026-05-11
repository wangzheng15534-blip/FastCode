"""
Snapshot-scoped canonical unit index and anchor resolver.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..ir.types import IRSnapshot


@dataclass
class SnapshotSymbolMaps:
    canonical_by_alias: dict[str, str] = field(default_factory=dict)
    aliases_by_canonical: dict[str, set[str]] = field(default_factory=dict)
    symbols_by_name: dict[str, set[str]] = field(default_factory=dict)
    symbols_by_path: dict[str, set[str]] = field(default_factory=dict)


class SnapshotSymbolIndex:
    def __init__(self) -> None:
        self._by_snapshot: dict[str, SnapshotSymbolMaps] = {}

    @staticmethod
    def _sequence_items(value: Any) -> Sequence[Any]:
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            return value
        if isinstance(value, (set, frozenset)):
            return tuple(value)
        return ()

    @classmethod
    def _string_list(cls, value: Any) -> list[str]:
        return [str(item) for item in cls._sequence_items(value) if item]

    @classmethod
    def _register_symbol_item(
        cls,
        maps: SnapshotSymbolMaps,
        *,
        canonical: str,
        aliases: Any,
        names: Any,
        path: Any,
    ) -> None:
        if not canonical:
            return
        alias_ids = {canonical}
        alias_ids.update(cls._string_list(aliases))
        for alias in alias_ids:
            if not alias:
                continue
            maps.canonical_by_alias[alias] = canonical
            maps.aliases_by_canonical.setdefault(canonical, set()).add(alias)

        for name in cls._string_list(names):
            maps.symbols_by_name.setdefault(name, set()).add(canonical)
        if path:
            maps.symbols_by_path.setdefault(str(path), set()).add(canonical)

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
            _md_aliases = (unit.metadata or {}).get("aliases", [])
            if isinstance(_md_aliases, list):
                aliases.update(_md_aliases)

            self._register_symbol_item(
                maps,
                canonical=canonical,
                aliases=aliases,
                names=[unit.display_name, unit.qualified_name],
                path=unit.path,
            )

        self._by_snapshot[snapshot.snapshot_id] = maps

    def register_snapshot_symbol_payload(self, payload: Mapping[str, Any]) -> bool:
        snapshot_id = str(payload.get("snapshot_id") or "")
        if not snapshot_id:
            return False

        maps = SnapshotSymbolMaps()
        for item in self._sequence_items(payload.get("symbols")):
            if not isinstance(item, Mapping):
                continue
            canonical = str(item.get("canonical") or "")
            self._register_symbol_item(
                maps,
                canonical=canonical,
                aliases=item.get("aliases"),
                names=item.get("names"),
                path=item.get("path"),
            )

        self._by_snapshot[snapshot_id] = maps
        return True

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
