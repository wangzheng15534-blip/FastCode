"""Runtime counters for explicit materialization boundaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

BOUNDARY_JSON_ENCODE = "json_encode"
BOUNDARY_JSON_DECODE = "json_decode"
BOUNDARY_PICKLE_DUMP = "pickle_dump"
BOUNDARY_PICKLE_LOAD = "pickle_load"
BOUNDARY_NETWORKX_CONVERSION = "networkx_conversion"
BOUNDARY_VECTOR_LIST_CONVERSION = "vector_list_conversion"
BOUNDARY_SNAPSHOT_FULL_LOAD = "snapshot_full_load"
BOUNDARY_GRAPH_FULL_LOAD = "graph_full_load"
BOUNDARY_SEMANTIC_PATCH_PRESERVED_OBJECTS = "semantic_patch_preserved_objects"
BOUNDARY_SEMANTIC_PATCH_CHANGED_OBJECTS = "semantic_patch_changed_objects"


def _counter() -> Counter[str]:
    return Counter()


@dataclass
class MaterializationCounters:
    """Count explicit object/native materialization boundaries in one operation."""

    counts: Counter[str] = field(default_factory=_counter)
    items: Counter[str] = field(default_factory=_counter)
    bytes_count: Counter[str] = field(default_factory=_counter)

    def increment(
        self,
        boundary: str,
        *,
        items: int = 0,
        bytes_count: int = 0,
    ) -> None:
        self.counts[boundary] += 1
        if items > 0:
            self.items[boundary] += int(items)
        if bytes_count > 0:
            self.bytes_count[boundary] += int(bytes_count)

    def as_metrics(self) -> dict[str, dict[str, int]]:
        return {
            "materialization_boundary_counts": dict(sorted(self.counts.items())),
            "materialization_boundary_items": dict(sorted(self.items.items())),
            "materialization_boundary_bytes": dict(sorted(self.bytes_count.items())),
        }


_CURRENT_COUNTERS: ContextVar[MaterializationCounters | None] = ContextVar(
    "fastcode_materialization_counters",
    default=None,
)


def set_materialization_counters(
    counters: MaterializationCounters | None,
) -> Token[MaterializationCounters | None]:
    return _CURRENT_COUNTERS.set(counters)


def reset_materialization_counters(
    token: Token[MaterializationCounters | None],
) -> None:
    _CURRENT_COUNTERS.reset(token)


def current_materialization_counters() -> MaterializationCounters | None:
    return _CURRENT_COUNTERS.get()


def increment_materialization_boundary(
    boundary: str,
    *,
    items: int = 0,
    bytes_count: int = 0,
) -> None:
    counters = current_materialization_counters()
    if counters is not None:
        counters.increment(boundary, items=items, bytes_count=bytes_count)


@contextmanager
def collect_materialization_counters() -> Iterator[MaterializationCounters]:
    counters = MaterializationCounters()
    token = set_materialization_counters(counters)
    try:
        yield counters
    finally:
        reset_materialization_counters(token)
