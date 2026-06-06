"""Domain-independent hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def projection_params_hash(scope_dict: dict[str, Any], version: str = "v1") -> str:
    """Hash projection parameters for cache key."""
    payload = json.dumps(
        {"scope": scope_dict, "projection_algo_version": version},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def deterministic_event_id(snapshot_id: str, payload: str) -> str:
    """Generate a deterministic event ID from snapshot_id + payload hash."""
    h = hashlib.sha256(f"{snapshot_id}:{payload}".encode()).hexdigest()[:32]
    return f"outbox:{snapshot_id}:{h}"
