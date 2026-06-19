"""Effectful key/digest helpers for the code-status pack (use_flow owner).

This module owns the serialization + hashing boundary work that the pure
meaning_core pack builder (`fastcode.ir.code_status.build_code_status_pack`)
must not perform: `import json`, `import hashlib`, and the
serde-equivalent `safe_jsonable` normalizer. The callables here reproduce the
historical pack algorithm exactly, so pack output is byte-for-byte unchanged.

`build_code_status_pack` is pure by accepting these as injected callables;
`StoreFacade.get_code_status_pack` wires them in via `default_code_status_keys`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastcode.utils.json import safe_jsonable


def _stable_json(payload: Any) -> str:
    return json.dumps(
        safe_jsonable(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def default_normalize(value: Any) -> Any:
    """JSON-safe normalization (the historical `safe_jsonable` pass)."""
    return safe_jsonable(value)


def default_source_id(path: str) -> str:
    return f"source:{hashlib.sha256(path.encode()).hexdigest()[:16]}"


def default_span_id(
    owner_kind: str, owner_id: str, path: str, start: Any, end: Any
) -> str:
    payload = {
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "path": path,
        "start_line": start,
        "end_line": end,
    }
    return f"span:{hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:20]}"


def default_digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(value).encode()).hexdigest()}"


def default_code_status_keys() -> dict[str, Any]:
    """Bundle of injected callables for `build_code_status_pack`."""
    return {
        "normalize": default_normalize,
        "source_id_fn": default_source_id,
        "span_id_fn": default_span_id,
        "digest_fn": default_digest,
    }
