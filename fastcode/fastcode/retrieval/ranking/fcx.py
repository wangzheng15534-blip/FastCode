"""FastCode Context (FCX) render and parse helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return "-"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[Any], value)
        return ",".join(_render_value(item) for item in sequence if item is not None)
    text = str(value)
    if not text:
        return "-"
    if any(ch.isspace() for ch in text) or "|" in text:
        return json.dumps(text, ensure_ascii=True)
    return text


def render_record(
    tag: str,
    record_id: str | None = None,
    fields: dict[str, Any] | None = None,
    tail: str | None = None,
) -> str:
    parts = [tag]
    if record_id:
        parts.append(record_id)
    for key, value in (fields or {}).items():
        parts.append(f"{key}={_render_value(value)}")
    line = " ".join(parts)
    if tail:
        return f"{line} | {tail}"
    return line


def render_block(
    *,
    mode: str,
    header_fields: dict[str, Any],
    records: list[str],
) -> str:
    header = render_record("@fcx", fields={"mode": mode, **header_fields})
    body = "\n".join(records)
    return f"{header}\n{body}" if body else header


def parse_block(text: str) -> dict[str, Any]:
    header: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        msg = "empty FCX block"
        raise ValueError(msg)

    for index, raw_line in enumerate(lines):
        left, separator, tail = raw_line.partition("|")
        tokens = left.strip().split()
        if not tokens:
            continue
        tag = tokens[0]
        record_id: str | None = None
        fields: dict[str, Any] = {}
        for token in tokens[1:]:
            if "=" not in token and record_id is None:
                record_id = token
                continue
            key, value = token.split("=", 1)
            fields[key] = _parse_value(value)

        if index == 0:
            if tag != "@fcx":
                msg = "FCX block must start with @fcx header"
                raise ValueError(msg)
            header = fields
            continue

        if record_id:
            if record_id in seen_ids:
                msg = f"duplicate FCX record id: {record_id}"
                raise ValueError(msg)
            seen_ids.add(record_id)

        record = {"tag": tag, "id": record_id, "fields": fields}
        if separator:
            record["tail"] = tail.strip()
        records.append(record)

    if not any(record.get("tag") == "END" for record in records):
        msg = "FCX block missing END record"
        raise ValueError(msg)
    return {"header": header, "records": records}


def _parse_value(value: str) -> Any:
    if value == "-":
        return None
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    if "," in value:
        return [item for item in value.split(",") if item]
    if value in {"0", "1"}:
        return value == "1"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
