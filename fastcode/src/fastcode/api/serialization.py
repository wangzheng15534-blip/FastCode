"""Explicit API serializers for stable response payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from fastcode.schemas.core_types import QuerySourceRecord


def _field_get(record: Any, field_name: str) -> Any:
    if isinstance(record, Mapping):
        return cast(Mapping[str, Any], record).get(field_name)
    return getattr(record, field_name, None)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, *, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sequence_items(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return cast(Sequence[Any], value)
    return ()


def _line_bounds(source: Any) -> tuple[int, int, str]:
    start_line = _int_or_none(_field_get(source, "start_line"))
    end_line = _int_or_none(_field_get(source, "end_line"))
    lines = _string_or_empty(_field_get(source, "lines"))

    if (start_line is None or end_line is None) and lines:
        start_text, _, end_text = lines.partition("-")
        if start_line is None:
            start_line = _int_or_none(start_text)
        if end_line is None and end_text:
            end_line = _int_or_none(end_text)
        if end_line is None:
            end_line = start_line

    start_line = start_line or 0
    end_line = end_line or 0
    normalized_lines = lines or f"{start_line}-{end_line}"
    if lines and "-" not in lines and start_line and end_line:
        normalized_lines = f"{start_line}-{end_line}"
    return start_line, end_line, normalized_lines


def serialize_query_source_record(source: Any) -> QuerySourceRecord:
    repo_name = _string_or_empty(
        _field_get(source, "repository")
        or _field_get(source, "repo")
        or _field_get(source, "repo_name")
    )
    start_line, end_line, lines = _line_bounds(source)
    return QuerySourceRecord(
        repository=repo_name,
        file=_string_or_empty(
            _field_get(source, "file")
            or _field_get(source, "relative_path")
            or _field_get(source, "path")
        ),
        name=_string_or_empty(_field_get(source, "name")),
        source_type=_string_or_empty(_field_get(source, "type")),
        lines=lines,
        start_line=start_line,
        end_line=end_line,
        score=_float_or_default(_field_get(source, "score")),
    )


def serialize_query_source_record_payload(
    source: QuerySourceRecord,
) -> dict[str, Any]:
    """Materialize the stable API source payload field-by-field."""
    return {
        "repository": source.repository,
        "repo": source.repository,
        "file": source.file,
        "name": source.name,
        "type": source.source_type,
        "lines": source.lines,
        "start_line": source.start_line,
        "end_line": source.end_line,
        "score": source.score,
    }


def serialize_query_source(source: Any) -> dict[str, Any]:
    return serialize_query_source_record_payload(serialize_query_source_record(source))


def serialize_query_sources(sources: Sequence[Any] | None) -> list[dict[str, Any]]:
    return [serialize_query_source(source) for source in _sequence_items(sources)]


def serialize_dialogue_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    mapping = cast(Mapping[str, Any], metadata)
    return {
        "intent": _string_or_empty(mapping.get("intent")),
        "keywords": [
            _string_or_empty(item) for item in _sequence_items(mapping.get("keywords"))
        ],
        "repo_filter": [
            _string_or_empty(item)
            for item in _sequence_items(mapping.get("repo_filter"))
        ],
        "multi_turn": bool(mapping.get("multi_turn", False)),
    }


def serialize_dialogue_turn(turn: Any) -> dict[str, Any]:
    return {
        "session_id": _string_or_empty(_field_get(turn, "session_id")),
        "turn_number": _int_or_none(_field_get(turn, "turn_number")) or 0,
        "timestamp": _float_or_default(_field_get(turn, "timestamp")),
        "query": _string_or_empty(_field_get(turn, "query")),
        "answer": _string_or_empty(_field_get(turn, "answer")),
        "summary": _string_or_empty(_field_get(turn, "summary")),
        "retrieved_elements": serialize_query_sources(
            _field_get(turn, "retrieved_elements")
        ),
        "metadata": serialize_dialogue_metadata(_field_get(turn, "metadata")),
    }


def serialize_dialogue_history(history: Sequence[Any] | None) -> list[dict[str, Any]]:
    return [serialize_dialogue_turn(turn) for turn in _sequence_items(history)]
