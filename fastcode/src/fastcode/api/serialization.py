"""Explicit API serializers for stable response payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from fastcode.api.outbound import (
    ApiStatus,
    DiagnosticBundleRecord,
    DiagnosticBundleResponse,
    IndexRunResponse,
    IndexRunResponseRecord,
    NewSessionRecord,
    NewSessionResponse,
    OpenMappingRecord,
    QueryResponse,
    QueryResponseRecord,
    QuerySourceDTO,
    QuerySourceRecord,
    ResolverDiagnosticDTO,
    ResolverDiagnosticRecord,
    StatusResponse,
    StatusResponseRecord,
)


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


def _mapping_items(value: Any) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return ()
    items: list[Mapping[str, Any]] = []
    for item in cast(Sequence[Any], value):
        if isinstance(item, Mapping):
            items.append(cast(Mapping[str, Any], item))
    return items


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in cast(Mapping[Any, Any], value).items()}


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


def serialize_query_source_dto(source: QuerySourceRecord) -> QuerySourceDTO:
    return QuerySourceDTO(
        repository=source.repository,
        repo=source.repository,
        file=source.file,
        name=source.name,
        type=source.source_type,
        lines=source.lines,
        start_line=source.start_line,
        end_line=source.end_line,
        score=source.score,
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


def serialize_query_source_dtos(
    sources: Sequence[Any] | None,
) -> list[QuerySourceDTO]:
    return [
        serialize_query_source_dto(serialize_query_source_record(source))
        for source in _sequence_items(sources)
    ]


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


def serialize_open_mapping_record(value: Any) -> OpenMappingRecord:
    return OpenMappingRecord(payload=_mapping_or_empty(value))


def serialize_open_mapping_payload(record: OpenMappingRecord) -> dict[str, Any]:
    return dict(record.payload)


def serialize_open_mapping_payloads(
    values: Sequence[Any] | None,
) -> list[dict[str, Any]]:
    return [
        serialize_open_mapping_payload(serialize_open_mapping_record(value))
        for value in _sequence_items(values)
    ]


def serialize_resolver_diagnostic_record(value: Any) -> ResolverDiagnosticRecord:
    mapping = _mapping_or_empty(value)
    return ResolverDiagnosticRecord(
        name=_string_or_empty(mapping.get("name")),
        source=_string_or_empty(mapping.get("source")),
        status=_string_or_empty(mapping.get("status")),
        reason=(
            _string_or_empty(mapping.get("reason"))
            if mapping.get("reason") is not None
            else None
        ),
        warnings=tuple(
            _string_or_empty(item) for item in _sequence_items(mapping.get("warnings"))
        ),
        metrics=_mapping_or_empty(mapping.get("metrics")),
    )


def serialize_resolver_diagnostic_dto(
    record: ResolverDiagnosticRecord,
) -> ResolverDiagnosticDTO:
    return ResolverDiagnosticDTO(
        name=record.name,
        source=record.source,
        status=record.status,
        reason=record.reason,
        warnings=list(record.warnings),
        metrics=dict(record.metrics),
    )


def _resolver_diagnostic_records(
    pipeline_layers: Sequence[Any] | None,
) -> tuple[ResolverDiagnosticRecord, ...]:
    diagnostics: list[ResolverDiagnosticRecord] = []
    for layer in _mapping_items(pipeline_layers):
        name = _string_or_empty(layer.get("name"))
        source = _string_or_empty(layer.get("source"))
        description = _string_or_empty(layer.get("description"))
        searchable = f"{name} {source} {description}".lower()
        if "resolver" not in searchable and "semantic" not in searchable:
            continue
        diagnostics.append(serialize_resolver_diagnostic_record(layer))
    return tuple(diagnostics)


def serialize_index_run_response_record(result: Any) -> IndexRunResponseRecord:
    result_payload = _mapping_or_empty(result)
    pipeline_layers = tuple(
        serialize_open_mapping_record(item)
        for item in _mapping_items(result_payload.get("pipeline_layers"))
    )
    return IndexRunResponseRecord(
        status=ApiStatus.SUCCESS,
        result=result_payload,
        index_status=(
            _string_or_empty(result_payload.get("status"))
            if result_payload.get("status") is not None
            else None
        ),
        run_id=(
            _string_or_empty(result_payload.get("run_id"))
            if result_payload.get("run_id") is not None
            else None
        ),
        repo_name=(
            _string_or_empty(result_payload.get("repo_name"))
            if result_payload.get("repo_name") is not None
            else None
        ),
        snapshot_id=(
            _string_or_empty(result_payload.get("snapshot_id"))
            if result_payload.get("snapshot_id") is not None
            else None
        ),
        artifact_key=(
            _string_or_empty(result_payload.get("artifact_key"))
            if result_payload.get("artifact_key") is not None
            else None
        ),
        warnings=tuple(
            _string_or_empty(item) for item in _sequence_items(result_payload.get("warnings"))
        ),
        pipeline_layers=pipeline_layers,
        pipeline_metrics=_mapping_or_empty(result_payload.get("pipeline_metrics")),
        resolver_diagnostics=_resolver_diagnostic_records(
            result_payload.get("pipeline_layers")
        ),
    )


def serialize_index_run_response(
    record: IndexRunResponseRecord,
) -> IndexRunResponse:
    return IndexRunResponse(
        status=record.status,
        result=dict(record.result),
        index_status=record.index_status,
        run_id=record.run_id,
        repo_name=record.repo_name,
        snapshot_id=record.snapshot_id,
        artifact_key=record.artifact_key,
        warnings=list(record.warnings),
        pipeline_layers=[
            serialize_open_mapping_payload(layer) for layer in record.pipeline_layers
        ],
        pipeline_metrics=dict(record.pipeline_metrics),
        resolver_diagnostics=[
            serialize_resolver_diagnostic_dto(item)
            for item in record.resolver_diagnostics
        ],
    )


def serialize_status_response_record(
    *,
    status: ApiStatus | str,
    repo_loaded: bool,
    repo_indexed: bool,
    repo_info: Any,
    graph_expansion_backend: str | None = None,
    storage_backend: str | None = None,
    retrieval_backend: str | None = None,
    available_repositories: Sequence[Any] | None = None,
    loaded_repositories: Sequence[Any] | None = None,
) -> StatusResponseRecord:
    return StatusResponseRecord(
        status=ApiStatus(status),
        repo_loaded=repo_loaded,
        repo_indexed=repo_indexed,
        repo_info=_mapping_or_empty(repo_info),
        graph_expansion_backend=graph_expansion_backend,
        storage_backend=storage_backend,
        retrieval_backend=retrieval_backend,
        available_repositories=tuple(
            serialize_open_mapping_record(item)
            for item in _mapping_items(available_repositories)
        ),
        loaded_repositories=tuple(
            serialize_open_mapping_record(item)
            for item in _mapping_items(loaded_repositories)
        ),
    )


def serialize_status_response(record: StatusResponseRecord) -> StatusResponse:
    return StatusResponse(
        status=record.status,
        repo_loaded=record.repo_loaded,
        repo_indexed=record.repo_indexed,
        repo_info=dict(record.repo_info),
        graph_expansion_backend=record.graph_expansion_backend,
        storage_backend=record.storage_backend,
        retrieval_backend=record.retrieval_backend,
        available_repositories=[
            serialize_open_mapping_payload(item) for item in record.available_repositories
        ],
        loaded_repositories=[
            serialize_open_mapping_payload(item) for item in record.loaded_repositories
        ],
    )


def serialize_diagnostic_bundle_record(bundle: Any) -> DiagnosticBundleRecord:
    return DiagnosticBundleRecord(
        status=ApiStatus.SUCCESS,
        bundle=_mapping_or_empty(bundle),
    )


def serialize_diagnostic_bundle_response(
    record: DiagnosticBundleRecord,
) -> DiagnosticBundleResponse:
    return DiagnosticBundleResponse(status=record.status, bundle=dict(record.bundle))


def serialize_query_response_record(
    result: Any,
    *,
    session_id: str | None,
) -> QueryResponseRecord:
    payload = _mapping_or_empty(result)
    prompt_tokens = _int_or_none(payload.get("prompt_tokens"))
    completion_tokens = _int_or_none(payload.get("completion_tokens"))
    total_tokens = _int_or_none(payload.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return QueryResponseRecord(
        answer=_string_or_empty(payload.get("answer")),
        query=_string_or_empty(payload.get("query")),
        context_elements=_int_or_none(payload.get("context_elements")) or 0,
        sources=tuple(
            serialize_query_source_record(source)
            for source in _sequence_items(payload.get("sources"))
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        session_id=session_id,
        turn_number=_int_or_none(payload.get("turn_number")),
    )


def serialize_query_response(record: QueryResponseRecord) -> QueryResponse:
    return QueryResponse(
        answer=record.answer,
        query=record.query,
        context_elements=record.context_elements,
        sources=[serialize_query_source_dto(source) for source in record.sources],
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        session_id=record.session_id,
        turn_number=record.turn_number,
    )


def serialize_new_session_response(record: NewSessionRecord) -> NewSessionResponse:
    return NewSessionResponse(session_id=record.session_id)
