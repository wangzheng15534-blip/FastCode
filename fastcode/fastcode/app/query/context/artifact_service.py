"""Context artifact CRUD: loading, parsing, and serializing context bundles.

Moved from main/fastcode.py (assembly_root) to use_flow (app/query)
because context artifact operations are query workflow logic,
not composition root wiring.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

from fastcode.app.query.context_payloads import (
    activation_payload,
    context_bundle_from_payload,
    context_bundle_payload,
    handoff_from_payload,
    handoff_payload,
    turn_journal_from_payload,
    working_memory_from_payload,
    working_memory_payload,
)
from fastcode.app.store.cache.contracts import (
    ContextActivationRecord,
    HandoffArtifactRecord,
)
from fastcode.app.store.cache.service import CacheManager
from fastcode.retrieval.context.agent_context import (
    ContextBundle,
    HandoffArtifact,
    TurnJournal,
    WorkingMemoryArtifact,
)
from fastcode.retrieval.context.context_compiler import (
    build_activation_record,
    build_context_bundle,
    build_handoff_from_working_memory,
    expand_bundle_source_ref,
    render_context_bundle,
)


def _parse_working_memory_payload(record: Any) -> WorkingMemoryArtifact:
    payload = json.loads(str(record.payload_json or "{}"))
    if not isinstance(payload, dict):
        msg = "working memory payload is invalid"
        raise RuntimeError(msg)
    return working_memory_from_payload(payload)


def _parse_handoff_payload(record: Any) -> HandoffArtifact:
    payload = json.loads(str(record.payload_json or "{}"))
    if not isinstance(payload, dict):
        msg = "handoff artifact payload is invalid"
        raise RuntimeError(msg)
    return handoff_from_payload(payload)


def _parse_turn_journal_payload(record: Any) -> TurnJournal:
    payload = json.loads(str(record.payload_json or "{}"))
    if not isinstance(payload, dict):
        msg = "turn journal payload is invalid"
        raise RuntimeError(msg)
    return turn_journal_from_payload(payload)


def _parse_context_bundle_payload(record: Any) -> ContextBundle:
    payload = json.loads(str(record.payload_json or "{}"))
    if not isinstance(payload, dict):
        msg = "context bundle payload is invalid"
        raise RuntimeError(msg)
    return context_bundle_from_payload(payload)


def _optional_string_tuple(
    value: Iterable[str] | str | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_context_bundle(
    cache_manager: CacheManager,
    *,
    session_id: str | None = None,
    turn_number: int | None = None,
    bundle_id: str | None = None,
) -> ContextBundle:
    """Load a context bundle from cache by bundle_id, session+turn, or latest."""
    if bundle_id:
        get_by_id = getattr(cache_manager, "get_context_bundle_record_by_id", None)
        record = get_by_id(bundle_id) if callable(get_by_id) else None
        if record is None:
            msg = f"context bundle not found: {bundle_id}"
            raise RuntimeError(msg)
        return _parse_context_bundle_payload(record)

    if not session_id:
        msg = "session_id is required when bundle_id is omitted"
        raise RuntimeError(msg)

    record = None
    if turn_number is None:
        get_latest = getattr(cache_manager, "get_latest_context_bundle_record", None)
        record = get_latest(session_id) if callable(get_latest) else None
    else:
        get_record = getattr(cache_manager, "get_context_bundle_record", None)
        record = get_record(session_id, turn_number) if callable(get_record) else None
    if record is not None:
        return _parse_context_bundle_payload(record)

    working_memory_record = (
        cache_manager.get_latest_working_memory_record(session_id)
        if turn_number is None
        else cache_manager.get_working_memory_record(session_id, turn_number)
    )
    if working_memory_record is None:
        msg = f"context bundle not found for session={session_id}, turn={turn_number}"
        raise RuntimeError(msg)
    resolved_turn = int(working_memory_record.turn_number)
    journal_record = cache_manager.get_turn_journal_record(session_id, resolved_turn)
    if journal_record is None:
        msg = f"turn journal not found for session={session_id}, turn={resolved_turn}"
        raise RuntimeError(msg)
    return build_context_bundle(
        working_memory=_parse_working_memory_payload(working_memory_record),
        turn_journal=_parse_turn_journal_payload(journal_record),
    )


def context_bundle_response(
    bundle: ContextBundle,
    *,
    output_format: str,
    token_budget: int,
) -> dict[str, Any]:
    """Build an API response dict from a ContextBundle."""
    response: dict[str, Any] = {
        "bundle_id": bundle.bundle_id,
        "session_id": bundle.session_id,
        "turn_number": bundle.turn_number,
        "snapshot_id": bundle.snapshot_id,
        "artifact_key": bundle.artifact_key,
        "compiler_fingerprint": bundle.compiler_fingerprint,
        "format": output_format,
        "invalidation_key": bundle.distillation.invalidation_key,
        "activation_id": bundle.activation.activation_id,
        "distillation_id": bundle.distillation.distillation_id,
    }
    if output_format == "json":
        response["bundle"] = context_bundle_payload(bundle)
        return response
    if output_format == "rendered":
        response["rendered"] = render_context_bundle(
            bundle,
            token_budget=token_budget,
        )
        return response
    msg = "format must be one of: json, rendered"
    raise RuntimeError(msg)


def get_turn_context(
    cache_manager: CacheManager,
    session_id: str,
    turn_number: int | None = None,
    output_format: str = "fcx",
) -> dict[str, Any]:
    """Get working memory context for a turn."""
    record = (
        cache_manager.get_latest_working_memory_record(session_id)
        if turn_number is None
        else cache_manager.get_working_memory_record(session_id, turn_number)
    )
    if record is None:
        msg = f"working memory not found for session={session_id}, turn={turn_number}"
        raise RuntimeError(msg)
    artifact = _parse_working_memory_payload(record)
    response: dict[str, Any] = {
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "compiler_fingerprint": record.compiler_fingerprint,
        "format": output_format,
    }
    if output_format == "fcx":
        response["stable_fcx"] = record.stable_fcx
        response["turn_fcx"] = record.turn_fcx
        response["obs_fcx"] = record.obs_fcx
        response["full_fcx"] = record.full_fcx
        return response
    if output_format == "json":
        response["artifact"] = working_memory_payload(artifact)
        return response
    msg = "format must be one of: fcx, json"
    raise RuntimeError(msg)


def create_handoff(
    cache_manager: CacheManager,
    session_id: str,
    turn_number: int | None = None,
    mode: str = "delegate",
) -> dict[str, Any]:
    """Create a handoff artifact from working memory."""
    record = (
        cache_manager.get_latest_working_memory_record(session_id)
        if turn_number is None
        else cache_manager.get_working_memory_record(session_id, turn_number)
    )
    if record is None:
        msg = f"working memory not found for session={session_id}, turn={turn_number}"
        raise RuntimeError(msg)
    artifact = build_handoff_from_working_memory(
        working_memory=_parse_working_memory_payload(record),
        mode=mode,
    )
    payload_json = json.dumps(
        handoff_payload(artifact),
        separators=(",", ":"),
        sort_keys=True,
    )
    cache_manager.save_handoff_artifact_record(
        HandoffArtifactRecord(
            artifact_id=artifact.artifact_id,
            session_id=artifact.session_id,
            turn_number=artifact.turn_number,
            snapshot_id=artifact.snapshot_id,
            compiler_fingerprint=artifact.compiler_fingerprint,
            mode=artifact.mode,
            payload_json=payload_json,
            full_fcx=artifact.full_fcx,
            created_at=artifact.created_at,
        )
    )
    return handoff_payload(artifact)


def get_handoff_artifact(
    cache_manager: CacheManager,
    artifact_id: str,
) -> dict[str, Any]:
    """Retrieve a handoff artifact by ID."""
    record = cache_manager.get_handoff_artifact_record(artifact_id)
    if record is None:
        msg = f"handoff artifact not found: {artifact_id}"
        raise RuntimeError(msg)
    return handoff_payload(_parse_handoff_payload(record))


def get_context_bundle(
    cache_manager: CacheManager,
    session_id: str,
    turn_number: int | None = None,
    output_format: str = "json",
    token_budget: int = 2048,
) -> dict[str, Any]:
    """Get a context bundle by session and optional turn."""
    bundle = load_context_bundle(
        cache_manager,
        session_id=session_id,
        turn_number=turn_number,
    )
    return context_bundle_response(
        bundle, output_format=output_format, token_budget=token_budget
    )


def get_context_bundle_by_id(
    cache_manager: CacheManager,
    bundle_id: str,
    output_format: str = "json",
    token_budget: int = 2048,
) -> dict[str, Any]:
    """Get a context bundle by its ID."""
    bundle = load_context_bundle(cache_manager, bundle_id=bundle_id)
    return context_bundle_response(
        bundle, output_format=output_format, token_budget=token_budget
    )


def expand_context_bundle_ref(
    cache_manager: CacheManager,
    ref_id: str,
    *,
    session_id: str | None = None,
    turn_number: int | None = None,
    bundle_id: str | None = None,
    depth: str = "L2",
) -> dict[str, Any]:
    """Expand a reference within a context bundle."""
    bundle = load_context_bundle(
        cache_manager,
        session_id=session_id,
        turn_number=turn_number,
        bundle_id=bundle_id,
    )
    expanded = expand_bundle_source_ref(bundle, ref_id, depth=depth)
    if expanded is None:
        msg = f"context bundle ref not found: {ref_id}"
        raise RuntimeError(msg)
    return expanded


def create_context_activation(
    cache_manager: CacheManager,
    *,
    session_id: str | None = None,
    turn_number: int | None = None,
    bundle_id: str | None = None,
    active_ref_ids: Iterable[str] | str | None = None,
    active_fact_ids: Iterable[str] | str | None = None,
    active_hypothesis_ids: Iterable[str] | str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Create a context activation record."""
    bundle = load_context_bundle(
        cache_manager,
        session_id=session_id,
        turn_number=turn_number,
        bundle_id=bundle_id,
    )
    activation = build_activation_record(
        bundle_id=bundle.bundle_id,
        working_memory=bundle.working_memory,
        active_ref_ids=_optional_string_tuple(active_ref_ids),
        active_fact_ids=_optional_string_tuple(active_fact_ids),
        active_hypothesis_ids=_optional_string_tuple(active_hypothesis_ids),
        reason=reason,
        created_at=time.time(),
    )
    payload_json = json.dumps(
        activation_payload(activation),
        separators=(",", ":"),
        sort_keys=True,
    )
    cache_manager.save_context_activation_record(
        ContextActivationRecord(
            activation_id=activation.activation_id,
            bundle_id=activation.bundle_id,
            session_id=activation.session_id,
            turn_number=activation.turn_number,
            snapshot_id=activation.snapshot_id,
            compiler_fingerprint=activation.compiler_fingerprint,
            active_ref_ids=activation.active_ref_ids,
            active_fact_ids=activation.active_fact_ids,
            active_hypothesis_ids=activation.active_hypothesis_ids,
            reason=activation.reason,
            payload_json=payload_json,
            created_at=activation.created_at,
        )
    )
    return activation_payload(activation)


def expand_context_ref(
    cache_manager: CacheManager,
    session_id: str,
    turn_number: int,
    ref_id: str,
    depth: str = "L2",
) -> dict[str, Any]:
    """Expand a single context reference from working memory."""
    record = cache_manager.get_working_memory_record(session_id, turn_number)
    if record is None:
        msg = f"working memory not found for session={session_id}, turn={turn_number}"
        raise RuntimeError(msg)
    artifact = _parse_working_memory_payload(record)
    for ref in artifact.evidence_refs:
        if ref.ref_id != ref_id:
            continue
        return {
            "session_id": session_id,
            "turn_number": turn_number,
            "depth": depth,
            "ref_id": ref.ref_id,
            "kind": ref.kind,
            "repo_name": ref.repo_name,
            "snapshot_id": ref.snapshot_id,
            "path": ref.path,
            "symbol_id": ref.symbol_id,
            "lines": ref.lines,
            "label": ref.label,
            "score": ref.score,
            "source": ref.source,
            "fresh": ref.fresh,
        }
    msg = f"context ref not found: {ref_id}"
    raise RuntimeError(msg)
