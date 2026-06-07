"""ContextFacade — session/context logic extracted from FastCode.

Owns all session, working-memory, handoff, context-bundle, and activation
operations.  Wraps CacheManager and delegates persistence; uses pure
payload builders from context_payloads and context_compiler.
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


class ContextFacade:
    """Aggregated session/context API for entry frames."""

    def __init__(self, cache_manager: CacheManager) -> None:
        self._cache_manager = cache_manager

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_session_history(self, session_id: str) -> list[Any]:
        """Get dialogue history for a session."""
        return self._cache_manager.get_dialogue_history_records(session_id)

    def get_turn_context(
        self,
        session_id: str,
        turn_number: int | None = None,
        output_format: str = "fcx",
    ) -> dict[str, Any]:
        """Fetch the typed working-memory artifact for a session turn."""
        record = (
            self._cache_manager.get_latest_working_memory_record(session_id)
            if turn_number is None
            else self._cache_manager.get_working_memory_record(session_id, turn_number)
        )
        if record is None:
            msg = (
                f"working memory not found for session={session_id}, turn={turn_number}"
            )
            raise RuntimeError(msg)
        artifact = self._parse_working_memory_record_payload(record)
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
        self,
        session_id: str,
        turn_number: int | None = None,
        mode: str = "delegate",
    ) -> dict[str, Any]:
        """Create and persist a handoff artifact from a session turn."""
        record = (
            self._cache_manager.get_latest_working_memory_record(session_id)
            if turn_number is None
            else self._cache_manager.get_working_memory_record(session_id, turn_number)
        )
        if record is None:
            msg = (
                f"working memory not found for session={session_id}, turn={turn_number}"
            )
            raise RuntimeError(msg)
        artifact = build_handoff_from_working_memory(
            working_memory=self._parse_working_memory_record_payload(record),
            mode=mode,
        )
        payload_json = json.dumps(
            handoff_payload(artifact),
            separators=(",", ":"),
            sort_keys=True,
        )
        self._cache_manager.save_handoff_artifact_record(
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

    def get_handoff_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Retrieve a persisted handoff artifact by ID."""
        record = self._cache_manager.get_handoff_artifact_record(artifact_id)
        if record is None:
            msg = f"handoff artifact not found: {artifact_id}"
            raise RuntimeError(msg)
        return handoff_payload(self._parse_handoff_record_payload(record))

    def get_context_bundle(
        self,
        session_id: str,
        turn_number: int | None = None,
        output_format: str = "json",
        token_budget: int = 2048,
    ) -> dict[str, Any]:
        """Fetch a durable context bundle for a session turn."""
        bundle = self._load_context_bundle_artifact(
            session_id=session_id,
            turn_number=turn_number,
        )
        return self._context_bundle_response(
            bundle,
            output_format=output_format,
            token_budget=token_budget,
        )

    def get_context_bundle_by_id(
        self,
        bundle_id: str,
        output_format: str = "json",
        token_budget: int = 2048,
    ) -> dict[str, Any]:
        """Fetch a durable context bundle by bundle ID."""
        bundle = self._load_context_bundle_artifact(bundle_id=bundle_id)
        return self._context_bundle_response(
            bundle,
            output_format=output_format,
            token_budget=token_budget,
        )

    def expand_context_bundle_ref(
        self,
        ref_id: str,
        *,
        session_id: str | None = None,
        turn_number: int | None = None,
        bundle_id: str | None = None,
        depth: str = "L2",
    ) -> dict[str, Any]:
        """Expand a single source ref from a durable context bundle."""
        bundle = self._load_context_bundle_artifact(
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
        self,
        session_id: str | None = None,
        turn_number: int | None = None,
        bundle_id: str | None = None,
        active_ref_ids: Iterable[str] | str | None = None,
        active_fact_ids: Iterable[str] | str | None = None,
        active_hypothesis_ids: Iterable[str] | str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Create and persist an activation record for a context bundle."""
        bundle = self._load_context_bundle_artifact(
            session_id=session_id,
            turn_number=turn_number,
            bundle_id=bundle_id,
        )
        activation = build_activation_record(
            bundle_id=bundle.bundle_id,
            working_memory=bundle.working_memory,
            active_ref_ids=self._optional_string_tuple(active_ref_ids),
            active_fact_ids=self._optional_string_tuple(active_fact_ids),
            active_hypothesis_ids=self._optional_string_tuple(active_hypothesis_ids),
            reason=reason,
            created_at=time.time(),
        )
        payload_json = json.dumps(
            activation_payload(activation),
            separators=(",", ":"),
            sort_keys=True,
        )
        self._cache_manager.save_context_activation_record(
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
        self,
        session_id: str,
        turn_number: int,
        ref_id: str,
        depth: str = "L2",
    ) -> dict[str, Any]:
        """Expand a single evidence ref from working memory."""
        record = self._cache_manager.get_working_memory_record(session_id, turn_number)
        if record is None:
            msg = (
                f"working memory not found for session={session_id}, turn={turn_number}"
            )
            raise RuntimeError(msg)
        artifact = self._parse_working_memory_record_payload(record)
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

    def delete_session(self, session_id: str) -> bool:
        """Delete a dialogue session."""
        return self._cache_manager.delete_session(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all dialogue sessions with enriched metadata."""
        enriched_sessions: list[dict[str, Any]] = []
        for session in self._cache_manager.list_session_records():
            session_id = str(session.session_id)
            title = "Unknown Session"
            if session_id:
                first_turn = self._cache_manager.get_dialogue_turn_record(session_id, 1)
                if first_turn is not None:
                    first_query = str(first_turn.query)
                    title = (
                        first_query[:77] + "..."
                        if len(first_query) > 80
                        else first_query
                    )
                else:
                    title = f"Session {session_id}"
            enriched_sessions.append(
                {
                    "session_id": session_id,
                    "created_at": float(session.created_at),
                    "total_turns": int(session.total_turns),
                    "last_updated": float(session.last_updated),
                    "multi_turn": bool(session.multi_turn),
                    "title": title,
                }
            )

        return enriched_sessions

    def get_session_multi_turn(self, session_id: str) -> bool:
        """Return whether a session is multi-turn."""
        record = self._cache_manager.get_session_index_record(session_id)
        return bool(record.multi_turn) if record is not None else False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_working_memory_record_payload(
        record: Any,
    ) -> WorkingMemoryArtifact:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            msg = "working memory payload is invalid"
            raise RuntimeError(msg)
        return working_memory_from_payload(payload)

    @staticmethod
    def _parse_handoff_record_payload(record: Any) -> HandoffArtifact:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            msg = "handoff artifact payload is invalid"
            raise RuntimeError(msg)
        return handoff_from_payload(payload)

    @staticmethod
    def _parse_turn_journal_record_payload(record: Any) -> TurnJournal:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            msg = "turn journal payload is invalid"
            raise RuntimeError(msg)
        return turn_journal_from_payload(payload)

    @staticmethod
    def _parse_context_bundle_record_payload(record: Any) -> ContextBundle:
        payload = json.loads(str(record.payload_json or "{}"))
        if not isinstance(payload, dict):
            msg = "context bundle payload is invalid"
            raise RuntimeError(msg)
        return context_bundle_from_payload(payload)

    @staticmethod
    def _optional_string_tuple(
        value: Iterable[str] | str | None,
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

    def _load_context_bundle_artifact(
        self,
        *,
        session_id: str | None = None,
        turn_number: int | None = None,
        bundle_id: str | None = None,
    ) -> ContextBundle:
        if bundle_id:
            get_by_id = getattr(
                self._cache_manager, "get_context_bundle_record_by_id", None
            )
            record = get_by_id(bundle_id) if callable(get_by_id) else None
            if record is None:
                msg = f"context bundle not found: {bundle_id}"
                raise RuntimeError(msg)
            return self._parse_context_bundle_record_payload(record)

        if not session_id:
            msg = "session_id is required when bundle_id is omitted"
            raise RuntimeError(msg)

        record = None
        if turn_number is None:
            get_latest = getattr(
                self._cache_manager, "get_latest_context_bundle_record", None
            )
            record = get_latest(session_id) if callable(get_latest) else None
        else:
            get_record = getattr(self._cache_manager, "get_context_bundle_record", None)
            record = (
                get_record(session_id, turn_number) if callable(get_record) else None
            )
        if record is not None:
            return self._parse_context_bundle_record_payload(record)

        working_memory_record = (
            self._cache_manager.get_latest_working_memory_record(session_id)
            if turn_number is None
            else self._cache_manager.get_working_memory_record(session_id, turn_number)
        )
        if working_memory_record is None:
            msg = (
                f"context bundle not found for session={session_id}, turn={turn_number}"
            )
            raise RuntimeError(msg)
        resolved_turn = int(working_memory_record.turn_number)
        journal_record = self._cache_manager.get_turn_journal_record(
            session_id, resolved_turn
        )
        if journal_record is None:
            msg = (
                f"turn journal not found for session={session_id}, turn={resolved_turn}"
            )
            raise RuntimeError(msg)
        return build_context_bundle(
            working_memory=self._parse_working_memory_record_payload(
                working_memory_record
            ),
            turn_journal=self._parse_turn_journal_record_payload(journal_record),
        )

    @staticmethod
    def _context_bundle_response(
        bundle: ContextBundle,
        *,
        output_format: str,
        token_budget: int,
    ) -> dict[str, Any]:
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
