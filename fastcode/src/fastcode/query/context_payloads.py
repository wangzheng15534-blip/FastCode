"""Explicit query/cache payload mappers for agent-context records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from fastcode.retrieval.agent_context import (
    AcceptanceContract,
    AcceptedFact,
    ActivationRecord,
    ContextBundle,
    DistillationRecord,
    EvidenceRef,
    Hypothesis,
    RejectedHypothesisEntry,
    RiskState,
    ToolObservation,
    TurnIntent,
    TurnJournal,
    TurnPlan,
    WorkingMemoryArtifact,
    WorkingSet,
)


def _string(value: Any) -> str:
    return str(value or "")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in cast(Sequence[Any], value))
    if value is None:
        return ()
    return (str(value),)


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item for key, item in cast(Mapping[Any, Any], value).items()
    }


def _mapping_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    items: list[dict[str, Any]] = []
    for item in cast(Sequence[Any], value):
        if isinstance(item, Mapping):
            items.append(_mapping(item))
    return tuple(items)


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _float_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def evidence_ref_payload(record: EvidenceRef) -> dict[str, Any]:
    return {
        "ref_id": record.ref_id,
        "kind": record.kind,
        "repo_name": record.repo_name,
        "snapshot_id": record.snapshot_id,
        "path": record.path,
        "symbol_id": record.symbol_id,
        "lines": record.lines,
        "label": record.label,
        "score": record.score,
        "source": record.source,
        "fresh": record.fresh,
    }


def evidence_ref_from_payload(payload: Mapping[str, Any]) -> EvidenceRef:
    score_value = _optional_float(payload.get("score"))
    return EvidenceRef(
        ref_id=_string(payload.get("ref_id")),
        kind=_string(payload.get("kind")),
        repo_name=_string_or_none(payload.get("repo_name")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        path=_string_or_none(payload.get("path")),
        symbol_id=_string_or_none(payload.get("symbol_id")),
        lines=_string_or_none(payload.get("lines")),
        label=_string_or_none(payload.get("label")),
        score=score_value,
        source=_string_or_none(payload.get("source")),
        fresh=_string(payload.get("fresh") or "unknown"),
    )


def tool_observation_payload(record: ToolObservation) -> dict[str, Any]:
    return {
        "observation_id": record.observation_id,
        "tool": record.tool,
        "ok": record.ok,
        "parameters": dict(record.parameters),
        "ref_ids": list(record.ref_ids),
        "cost": record.cost,
        "fresh": record.fresh,
        "warnings": list(record.warnings),
        "round_number": record.round_number,
        "summary": record.summary,
    }


def tool_observation_from_payload(payload: Mapping[str, Any]) -> ToolObservation:
    return ToolObservation(
        observation_id=_string(payload.get("observation_id")),
        tool=_string(payload.get("tool")),
        ok=bool(payload.get("ok", False)),
        parameters=_mapping(payload.get("parameters")),
        ref_ids=_string_tuple(payload.get("ref_ids")),
        cost=_int_value(payload.get("cost")),
        fresh=_string(payload.get("fresh") or "unknown"),
        warnings=_string_tuple(payload.get("warnings")),
        round_number=_int_value(payload.get("round_number")),
        summary=_string_or_none(payload.get("summary")),
    )


def hypothesis_payload(record: Hypothesis) -> dict[str, Any]:
    return {
        "hypothesis_id": record.hypothesis_id,
        "statement": record.statement,
        "confidence": record.confidence,
        "support_ref_ids": list(record.support_ref_ids),
        "conflict_ref_ids": list(record.conflict_ref_ids),
        "state": record.state,
    }


def hypothesis_from_payload(payload: Mapping[str, Any]) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=_string(payload.get("hypothesis_id")),
        statement=_string(payload.get("statement")),
        confidence=_float_value(payload.get("confidence")),
        support_ref_ids=_string_tuple(payload.get("support_ref_ids")),
        conflict_ref_ids=_string_tuple(payload.get("conflict_ref_ids")),
        state=_string(payload.get("state") or "open"),
    )


def rejected_hypothesis_payload(record: RejectedHypothesisEntry) -> dict[str, Any]:
    return {
        "entry_id": record.entry_id,
        "hypothesis_id": record.hypothesis_id,
        "killed_by_ref_ids": list(record.killed_by_ref_ids),
        "reason_code": record.reason_code,
        "snapshot_id": record.snapshot_id,
        "reopen_condition": record.reopen_condition,
    }


def rejected_hypothesis_from_payload(
    payload: Mapping[str, Any],
) -> RejectedHypothesisEntry:
    return RejectedHypothesisEntry(
        entry_id=_string(payload.get("entry_id")),
        hypothesis_id=_string(payload.get("hypothesis_id")),
        killed_by_ref_ids=_string_tuple(payload.get("killed_by_ref_ids")),
        reason_code=_string(payload.get("reason_code")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        reopen_condition=_string(payload.get("reopen_condition") or "-"),
    )


def accepted_fact_payload(record: AcceptedFact) -> dict[str, Any]:
    return {
        "fact_id": record.fact_id,
        "statement": record.statement,
        "ref_ids": list(record.ref_ids),
        "scope": record.scope,
    }


def accepted_fact_from_payload(payload: Mapping[str, Any]) -> AcceptedFact:
    return AcceptedFact(
        fact_id=_string(payload.get("fact_id")),
        statement=_string(payload.get("statement")),
        ref_ids=_string_tuple(payload.get("ref_ids")),
        scope=_string(payload.get("scope") or "turn"),
    )


def risk_state_payload(record: RiskState) -> dict[str, Any]:
    return {
        "evidence_gap": record.evidence_gap,
        "conflict_level": record.conflict_level,
        "freshness_risk": record.freshness_risk,
        "requirement_ambiguity": record.requirement_ambiguity,
        "execution_risk": record.execution_risk,
        "verifier_status": record.verifier_status,
        "action_bias": record.action_bias,
    }


def risk_state_from_payload(payload: Mapping[str, Any]) -> RiskState:
    return RiskState(
        evidence_gap=_int_value(payload.get("evidence_gap")),
        conflict_level=_int_value(payload.get("conflict_level")),
        freshness_risk=_int_value(payload.get("freshness_risk")),
        requirement_ambiguity=_int_value(payload.get("requirement_ambiguity")),
        execution_risk=_int_value(payload.get("execution_risk")),
        verifier_status=_string(payload.get("verifier_status") or "pending"),
        action_bias=_string(payload.get("action_bias") or "retrieve"),
    )


def acceptance_contract_payload(record: AcceptanceContract) -> dict[str, Any]:
    return {
        "requested_outcome": record.requested_outcome,
        "required_evidence_kinds": list(record.required_evidence_kinds),
        "required_verifiers": list(record.required_verifiers),
        "allowed_tools": list(record.allowed_tools),
        "allowed_write_scope": list(record.allowed_write_scope),
        "done_condition": record.done_condition,
        "must_ask_before": list(record.must_ask_before),
        "must_abstain_when": list(record.must_abstain_when),
    }


def acceptance_contract_from_payload(
    payload: Mapping[str, Any],
) -> AcceptanceContract:
    return AcceptanceContract(
        requested_outcome=_string(payload.get("requested_outcome")),
        required_evidence_kinds=_string_tuple(
            payload.get("required_evidence_kinds")
        ),
        required_verifiers=_string_tuple(payload.get("required_verifiers")),
        allowed_tools=_string_tuple(payload.get("allowed_tools")),
        allowed_write_scope=_string_tuple(payload.get("allowed_write_scope")),
        done_condition=_string(payload.get("done_condition")),
        must_ask_before=_string_tuple(payload.get("must_ask_before")),
        must_abstain_when=_string_tuple(payload.get("must_abstain_when")),
    )


def turn_intent_payload(record: TurnIntent) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "question": record.question,
        "kind": record.kind,
        "requested_outcome": record.requested_outcome,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "repo_filter": list(record.repo_filter),
    }


def turn_intent_from_payload(payload: Mapping[str, Any]) -> TurnIntent:
    return TurnIntent(
        session_id=_string(payload.get("session_id")),
        turn_number=_int_value(payload.get("turn_number")),
        question=_string(payload.get("question")),
        kind=_string(payload.get("kind")),
        requested_outcome=_string(payload.get("requested_outcome")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        artifact_key=_string_or_none(payload.get("artifact_key")),
        repo_filter=_string_tuple(payload.get("repo_filter")),
    )


def turn_plan_payload(record: TurnPlan) -> dict[str, Any]:
    return {
        "step": record.step,
        "action": record.action,
        "why": record.why,
        "stop_condition": record.stop_condition,
        "allowed_actions": list(record.allowed_actions),
        "allowed_tools": list(record.allowed_tools),
        "remaining_budget": record.remaining_budget,
    }


def turn_plan_from_payload(payload: Mapping[str, Any]) -> TurnPlan:
    remaining_value = payload.get("remaining_budget")
    return TurnPlan(
        step=_int_value(payload.get("step")),
        action=_string(payload.get("action")),
        why=_string(payload.get("why")),
        stop_condition=_string(payload.get("stop_condition")),
        allowed_actions=_string_tuple(payload.get("allowed_actions")),
        allowed_tools=_string_tuple(payload.get("allowed_tools")),
        remaining_budget=remaining_value if isinstance(remaining_value, int) else None,
    )


def working_set_payload(record: WorkingSet) -> dict[str, Any]:
    return {
        "keep_ids": list(record.keep_ids),
        "drop_ids": list(record.drop_ids),
        "protect_ids": list(record.protect_ids),
        "reason": record.reason,
    }


def working_set_from_payload(payload: Mapping[str, Any]) -> WorkingSet:
    return WorkingSet(
        keep_ids=_string_tuple(payload.get("keep_ids")),
        drop_ids=_string_tuple(payload.get("drop_ids")),
        protect_ids=_string_tuple(payload.get("protect_ids")),
        reason=_string(payload.get("reason")),
    )


def working_memory_payload(record: WorkingMemoryArtifact) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "compiler_fingerprint": record.compiler_fingerprint,
        "stable_fcx": record.stable_fcx,
        "turn_fcx": record.turn_fcx,
        "obs_fcx": record.obs_fcx,
        "full_fcx": record.full_fcx,
        "evidence_refs": [
            evidence_ref_payload(item) for item in record.evidence_refs
        ],
        "accepted_facts": [
            accepted_fact_payload(item) for item in record.accepted_facts
        ],
        "hypotheses": [hypothesis_payload(item) for item in record.hypotheses],
        "rejected_hypotheses": [
            rejected_hypothesis_payload(item)
            for item in record.rejected_hypotheses
        ],
        "unresolved_questions": list(record.unresolved_questions),
        "risk_state": risk_state_payload(record.risk_state),
        "acceptance_contract": acceptance_contract_payload(
            record.acceptance_contract
        ),
        "working_set": working_set_payload(record.working_set),
        "created_at": record.created_at,
    }


def working_memory_from_payload(
    payload: Mapping[str, Any],
) -> WorkingMemoryArtifact:
    return WorkingMemoryArtifact(
        session_id=_string(payload.get("session_id")),
        turn_number=_int_value(payload.get("turn_number")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        artifact_key=_string_or_none(payload.get("artifact_key")),
        compiler_fingerprint=_string(payload.get("compiler_fingerprint")),
        stable_fcx=_string(payload.get("stable_fcx")),
        turn_fcx=_string(payload.get("turn_fcx")),
        obs_fcx=_string(payload.get("obs_fcx")),
        full_fcx=_string(payload.get("full_fcx")),
        evidence_refs=tuple(
            evidence_ref_from_payload(item)
            for item in _mapping_tuple(payload.get("evidence_refs"))
        ),
        accepted_facts=tuple(
            accepted_fact_from_payload(item)
            for item in _mapping_tuple(payload.get("accepted_facts"))
        ),
        hypotheses=tuple(
            hypothesis_from_payload(item)
            for item in _mapping_tuple(payload.get("hypotheses"))
        ),
        rejected_hypotheses=tuple(
            rejected_hypothesis_from_payload(item)
            for item in _mapping_tuple(payload.get("rejected_hypotheses"))
        ),
        unresolved_questions=_string_tuple(payload.get("unresolved_questions")),
        risk_state=risk_state_from_payload(_mapping(payload.get("risk_state"))),
        acceptance_contract=acceptance_contract_from_payload(
            _mapping(payload.get("acceptance_contract"))
        ),
        working_set=working_set_from_payload(_mapping(payload.get("working_set"))),
        created_at=_float_value(payload.get("created_at")),
    )


def turn_journal_payload(record: TurnJournal) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "compiler_fingerprint": record.compiler_fingerprint,
        "intent": turn_intent_payload(record.intent),
        "plan": turn_plan_payload(record.plan),
        "observations": [
            tool_observation_payload(item) for item in record.observations
        ],
        "evidence_refs": [
            evidence_ref_payload(item) for item in record.evidence_refs
        ],
        "risk_state": risk_state_payload(record.risk_state),
        "acceptance_contract": acceptance_contract_payload(
            record.acceptance_contract
        ),
        "hypotheses": [hypothesis_payload(item) for item in record.hypotheses],
        "rejected_hypotheses": [
            rejected_hypothesis_payload(item)
            for item in record.rejected_hypotheses
        ],
        "accepted_facts": [
            accepted_fact_payload(item) for item in record.accepted_facts
        ],
        "working_set": working_set_payload(record.working_set),
        "answer_summary": record.answer_summary,
        "created_at": record.created_at,
    }


def turn_journal_from_payload(payload: Mapping[str, Any]) -> TurnJournal:
    return TurnJournal(
        session_id=_string(payload.get("session_id")),
        turn_number=_int_value(payload.get("turn_number")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        artifact_key=_string_or_none(payload.get("artifact_key")),
        compiler_fingerprint=_string(payload.get("compiler_fingerprint")),
        intent=turn_intent_from_payload(_mapping(payload.get("intent"))),
        plan=turn_plan_from_payload(_mapping(payload.get("plan"))),
        observations=tuple(
            tool_observation_from_payload(item)
            for item in _mapping_tuple(payload.get("observations"))
        ),
        evidence_refs=tuple(
            evidence_ref_from_payload(item)
            for item in _mapping_tuple(payload.get("evidence_refs"))
        ),
        risk_state=risk_state_from_payload(_mapping(payload.get("risk_state"))),
        acceptance_contract=acceptance_contract_from_payload(
            _mapping(payload.get("acceptance_contract"))
        ),
        hypotheses=tuple(
            hypothesis_from_payload(item)
            for item in _mapping_tuple(payload.get("hypotheses"))
        ),
        rejected_hypotheses=tuple(
            rejected_hypothesis_from_payload(item)
            for item in _mapping_tuple(payload.get("rejected_hypotheses"))
        ),
        accepted_facts=tuple(
            accepted_fact_from_payload(item)
            for item in _mapping_tuple(payload.get("accepted_facts"))
        ),
        working_set=working_set_from_payload(_mapping(payload.get("working_set"))),
        answer_summary=_string_or_none(payload.get("answer_summary")),
        created_at=_float_value(payload.get("created_at")),
    )


def distillation_payload(record: DistillationRecord) -> dict[str, Any]:
    return {
        "distillation_id": record.distillation_id,
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "compiler_fingerprint": record.compiler_fingerprint,
        "summary": record.summary,
        "source_refs": [
            evidence_ref_payload(item) for item in record.source_refs
        ],
        "accepted_facts": [
            accepted_fact_payload(item) for item in record.accepted_facts
        ],
        "reused_from_distillation_id": record.reused_from_distillation_id,
        "invalidation_key": record.invalidation_key,
        "created_at": record.created_at,
        "projection_fingerprint": record.projection_fingerprint,
        "embedding_fingerprint": record.embedding_fingerprint,
        "retrieval_policy_fingerprint": record.retrieval_policy_fingerprint,
        "distillation_prompt_fingerprint": (
            record.distillation_prompt_fingerprint
        ),
        "budget_fingerprint": record.budget_fingerprint,
    }


def distillation_from_payload(payload: Mapping[str, Any]) -> DistillationRecord:
    return DistillationRecord(
        distillation_id=_string(payload.get("distillation_id")),
        session_id=_string(payload.get("session_id")),
        turn_number=_int_value(payload.get("turn_number")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        compiler_fingerprint=_string(payload.get("compiler_fingerprint")),
        summary=_string(payload.get("summary")),
        source_refs=tuple(
            evidence_ref_from_payload(item)
            for item in _mapping_tuple(payload.get("source_refs"))
        ),
        accepted_facts=tuple(
            accepted_fact_from_payload(item)
            for item in _mapping_tuple(payload.get("accepted_facts"))
        ),
        reused_from_distillation_id=_string_or_none(
            payload.get("reused_from_distillation_id")
        ),
        invalidation_key=_string(payload.get("invalidation_key")),
        created_at=_float_value(payload.get("created_at")),
        projection_fingerprint=_string(
            payload.get("projection_fingerprint") or "projection:none"
        ),
        embedding_fingerprint=_string(
            payload.get("embedding_fingerprint") or "embedding:unknown"
        ),
        retrieval_policy_fingerprint=_string(
            payload.get("retrieval_policy_fingerprint") or "retrieval:default"
        ),
        distillation_prompt_fingerprint=_string(
            payload.get("distillation_prompt_fingerprint") or "distill:v1"
        ),
        budget_fingerprint=_string(
            payload.get("budget_fingerprint") or "budget:default"
        ),
    )


def activation_payload(record: ActivationRecord) -> dict[str, Any]:
    return {
        "activation_id": record.activation_id,
        "bundle_id": record.bundle_id,
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "compiler_fingerprint": record.compiler_fingerprint,
        "active_ref_ids": list(record.active_ref_ids),
        "active_fact_ids": list(record.active_fact_ids),
        "active_hypothesis_ids": list(record.active_hypothesis_ids),
        "reason": record.reason,
        "created_at": record.created_at,
    }


def activation_from_payload(payload: Mapping[str, Any]) -> ActivationRecord:
    return ActivationRecord(
        activation_id=_string(payload.get("activation_id")),
        bundle_id=_string(payload.get("bundle_id")),
        session_id=_string(payload.get("session_id")),
        turn_number=_int_value(payload.get("turn_number")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        compiler_fingerprint=_string(payload.get("compiler_fingerprint")),
        active_ref_ids=_string_tuple(payload.get("active_ref_ids")),
        active_fact_ids=_string_tuple(payload.get("active_fact_ids")),
        active_hypothesis_ids=_string_tuple(payload.get("active_hypothesis_ids")),
        reason=_string(payload.get("reason")),
        created_at=_float_value(payload.get("created_at")),
    )


def context_bundle_payload(record: ContextBundle) -> dict[str, Any]:
    return {
        "bundle_id": record.bundle_id,
        "session_id": record.session_id,
        "turn_number": record.turn_number,
        "snapshot_id": record.snapshot_id,
        "artifact_key": record.artifact_key,
        "compiler_fingerprint": record.compiler_fingerprint,
        "working_memory": working_memory_payload(record.working_memory),
        "turn_journal": turn_journal_payload(record.turn_journal),
        "distillation": distillation_payload(record.distillation),
        "activation": activation_payload(record.activation),
        "created_at": record.created_at,
        "projection_fingerprint": record.projection_fingerprint,
        "embedding_fingerprint": record.embedding_fingerprint,
        "retrieval_policy_fingerprint": record.retrieval_policy_fingerprint,
        "distillation_prompt_fingerprint": (
            record.distillation_prompt_fingerprint
        ),
        "budget_fingerprint": record.budget_fingerprint,
    }


def context_bundle_from_payload(payload: Mapping[str, Any]) -> ContextBundle:
    return ContextBundle(
        bundle_id=_string(payload.get("bundle_id")),
        session_id=_string(payload.get("session_id")),
        turn_number=_int_value(payload.get("turn_number")),
        snapshot_id=_string_or_none(payload.get("snapshot_id")),
        artifact_key=_string_or_none(payload.get("artifact_key")),
        compiler_fingerprint=_string(payload.get("compiler_fingerprint")),
        working_memory=working_memory_from_payload(
            _mapping(payload.get("working_memory"))
        ),
        turn_journal=turn_journal_from_payload(_mapping(payload.get("turn_journal"))),
        distillation=distillation_from_payload(_mapping(payload.get("distillation"))),
        activation=activation_from_payload(_mapping(payload.get("activation"))),
        created_at=_float_value(payload.get("created_at")),
        projection_fingerprint=_string(
            payload.get("projection_fingerprint") or "projection:none"
        ),
        embedding_fingerprint=_string(
            payload.get("embedding_fingerprint") or "embedding:unknown"
        ),
        retrieval_policy_fingerprint=_string(
            payload.get("retrieval_policy_fingerprint") or "retrieval:default"
        ),
        distillation_prompt_fingerprint=_string(
            payload.get("distillation_prompt_fingerprint") or "distill:v1"
        ),
        budget_fingerprint=_string(
            payload.get("budget_fingerprint") or "budget:default"
        ),
    )
