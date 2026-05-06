"""Typed agent-context records and deterministic policy helpers.

This module is pure: no I/O, no logging, no framework imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _string(value: Any) -> str:
    return str(value or "")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if value is None:
        return ()
    return (str(value),)


def _dict_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            items.append({str(key): sub_item for key, sub_item in item.items()})
    return tuple(items)


@dataclass(frozen=True)
class EvidenceRef:
    ref_id: str
    kind: str
    repo_name: str | None = None
    snapshot_id: str | None = None
    path: str | None = None
    symbol_id: str | None = None
    lines: str | None = None
    label: str | None = None
    score: float | None = None
    source: str | None = None
    fresh: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "kind": self.kind,
            "repo_name": self.repo_name,
            "snapshot_id": self.snapshot_id,
            "path": self.path,
            "symbol_id": self.symbol_id,
            "lines": self.lines,
            "label": self.label,
            "score": self.score,
            "source": self.source,
            "fresh": self.fresh,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRef:
        score_value = data.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        return cls(
            ref_id=_string(data.get("ref_id")),
            kind=_string(data.get("kind")),
            repo_name=_string_or_none(data.get("repo_name")),
            snapshot_id=_string_or_none(data.get("snapshot_id")),
            path=_string_or_none(data.get("path")),
            symbol_id=_string_or_none(data.get("symbol_id")),
            lines=_string_or_none(data.get("lines")),
            label=_string_or_none(data.get("label")),
            score=score,
            source=_string_or_none(data.get("source")),
            fresh=_string(data.get("fresh") or "unknown"),
        )


@dataclass(frozen=True)
class ToolObservation:
    observation_id: str
    tool: str
    ok: bool
    parameters: dict[str, Any]
    ref_ids: tuple[str, ...]
    cost: int = 0
    fresh: str = "unknown"
    warnings: tuple[str, ...] = ()
    round_number: int = 0
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "tool": self.tool,
            "ok": self.ok,
            "parameters": dict(self.parameters),
            "ref_ids": list(self.ref_ids),
            "cost": self.cost,
            "fresh": self.fresh,
            "warnings": list(self.warnings),
            "round_number": self.round_number,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolObservation:
        return cls(
            observation_id=_string(data.get("observation_id")),
            tool=_string(data.get("tool")),
            ok=bool(data.get("ok", False)),
            parameters=_dict_value(data.get("parameters")),
            ref_ids=_string_tuple(data.get("ref_ids")),
            cost=int(data.get("cost") or 0),
            fresh=_string(data.get("fresh") or "unknown"),
            warnings=_string_tuple(data.get("warnings")),
            round_number=int(data.get("round_number") or 0),
            summary=_string_or_none(data.get("summary")),
        )


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    confidence: float
    support_ref_ids: tuple[str, ...]
    conflict_ref_ids: tuple[str, ...]
    state: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "confidence": self.confidence,
            "support_ref_ids": list(self.support_ref_ids),
            "conflict_ref_ids": list(self.conflict_ref_ids),
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        confidence_value = data.get("confidence")
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float))
            else 0.0
        )
        return cls(
            hypothesis_id=_string(data.get("hypothesis_id")),
            statement=_string(data.get("statement")),
            confidence=confidence,
            support_ref_ids=_string_tuple(data.get("support_ref_ids")),
            conflict_ref_ids=_string_tuple(data.get("conflict_ref_ids")),
            state=_string(data.get("state") or "open"),
        )


@dataclass(frozen=True)
class RejectedHypothesisEntry:
    entry_id: str
    hypothesis_id: str
    killed_by_ref_ids: tuple[str, ...]
    reason_code: str
    snapshot_id: str | None = None
    reopen_condition: str = "-"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "hypothesis_id": self.hypothesis_id,
            "killed_by_ref_ids": list(self.killed_by_ref_ids),
            "reason_code": self.reason_code,
            "snapshot_id": self.snapshot_id,
            "reopen_condition": self.reopen_condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RejectedHypothesisEntry:
        return cls(
            entry_id=_string(data.get("entry_id")),
            hypothesis_id=_string(data.get("hypothesis_id")),
            killed_by_ref_ids=_string_tuple(data.get("killed_by_ref_ids")),
            reason_code=_string(data.get("reason_code")),
            snapshot_id=_string_or_none(data.get("snapshot_id")),
            reopen_condition=_string(data.get("reopen_condition") or "-"),
        )


@dataclass(frozen=True)
class AcceptedFact:
    fact_id: str
    statement: str
    ref_ids: tuple[str, ...]
    scope: str = "turn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "statement": self.statement,
            "ref_ids": list(self.ref_ids),
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcceptedFact:
        return cls(
            fact_id=_string(data.get("fact_id")),
            statement=_string(data.get("statement")),
            ref_ids=_string_tuple(data.get("ref_ids")),
            scope=_string(data.get("scope") or "turn"),
        )


@dataclass(frozen=True)
class RiskState:
    evidence_gap: int
    conflict_level: int
    freshness_risk: int
    requirement_ambiguity: int
    execution_risk: int
    verifier_status: str
    action_bias: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_gap": self.evidence_gap,
            "conflict_level": self.conflict_level,
            "freshness_risk": self.freshness_risk,
            "requirement_ambiguity": self.requirement_ambiguity,
            "execution_risk": self.execution_risk,
            "verifier_status": self.verifier_status,
            "action_bias": self.action_bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskState:
        return cls(
            evidence_gap=int(data.get("evidence_gap") or 0),
            conflict_level=int(data.get("conflict_level") or 0),
            freshness_risk=int(data.get("freshness_risk") or 0),
            requirement_ambiguity=int(data.get("requirement_ambiguity") or 0),
            execution_risk=int(data.get("execution_risk") or 0),
            verifier_status=_string(data.get("verifier_status") or "pending"),
            action_bias=_string(data.get("action_bias") or "retrieve"),
        )


@dataclass(frozen=True)
class AcceptanceContract:
    requested_outcome: str
    required_evidence_kinds: tuple[str, ...]
    required_verifiers: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    allowed_write_scope: tuple[str, ...]
    done_condition: str
    must_ask_before: tuple[str, ...]
    must_abstain_when: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_outcome": self.requested_outcome,
            "required_evidence_kinds": list(self.required_evidence_kinds),
            "required_verifiers": list(self.required_verifiers),
            "allowed_tools": list(self.allowed_tools),
            "allowed_write_scope": list(self.allowed_write_scope),
            "done_condition": self.done_condition,
            "must_ask_before": list(self.must_ask_before),
            "must_abstain_when": list(self.must_abstain_when),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcceptanceContract:
        return cls(
            requested_outcome=_string(data.get("requested_outcome")),
            required_evidence_kinds=_string_tuple(data.get("required_evidence_kinds")),
            required_verifiers=_string_tuple(data.get("required_verifiers")),
            allowed_tools=_string_tuple(data.get("allowed_tools")),
            allowed_write_scope=_string_tuple(data.get("allowed_write_scope")),
            done_condition=_string(data.get("done_condition")),
            must_ask_before=_string_tuple(data.get("must_ask_before")),
            must_abstain_when=_string_tuple(data.get("must_abstain_when")),
        )


@dataclass(frozen=True)
class TurnIntent:
    session_id: str
    turn_number: int
    question: str
    kind: str
    requested_outcome: str
    snapshot_id: str | None = None
    artifact_key: str | None = None
    repo_filter: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "question": self.question,
            "kind": self.kind,
            "requested_outcome": self.requested_outcome,
            "snapshot_id": self.snapshot_id,
            "artifact_key": self.artifact_key,
            "repo_filter": list(self.repo_filter),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnIntent:
        return cls(
            session_id=_string(data.get("session_id")),
            turn_number=int(data.get("turn_number") or 0),
            question=_string(data.get("question")),
            kind=_string(data.get("kind")),
            requested_outcome=_string(data.get("requested_outcome")),
            snapshot_id=_string_or_none(data.get("snapshot_id")),
            artifact_key=_string_or_none(data.get("artifact_key")),
            repo_filter=_string_tuple(data.get("repo_filter")),
        )


@dataclass(frozen=True)
class TurnPlan:
    step: int
    action: str
    why: str
    stop_condition: str
    allowed_actions: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    remaining_budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "why": self.why,
            "stop_condition": self.stop_condition,
            "allowed_actions": list(self.allowed_actions),
            "allowed_tools": list(self.allowed_tools),
            "remaining_budget": self.remaining_budget,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnPlan:
        budget_value = data.get("remaining_budget")
        budget = int(budget_value) if isinstance(budget_value, int) else None
        return cls(
            step=int(data.get("step") or 0),
            action=_string(data.get("action")),
            why=_string(data.get("why")),
            stop_condition=_string(data.get("stop_condition")),
            allowed_actions=_string_tuple(data.get("allowed_actions")),
            allowed_tools=_string_tuple(data.get("allowed_tools")),
            remaining_budget=budget,
        )


@dataclass(frozen=True)
class WorkingSet:
    keep_ids: tuple[str, ...]
    drop_ids: tuple[str, ...]
    protect_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep_ids": list(self.keep_ids),
            "drop_ids": list(self.drop_ids),
            "protect_ids": list(self.protect_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkingSet:
        return cls(
            keep_ids=_string_tuple(data.get("keep_ids")),
            drop_ids=_string_tuple(data.get("drop_ids")),
            protect_ids=_string_tuple(data.get("protect_ids")),
            reason=_string(data.get("reason")),
        )


@dataclass(frozen=True)
class WorkingMemoryArtifact:
    session_id: str
    turn_number: int
    snapshot_id: str | None
    artifact_key: str | None
    compiler_fingerprint: str
    stable_fcx: str
    turn_fcx: str
    obs_fcx: str
    full_fcx: str
    evidence_refs: tuple[EvidenceRef, ...]
    accepted_facts: tuple[AcceptedFact, ...]
    hypotheses: tuple[Hypothesis, ...]
    rejected_hypotheses: tuple[RejectedHypothesisEntry, ...]
    unresolved_questions: tuple[str, ...]
    risk_state: RiskState
    acceptance_contract: AcceptanceContract
    working_set: WorkingSet
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "artifact_key": self.artifact_key,
            "compiler_fingerprint": self.compiler_fingerprint,
            "stable_fcx": self.stable_fcx,
            "turn_fcx": self.turn_fcx,
            "obs_fcx": self.obs_fcx,
            "full_fcx": self.full_fcx,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "accepted_facts": [item.to_dict() for item in self.accepted_facts],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "rejected_hypotheses": [
                item.to_dict() for item in self.rejected_hypotheses
            ],
            "unresolved_questions": list(self.unresolved_questions),
            "risk_state": self.risk_state.to_dict(),
            "acceptance_contract": self.acceptance_contract.to_dict(),
            "working_set": self.working_set.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkingMemoryArtifact:
        created_at_value = data.get("created_at")
        return cls(
            session_id=_string(data.get("session_id")),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=_string_or_none(data.get("snapshot_id")),
            artifact_key=_string_or_none(data.get("artifact_key")),
            compiler_fingerprint=_string(data.get("compiler_fingerprint")),
            stable_fcx=_string(data.get("stable_fcx")),
            turn_fcx=_string(data.get("turn_fcx")),
            obs_fcx=_string(data.get("obs_fcx")),
            full_fcx=_string(data.get("full_fcx")),
            evidence_refs=tuple(
                EvidenceRef.from_dict(item)
                for item in _dict_tuple(data.get("evidence_refs"))
            ),
            accepted_facts=tuple(
                AcceptedFact.from_dict(item)
                for item in _dict_tuple(data.get("accepted_facts"))
            ),
            hypotheses=tuple(
                Hypothesis.from_dict(item)
                for item in _dict_tuple(data.get("hypotheses"))
            ),
            rejected_hypotheses=tuple(
                RejectedHypothesisEntry.from_dict(item)
                for item in _dict_tuple(data.get("rejected_hypotheses"))
            ),
            unresolved_questions=_string_tuple(data.get("unresolved_questions")),
            risk_state=RiskState.from_dict(_dict_value(data.get("risk_state"))),
            acceptance_contract=AcceptanceContract.from_dict(
                _dict_value(data.get("acceptance_contract"))
            ),
            working_set=WorkingSet.from_dict(_dict_value(data.get("working_set"))),
            created_at=(
                float(created_at_value)
                if isinstance(created_at_value, (int, float))
                else 0.0
            ),
        )


@dataclass(frozen=True)
class HandoffArtifact:
    artifact_id: str
    session_id: str
    turn_number: int
    mode: str
    snapshot_id: str | None
    compiler_fingerprint: str
    full_fcx: str
    intent: TurnIntent
    acceptance_contract: AcceptanceContract
    accepted_facts: tuple[AcceptedFact, ...]
    surviving_hypotheses: tuple[Hypothesis, ...]
    rejected_hypotheses: tuple[RejectedHypothesisEntry, ...]
    unresolved_questions: tuple[str, ...]
    keep_ids: tuple[str, ...]
    recommended_action: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "mode": self.mode,
            "snapshot_id": self.snapshot_id,
            "compiler_fingerprint": self.compiler_fingerprint,
            "full_fcx": self.full_fcx,
            "intent": self.intent.to_dict(),
            "acceptance_contract": self.acceptance_contract.to_dict(),
            "accepted_facts": [item.to_dict() for item in self.accepted_facts],
            "surviving_hypotheses": [
                item.to_dict() for item in self.surviving_hypotheses
            ],
            "rejected_hypotheses": [
                item.to_dict() for item in self.rejected_hypotheses
            ],
            "unresolved_questions": list(self.unresolved_questions),
            "keep_ids": list(self.keep_ids),
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffArtifact:
        created_at_value = data.get("created_at")
        return cls(
            artifact_id=_string(data.get("artifact_id")),
            session_id=_string(data.get("session_id")),
            turn_number=int(data.get("turn_number") or 0),
            mode=_string(data.get("mode")),
            snapshot_id=_string_or_none(data.get("snapshot_id")),
            compiler_fingerprint=_string(data.get("compiler_fingerprint")),
            full_fcx=_string(data.get("full_fcx")),
            intent=TurnIntent.from_dict(_dict_value(data.get("intent"))),
            acceptance_contract=AcceptanceContract.from_dict(
                _dict_value(data.get("acceptance_contract"))
            ),
            accepted_facts=tuple(
                AcceptedFact.from_dict(item)
                for item in _dict_tuple(data.get("accepted_facts"))
            ),
            surviving_hypotheses=tuple(
                Hypothesis.from_dict(item)
                for item in _dict_tuple(data.get("surviving_hypotheses"))
            ),
            rejected_hypotheses=tuple(
                RejectedHypothesisEntry.from_dict(item)
                for item in _dict_tuple(data.get("rejected_hypotheses"))
            ),
            unresolved_questions=_string_tuple(data.get("unresolved_questions")),
            keep_ids=_string_tuple(data.get("keep_ids")),
            recommended_action=_string(data.get("recommended_action")),
            created_at=(
                float(created_at_value)
                if isinstance(created_at_value, (int, float))
                else 0.0
            ),
        )


@dataclass(frozen=True)
class TurnJournal:
    session_id: str
    turn_number: int
    snapshot_id: str | None
    artifact_key: str | None
    compiler_fingerprint: str
    intent: TurnIntent
    plan: TurnPlan
    observations: tuple[ToolObservation, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    risk_state: RiskState
    acceptance_contract: AcceptanceContract
    hypotheses: tuple[Hypothesis, ...]
    rejected_hypotheses: tuple[RejectedHypothesisEntry, ...]
    accepted_facts: tuple[AcceptedFact, ...]
    working_set: WorkingSet
    answer_summary: str | None
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "snapshot_id": self.snapshot_id,
            "artifact_key": self.artifact_key,
            "compiler_fingerprint": self.compiler_fingerprint,
            "intent": self.intent.to_dict(),
            "plan": self.plan.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "risk_state": self.risk_state.to_dict(),
            "acceptance_contract": self.acceptance_contract.to_dict(),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "rejected_hypotheses": [
                item.to_dict() for item in self.rejected_hypotheses
            ],
            "accepted_facts": [item.to_dict() for item in self.accepted_facts],
            "working_set": self.working_set.to_dict(),
            "answer_summary": self.answer_summary,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnJournal:
        created_at_value = data.get("created_at")
        return cls(
            session_id=_string(data.get("session_id")),
            turn_number=int(data.get("turn_number") or 0),
            snapshot_id=_string_or_none(data.get("snapshot_id")),
            artifact_key=_string_or_none(data.get("artifact_key")),
            compiler_fingerprint=_string(data.get("compiler_fingerprint")),
            intent=TurnIntent.from_dict(_dict_value(data.get("intent"))),
            plan=TurnPlan.from_dict(_dict_value(data.get("plan"))),
            observations=tuple(
                ToolObservation.from_dict(item)
                for item in _dict_tuple(data.get("observations"))
            ),
            evidence_refs=tuple(
                EvidenceRef.from_dict(item)
                for item in _dict_tuple(data.get("evidence_refs"))
            ),
            risk_state=RiskState.from_dict(_dict_value(data.get("risk_state"))),
            acceptance_contract=AcceptanceContract.from_dict(
                _dict_value(data.get("acceptance_contract"))
            ),
            hypotheses=tuple(
                Hypothesis.from_dict(item)
                for item in _dict_tuple(data.get("hypotheses"))
            ),
            rejected_hypotheses=tuple(
                RejectedHypothesisEntry.from_dict(item)
                for item in _dict_tuple(data.get("rejected_hypotheses"))
            ),
            accepted_facts=tuple(
                AcceptedFact.from_dict(item)
                for item in _dict_tuple(data.get("accepted_facts"))
            ),
            working_set=WorkingSet.from_dict(_dict_value(data.get("working_set"))),
            answer_summary=_string_or_none(data.get("answer_summary")),
            created_at=(
                float(created_at_value)
                if isinstance(created_at_value, (int, float))
                else 0.0
            ),
        )


def build_acceptance_contract(
    requested_outcome: str = "answer",
    allowed_tools: tuple[str, ...] = (
        "search_codebase",
        "list_directory",
        "projection",
        "graph",
        "retrieve",
    ),
) -> AcceptanceContract:
    return AcceptanceContract(
        requested_outcome=requested_outcome,
        required_evidence_kinds=("repo_evidence",),
        required_verifiers=("source_ref_coverage", "fcx_parse", "snapshot_freshness"),
        allowed_tools=allowed_tools,
        allowed_write_scope=(),
        done_condition="cited_answer",
        must_ask_before=(),
        must_abstain_when=("missing_primary_support", "unresolved_snapshot_scope"),
    )


def compute_risk_state(
    *,
    question: str,
    snapshot_id: str | None,
    evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef],
    hypotheses: tuple[Hypothesis, ...] | list[Hypothesis],
    accepted_facts: tuple[AcceptedFact, ...] | list[AcceptedFact],
    rejected_hypotheses: tuple[RejectedHypothesisEntry, ...]
    | list[RejectedHypothesisEntry],
) -> RiskState:
    evidence_count = len(evidence_refs)
    accepted_count = len(accepted_facts)
    live_hypotheses = [
        item for item in hypotheses if item.state not in {"rejected", "blocked"}
    ]
    rejected_count = len(rejected_hypotheses)

    if accepted_count >= 2:
        evidence_gap = 0
    elif evidence_count >= 2:
        evidence_gap = 1
    elif evidence_count == 1:
        evidence_gap = 2
    else:
        evidence_gap = 3

    if len(live_hypotheses) <= 1 and rejected_count == 0:
        conflict_level = 0
    elif len(live_hypotheses) <= 2:
        conflict_level = 1
    elif len(live_hypotheses) == 3:
        conflict_level = 2
    else:
        conflict_level = 3

    freshness_risk = 0 if snapshot_id else 1
    requirement_ambiguity = 0 if len(question.strip()) >= 12 else 2
    execution_risk = 0

    if conflict_level >= 2:
        verifier_status = "mixed"
        action_bias = "reset"
    elif evidence_gap == 0:
        verifier_status = "clean"
        action_bias = "answer"
    elif evidence_gap >= 2:
        verifier_status = "pending"
        action_bias = "retrieve"
    else:
        verifier_status = "pending"
        action_bias = "verify"

    return RiskState(
        evidence_gap=evidence_gap,
        conflict_level=conflict_level,
        freshness_risk=freshness_risk,
        requirement_ambiguity=requirement_ambiguity,
        execution_risk=execution_risk,
        verifier_status=verifier_status,
        action_bias=action_bias,
    )


def promote_observations(
    *,
    question: str,
    evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef],
    observations: tuple[ToolObservation, ...] | list[ToolObservation],
    snapshot_id: str | None,
) -> tuple[
    tuple[AcceptedFact, ...],
    tuple[Hypothesis, ...],
    tuple[RejectedHypothesisEntry, ...],
]:
    ref_ids = tuple(item.ref_id for item in evidence_refs)
    supported_refs = ref_ids[:3]
    supported_statement = (
        f"Relevant repository evidence was collected for: {question}"
        if supported_refs
        else f"Repository evidence still needs expansion for: {question}"
    )
    primary_hypothesis = Hypothesis(
        hypothesis_id="h1",
        statement=supported_statement,
        confidence=0.82 if len(supported_refs) >= 2 else 0.45,
        support_ref_ids=supported_refs,
        conflict_ref_ids=(),
        state="favored" if supported_refs else "open",
    )

    rejected: list[RejectedHypothesisEntry] = []
    for item in observations:
        if item.ok:
            continue
        rejected.append(
            RejectedHypothesisEntry(
                entry_id=f"x_{item.observation_id}",
                hypothesis_id=primary_hypothesis.hypothesis_id,
                killed_by_ref_ids=item.ref_ids,
                reason_code="tool_failed",
                snapshot_id=snapshot_id,
                reopen_condition="new_evidence",
            )
        )

    accepted: list[AcceptedFact] = []
    if len(supported_refs) >= 2:
        accepted.append(
            AcceptedFact(
                fact_id="f1",
                statement="The current answer should be grounded in retrieved repository evidence.",
                ref_ids=supported_refs,
                scope="turn",
            )
        )

    return tuple(accepted), (primary_hypothesis,), tuple(rejected)


def build_handoff_artifact(
    *,
    artifact_id: str,
    mode: str,
    working_memory: WorkingMemoryArtifact,
) -> HandoffArtifact:
    return HandoffArtifact(
        artifact_id=artifact_id,
        session_id=working_memory.session_id,
        turn_number=working_memory.turn_number,
        mode=mode,
        snapshot_id=working_memory.snapshot_id,
        compiler_fingerprint=working_memory.compiler_fingerprint,
        full_fcx=working_memory.full_fcx,
        intent=TurnIntent(
            session_id=working_memory.session_id,
            turn_number=working_memory.turn_number,
            question=working_memory.unresolved_questions[0]
            if working_memory.unresolved_questions
            else "",
            kind="research",
            requested_outcome=working_memory.acceptance_contract.requested_outcome,
            snapshot_id=working_memory.snapshot_id,
            artifact_key=working_memory.artifact_key,
        ),
        acceptance_contract=working_memory.acceptance_contract,
        accepted_facts=working_memory.accepted_facts,
        surviving_hypotheses=tuple(
            item
            for item in working_memory.hypotheses
            if item.state not in {"rejected", "blocked"}
        ),
        rejected_hypotheses=working_memory.rejected_hypotheses,
        unresolved_questions=working_memory.unresolved_questions,
        keep_ids=working_memory.working_set.keep_ids,
        recommended_action=working_memory.risk_state.action_bias,
        created_at=working_memory.created_at,
    )


def should_reopen_hypothesis(
    entry: RejectedHypothesisEntry,
    *,
    new_snapshot_id: str | None,
    new_ref_ids: tuple[str, ...] | list[str] = (),
    contract_changed: bool = False,
) -> bool:
    if contract_changed:
        return True
    if entry.snapshot_id and new_snapshot_id and entry.snapshot_id != new_snapshot_id:
        return True
    return bool(set(entry.killed_by_ref_ids).difference(set(new_ref_ids)))
