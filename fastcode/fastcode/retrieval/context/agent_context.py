"""Typed agent-context records and deterministic policy helpers.

This module is pure retrieval-domain code: no I/O, logging, framework imports,
or persistence serializers.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class ToolObservation:
    observation_id: str
    tool: str
    ok: bool
    parameters: dict[str, object]
    ref_ids: tuple[str, ...]
    cost: int = 0
    fresh: str = "unknown"
    warnings: tuple[str, ...] = ()
    round_number: int = 0
    summary: str | None = None


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    confidence: float
    support_ref_ids: tuple[str, ...]
    conflict_ref_ids: tuple[str, ...]
    state: str = "open"


@dataclass(frozen=True)
class RejectedHypothesisEntry:
    entry_id: str
    hypothesis_id: str
    killed_by_ref_ids: tuple[str, ...]
    reason_code: str
    snapshot_id: str | None = None
    reopen_condition: str = "-"


@dataclass(frozen=True)
class AcceptedFact:
    fact_id: str
    statement: str
    ref_ids: tuple[str, ...]
    scope: str = "turn"


@dataclass(frozen=True)
class RiskState:
    evidence_gap: int
    conflict_level: int
    freshness_risk: int
    requirement_ambiguity: int
    execution_risk: int
    verifier_status: str
    action_bias: str


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


@dataclass(frozen=True)
class TurnPlan:
    step: int
    action: str
    why: str
    stop_condition: str
    allowed_actions: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    remaining_budget: int | None = None


@dataclass(frozen=True)
class WorkingSet:
    keep_ids: tuple[str, ...]
    drop_ids: tuple[str, ...]
    protect_ids: tuple[str, ...]
    reason: str


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


@dataclass(frozen=True)
class DistillationRecord:
    distillation_id: str
    session_id: str
    turn_number: int
    snapshot_id: str | None
    compiler_fingerprint: str
    summary: str
    source_refs: tuple[EvidenceRef, ...]
    accepted_facts: tuple[AcceptedFact, ...]
    reused_from_distillation_id: str | None
    invalidation_key: str
    created_at: float
    projection_fingerprint: str = "projection:none"
    embedding_fingerprint: str = "embedding:unknown"
    retrieval_policy_fingerprint: str = "retrieval:default"
    distillation_prompt_fingerprint: str = "distill:v1"
    budget_fingerprint: str = "budget:default"


@dataclass(frozen=True)
class ActivationRecord:
    activation_id: str
    bundle_id: str
    session_id: str
    turn_number: int
    snapshot_id: str | None
    compiler_fingerprint: str
    active_ref_ids: tuple[str, ...]
    active_fact_ids: tuple[str, ...]
    active_hypothesis_ids: tuple[str, ...]
    reason: str
    created_at: float


@dataclass(frozen=True)
class ContextBundle:
    bundle_id: str
    session_id: str
    turn_number: int
    snapshot_id: str | None
    artifact_key: str | None
    compiler_fingerprint: str
    working_memory: WorkingMemoryArtifact
    turn_journal: TurnJournal
    distillation: DistillationRecord
    activation: ActivationRecord
    created_at: float
    projection_fingerprint: str = "projection:none"
    embedding_fingerprint: str = "embedding:unknown"
    retrieval_policy_fingerprint: str = "retrieval:default"
    distillation_prompt_fingerprint: str = "distill:v1"
    budget_fingerprint: str = "budget:default"


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
                statement=(
                    "The current answer should be grounded in retrieved repository "
                    "evidence."
                ),
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
            question=(
                working_memory.unresolved_questions[0]
                if working_memory.unresolved_questions
                else ""
            ),
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
