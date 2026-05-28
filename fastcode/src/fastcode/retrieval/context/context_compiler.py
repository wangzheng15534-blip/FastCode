"""Compile typed agent context into FCX working-memory artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any, cast

from fastcode.retrieval.contracts import SourceCitation
from fastcode.retrieval.ranking import fcx as _fcx

from .agent_context import (
    AcceptanceContract,
    AcceptedFact,
    ActivationRecord,
    ContextBundle,
    DistillationRecord,
    EvidenceRef,
    HandoffArtifact,
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

COMPILER_FINGERPRINT = "fcx-v1"
DEFAULT_PROJECTION_FINGERPRINT = "projection:none"
DEFAULT_EMBEDDING_FINGERPRINT = "embedding:unknown"
DEFAULT_RETRIEVAL_POLICY_FINGERPRINT = "retrieval:default"
DEFAULT_DISTILLATION_PROMPT_FINGERPRINT = "distill:v1"
DEFAULT_BUDGET_FINGERPRINT = "budget:default"


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _hash_payload(prefix: str, payload: Any, length: int = 16) -> str:
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def _mapping_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    mapping = cast(Mapping[str, Any], value)
    return str(mapping.get("summary") or "")


def build_evidence_refs_from_sources(
    sources: tuple[SourceCitation, ...] | list[SourceCitation],
    *,
    snapshot_id: str | None,
    source_kind: str = "range",
) -> tuple[EvidenceRef, ...]:
    refs: list[EvidenceRef] = []
    for index, item in enumerate(sources, 1):
        repo_name = item.repository
        path = item.file
        lines = item.lines
        refs.append(
            EvidenceRef(
                ref_id=f"e{index}",
                kind=source_kind,
                repo_name=repo_name or None,
                snapshot_id=snapshot_id,
                path=path or None,
                lines=lines or None,
                label=item.name or path or f"source-{index}",
                score=item.score,
                source="retrieval",
                fresh="ok" if snapshot_id else "unknown",
            )
        )
    return tuple(refs)


def build_tool_observation(
    *,
    observation_id: str,
    tool: str,
    ok: bool,
    parameters: dict[str, Any],
    ref_ids: tuple[str, ...],
    summary: str,
    round_number: int = 0,
) -> ToolObservation:
    return ToolObservation(
        observation_id=observation_id,
        tool=tool,
        ok=ok,
        parameters=dict(parameters),
        ref_ids=ref_ids,
        cost=len(ref_ids),
        fresh="ok",
        warnings=(),
        round_number=round_number,
        summary=summary,
    )


def build_working_set(
    *,
    accepted_facts: tuple[AcceptedFact, ...],
    hypotheses: tuple[Hypothesis, ...],
    rejected_hypotheses: tuple[RejectedHypothesisEntry, ...],
) -> WorkingSet:
    keep_ids = tuple(
        [item.fact_id for item in accepted_facts]
        + [item.hypothesis_id for item in hypotheses if item.state != "rejected"]
    )
    drop_ids = tuple(
        item.hypothesis_id for item in hypotheses if item.state == "rejected"
    )
    protect_ids = tuple(item.fact_id for item in accepted_facts)
    reason = "compiled_context"
    if rejected_hypotheses:
        reason = "compiled_context_with_rejections"
    return WorkingSet(
        keep_ids=keep_ids,
        drop_ids=drop_ids,
        protect_ids=protect_ids,
        reason=reason,
    )


def build_turn_plan(
    *,
    risk_state: RiskState,
    contract: AcceptanceContract,
) -> TurnPlan:
    action = risk_state.action_bias
    allowed_actions = ("expand", "verify", "answer", "abstain")
    if risk_state.action_bias == "reset":
        allowed_actions = ("expand", "verify", "reset", "abstain")
    return TurnPlan(
        step=1,
        action=action,
        why="typed_context_policy",
        stop_condition=contract.done_condition,
        allowed_actions=allowed_actions,
        allowed_tools=contract.allowed_tools,
        remaining_budget=None,
    )


def compile_working_memory(
    *,
    intent: TurnIntent,
    contract: AcceptanceContract,
    risk_state: RiskState,
    plan: TurnPlan,
    evidence_refs: tuple[EvidenceRef, ...],
    observations: tuple[ToolObservation, ...],
    accepted_facts: tuple[AcceptedFact, ...],
    hypotheses: tuple[Hypothesis, ...],
    rejected_hypotheses: tuple[RejectedHypothesisEntry, ...],
    unresolved_questions: tuple[str, ...],
    session_prefix: dict[str, Any] | None,
    created_at: float | None = None,
) -> WorkingMemoryArtifact:
    timestamp = created_at if created_at is not None else time.time()
    working_set = build_working_set(
        accepted_facts=accepted_facts,
        hypotheses=hypotheses,
        rejected_hypotheses=rejected_hypotheses,
    )
    header = {
        "v": 1,
        "sid": intent.session_id,
        "turn": intent.turn_number,
        "snap": intent.snapshot_id or "-",
        "art": intent.artifact_key or "-",
        "fp": COMPILER_FINGERPRINT,
    }

    stable_records: list[str] = []
    prefix: Mapping[str, Any] = session_prefix or {}
    l0 = prefix.get("l0")
    l1 = prefix.get("l1")
    projection_id = str(prefix.get("projection_id") or "")
    if isinstance(l0, Mapping):
        summary = _mapping_summary(l0)
        stable_records.append(
            _fcx.render_record(
                "L0",
                fields={
                    "p": projection_id or "proj",
                    "tok": len(summary),
                },
                tail=summary,
            )
        )
    if isinstance(l1, Mapping):
        summary = _mapping_summary(l1)
        stable_records.append(
            _fcx.render_record(
                "L1",
                fields={
                    "p": projection_id or "proj",
                    "tok": len(summary),
                },
                tail=summary,
            )
        )
    stable_records.append(
        _fcx.render_record(
            "C",
            fields={
                "out": contract.requested_outcome,
                "need": contract.required_evidence_kinds,
                "allow": contract.allowed_tools,
                "done": contract.done_condition,
                "fail": contract.must_abstain_when,
            },
        )
    )
    for fact in accepted_facts:
        stable_records.append(
            _fcx.render_record(
                "F",
                fact.fact_id,
                fields={"scope": fact.scope, "refs": fact.ref_ids},
                tail=fact.statement,
            )
        )
    stable_records.append(
        _fcx.render_record("END", fields={"refs": len(stable_records)})
    )
    stable_fcx = (
        "<fcx:stable>\n"
        + _fcx.render_block(mode="stable", header_fields=header, records=stable_records)
        + "\n</fcx:stable>"
    )

    turn_records: list[str] = [
        _fcx.render_record(
            "I",
            "i1",
            fields={
                "kind": intent.kind,
                "out": intent.requested_outcome,
                "q": intent.question,
            },
        ),
        _fcx.render_record(
            "R",
            fields={
                "eg": risk_state.evidence_gap,
                "cf": risk_state.conflict_level,
                "fr": risk_state.freshness_risk,
                "ra": risk_state.requirement_ambiguity,
                "xr": risk_state.execution_risk,
                "vs": risk_state.verifier_status,
                "act": risk_state.action_bias,
            },
        ),
        _fcx.render_record(
            "P",
            "p1",
            fields={
                "step": plan.step,
                "act": plan.action,
                "why": plan.why,
                "stop": plan.stop_condition,
            },
        ),
    ]
    for hypothesis in hypotheses:
        turn_records.append(
            _fcx.render_record(
                "H",
                hypothesis.hypothesis_id,
                fields={
                    "p": round(hypothesis.confidence, 2),
                    "state": hypothesis.state,
                    "s": hypothesis.support_ref_ids,
                    "c": hypothesis.conflict_ref_ids or "-",
                },
                tail=hypothesis.statement,
            )
        )
    for question_index, question in enumerate(unresolved_questions, 1):
        turn_records.append(
            _fcx.render_record(
                "Q",
                f"q{question_index}",
                fields={"need": "evidence"},
                tail=question,
            )
        )
    for rejected in rejected_hypotheses:
        turn_records.append(
            _fcx.render_record(
                "X",
                rejected.entry_id,
                fields={
                    "from": rejected.hypothesis_id,
                    "by": rejected.killed_by_ref_ids,
                    "why": rejected.reason_code,
                    "reopen": rejected.reopen_condition,
                },
            )
        )
    turn_records.append(
        _fcx.render_record(
            "W",
            fields={
                "keep": working_set.keep_ids,
                "drop": working_set.drop_ids or "-",
                "protect": working_set.protect_ids or "-",
                "reason": working_set.reason,
            },
        )
    )
    turn_records.append(
        _fcx.render_record(
            "N",
            fields={
                "act": plan.allowed_actions,
                "expand": [item.ref_id for item in evidence_refs[:4]],
                "tool": plan.allowed_tools,
                "verify": contract.required_verifiers,
                "ask": "-",
            },
        )
    )
    turn_records.append(_fcx.render_record("END", fields={"refs": len(turn_records)}))
    turn_fcx = (
        "<fcx:turn>\n"
        + _fcx.render_block(mode="turn", header_fields=header, records=turn_records)
        + "\n</fcx:turn>"
    )

    obs_records: list[str] = []
    for observation in observations:
        obs_records.append(
            _fcx.render_record(
                "O",
                observation.observation_id,
                fields={
                    "tool": observation.tool,
                    "ok": observation.ok,
                    "refs": observation.ref_ids,
                    "cost": observation.cost,
                    "fresh": observation.fresh,
                    "warn": observation.warnings or "-",
                },
                tail=observation.summary,
            )
        )
    obs_records.append(_fcx.render_record("END", fields={"refs": len(obs_records)}))
    obs_fcx = (
        "<fcx:obs>\n"
        + _fcx.render_block(mode="tool", header_fields=header, records=obs_records)
        + "\n</fcx:obs>"
    )

    full_fcx = "\n\n".join((stable_fcx, turn_fcx, obs_fcx))
    return WorkingMemoryArtifact(
        session_id=intent.session_id,
        turn_number=intent.turn_number,
        snapshot_id=intent.snapshot_id,
        artifact_key=intent.artifact_key,
        compiler_fingerprint=COMPILER_FINGERPRINT,
        stable_fcx=stable_fcx,
        turn_fcx=turn_fcx,
        obs_fcx=obs_fcx,
        full_fcx=full_fcx,
        evidence_refs=evidence_refs,
        accepted_facts=accepted_facts,
        hypotheses=hypotheses,
        rejected_hypotheses=rejected_hypotheses,
        unresolved_questions=unresolved_questions,
        risk_state=risk_state,
        acceptance_contract=contract,
        working_set=working_set,
        created_at=timestamp,
    )


def build_turn_journal(
    *,
    intent: TurnIntent,
    plan: TurnPlan,
    observations: tuple[ToolObservation, ...],
    evidence_refs: tuple[EvidenceRef, ...],
    risk_state: RiskState,
    acceptance_contract: AcceptanceContract,
    hypotheses: tuple[Hypothesis, ...],
    rejected_hypotheses: tuple[RejectedHypothesisEntry, ...],
    accepted_facts: tuple[AcceptedFact, ...],
    working_set: WorkingSet,
    answer_summary: str | None,
    created_at: float | None = None,
) -> TurnJournal:
    timestamp = created_at if created_at is not None else time.time()
    return TurnJournal(
        session_id=intent.session_id,
        turn_number=intent.turn_number,
        snapshot_id=intent.snapshot_id,
        artifact_key=intent.artifact_key,
        compiler_fingerprint=COMPILER_FINGERPRINT,
        intent=intent,
        plan=plan,
        observations=observations,
        evidence_refs=evidence_refs,
        risk_state=risk_state,
        acceptance_contract=acceptance_contract,
        hypotheses=hypotheses,
        rejected_hypotheses=rejected_hypotheses,
        accepted_facts=accepted_facts,
        working_set=working_set,
        answer_summary=answer_summary,
        created_at=timestamp,
    )


def build_handoff_artifact_id(
    session_id: str,
    turn_number: int,
    mode: str,
) -> str:
    raw = f"{session_id}:{turn_number}:{mode}:{COMPILER_FINGERPRINT}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"hf_{digest}"


def build_context_invalidation_key(
    *,
    session_id: str,
    snapshot_id: str | None,
    artifact_key: str | None,
    evidence_refs: tuple[EvidenceRef, ...],
    compiler_fingerprint: str = COMPILER_FINGERPRINT,
    projection_fingerprint: str = DEFAULT_PROJECTION_FINGERPRINT,
    embedding_fingerprint: str = DEFAULT_EMBEDDING_FINGERPRINT,
    retrieval_policy_fingerprint: str = DEFAULT_RETRIEVAL_POLICY_FINGERPRINT,
    distillation_prompt_fingerprint: str = DEFAULT_DISTILLATION_PROMPT_FINGERPRINT,
    budget_fingerprint: str = DEFAULT_BUDGET_FINGERPRINT,
) -> str:
    payload = {
        "session_id": session_id,
        "snapshot_id": snapshot_id,
        "artifact_key": artifact_key,
        "compiler_fingerprint": compiler_fingerprint,
        "projection_fingerprint": projection_fingerprint,
        "embedding_fingerprint": embedding_fingerprint,
        "retrieval_policy_fingerprint": retrieval_policy_fingerprint,
        "distillation_prompt_fingerprint": distillation_prompt_fingerprint,
        "budget_fingerprint": budget_fingerprint,
        "source_refs": [
            {
                "ref_id": ref.ref_id,
                "kind": ref.kind,
                "repo_name": ref.repo_name,
                "snapshot_id": ref.snapshot_id,
                "path": ref.path,
                "symbol_id": ref.symbol_id,
                "lines": ref.lines,
                "fresh": ref.fresh,
            }
            for ref in evidence_refs
        ],
    }
    return _hash_payload("ctxinv", payload)


def build_context_bundle_id(
    *,
    session_id: str,
    turn_number: int,
    invalidation_key: str,
    compiler_fingerprint: str = COMPILER_FINGERPRINT,
) -> str:
    return _hash_payload(
        "ctxb",
        {
            "session_id": session_id,
            "turn_number": turn_number,
            "invalidation_key": invalidation_key,
            "compiler_fingerprint": compiler_fingerprint,
        },
    )


def build_distillation_record(
    *,
    working_memory: WorkingMemoryArtifact,
    invalidation_key: str,
    previous: DistillationRecord | None = None,
    projection_fingerprint: str = DEFAULT_PROJECTION_FINGERPRINT,
    embedding_fingerprint: str = DEFAULT_EMBEDDING_FINGERPRINT,
    retrieval_policy_fingerprint: str = DEFAULT_RETRIEVAL_POLICY_FINGERPRINT,
    distillation_prompt_fingerprint: str = DEFAULT_DISTILLATION_PROMPT_FINGERPRINT,
    budget_fingerprint: str = DEFAULT_BUDGET_FINGERPRINT,
) -> DistillationRecord:
    if (
        previous is not None
        and previous.invalidation_key == invalidation_key
        and previous.compiler_fingerprint == working_memory.compiler_fingerprint
    ):
        return DistillationRecord(
            distillation_id=previous.distillation_id,
            session_id=working_memory.session_id,
            turn_number=working_memory.turn_number,
            snapshot_id=working_memory.snapshot_id,
            compiler_fingerprint=working_memory.compiler_fingerprint,
            summary=previous.summary,
            source_refs=previous.source_refs,
            accepted_facts=previous.accepted_facts,
            reused_from_distillation_id=previous.distillation_id,
            invalidation_key=invalidation_key,
            created_at=working_memory.created_at,
            projection_fingerprint=projection_fingerprint,
            embedding_fingerprint=embedding_fingerprint,
            retrieval_policy_fingerprint=retrieval_policy_fingerprint,
            distillation_prompt_fingerprint=distillation_prompt_fingerprint,
            budget_fingerprint=budget_fingerprint,
        )

    statements = [fact.statement for fact in working_memory.accepted_facts]
    if not statements:
        statements = [
            ref.label or ref.path or ref.ref_id for ref in working_memory.evidence_refs
        ]
    summary = " ".join(item for item in statements if item).strip()
    if not summary:
        summary = "No durable facts have been accepted for this turn."
    distillation_id = _hash_payload(
        "dist",
        {
            "session_id": working_memory.session_id,
            "turn_number": working_memory.turn_number,
            "invalidation_key": invalidation_key,
            "summary": summary,
        },
    )
    return DistillationRecord(
        distillation_id=distillation_id,
        session_id=working_memory.session_id,
        turn_number=working_memory.turn_number,
        snapshot_id=working_memory.snapshot_id,
        compiler_fingerprint=working_memory.compiler_fingerprint,
        summary=summary,
        source_refs=working_memory.evidence_refs,
        accepted_facts=working_memory.accepted_facts,
        reused_from_distillation_id=None,
        invalidation_key=invalidation_key,
        created_at=working_memory.created_at,
        projection_fingerprint=projection_fingerprint,
        embedding_fingerprint=embedding_fingerprint,
        retrieval_policy_fingerprint=retrieval_policy_fingerprint,
        distillation_prompt_fingerprint=distillation_prompt_fingerprint,
        budget_fingerprint=budget_fingerprint,
    )


def build_activation_record(
    *,
    bundle_id: str,
    working_memory: WorkingMemoryArtifact,
    active_ref_ids: tuple[str, ...] | None = None,
    active_fact_ids: tuple[str, ...] | None = None,
    active_hypothesis_ids: tuple[str, ...] | None = None,
    reason: str | None = None,
    created_at: float | None = None,
) -> ActivationRecord:
    default_active_hypotheses = tuple(
        item.hypothesis_id
        for item in working_memory.hypotheses
        if item.state not in {"rejected", "blocked"}
    )
    ref_ids = (
        active_ref_ids
        if active_ref_ids is not None
        else tuple(ref.ref_id for ref in working_memory.evidence_refs)
    )
    fact_ids = (
        active_fact_ids
        if active_fact_ids is not None
        else tuple(fact.fact_id for fact in working_memory.accepted_facts)
    )
    hypothesis_ids = (
        active_hypothesis_ids
        if active_hypothesis_ids is not None
        else default_active_hypotheses
    )
    activation_reason = reason or working_memory.working_set.reason
    activation_id = _hash_payload(
        "act",
        {
            "bundle_id": bundle_id,
            "refs": ref_ids,
            "facts": fact_ids,
            "hypotheses": hypothesis_ids,
            "reason": activation_reason,
        },
    )
    return ActivationRecord(
        activation_id=activation_id,
        bundle_id=bundle_id,
        session_id=working_memory.session_id,
        turn_number=working_memory.turn_number,
        snapshot_id=working_memory.snapshot_id,
        compiler_fingerprint=working_memory.compiler_fingerprint,
        active_ref_ids=ref_ids,
        active_fact_ids=fact_ids,
        active_hypothesis_ids=hypothesis_ids,
        reason=activation_reason,
        created_at=created_at if created_at is not None else working_memory.created_at,
    )


def build_context_bundle(
    *,
    working_memory: WorkingMemoryArtifact,
    turn_journal: TurnJournal,
    previous_distillation: DistillationRecord | None = None,
    projection_fingerprint: str = DEFAULT_PROJECTION_FINGERPRINT,
    embedding_fingerprint: str = DEFAULT_EMBEDDING_FINGERPRINT,
    retrieval_policy_fingerprint: str = DEFAULT_RETRIEVAL_POLICY_FINGERPRINT,
    distillation_prompt_fingerprint: str = DEFAULT_DISTILLATION_PROMPT_FINGERPRINT,
    budget_fingerprint: str = DEFAULT_BUDGET_FINGERPRINT,
) -> ContextBundle:
    invalidation_key = build_context_invalidation_key(
        session_id=working_memory.session_id,
        snapshot_id=working_memory.snapshot_id,
        artifact_key=working_memory.artifact_key,
        evidence_refs=working_memory.evidence_refs,
        compiler_fingerprint=working_memory.compiler_fingerprint,
        projection_fingerprint=projection_fingerprint,
        embedding_fingerprint=embedding_fingerprint,
        retrieval_policy_fingerprint=retrieval_policy_fingerprint,
        distillation_prompt_fingerprint=distillation_prompt_fingerprint,
        budget_fingerprint=budget_fingerprint,
    )
    bundle_id = build_context_bundle_id(
        session_id=working_memory.session_id,
        turn_number=working_memory.turn_number,
        invalidation_key=invalidation_key,
        compiler_fingerprint=working_memory.compiler_fingerprint,
    )
    distillation = build_distillation_record(
        working_memory=working_memory,
        invalidation_key=invalidation_key,
        previous=previous_distillation,
        projection_fingerprint=projection_fingerprint,
        embedding_fingerprint=embedding_fingerprint,
        retrieval_policy_fingerprint=retrieval_policy_fingerprint,
        distillation_prompt_fingerprint=distillation_prompt_fingerprint,
        budget_fingerprint=budget_fingerprint,
    )
    activation = build_activation_record(
        bundle_id=bundle_id,
        working_memory=working_memory,
    )
    return ContextBundle(
        bundle_id=bundle_id,
        session_id=working_memory.session_id,
        turn_number=working_memory.turn_number,
        snapshot_id=working_memory.snapshot_id,
        artifact_key=working_memory.artifact_key,
        compiler_fingerprint=working_memory.compiler_fingerprint,
        working_memory=working_memory,
        turn_journal=turn_journal,
        distillation=distillation,
        activation=activation,
        created_at=working_memory.created_at,
        projection_fingerprint=projection_fingerprint,
        embedding_fingerprint=embedding_fingerprint,
        retrieval_policy_fingerprint=retrieval_policy_fingerprint,
        distillation_prompt_fingerprint=distillation_prompt_fingerprint,
        budget_fingerprint=budget_fingerprint,
    )


def render_context_bundle(
    bundle: ContextBundle,
    *,
    token_budget: int = 2048,
) -> dict[str, Any]:
    budget = max(1, int(token_budget))
    lines = [
        f"bundle {bundle.bundle_id}",
        f"session {bundle.session_id} turn {bundle.turn_number}",
        f"snapshot {bundle.snapshot_id or '-'} artifact {bundle.artifact_key or '-'}",
        f"distillation {bundle.distillation.distillation_id}",
        bundle.distillation.summary,
    ]
    for fact in bundle.distillation.accepted_facts:
        refs = ",".join(fact.ref_ids) or "non-code:accepted-fact"
        lines.append(f"fact {fact.fact_id} refs={refs}: {fact.statement}")
    for ref in bundle.distillation.source_refs:
        source = ref.path or ref.symbol_id or ref.label or "non-code:source"
        lines.append(f"source {ref.ref_id} {source} {ref.lines or '-'}")

    tokens_used = 0
    rendered: list[str] = []
    truncated = False
    for line in lines:
        words = line.split()
        if tokens_used + len(words) > budget:
            remaining = budget - tokens_used
            if remaining > 0:
                rendered.append(" ".join(words[:remaining]))
                tokens_used += remaining
            truncated = True
            break
        rendered.append(line)
        tokens_used += len(words)

    return {
        "bundle_id": bundle.bundle_id,
        "token_budget": budget,
        "tokens_used": tokens_used,
        "truncated": truncated,
        "text": "\n".join(rendered),
    }


def expand_bundle_source_ref(
    bundle: ContextBundle,
    ref_id: str,
    *,
    depth: str = "L2",
) -> dict[str, Any] | None:
    for ref in bundle.distillation.source_refs:
        if ref.ref_id != ref_id:
            continue
        return {
            "bundle_id": bundle.bundle_id,
            "session_id": bundle.session_id,
            "turn_number": bundle.turn_number,
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
    return None


def build_lines_index(
    evidence_refs: tuple[EvidenceRef, ...],
) -> dict[str, dict[str, Any]]:
    return {
        item.ref_id: {
            "kind": item.kind,
            "repo_name": item.repo_name,
            "path": item.path,
            "lines": item.lines,
            "label": item.label,
        }
        for item in evidence_refs
    }


def build_handoff_from_working_memory(
    *,
    working_memory: WorkingMemoryArtifact,
    mode: str = "delegate",
) -> HandoffArtifact:
    from .agent_context import build_handoff_artifact

    artifact_id = build_handoff_artifact_id(
        working_memory.session_id, working_memory.turn_number, mode
    )
    return build_handoff_artifact(
        artifact_id=artifact_id,
        mode=mode,
        working_memory=working_memory,
    )
