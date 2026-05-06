"""Compile typed agent context into FCX working-memory artifacts."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from . import fcx as _fcx
from .agent_context import (
    AcceptanceContract,
    AcceptedFact,
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


def build_evidence_refs_from_sources(
    sources: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    snapshot_id: str | None,
    source_kind: str = "range",
) -> tuple[EvidenceRef, ...]:
    refs: list[EvidenceRef] = []
    for index, item in enumerate(sources, 1):
        repo_name = str(item.get("repository") or item.get("repo") or "")
        path = str(item.get("file") or item.get("path") or "")
        lines_value = item.get("lines")
        start_line = item.get("start_line")
        end_line = item.get("end_line")
        if (
            not lines_value
            and isinstance(start_line, int)
            and isinstance(end_line, int)
        ):
            lines_value = f"{start_line}-{end_line}"
        lines = str(lines_value or "")
        score_value = item.get("score")
        if score_value is None:
            score_value = item.get("total_score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        refs.append(
            EvidenceRef(
                ref_id=f"e{index}",
                kind=source_kind,
                repo_name=repo_name or None,
                snapshot_id=snapshot_id,
                path=path or None,
                lines=lines or None,
                label=str(item.get("name") or path or f"source-{index}"),
                score=score,
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
    l0 = (session_prefix or {}).get("l0")
    l1 = (session_prefix or {}).get("l1")
    projection_id = str((session_prefix or {}).get("projection_id") or "")
    if isinstance(l0, dict):
        stable_records.append(
            _fcx.render_record(
                "L0",
                fields={
                    "p": projection_id or "proj",
                    "tok": len(str(l0.get("summary") or "")),
                },
                tail=str(l0.get("summary") or ""),
            )
        )
    if isinstance(l1, dict):
        stable_records.append(
            _fcx.render_record(
                "L1",
                fields={
                    "p": projection_id or "proj",
                    "tok": len(str(l1.get("summary") or "")),
                },
                tail=str(l1.get("summary") or ""),
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
