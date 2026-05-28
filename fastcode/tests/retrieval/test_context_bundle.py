from __future__ import annotations

from fastcode.retrieval.context.agent_context import (
    AcceptedFact,
    EvidenceRef,
    Hypothesis,
    RiskState,
    TurnIntent,
    TurnJournal,
    WorkingMemoryArtifact,
    build_acceptance_contract,
)
from fastcode.retrieval.context.context_compiler import (
    build_context_bundle,
    build_context_invalidation_key,
    build_tool_observation,
    build_turn_journal,
    build_turn_plan,
    compile_working_memory,
)


def _working_memory(
    *,
    turn_number: int,
    evidence_ref: EvidenceRef,
) -> tuple[WorkingMemoryArtifact, TurnJournal]:
    intent = TurnIntent(
        session_id="session-agent",
        turn_number=turn_number,
        question="Where is auth handled?",
        kind="debug",
        requested_outcome="answer",
        snapshot_id="snap:1",
        artifact_key="art:1",
        repo_filter=("repo",),
    )
    contract = build_acceptance_contract(requested_outcome="answer")
    risk_state = RiskState(
        evidence_gap=0,
        conflict_level=0,
        freshness_risk=0,
        requirement_ambiguity=0,
        execution_risk=0,
        verifier_status="clean",
        action_bias="answer",
    )
    plan = build_turn_plan(risk_state=risk_state, contract=contract)
    observations = (
        build_tool_observation(
            observation_id="o1",
            tool="retrieve",
            ok=True,
            parameters={"mode": "standard"},
            ref_ids=(evidence_ref.ref_id,),
            summary="Retrieved source evidence.",
        ),
    )
    accepted_facts = (
        AcceptedFact(
            fact_id="f1",
            statement="Auth is grounded in the cited source range.",
            ref_ids=(evidence_ref.ref_id,),
            scope="turn",
        ),
    )
    hypotheses = (
        Hypothesis(
            hypothesis_id="h1",
            statement="Auth logic lives in the cited source range.",
            confidence=0.91,
            support_ref_ids=(evidence_ref.ref_id,),
            conflict_ref_ids=(),
            state="favored",
        ),
    )
    working_memory = compile_working_memory(
        intent=intent,
        contract=contract,
        risk_state=risk_state,
        plan=plan,
        evidence_refs=(evidence_ref,),
        observations=observations,
        accepted_facts=accepted_facts,
        hypotheses=hypotheses,
        rejected_hypotheses=(),
        unresolved_questions=(),
        session_prefix=None,
        created_at=1234.5 + turn_number,
    )
    journal = build_turn_journal(
        intent=intent,
        plan=plan,
        observations=observations,
        evidence_refs=(evidence_ref,),
        risk_state=risk_state,
        acceptance_contract=contract,
        hypotheses=hypotheses,
        rejected_hypotheses=(),
        accepted_facts=accepted_facts,
        working_set=working_memory.working_set,
        answer_summary="Auth is grounded in the cited source range.",
        created_at=working_memory.created_at,
    )
    return working_memory, journal


def test_context_bundle_distillation_reuse_preserves_source_refs() -> None:
    source_ref = EvidenceRef(
        ref_id="e1",
        kind="range",
        repo_name="repo",
        snapshot_id="snap:1",
        path="src/auth.py",
        lines="10-20",
        label="auth.py",
        source="retrieval",
        fresh="ok",
    )
    working_memory, journal = _working_memory(
        turn_number=1,
        evidence_ref=source_ref,
    )
    first_bundle = build_context_bundle(
        working_memory=working_memory,
        turn_journal=journal,
    )

    next_memory, next_journal = _working_memory(
        turn_number=2,
        evidence_ref=source_ref,
    )
    reused_bundle = build_context_bundle(
        working_memory=next_memory,
        turn_journal=next_journal,
        previous_distillation=first_bundle.distillation,
    )

    assert (
        reused_bundle.distillation.reused_from_distillation_id
        == first_bundle.distillation.distillation_id
    )
    assert reused_bundle.distillation.source_refs[0].path == "src/auth.py"
    assert reused_bundle.distillation.source_refs[0].lines == "10-20"
    assert reused_bundle.activation.active_ref_ids == ("e1",)
    assert reused_bundle.distillation.accepted_facts[0].ref_ids == ("e1",)


def test_context_bundle_distillation_invalidation_changes_with_source_refs() -> None:
    auth_ref = EvidenceRef(
        ref_id="e1",
        kind="range",
        repo_name="repo",
        snapshot_id="snap:1",
        path="src/auth.py",
        lines="10-20",
        label="auth.py",
        source="retrieval",
        fresh="ok",
    )
    config_ref = EvidenceRef(
        ref_id="e1",
        kind="range",
        repo_name="repo",
        snapshot_id="snap:1",
        path="src/config.py",
        lines="5-8",
        label="config.py",
        source="retrieval",
        fresh="ok",
    )
    auth_memory, auth_journal = _working_memory(turn_number=1, evidence_ref=auth_ref)
    auth_bundle = build_context_bundle(
        working_memory=auth_memory,
        turn_journal=auth_journal,
    )
    config_memory, config_journal = _working_memory(
        turn_number=2,
        evidence_ref=config_ref,
    )
    config_bundle = build_context_bundle(
        working_memory=config_memory,
        turn_journal=config_journal,
        previous_distillation=auth_bundle.distillation,
    )

    auth_key = build_context_invalidation_key(
        session_id=auth_memory.session_id,
        snapshot_id=auth_memory.snapshot_id,
        artifact_key=auth_memory.artifact_key,
        evidence_refs=auth_memory.evidence_refs,
        compiler_fingerprint=auth_memory.compiler_fingerprint,
    )
    config_key = build_context_invalidation_key(
        session_id=config_memory.session_id,
        snapshot_id=config_memory.snapshot_id,
        artifact_key=config_memory.artifact_key,
        evidence_refs=config_memory.evidence_refs,
        compiler_fingerprint=config_memory.compiler_fingerprint,
    )

    assert auth_key != config_key
    assert config_bundle.distillation.reused_from_distillation_id is None
    assert config_bundle.distillation.source_refs[0].path == "src/config.py"
