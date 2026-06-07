"""Scenario contracts for session-prefix projection consumption."""

from __future__ import annotations

from fastcode.retrieval.context.agent_context import (
    AcceptedFact,
    EvidenceRef,
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


def _given_agent_turn_with_l0_l1_projection_prefix() -> tuple[
    TurnIntent, dict[str, object]
]:
    intent = TurnIntent(
        session_id="session-prefix",
        turn_number=1,
        question="Where does authentication start?",
        kind="where",
        requested_outcome="answer",
        snapshot_id="snap:auth",
        artifact_key="artifact:auth",
        repo_filter=("repo",),
    )
    session_prefix = {
        "projection_id": "proj:auth",
        "l0": {
            "summary": "Auth service coordinates login and token storage.",
            "clusters": ["auth", "storage"],
        },
        "l1": {
            "summary": "Auth flow calls TokenStore.save.",
            "auth": ["AuthService.login", "TokenStore.save"],
            "edges": ["AuthService.login -> TokenStore.save"],
        },
    }
    return intent, session_prefix


def _when_working_memory_is_compiled(
    intent: TurnIntent,
    session_prefix: dict[str, object],
) -> tuple[WorkingMemoryArtifact, TurnJournal]:
    contract = build_acceptance_contract(requested_outcome=intent.requested_outcome)
    risk_state = RiskState(
        evidence_gap=1,
        conflict_level=0,
        freshness_risk=0,
        requirement_ambiguity=0,
        execution_risk=0,
        verifier_status="pending",
        action_bias="retrieve",
    )
    plan = build_turn_plan(risk_state=risk_state, contract=contract)
    evidence_ref = EvidenceRef(
        ref_id="e1",
        kind="projection",
        repo_name="repo",
        snapshot_id=intent.snapshot_id,
        label="L0/L1 projection",
        source="projection",
        fresh="ok",
    )
    observations = (
        build_tool_observation(
            observation_id="o1",
            tool="projection",
            ok=True,
            parameters={"layers": ["L0", "L1"]},
            ref_ids=(evidence_ref.ref_id,),
            summary="Loaded compact architectural prefix.",
        ),
    )
    accepted_facts = (
        AcceptedFact(
            fact_id="f1",
            statement="Authentication architecture is available in the projection prefix.",
            ref_ids=(evidence_ref.ref_id,),
            scope="session",
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
        hypotheses=(),
        rejected_hypotheses=(),
        unresolved_questions=(),
        session_prefix=session_prefix,
        created_at=1234.5,
    )
    journal = build_turn_journal(
        intent=intent,
        plan=plan,
        observations=observations,
        evidence_refs=(evidence_ref,),
        risk_state=risk_state,
        acceptance_contract=contract,
        hypotheses=(),
        rejected_hypotheses=(),
        accepted_facts=accepted_facts,
        working_set=working_memory.working_set,
        answer_summary=None,
        created_at=working_memory.created_at,
    )
    return working_memory, journal


def _then_l0_l1_prefix_is_stable_session_context(
    working_memory: WorkingMemoryArtifact,
) -> None:
    assert "\nL0 " in working_memory.stable_fcx
    assert "\nL1 " in working_memory.stable_fcx
    assert "proj:auth" in working_memory.stable_fcx
    assert "Auth service coordinates login" in working_memory.stable_fcx
    assert "Auth flow calls TokenStore.save" in working_memory.stable_fcx
    assert "\nL0 " not in working_memory.turn_fcx
    assert "\nL1 " not in working_memory.obs_fcx


def test_l0_l1_projection_prefix_loads_as_stable_session_context() -> None:
    """Scenario: session startup projection becomes stable agent context."""
    intent, session_prefix = _given_agent_turn_with_l0_l1_projection_prefix()

    working_memory, _journal = _when_working_memory_is_compiled(intent, session_prefix)

    _then_l0_l1_prefix_is_stable_session_context(working_memory)


def test_projection_fingerprint_controls_context_bundle_reuse() -> None:
    """Scenario: changing the projection prefix invalidates context reuse."""
    intent, session_prefix = _given_agent_turn_with_l0_l1_projection_prefix()
    working_memory, journal = _when_working_memory_is_compiled(intent, session_prefix)

    old_key = build_context_invalidation_key(
        session_id=working_memory.session_id,
        snapshot_id=working_memory.snapshot_id,
        artifact_key=working_memory.artifact_key,
        evidence_refs=working_memory.evidence_refs,
        compiler_fingerprint=working_memory.compiler_fingerprint,
        projection_fingerprint="projection:old",
    )
    new_key = build_context_invalidation_key(
        session_id=working_memory.session_id,
        snapshot_id=working_memory.snapshot_id,
        artifact_key=working_memory.artifact_key,
        evidence_refs=working_memory.evidence_refs,
        compiler_fingerprint=working_memory.compiler_fingerprint,
        projection_fingerprint="projection:new",
    )
    bundle = build_context_bundle(
        working_memory=working_memory,
        turn_journal=journal,
        projection_fingerprint="projection:new",
    )

    assert old_key != new_key
    assert bundle.distillation.invalidation_key == new_key
    assert bundle.projection_fingerprint == "projection:new"
