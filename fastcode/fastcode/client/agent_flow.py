"""AgentFlow -- AI-agent-optimized workflow surface for the client axis.

Semantic ``use_flow`` over the FastCode agent-context HTTP API. Wraps session
lifecycle, the ask workflow, durable token-budgeted context bundles, working
memory, activation, and handoff into an agent-friendly interface.

This layer is the FastCode analogue of ``app/query/facade.py`` but on the
client axis: a pure workflow layer that delegates every call to the run_kit
``FastCodeClient`` (httpx transport). It owns NO httpx imports, NO FastCode
server internals, and NO env/config reading. Agents import it directly
(``from fastcode.client.agent_flow import AgentFlow``) or the cli assembly_root
constructs it via ``AgentFlow.for_base_url``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastcode.client.http_client import FastCodeClient


class AgentFlow:
    """AI-agent-optimized client-axis workflow over a run_kit HTTP transport.

    Construct with an existing :class:`FastCodeClient` (run_kit) or via
    :meth:`for_base_url` from the cli assembly_root.
    """

    def __init__(self, client: "FastCodeClient") -> None:
        self._client = client

    @classmethod
    def for_base_url(cls, base_url: str, timeout: float = 300.0) -> "AgentFlow":
        """Build an :class:`AgentFlow` from a server base URL (cli entry point)."""
        from fastcode.client.http_client import FastCodeClient

        return cls(FastCodeClient(base_url=base_url, timeout=timeout))

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, *, clear_session_id: str | None = None) -> dict[str, Any]:
        """Open (or reset) a dialogue session; returns the new session descriptor."""
        return self._client.new_session(clear_session_id=clear_session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return the list of dialogue sessions."""
        return self._client.list_sessions().get("sessions", [])

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return the dialogue history for a session."""
        return self._client.get_session(session_id)

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete a dialogue session."""
        return self._client.delete_session(session_id)

    # ------------------------------------------------------------------
    # Core ask workflow
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        repo_filter: list[str] | None = None,
        multi_turn: bool = False,
    ) -> dict[str, Any]:
        """Ask one question. In ``multi_turn`` mode the answer carries a
        ``turn_number`` for subsequent context-retrieval calls."""
        return self._client.query(
            question=question,
            repo_filter=repo_filter,
            session_id=session_id if multi_turn else None,
            multi_turn=multi_turn,
        )

    # ------------------------------------------------------------------
    # Agent-context retrieval (LLM-ready)
    # ------------------------------------------------------------------

    def get_context_for_llm(
        self,
        session_id: str,
        *,
        turn_number: int | None = None,
        token_budget: int = 2048,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """Fetch the durable, token-budgeted context bundle ready to drop into
        an LLM prompt. This is the primary AI-agent-optimized surface."""
        return self._client.get_context_bundle(
            session_id=session_id,
            turn_number=turn_number,
            token_budget=token_budget,
            format=output_format,
        )

    def get_turn_context(
        self,
        session_id: str,
        *,
        turn_number: int | None = None,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """Fetch the typed working-memory artifact for the latest (or given) turn."""
        return self._client.get_turn_context(
            session_id=session_id,
            turn_number=turn_number,
            output_format=output_format,
        )

    # ------------------------------------------------------------------
    # Artifact persistence / handoff workflow
    # ------------------------------------------------------------------

    def activate(
        self,
        *,
        session_id: str,
        turn_number: int | None = None,
        bundle_id: str | None = None,
        active_ref_ids: list[str] | None = None,
        active_fact_ids: list[str] | None = None,
        active_hypothesis_ids: list[str] | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Record an activation (which refs/facts/hypotheses are live) for a bundle."""
        return self._client.create_activation(
            session_id=session_id,
            turn_number=turn_number,
            bundle_id=bundle_id,
            active_ref_ids=active_ref_ids,
            active_fact_ids=active_fact_ids,
            active_hypothesis_ids=active_hypothesis_ids,
            reason=reason,
        )

    def handoff_artifact(
        self,
        *,
        session_id: str,
        turn_number: int | None = None,
        mode: str = "summary",
    ) -> dict[str, Any]:
        """Persist a handoff artifact for a session turn; returns the artifact descriptor."""
        return self._client.create_handoff(
            session_id=session_id, turn_number=turn_number, mode=mode
        )

    def get_handoff(self, artifact_id: str) -> dict[str, Any]:
        """Retrieve a persisted handoff artifact by id."""
        return self._client.get_handoff(artifact_id)

    # ------------------------------------------------------------------
    # Convenience: a full single-shot agent turn
    # ------------------------------------------------------------------

    def run_turn(
        self,
        question: str,
        session_id: str,
        *,
        token_budget: int = 2048,
    ) -> dict[str, Any]:
        """Ask, then return the LLM-ready context bundle for the turn produced.

        The all-in-one agent entry point: one question, one answer, plus the
        durable context ready for the next model call.
        """
        answer = self.ask(question, session_id=session_id, multi_turn=True)
        turn = answer.get("turn_number")
        context_bundle = self.get_context_for_llm(
            session_id, turn_number=turn, token_budget=token_budget
        )
        return {"answer": answer, "context_bundle": context_bundle}
