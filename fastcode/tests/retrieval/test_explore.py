from __future__ import annotations

import pytest

from fastcode.retrieval.context.explore import build_explore_code_payload
from fastcode.retrieval.contracts import (
    Hit,
    ProjectionEvidence,
    RetrievalSource,
    TraceLink,
)


def _hit(
    element_id: str,
    *,
    repo: str = "repo",
    path: str = "src/app.py",
    name: str = "run",
    start_line: int = 10,
    code: str = "def run():\n    return True",
    score: float = 1.0,
) -> Hit:
    return Hit.from_element(
        {
            "id": element_id,
            "type": "function",
            "name": name,
            "repo_name": repo,
            "relative_path": path,
            "language": "python",
            "start_line": start_line,
            "end_line": start_line + 1,
            "code": code,
            "signature": f"def {name}()",
        },
        score=score,
        source=RetrievalSource.SEMANTIC,
    )


def test_explore_payload_groups_hits_by_repo_and_file_with_line_numbered_code() -> None:
    payload = build_explore_code_payload(
        question="Where does app run?",
        snapshot_id="snap:repo:1",
        artifact_key="artifact:1",
        repo_filter=("repo",),
        hits=[
            _hit("u:helper", name="helper", start_line=30, score=0.7),
            _hit("u:run", name="run", start_line=10, score=0.9),
        ],
    )

    assert payload["freshness"]["state"] == "fresh"
    assert payload["completeness"] == {
        "state": "complete",
        "returned_snippets": 2,
        "omitted_snippets": 0,
        "returned_groups": 1,
    }
    assert payload["groups"][0]["ref_id"] == "g1"
    snippets = payload["groups"][0]["snippets"]
    assert [snippet["ref_id"] for snippet in snippets] == ["e1", "e2"]
    assert [snippet["name"] for snippet in snippets] == ["run", "helper"]
    assert "  10 | def run():" in snippets[0]["code"]
    assert snippets[0]["element_id"] == "u:run"
    assert snippets[0]["evidence_refs"][0] == {
        "ref_id": "e1",
        "kind": "retrieval_hit",
        "element_id": "u:run",
        "source": "semantic",
        "score": 0.9,
        "scores": {
            "semantic": 0.0,
            "keyword": 0.0,
            "pseudocode": 0.0,
            "graph": 0.0,
        },
    }
    assert snippets[0]["expansion"]["tool"] == "explore_code"


def test_explore_payload_reports_omissions_and_minimal_keeps_refs_without_code() -> (
    None
):
    payload = build_explore_code_payload(
        question="Find auth",
        detail_level="minimal",
        max_snippets=1,
        hits=[
            _hit("u:a", path="src/a.py", score=0.9),
            _hit("u:b", path="src/b.py", score=0.8),
        ],
    )

    assert payload["freshness"]["state"] == "unknown"
    assert payload["completeness"]["state"] == "partial"
    assert payload["completeness"]["omitted_snippets"] == 1
    snippet = payload["groups"][0]["snippets"][0]
    assert snippet["ref_id"] == "e1"
    assert "code" not in snippet


def test_explore_payload_rejects_unknown_detail_level() -> None:
    with pytest.raises(ValueError, match="detail_level"):
        build_explore_code_payload(
            question="Find auth",
            detail_level="verbose",
            hits=[],
        )


def test_explore_payload_includes_graph_relationships_and_projection_evidence() -> None:
    trace = ProjectionEvidence(
        link=TraceLink(
            unit_id="unit:auth",
            weight=0.7,
            evidence_type="design_ref",
            chunk_id="doc:chunk:1",
            symbol_name="authenticate",
            confidence="high",
        ),
        doc_id="doc:adr",
        doc_score=0.81,
        contribution=0.42,
    )
    hit = _hit(
        "u:auth",
        path="src/auth.py",
        name="authenticate",
        score=0.9,
    ).with_scores(
        semantic_score=0.6,
        keyword_score=0.2,
        graph_score=0.4,
        traceability=(trace,),
    )
    hit = Hit.from_retrieval_row(
        {
            **hit.to_retrieval_row(),
            "related_to": "u:caller",
        }
    )

    payload = build_explore_code_payload(
        question="Where is auth?",
        hits=[hit],
    )

    snippet = payload["groups"][0]["snippets"][0]
    assert snippet["graph_relationships"] == [
        {
            "source_element_id": "u:caller",
            "target_element_id": "u:auth",
            "relationship": "related",
            "provenance": "semantic",
            "score": 0.4,
        }
    ]
    assert snippet["evidence_refs"][1] == {
        "ref_id": "e1:trace:1",
        "kind": "projection_trace",
        "doc_id": "doc:adr",
        "unit_id": "unit:auth",
        "evidence_type": "design_ref",
        "chunk_id": "doc:chunk:1",
        "confidence": "high",
        "score": 0.81,
        "contribution": 0.42,
    }
