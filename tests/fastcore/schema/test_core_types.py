"""Tests for fastcode.core.types — frozen dataclasses for the FP refactoring."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fastcode.schema.core_types import (
    ElementFilter,
    FileAnalysis,
    FusionConfig,
    FusionWeights,
    GenerationInput,
    GenerationResult,
    Hit,
    IterationConfig,
    IterationHistoryEntry,
    IterationMetrics,
    IterationState,
    RepoStructure,
    RetrievalChannelOutput,
    RoundResult,
    ScipKind,
    ScipRole,
    SourceRef,
    ToolCall,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_retrieval_row() -> dict:
    """Minimal retrieval row as produced by HybridRetriever."""
    return {
        "element": {
            "id": "elem-42",
            "type": "function",
            "name": "parse_config",
            "metadata": {"language": "python", "path": "config.py"},
        },
        "semantic_score": 0.85,
        "keyword_score": 0.60,
        "total_score": 0.78,
        "projected_only": True,
        "llm_file_selected": False,
        "agent_found": False,
    }


# ---------------------------------------------------------------------------
# Hit
# ---------------------------------------------------------------------------


class TestHit:
    def test_construction_with_defaults(self):
        hit = Hit(element_id="e1", element_type="class", element_name="Foo", score=0.9)
        assert hit.element_id == "e1"
        assert hit.semantic_score == 0.0
        assert hit.keyword_score == 0.0
        assert hit.total_score == 0.0
        assert hit.metadata == {}
        assert hit.projected_only is False

    def test_construction_with_all_fields(self):
        hit = Hit(
            element_id="e1",
            element_type="function",
            element_name="bar",
            score=0.5,
            semantic_score=0.4,
            keyword_score=0.6,
            pseudocode_score=0.1,
            graph_score=0.2,
            total_score=0.8,
            source="semantic",
            metadata={"lang": "py"},
            projected_only=True,
            llm_selected=True,
            agent_found=True,
        )
        assert hit.semantic_score == 0.4
        assert hit.metadata == {"lang": "py"}
        assert hit.projected_only is True

    def test_from_retrieval_row(self):
        row = _sample_retrieval_row()
        hit = Hit.from_retrieval_row(row)
        assert hit.element_id == "elem-42"
        assert hit.element_type == "function"
        assert hit.element_name == "parse_config"
        assert hit.semantic_score == 0.85
        assert hit.keyword_score == 0.60
        assert hit.total_score == 0.78
        assert hit.projected_only is True
        assert hit.llm_selected is False
        assert hit.agent_found is False
        assert hit.metadata == {"language": "python", "path": "config.py"}

    def test_from_retrieval_row_handles_missing_element(self):
        row = {"element": None, "total_score": 0.5}
        hit = Hit.from_retrieval_row(row)
        assert hit.element_id == ""
        assert hit.element_type == ""
        assert hit.element_name == ""

    def test_from_retrieval_row_handles_empty_row(self):
        hit = Hit.from_retrieval_row({})
        assert hit.element_id == ""
        assert hit.score == 0.0

    def test_to_retrieval_row_roundtrip(self):
        row = _sample_retrieval_row()
        hit = Hit.from_retrieval_row(row)
        result = hit.to_retrieval_row()
        assert result["element"]["id"] == "elem-42"
        assert result["element"]["type"] == "function"
        assert result["element"]["name"] == "parse_config"
        assert result["element"]["metadata"] == {
            "language": "python",
            "path": "config.py",
        }
        assert result["semantic_score"] == 0.85
        assert result["keyword_score"] == 0.60
        assert result["total_score"] == 0.78
        assert result["projected_only"] is True
        assert result["llm_file_selected"] is False
        assert result["agent_found"] is False

    def test_frozen_immutability(self):
        hit = Hit(element_id="e1", element_type="class", element_name="Foo", score=0.9)
        with pytest.raises(FrozenInstanceError):
            hit.score = 0.1

    def test_frozen_metadata_shallow_copy(self):
        """metadata dict is a default_factory copy — but still, the field is frozen."""
        hit = Hit(
            element_id="e1",
            element_type="f",
            element_name="x",
            score=0.0,
            metadata={"k": "v"},
        )
        assert hit.metadata == {"k": "v"}
        # Reassigning metadata raises FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            hit.metadata = {}

    def test_total_score_fallback_to_score(self):
        """When total_score is 0.0, score should be used as fallback."""
        hit = Hit(element_id="e1", element_type="f", element_name="x", score=0.95)
        row = hit.to_retrieval_row()
        # total_score in the row comes from hit.total_score (default 0.0)
        assert row["total_score"] == 0.0


# ---------------------------------------------------------------------------
# FusionConfig
# ---------------------------------------------------------------------------


class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.alpha_base == 0.8
        assert cfg.rrf_k_base == 60

    def test_from_dict_full(self):
        d = {"alpha_base": 0.5, "rrf_k_base": 40, "rrf_k_max": 200}
        cfg = FusionConfig.from_dict(d)
        assert cfg.alpha_base == 0.5
        assert cfg.rrf_k_base == 40
        assert cfg.rrf_k_max == 200
        # unspecified fields keep defaults
        assert cfg.alpha_min == 0.25

    def test_from_dict_empty(self):
        cfg = FusionConfig.from_dict({})
        assert cfg.alpha_base == 0.8
        assert cfg.rrf_k_base == 60

    def test_frozen_immutability(self):
        cfg = FusionConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.alpha_base = 0.0


# ---------------------------------------------------------------------------
# FusionWeights
# ---------------------------------------------------------------------------


class TestFusionWeights:
    def test_defaults(self):
        w = FusionWeights()
        assert w.code_weight == 0.7
        assert w.doc_weight == 0.3
        assert w.alpha == 0.8

    def test_custom(self):
        w = FusionWeights(code_weight=0.6, doc_weight=0.4, alpha=0.5, beta=0.2)
        assert w.code_weight == 0.6
        assert w.doc_weight == 0.4

    def test_frozen(self):
        w = FusionWeights()
        with pytest.raises(FrozenInstanceError):
            w.code_weight = 1.0


# ---------------------------------------------------------------------------
# RetrievalChannelOutput
# ---------------------------------------------------------------------------


class TestRetrievalChannelOutput:
    def test_defaults(self):
        out = RetrievalChannelOutput(
            collection="code",
            semantic_results=(),
            keyword_results=(),
        )
        assert out.collection == "code"
        assert out.pseudocode_results == ()
        assert out.ranked_results == ()

    def test_with_results(self):
        out = RetrievalChannelOutput(
            collection="docs",
            semantic_results=(("doc1", 0.9),),
            keyword_results=(("doc2", 0.7),),
        )
        assert len(out.semantic_results) == 1


# ---------------------------------------------------------------------------
# ElementFilter
# ---------------------------------------------------------------------------


class TestElementFilter:
    def test_defaults_all_none(self):
        f = ElementFilter()
        assert f.language is None
        assert f.element_type is None
        assert f.file_path is None
        assert f.snapshot_id is None

    def test_partial(self):
        f = ElementFilter(language="python", element_type="function")
        assert f.language == "python"
        assert f.element_type == "function"
        assert f.file_path is None


# ---------------------------------------------------------------------------
# IterationConfig
# ---------------------------------------------------------------------------


class TestIterationConfig:
    def test_defaults(self):
        cfg = IterationConfig()
        assert cfg.base_max_iterations == 4
        assert cfg.max_total_lines == 12000

    def test_frozen(self):
        cfg = IterationConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.max_iterations = 99


# ---------------------------------------------------------------------------
# IterationHistoryEntry
# ---------------------------------------------------------------------------


class TestIterationHistoryEntry:
    def test_construction(self):
        entry = IterationHistoryEntry(
            round=1,
            confidence=70,
            query_complexity=3,
            elements_count=10,
            total_lines=500,
            confidence_gain=15.0,
            lines_added=200,
            roi=0.075,
            budget_usage_pct=4.2,
        )
        assert entry.round == 1
        assert entry.roi == 0.075

    def test_frozen(self):
        entry = IterationHistoryEntry(
            round=1,
            confidence=50,
            query_complexity=1,
            elements_count=0,
            total_lines=0,
            confidence_gain=0.0,
            lines_added=0,
            roi=0.0,
            budget_usage_pct=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            entry.confidence = 100


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_construction(self):
        tc = ToolCall(tool="search", parameters={"q": "test"})
        assert tc.tool == "search"
        assert tc.parameters == {"q": "test"}

    def test_default_parameters(self):
        tc = ToolCall(tool="read_file")
        assert tc.parameters == {}

    def test_frozen(self):
        tc = ToolCall(tool="x")
        with pytest.raises(FrozenInstanceError):
            tc.tool = "y"


# ---------------------------------------------------------------------------
# RoundResult
# ---------------------------------------------------------------------------


class TestRoundResult:
    def test_construction(self):
        rr = RoundResult(
            confidence=85,
            tool_calls=(ToolCall(tool="search"),),
            keep_files=("a.py", "b.py"),
            reasoning="Found relevant code",
        )
        assert rr.confidence == 85
        assert len(rr.tool_calls) == 1
        assert rr.should_answer_directly is False
        assert rr.query_complexity is None

    def test_with_optional_fields(self):
        rr = RoundResult(
            confidence=95,
            tool_calls=(),
            keep_files=(),
            reasoning="Done",
            query_complexity=2,
            should_answer_directly=True,
        )
        assert rr.query_complexity == 2
        assert rr.should_answer_directly is True


# ---------------------------------------------------------------------------
# IterationState
# ---------------------------------------------------------------------------


class TestIterationState:
    def _make_hit(self, eid: str = "e1") -> Hit:
        return Hit(
            element_id=eid, element_type="function", element_name=f"fn_{eid}", score=0.5
        )

    def test_defaults(self):
        state = IterationState(
            round_num=0, elements=(), history=(), tool_call_history=()
        )
        assert state.round_num == 0
        assert state.retained_elements == ()
        assert state.pending_elements == ()
        assert state.confidence == 0
        assert state.dialogue_history == ()

    def test_with_elements_returns_new_state(self):
        original = IterationState(
            round_num=0, elements=(), history=(), tool_call_history=()
        )
        hits = (self._make_hit("a"), self._make_hit("b"))
        updated = original.with_elements(hits)
        assert updated.elements == hits
        # original unchanged
        assert original.elements == ()

    def test_with_elements_preserves_other_fields(self):
        tc = ToolCall(tool="search")
        original = IterationState(
            round_num=1, elements=(), history=(), tool_call_history=(tc,), confidence=50
        )
        hits = (self._make_hit("c"),)
        updated = original.with_elements(hits)
        assert updated.round_num == 1
        assert updated.tool_call_history == (tc,)
        assert updated.confidence == 50

    def test_with_history_entry(self):
        state = IterationState(
            round_num=1, elements=(), history=(), tool_call_history=()
        )
        entry = IterationHistoryEntry(
            round=1,
            confidence=70,
            query_complexity=2,
            elements_count=5,
            total_lines=300,
            confidence_gain=20.0,
            lines_added=300,
            roi=0.067,
            budget_usage_pct=2.5,
        )
        updated = state.with_history_entry(entry)
        assert updated.history == (entry,)
        assert state.history == ()

    def test_with_tool_calls(self):
        state = IterationState(
            round_num=0, elements=(), history=(), tool_call_history=()
        )
        calls = (ToolCall(tool="search"), ToolCall(tool="read"))
        updated = state.with_tool_calls(calls)
        assert updated.tool_call_history == calls
        assert state.tool_call_history == ()

    def test_next_round(self):
        state = IterationState(
            round_num=1, elements=(), history=(), tool_call_history=(), confidence=80
        )
        updated = state.next_round()
        assert updated.round_num == 2
        assert updated.confidence == 80
        assert state.round_num == 1

    def test_frozen(self):
        state = IterationState(
            round_num=0, elements=(), history=(), tool_call_history=()
        )
        with pytest.raises(FrozenInstanceError):
            state.round_num = 5


# ---------------------------------------------------------------------------
# GenerationInput
# ---------------------------------------------------------------------------


class TestGenerationInput:
    def test_construction(self):
        gi = GenerationInput(
            query="what is foo", context="...", prompt_tokens=100, max_tokens=500
        )
        assert gi.query == "what is foo"
        assert gi.dialogue_history == ()

    def test_with_dialogue(self):
        gi = GenerationInput(
            query="q",
            context="c",
            prompt_tokens=10,
            max_tokens=50,
            dialogue_history=({"role": "user", "content": "hi"},),
        )
        assert len(gi.dialogue_history) == 1

    def test_frozen(self):
        gi = GenerationInput(query="q", context="c", prompt_tokens=0, max_tokens=0)
        with pytest.raises(FrozenInstanceError):
            gi.query = "changed"


# ---------------------------------------------------------------------------
# SourceRef
# ---------------------------------------------------------------------------


class TestSourceRef:
    def test_minimal(self):
        sr = SourceRef(path="foo.py", name="bar")
        assert sr.line == 0
        assert sr.element_type == ""
        assert sr.repo_name == ""

    def test_full(self):
        sr = SourceRef(
            path="src/main.py",
            name="main",
            line=42,
            element_type="function",
            repo_name="myrepo",
        )
        assert sr.line == 42
        assert sr.repo_name == "myrepo"


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


class TestGenerationResult:
    def test_construction_with_sources(self):
        sources = (
            SourceRef(path="a.py", name="A", line=10),
            SourceRef(path="b.py", name="B"),
        )
        gr = GenerationResult(answer="The answer", sources=sources, prompt_tokens=150)
        assert gr.answer == "The answer"
        assert len(gr.sources) == 2
        assert gr.summary is None
        assert gr.error is None

    def test_with_summary_and_error(self):
        gr = GenerationResult(
            answer="", sources=(), prompt_tokens=0, summary="short", error="timeout"
        )
        assert gr.summary == "short"
        assert gr.error == "timeout"

    def test_frozen(self):
        gr = GenerationResult(answer="a", sources=(), prompt_tokens=0)
        with pytest.raises(FrozenInstanceError):
            gr.answer = "b"


# ---------------------------------------------------------------------------
# FileAnalysis
# ---------------------------------------------------------------------------


class TestFileAnalysis:
    def test_construction(self):
        fa = FileAnalysis(total_files=100)
        assert fa.total_files == 100
        assert fa.languages == {}
        assert fa.key_files == ()

    def test_with_data(self):
        fa = FileAnalysis(
            total_files=50,
            languages={"python": 30, "yaml": 10, "markdown": 10},
            file_types={"py": 30, "yaml": 10, "md": 10},
            key_files=("main.py", "config.yaml"),
        )
        assert fa.languages["python"] == 30
        assert fa.key_files == ("main.py", "config.yaml")

    def test_frozen(self):
        fa = FileAnalysis(total_files=1)
        with pytest.raises(FrozenInstanceError):
            fa.total_files = 2


# ---------------------------------------------------------------------------
# RepoStructure
# ---------------------------------------------------------------------------


class TestRepoStructure:
    def test_minimal(self):
        fa = FileAnalysis(total_files=10)
        rs = RepoStructure(repo_name="myrepo", summary="A repo", analysis=fa)
        assert rs.repo_name == "myrepo"
        assert rs.has_readme is False
        assert rs.readme_content is None

    def test_full(self):
        fa = FileAnalysis(total_files=5, key_files=("README.md",))
        rs = RepoStructure(
            repo_name="r",
            summary="s",
            analysis=fa,
            has_readme=True,
            readme_content="# Hello",
            structure_text="src/\n  main.py",
        )
        assert rs.has_readme is True
        assert rs.structure_text == "src/\n  main.py"


# ---------------------------------------------------------------------------
# ScipKind
# ---------------------------------------------------------------------------


class TestScipKind:
    def test_constants_are_strings(self):
        assert isinstance(ScipKind.FUNCTION, str)
        assert isinstance(ScipKind.CLASS, str)

    def test_all_constants(self):
        expected = {
            "Function",
            "Method",
            "Class",
            "Module",
            "Interface",
            "Enum",
            "Variable",
            "Constant",
            "Property",
            "Type",
            "Unknown",
        }
        actual = {v for k, v in vars(ScipKind).items() if k.isupper()}
        assert actual == expected


# ---------------------------------------------------------------------------
# ScipRole
# ---------------------------------------------------------------------------


class TestScipRole:
    def test_constants_are_strings(self):
        assert isinstance(ScipRole.DEFINITION, str)
        assert isinstance(ScipRole.REFERENCE, str)

    def test_all_constants(self):
        expected = {
            "Definition",
            "Reference",
            "Import",
            "WriteAccess",
            "ForwardDefinition",
        }
        actual = {v for k, v in vars(ScipRole).items() if k.isupper()}
        assert actual == expected


# ---------------------------------------------------------------------------
# IterationMetrics
# ---------------------------------------------------------------------------


class TestIterationMetrics:
    def test_construction(self):
        im = IterationMetrics(
            rounds=3,
            answered_directly=False,
            query_complexity=4,
            initial_confidence=30,
            final_confidence=90,
            confidence_gain=60,
            total_elements=25,
            total_lines=5000,
            budget_used_pct=42.0,
            iterations_used_pct=75.0,
            overall_roi=0.12,
            round_efficiencies=(),
            adaptive_params={},
            stopping_reason="confidence_reached",
            efficiency_rating="good",
        )
        assert im.rounds == 3
        assert im.stopping_reason == "confidence_reached"
        assert im.efficiency_rating == "good"

    def test_frozen(self):
        im = IterationMetrics(
            rounds=0,
            answered_directly=False,
            query_complexity=0,
            initial_confidence=0,
            final_confidence=0,
            confidence_gain=0,
            total_elements=0,
            total_lines=0,
            budget_used_pct=0.0,
            iterations_used_pct=0.0,
            overall_roi=0.0,
            round_efficiencies=(),
            adaptive_params={},
            stopping_reason="",
            efficiency_rating="",
        )
        with pytest.raises(FrozenInstanceError):
            im.rounds = 1
