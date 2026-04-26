"""Tests for pure combination function extracted from retriever."""

from fastcode.core.combination import combine_results


def _mk_meta(elem_id: str, **extra) -> dict:
    meta = {"id": elem_id, "type": "function", "name": elem_id}
    meta.update(extra)
    return meta


class TestCombineResults:
    def test_merges_semantic_and_keyword(self):
        sem = [(_mk_meta("a"), 0.8)]
        kw = [(_mk_meta("b"), 5.0)]
        result = combine_results(sem, kw)
        ids = [r["element"]["id"] for r in result]
        assert "a" in ids
        assert "b" in ids

    def test_merges_same_element(self):
        sem = [(_mk_meta("a"), 0.8)]
        kw = [(_mk_meta("a"), 5.0)]
        result = combine_results(sem, kw)
        assert len(result) == 1
        assert result[0]["semantic_score"] > 0
        assert result[0]["keyword_score"] > 0

    def test_pseudocode_results(self):
        sem = [(_mk_meta("a"), 0.8)]
        pseudo = [(_mk_meta("b"), 0.6)]
        result = combine_results(sem, [], pseudo)
        assert len(result) == 2

    def test_source_priority_boost(self):
        sem_high = [(_mk_meta("a", metadata={"source_priority": 100}), 0.8)]
        sem_low = [(_mk_meta("b", metadata={"source_priority": 0}), 0.8)]
        result = combine_results(sem_high + sem_low, [])
        assert result[0]["element"]["id"] == "a"

    def test_empty_inputs(self):
        result = combine_results([], [])
        assert result == []

    def test_sorted_by_total_score(self):
        sem = [(_mk_meta("low"), 0.3), (_mk_meta("high"), 0.9)]
        result = combine_results(sem, [])
        assert result[0]["element"]["id"] == "high"

    def test_bm25_normalization(self):
        kw = [(_mk_meta("a"), 5.0), (_mk_meta("b"), 2.5)]
        result = combine_results([], kw)
        assert result[0]["keyword_score"] >= result[1]["keyword_score"]

    def test_semantic_weight_applied(self):
        sem = [(_mk_meta("a"), 1.0)]
        result = combine_results(sem, [], semantic_weight=0.5)
        assert abs(result[0]["semantic_score"] - 0.5) < 1e-9

    def test_keyword_weight_applied(self):
        kw = [(_mk_meta("a"), 2.0)]
        result = combine_results([], kw, keyword_weight=0.3)
        assert abs(result[0]["keyword_score"] - 0.3) < 1e-9
