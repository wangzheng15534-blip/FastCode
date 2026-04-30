"""Tests for fastcode.semantic_ir — computed properties, deduplication, legacy conversion.

Previous version: 1152 lines of Hypothesis round-trip tests.
This version: tests source_priority computation, occurrence dedup, confidence mapping,
              legacy payload conversion, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastcode.semantic_ir import (
    IRCodeUnit,
    IRDocument,
    IRRelation,
    IRSnapshot,
    IRUnitSupport,
    _confidence_to_resolution,
    _resolution_to_confidence,
    resolution_rank,
)

# --- source_priority computation ---


class TestSourcePriority:
    def test_scip_and_fc_structure_gives_100(self):
        unit = IRCodeUnit(
            unit_id="u1",
            kind="function",
            path="a.py",
            language="python",
            display_name="f",
            source_set={"scip", "fc_structure"},
        )
        assert unit.source_priority == 100

    def test_scip_only_gives_100(self):
        unit = IRCodeUnit(
            unit_id="u2",
            kind="function",
            path="a.py",
            language="python",
            display_name="f",
            source_set={"scip"},
        )
        assert unit.source_priority == 100

    def test_fc_structure_only_gives_50(self):
        unit = IRCodeUnit(
            unit_id="u3",
            kind="function",
            path="a.py",
            language="python",
            display_name="f",
            source_set={"fc_structure"},
        )
        assert unit.source_priority == 50

    def test_no_known_source_gives_0(self):
        unit = IRCodeUnit(
            unit_id="u4",
            kind="function",
            path="a.py",
            language="python",
            display_name="f",
            source_set=set(),
        )
        assert unit.source_priority == 0

    def test_unknown_source_gives_0(self):
        unit = IRCodeUnit(
            unit_id="u5",
            kind="function",
            path="a.py",
            language="python",
            display_name="f",
            source_set={"custom_extractor"},
        )
        assert unit.source_priority == 0


# --- confidence <-> resolution_state mapping ---


class TestConfidenceMapping:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("anchored", "precise"),
            ("semantic", "precise"),
            ("semantically_resolved", "precise"),
            ("structural", "resolved"),
            ("candidate", "heuristic"),
        ],
    )
    def test_known_states_map_correctly(self, state: str, expected: str) -> None:
        assert _resolution_to_confidence(state) == expected

    def test_unknown_state_falls_back_to_derived(self):
        assert _resolution_to_confidence("bogus_value") == "derived"

    def test_none_state_falls_back_to_derived(self):
        assert _resolution_to_confidence(None) == "derived"

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            ("precise", "anchored"),
            ("resolved", "structural"),
            ("heuristic", "candidate"),
        ],
    )
    def test_reverse_mapping_known(self, confidence: str, expected: str) -> None:
        assert _confidence_to_resolution(confidence) == expected

    def test_reverse_unknown_falls_back(self):
        assert _confidence_to_resolution("bogus") == "structural"


# --- IRRelation computed properties ---


class TestIRRelationProperties:
    def test_source_returns_first_sorted_support_source(self):
        rel = IRRelation(
            relation_id="r1",
            src_unit_id="a",
            dst_unit_id="b",
            relation_type="call",
            resolution_state="anchored",
            support_sources={"z_source", "a_source"},
        )
        assert rel.source == "a_source"

    def test_source_falls_back_to_metadata_when_no_support_sources(self):
        rel = IRRelation(
            relation_id="r2",
            src_unit_id="a",
            dst_unit_id="b",
            relation_type="call",
            resolution_state="anchored",
            metadata={"source": "meta_source"},
        )
        assert rel.source == "meta_source"

    def test_confidence_from_resolution_state(self):
        rel = IRRelation(
            relation_id="r3",
            src_unit_id="a",
            dst_unit_id="b",
            relation_type="call",
            resolution_state="anchored",
        )
        assert rel.confidence == "precise"

    def test_doc_id_from_metadata(self):
        rel = IRRelation(
            relation_id="r4",
            src_unit_id="a",
            dst_unit_id="b",
            relation_type="call",
            resolution_state="anchored",
            metadata={"doc_id": "doc:/src/main.py"},
        )
        assert rel.doc_id == "doc:/src/main.py"

    def test_doc_id_none_when_not_in_metadata(self):
        rel = IRRelation(
            relation_id="r5",
            src_unit_id="a",
            dst_unit_id="b",
            relation_type="call",
            resolution_state="anchored",
        )
        assert rel.doc_id is None


# --- Occurrence deduplication ---


class TestOccurrenceDeduplication:
    def _snapshot_with_occurrences(
        self, supports: list[IRUnitSupport], *, unit_source: str = "scip"
    ) -> IRSnapshot:
        """Build a snapshot with the given IRUnitSupport entries."""
        units = [
            IRCodeUnit(
                unit_id="sym:func_a",
                kind="function",
                path="a.py",
                language="python",
                display_name="func_a",
                source_set={unit_source},
            ),
            IRCodeUnit(
                unit_id="doc:a.py",
                kind="file",
                path="a.py",
                language="python",
                display_name="a.py",
                source_set={"scip"},
            ),
        ]
        return IRSnapshot(
            repo_name="test",
            snapshot_id="snap:test:c1",
            units=units,
            supports=supports,
        )

    def test_duplicate_occurrences_deduplicated(self):
        """Two occurrences with same (symbol, doc, role, position) should become one."""
        supports = [
            IRUnitSupport(
                support_id="sup1",
                unit_id="sym:func_a",
                source="scip",
                support_kind="occurrence",
                role="definition",
                start_line=10,
                start_col=0,
                end_line=10,
                end_col=10,
                metadata={"doc_id": "doc:a.py"},
            ),
            IRUnitSupport(
                support_id="sup2",
                unit_id="sym:func_a",
                source="fc_structure",
                support_kind="occurrence",
                role="definition",
                start_line=10,
                start_col=0,
                end_line=10,
                end_col=10,
                metadata={"doc_id": "doc:a.py"},
            ),
        ]
        snap = self._snapshot_with_occurrences(supports)
        assert len(snap.occurrences) == 1

    def test_scip_wins_over_non_scip_on_duplicate(self):
        """When deduplicating, SCIP source should be kept over non-SCIP."""
        supports = [
            IRUnitSupport(
                support_id="sup1",
                unit_id="sym:func_a",
                source="fc_structure",
                support_kind="occurrence",
                role="definition",
                start_line=10,
                start_col=0,
                end_line=10,
                end_col=10,
                metadata={"doc_id": "doc:a.py"},
            ),
            IRUnitSupport(
                support_id="sup2",
                unit_id="sym:func_a",
                source="scip",
                support_kind="occurrence",
                role="definition",
                start_line=10,
                start_col=0,
                end_line=10,
                end_col=10,
                metadata={"doc_id": "doc:a.py"},
            ),
        ]
        snap = self._snapshot_with_occurrences(supports)
        assert len(snap.occurrences) == 1
        assert snap.occurrences[0].source == "scip"

    def test_different_positions_not_deduplicated(self):
        supports = [
            IRUnitSupport(
                support_id="sup1",
                unit_id="sym:func_a",
                source="scip",
                support_kind="occurrence",
                role="definition",
                start_line=10,
                start_col=0,
                end_line=10,
                end_col=10,
                metadata={"doc_id": "doc:a.py"},
            ),
            IRUnitSupport(
                support_id="sup2",
                unit_id="sym:func_a",
                source="scip",
                support_kind="occurrence",
                role="reference",
                start_line=20,
                start_col=0,
                end_line=20,
                end_col=5,
                metadata={"doc_id": "doc:a.py"},
            ),
        ]
        snap = self._snapshot_with_occurrences(supports)
        assert len(snap.occurrences) == 2


# --- IRSnapshot legacy conversion ---


class TestLegacyConversion:
    def _legacy_payload(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "repo_name": "test",
            "snapshot_id": "snap:test:abc",
            "commit_id": "c1",
            "branch": "main",
            "documents": [
                IRDocument(
                    doc_id="doc:/a.py", path="/a.py", language="python"
                ).to_dict(),
            ],
            "symbols": [
                {
                    "symbol_id": "sym:f",
                    "display_name": "f",
                    "kind": "function",
                    "qualified_name": "mod.f",
                    "path": "/a.py",
                    "language": "python",
                    "start_line": 5,
                    "end_line": 10,
                    "external_symbol_id": None,
                },
            ],
            "occurrences": [],
            "edges": [],
            "attachments": [],
        }
        base.update(overrides)
        return base

    def test_legacy_creates_file_units_from_documents(self):
        snap = IRSnapshot.from_dict(self._legacy_payload())
        doc_units = [u for u in snap.units if u.kind == "file"]
        assert len(doc_units) == 1
        assert doc_units[0].path == "/a.py"

    def test_legacy_creates_symbol_units(self):
        snap = IRSnapshot.from_dict(self._legacy_payload())
        sym_units = [u for u in snap.units if u.kind == "function"]
        assert len(sym_units) == 1

    def test_legacy_with_empty_collections(self):
        snap = IRSnapshot.from_dict(
            self._legacy_payload(
                documents=[],
                symbols=[],
                occurrences=[],
                edges=[],
                attachments=[],
            )
        )
        assert len(snap.units) == 0
        assert len(snap.occurrences) == 0
        assert len(snap.edges) == 0

    def test_canonical_format_preserved(self):
        """When saving canonical format and reloading, units/supports survive."""
        snap = IRSnapshot(
            repo_name="test",
            snapshot_id="snap:test:abc",
            units=[
                IRCodeUnit(
                    unit_id="u1",
                    kind="function",
                    path="a.py",
                    language="python",
                    display_name="f",
                ),
            ],
            supports=[],
            relations=[],
        )
        data = snap.to_dict()
        assert data.get("schema_version") == "ir.v2"
        restored = IRSnapshot.from_dict(data)
        assert len(restored.units) == 1


# --- Smoke round-trip (keep exactly 1) ---


class TestSmokeRoundTrip:
    def test_snapshot_roundtrip_smoke(self):
        """One smoke test: snapshot survives to_dict -> from_dict."""
        snap = IRSnapshot(
            repo_name="r",
            snapshot_id="snap:r:c1",
            units=[
                IRCodeUnit(
                    unit_id="u1",
                    kind="function",
                    path="a.py",
                    language="python",
                    display_name="f",
                )
            ],
            supports=[],
            relations=[],
        )
        restored = IRSnapshot.from_dict(snap.to_dict())
        assert restored.repo_name == snap.repo_name
        assert len(restored.units) == len(snap.units)


# --- resolution_rank ordering ---


class TestResolutionRank:
    def test_candidate_is_lowest(self):
        assert resolution_rank("candidate") == 0

    def test_structural_above_candidate(self):
        assert resolution_rank("structural") > resolution_rank("candidate")

    def test_anchored_above_structural(self):
        assert resolution_rank("anchored") > resolution_rank("structural")

    def test_semantic_above_anchored(self):
        assert resolution_rank("semantic") > resolution_rank("anchored")

    def test_semantically_resolved_equals_semantic(self):
        assert resolution_rank("semantically_resolved") == resolution_rank("semantic")

    def test_unknown_defaults_to_zero(self):
        assert resolution_rank("bogus") == 0


# --- pending_capabilities ---


class TestPendingCapabilities:
    def test_defaults_to_empty_set(self):
        rel = IRRelation(
            relation_id="r1",
            src_unit_id="u1",
            dst_unit_id="u2",
            relation_type="call",
            resolution_state="structural",
        )
        assert rel.pending_capabilities == set()

    def test_roundtrip_preserves_pending_capabilities(self):
        rel = IRRelation(
            relation_id="r1",
            src_unit_id="u1",
            dst_unit_id="u2",
            relation_type="call",
            resolution_state="structural",
            pending_capabilities={"resolve_calls", "resolve_types"},
        )
        restored = IRRelation.from_dict(rel.to_dict())
        assert restored.pending_capabilities == {"resolve_calls", "resolve_types"}

    def test_backward_compat_missing_key(self):
        data = {
            "relation_id": "r1",
            "src_unit_id": "u1",
            "dst_unit_id": "u2",
            "relation_type": "call",
            "resolution_state": "structural",
        }
        restored = IRRelation.from_dict(data)
        assert restored.pending_capabilities == set()
