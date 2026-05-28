"""
Tests for incremental update via file identity diffing.

Each test is self-contained with its own factory functions.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from fastcode.app.indexing.pipeline.incremental import (
    FileChangeSet,
    PlanChanges,
    apply_incremental_update,
    diff_changed_files,
)
from fastcode.ir.types import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_file_unit(
    path: str,
    blob_oid: str | None,
    unit_id: str | None = None,
    content_hash: str | None = None,
) -> IRCodeUnit:
    """Create a file-level IRCodeUnit with identity metadata."""
    uid = unit_id or f"unit:file:{path}"
    metadata: dict[str, str] = {}
    if blob_oid is not None:
        metadata["blob_oid"] = blob_oid
    if content_hash is not None:
        metadata["content_hash"] = content_hash
    return IRCodeUnit(
        unit_id=uid,
        kind="file",
        path=path,
        language="python",
        display_name=path,
        source_set={"fc_structure"},
        metadata=metadata,
    )


def _make_symbol_unit(
    path: str, name: str, unit_id: str | None = None, kind: str = "function"
) -> IRCodeUnit:
    uid = unit_id or f"unit:sym:{path}:{name}"
    return IRCodeUnit(
        unit_id=uid,
        kind=kind,
        path=path,
        language="python",
        display_name=name,
        qualified_name=f"{path}:{name}",
        source_set={"fc_structure"},
        metadata={"stable_unit_id": f"stable:{path}:{kind}:{name}"},
    )


def _make_relation(
    src_id: str, dst_id: str, rel_type: str = "call", rel_id: str | None = None
) -> IRRelation:
    rid = rel_id or f"rel:{rel_type}:{src_id}->{dst_id}"
    return IRRelation(
        relation_id=rid,
        src_unit_id=src_id,
        dst_unit_id=dst_id,
        relation_type=rel_type,
        resolution_state="structural",
        support_sources={"fc_structure"},
    )


def _make_embedding(unit_id: str, vector: list[float] | None = None) -> IRUnitEmbedding:
    return IRUnitEmbedding(
        embedding_id=f"emb:{unit_id}",
        unit_id=unit_id,
        source="fc_structure",
        vector=vector or [0.1, 0.2, 0.3],
    )


def _make_snapshot(
    snapshot_id: str = "snap:test:abc123",
    repo_name: str = "test",
    units: list[IRCodeUnit] | None = None,
    relations: list[IRRelation] | None = None,
    embeddings: list[IRUnitEmbedding] | None = None,
    supports: list[IRUnitSupport] | None = None,
    branch: str = "main",
    commit_id: str = "abc123",
) -> IRSnapshot:
    return IRSnapshot(
        repo_name=repo_name,
        snapshot_id=snapshot_id,
        branch=branch,
        commit_id=commit_id,
        units=units or [],
        relations=relations or [],
        embeddings=embeddings or [],
        supports=supports or [],
    )


def _build_simple_snapshot(
    files: dict[str, str],
    extra_units: list[IRCodeUnit] | None = None,
    relations: list[IRRelation] | None = None,
    embeddings: list[IRUnitEmbedding] | None = None,
    supports: list[IRUnitSupport] | None = None,
    snapshot_id: str = "snap:test:old",
    commit_id: str = "old123",
) -> IRSnapshot:
    """Build a snapshot from {path: blob_oid} mapping, with optional extras."""
    units: list[IRCodeUnit] = []
    for path, blob in files.items():
        units.append(_make_file_unit(path, blob))
    if extra_units:
        units.extend(extra_units)
    return _make_snapshot(
        snapshot_id=snapshot_id,
        units=units,
        relations=relations,
        embeddings=embeddings,
        supports=supports,
        commit_id=commit_id,
    )


# ---------------------------------------------------------------------------
# Tests: diff_changed_files
# ---------------------------------------------------------------------------


class TestPlanChanges:
    def test_payload_and_legacy_prefilter_adapter_share_one_plan(self) -> None:
        plan = PlanChanges(
            previous_snapshot_id="snap:repo:prev",
            previous_artifact_key="snap_repo_prev",
            added_paths=("new.py",),
            modified_paths=("changed.py",),
            removed_paths=("deleted.py",),
            unchanged_paths=("same.py",),
            reused_elements=3,
            reindexed_elements=2,
            reused_changed_embeddings=1,
            semantic_frontier_widened=True,
            api_frontier_changed_paths=("changed.py",),
            package_scope_roots=(".",),
            change_kinds=("api_surface_hash", "embedding_text_hash"),
            interface_digest_changed_paths=("changed.py",),
            interface_digests={"changed.py": "iface:abc"},
            dependency_frontier={
                "radius": "dependent_neighborhood",
                "strategy": "package",
            },
            degraded_reasons=("semantic_frontier_widened",),
        )

        payload = plan.to_payload()
        legacy = plan.to_prefilter_payload()

        assert payload["schema_version"] == "fastcode.plan_changes.v1"
        assert payload["counts"] == {
            "added": 1,
            "modified": 1,
            "removed": 1,
            "unchanged": 1,
            "changed": 3,
        }
        assert payload["paths"] == {
            "added": ["new.py"],
            "modified": ["changed.py"],
            "removed": ["deleted.py"],
            "unchanged": ["same.py"],
            "changed": ["changed.py", "new.py"],
        }
        assert payload["reuse"] == {
            "reused_elements": 3,
            "reindexed_elements": 2,
            "reused_changed_embeddings": 1,
        }
        assert payload["frontier"]["degraded"] is True
        assert legacy["plan_changes"] == payload
        assert legacy["added"] == payload["counts"]["added"]
        assert legacy["modified"] == payload["counts"]["modified"]
        assert legacy["removed"] == payload["counts"]["removed"]
        assert legacy["unchanged"] == payload["counts"]["unchanged"]
        assert legacy["changed_paths"] == payload["paths"]["changed"]
        assert legacy["semantic_frontier_widened"] == 1
        assert legacy["api_frontier_changed"] == 1


class TestDiffChangedFiles:
    def test_no_changes(self) -> None:
        old = _build_simple_snapshot({"a.py": "h1", "b.py": "h2"})
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"}, snapshot_id="snap:test:new"
        )
        cs = diff_changed_files(old, new)
        assert cs.unchanged == ["a.py", "b.py"]
        assert cs.added == []
        assert cs.removed == []
        assert cs.modified == []

    def test_added_file(self) -> None:
        old = _build_simple_snapshot({"a.py": "h1"})
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"}, snapshot_id="snap:test:new"
        )
        cs = diff_changed_files(old, new)
        assert cs.added == ["b.py"]
        assert cs.unchanged == ["a.py"]
        assert cs.removed == []
        assert cs.modified == []

    def test_removed_file(self) -> None:
        old = _build_simple_snapshot({"a.py": "h1", "b.py": "h2"})
        new = _build_simple_snapshot({"a.py": "h1"}, snapshot_id="snap:test:new")
        cs = diff_changed_files(old, new)
        assert cs.removed == ["b.py"]
        assert cs.unchanged == ["a.py"]
        assert cs.added == []
        assert cs.modified == []

    def test_modified_file(self) -> None:
        old = _build_simple_snapshot({"a.py": "h1", "b.py": "h2"})
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_changed"}, snapshot_id="snap:test:new"
        )
        cs = diff_changed_files(old, new)
        assert cs.modified == ["b.py"]
        assert cs.unchanged == ["a.py"]
        assert cs.added == []
        assert cs.removed == []

    def test_all_change_types(self) -> None:
        old = _build_simple_snapshot({"a.py": "h1", "b.py": "h2", "c.py": "h3"})
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_mod", "d.py": "h4"}, snapshot_id="snap:test:new"
        )
        cs = diff_changed_files(old, new)
        assert cs.unchanged == ["a.py"]
        assert cs.modified == ["b.py"]
        assert cs.removed == ["c.py"]
        assert cs.added == ["d.py"]

    def test_empty_old_snapshot(self) -> None:
        old = _make_snapshot(snapshot_id="snap:test:old")
        new = _build_simple_snapshot({"a.py": "h1"}, snapshot_id="snap:test:new")
        cs = diff_changed_files(old, new)
        assert cs.added == ["a.py"]
        assert cs.unchanged == []

    def test_empty_new_snapshot(self) -> None:
        old = _build_simple_snapshot({"a.py": "h1"})
        new = _make_snapshot(snapshot_id="snap:test:new")
        cs = diff_changed_files(old, new)
        assert cs.removed == ["a.py"]
        assert cs.unchanged == []

    def test_none_blob_oid_treated_as_change(self) -> None:
        """A file with blob_oid=None should be considered different from a hash."""
        old = _build_simple_snapshot({"a.py": "h1"})
        new_units = [_make_file_unit("a.py", None)]
        new = _make_snapshot(snapshot_id="snap:test:new", units=new_units)
        cs = diff_changed_files(old, new)
        assert cs.modified == ["a.py"]

    def test_content_hash_fallback_preserves_unchanged_file(self) -> None:
        old = _make_snapshot(
            snapshot_id="snap:test:old",
            units=[_make_file_unit("a.py", None, content_hash="h1")],
        )
        new = _make_snapshot(
            snapshot_id="snap:test:new",
            units=[_make_file_unit("a.py", None, content_hash="h1")],
        )
        cs = diff_changed_files(old, new)
        assert cs.unchanged == ["a.py"]
        assert cs.modified == []

    def test_content_hash_fallback_detects_modified_file(self) -> None:
        old = _make_snapshot(
            snapshot_id="snap:test:old",
            units=[_make_file_unit("a.py", None, content_hash="h1")],
        )
        new = _make_snapshot(
            snapshot_id="snap:test:new",
            units=[_make_file_unit("a.py", None, content_hash="h2")],
        )
        cs = diff_changed_files(old, new)
        assert cs.modified == ["a.py"]
        assert cs.unchanged == []

    def test_blob_oid_takes_precedence_over_stale_content_hash(self) -> None:
        old = _make_snapshot(
            snapshot_id="snap:test:old",
            units=[_make_file_unit("a.py", "blob1", content_hash="stale")],
        )
        new = _make_snapshot(
            snapshot_id="snap:test:new",
            units=[_make_file_unit("a.py", "blob1", content_hash="different")],
        )
        cs = diff_changed_files(old, new)
        assert cs.unchanged == ["a.py"]
        assert cs.modified == []

    def test_missing_file_identity_is_conservative_change(self) -> None:
        old = _make_snapshot(
            snapshot_id="snap:test:old",
            units=[_make_file_unit("a.py", None)],
        )
        new = _make_snapshot(
            snapshot_id="snap:test:new",
            units=[_make_file_unit("a.py", None)],
        )
        cs = diff_changed_files(old, new)
        assert cs.modified == ["a.py"]
        assert cs.unchanged == []


# ---------------------------------------------------------------------------
# Tests: apply_incremental_update
# ---------------------------------------------------------------------------


class TestApplyIncrementalUpdate:
    def test_apply_defers_merged_component_materialization(
        self,
    ) -> None:
        class _ExplodingList(list[object]):
            def __iter__(self) -> Iterator[object]:
                raise AssertionError("component stream should stay lazy until access")

        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[_make_symbol_unit("a.py", "old_a")],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            extra_units=[_make_symbol_unit("b.py", "new_b")],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        old.supports = _ExplodingList()
        old.relations = _ExplodingList()
        old.embeddings = _ExplodingList()
        new.supports = _ExplodingList()
        new.relations = _ExplodingList()
        new.embeddings = _ExplodingList()

        result = apply_incremental_update(
            old,
            new,
            FileChangeSet(modified=["b.py"], unchanged=["a.py"]),
        )

        assert result.snapshot_id == "snap:test:new"
        assert [unit.path for unit in result.units] == ["a.py", "a.py", "b.py", "b.py"]

    def test_no_changes_preserves_everything(self) -> None:
        old = _build_simple_snapshot(
            {"a.py": "h1"},
            extra_units=[_make_symbol_unit("a.py", "foo")],
            relations=[
                _make_relation("unit:file:a.py", "unit:sym:a.py:foo", "contain")
            ],
            embeddings=[_make_embedding("unit:sym:a.py:foo")],
        )
        # Same extraction, same blob_oids
        new = _build_simple_snapshot(
            {"a.py": "h1"},
            extra_units=[_make_symbol_unit("a.py", "foo")],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        # Should preserve all units from old snapshot
        assert len(result.units) == 2  # file + symbol
        assert len(result.relations) == 1
        assert len(result.embeddings) == 1
        # Should take new snapshot identity
        assert result.snapshot_id == "snap:test:new"
        assert result.commit_id == "new123"

    def test_added_file_incorporates_new_units(self) -> None:
        old = _build_simple_snapshot({"a.py": "h1"})
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[_make_symbol_unit("b.py", "bar")],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(added=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        unit_paths = {u.path for u in result.units}
        assert "a.py" in unit_paths
        assert "b.py" in unit_paths
        # Should have file unit for a.py, file for b.py, symbol for bar
        assert len(result.units) == 3

    def test_modified_file_replaces_units(self) -> None:
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[_make_symbol_unit("b.py", "old_func")],
            embeddings=[_make_embedding("unit:sym:b.py:old_func")],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            extra_units=[_make_symbol_unit("b.py", "new_func")],
            embeddings=[_make_embedding("unit:sym:b.py:new_func")],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        # Old symbol from b.py should be gone, new one present
        unit_ids = {u.unit_id for u in result.units}
        assert "unit:sym:b.py:old_func" not in unit_ids
        assert "unit:sym:b.py:new_func" in unit_ids

        # Embeddings: old_func removed, new_func added
        emb_unit_ids = {e.unit_id for e in result.embeddings}
        assert "unit:sym:b.py:old_func" not in emb_unit_ids
        assert "unit:sym:b.py:new_func" in emb_unit_ids

    def test_reuses_unchanged_unit_objects_by_reference(self) -> None:
        stable_file = _make_file_unit("a.py", "h1")
        stable_symbol = _make_symbol_unit("a.py", "stable")
        old_changed_symbol = _make_symbol_unit("b.py", "old_func")
        new_changed_symbol = _make_symbol_unit("b.py", "new_func")
        old = _make_snapshot(
            snapshot_id="snap:test:old",
            units=[stable_file, stable_symbol, _make_file_unit("b.py", "h2"), old_changed_symbol],
            commit_id="old123",
        )
        new = _make_snapshot(
            snapshot_id="snap:test:new",
            units=[
                _make_file_unit("a.py", "h1"),
                _make_symbol_unit("a.py", "stable"),
                _make_file_unit("b.py", "h2_new"),
                new_changed_symbol,
            ],
            commit_id="new123",
        )

        result = apply_incremental_update(
            old,
            new,
            FileChangeSet(modified=["b.py"], unchanged=["a.py"]),
        )

        assert result.units[0] is stable_file
        assert result.units[1] is stable_symbol
        assert result.units[2].unit_id == "unit:file:b.py"
        assert result.units[2] is not old.units[2]
        assert result.units[3] is new_changed_symbol

    def test_removed_file_drops_units_and_relations(self) -> None:
        sym_b = _make_symbol_unit("b.py", "func_b")
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[sym_b],
            relations=[
                _make_relation("unit:file:a.py", "unit:file:b.py", "import"),
                _make_relation("unit:file:b.py", "unit:sym:b.py:func_b", "contain"),
            ],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1"},
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(removed=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        unit_paths = {u.path for u in result.units}
        assert "b.py" not in unit_paths
        # Relations referencing b.py units should be dropped
        rel_ids = {r.src_unit_id for r in result.relations}
        assert "unit:file:b.py" not in rel_ids

    def test_preserves_embeddings_for_unchanged_units(self) -> None:
        emb_a = _make_embedding("unit:sym:a.py:foo", vector=[1.0, 2.0, 3.0])
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[_make_symbol_unit("a.py", "foo")],
            embeddings=[emb_a],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_changed"},
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        # Embedding for a.py:foo should survive (it's in an unchanged file)
        emb_ids = {e.unit_id for e in result.embeddings}
        assert "unit:sym:a.py:foo" in emb_ids
        emb = next(e for e in result.embeddings if e.unit_id == "unit:sym:a.py:foo")
        assert emb.vector == [1.0, 2.0, 3.0]

    def test_reuses_unchanged_support_relation_and_embedding_objects_by_reference(
        self,
    ) -> None:
        caller = _make_symbol_unit("a.py", "caller")
        callee_old = _make_symbol_unit("b.py", "callee_old")
        callee_new = _make_symbol_unit("b.py", "callee_new")
        old_support = IRUnitSupport(
            support_id="sup:a",
            unit_id=caller.unit_id,
            source="scip",
            support_kind="occurrence",
            role="definition",
        )
        old_relation = _make_relation(
            "unit:file:a.py",
            caller.unit_id,
            "contain",
            rel_id="rel:contain:a",
        )
        old_embedding = _make_embedding(caller.unit_id, vector=[1.0, 2.0, 3.0])
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[caller, callee_old],
            supports=[old_support],
            relations=[old_relation],
            embeddings=[old_embedding],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            extra_units=[callee_new],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )

        result = apply_incremental_update(
            old,
            new,
            FileChangeSet(modified=["b.py"], unchanged=["a.py"]),
        )

        assert result.supports[0] is old_support
        assert result.relations[0] is old_relation
        assert result.embeddings[0] is old_embedding

    def test_cross_file_relation_preserved_if_source_unchanged(self) -> None:
        """A relation from an unchanged file to a changed file's unit should be dropped
        (dst is tombstoned). A relation from a changed file to unchanged should also be dropped
        (src is tombstoned). Only relations between unchanged units survive."""
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[
                _make_symbol_unit("a.py", "caller"),
                _make_symbol_unit("b.py", "callee"),
            ],
            relations=[
                # a.py calls b.py — dst is in changed file
                _make_relation("unit:sym:a.py:caller", "unit:sym:b.py:callee", "call"),
            ],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            extra_units=[
                _make_symbol_unit("b.py", "new_callee"),
            ],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        # Old relation a:caller -> b:callee should be dropped (dst tombstoned)
        old_rel_ids = {r.relation_id for r in result.relations}
        assert "rel:call:unit:sym:a.py:caller->unit:sym:b.py:callee" not in old_rel_ids

    def test_cross_file_relation_relinked_when_changed_target_keeps_stable_identity(
        self,
    ) -> None:
        old_callee = _make_symbol_unit(
            "b.py",
            "callee",
            unit_id="unit:sym:b.py:callee:old",
        )
        new_callee = _make_symbol_unit(
            "b.py",
            "callee",
            unit_id="unit:sym:b.py:callee:new",
        )
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[
                _make_symbol_unit("a.py", "caller"),
                old_callee,
            ],
            relations=[
                _make_relation(
                    "unit:sym:a.py:caller",
                    "unit:sym:b.py:callee:old",
                    "call",
                ),
            ],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            extra_units=[new_callee],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        assert any(
            relation.src_unit_id == "unit:sym:a.py:caller"
            and relation.dst_unit_id == "unit:sym:b.py:callee:new"
            and relation.relation_type == "call"
            for relation in result.relations
        )

    def test_file_import_relation_relinked_when_changed_file_unit_replaced(
        self,
    ) -> None:
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            relations=[
                _make_relation("unit:file:a.py", "unit:file:b.py", "import"),
            ],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        assert any(
            relation.src_unit_id == "unit:file:a.py"
            and relation.dst_unit_id == "unit:file:b.py"
            and relation.relation_type == "import"
            for relation in result.relations
        )

    def test_relation_dedup_by_tuple_key(self) -> None:
        """If old and new both have the same (src, dst, type) relation, only one survives."""
        rel = _make_relation("unit:file:a.py", "unit:sym:a.py:foo", "contain")
        old = _build_simple_snapshot(
            {"a.py": "h1"},
            extra_units=[_make_symbol_unit("a.py", "foo")],
            relations=[rel],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1"},
            extra_units=[_make_symbol_unit("a.py", "foo")],
            relations=[
                _make_relation(
                    "unit:file:a.py",
                    "unit:sym:a.py:foo",
                    "contain",
                    rel_id="rel:dup",
                )
            ],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        # Both have same (src, dst, type) — only one should survive
        keys = [
            (r.src_unit_id, r.dst_unit_id, r.relation_type) for r in result.relations
        ]
        assert keys.count(("unit:file:a.py", "unit:sym:a.py:foo", "contain")) == 1

    def test_supports_preserved_for_unchanged_units(self) -> None:
        support = IRUnitSupport(
            support_id="sup:1",
            unit_id="unit:sym:a.py:foo",
            source="scip",
            support_kind="occurrence",
            role="definition",
            start_line=10,
            start_col=0,
            end_line=10,
            end_col=20,
        )
        old = _build_simple_snapshot(
            {"a.py": "h1"},
            extra_units=[_make_symbol_unit("a.py", "foo")],
            supports=[support],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1"},
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        assert len(result.supports) == 1
        assert result.supports[0].support_id == "sup:1"

    def test_supports_replaced_for_changed_units(self) -> None:
        old_support = IRUnitSupport(
            support_id="sup:old",
            unit_id="unit:sym:b.py:func",
            source="scip",
            support_kind="occurrence",
            role="definition",
            start_line=5,
            start_col=0,
            end_line=5,
            end_col=10,
        )
        new_support = IRUnitSupport(
            support_id="sup:new",
            unit_id="unit:sym:b.py:func",
            source="scip",
            support_kind="occurrence",
            role="definition",
            start_line=5,
            start_col=0,
            end_line=5,
            end_col=12,
        )
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[_make_symbol_unit("b.py", "func")],
            supports=[old_support],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            extra_units=[_make_symbol_unit("b.py", "func")],
            supports=[new_support],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"], unchanged=["a.py"])
        result = apply_incremental_update(old, new, cs)

        # Old support should be replaced by new support for the changed file's unit
        support_ids = {s.support_id for s in result.supports}
        assert "sup:new" in support_ids

    def test_preserve_source_owned_supports_for_stable_changed_units(self) -> None:
        old_func = _make_symbol_unit(
            "b.py",
            "func",
            unit_id="unit:sym:b.py:func:old",
        )
        new_func = _make_symbol_unit(
            "b.py",
            "func",
            unit_id="unit:sym:b.py:func:new",
        )
        old_support = IRUnitSupport(
            support_id="sup:scip:old",
            unit_id="unit:sym:b.py:func:old",
            source="scip",
            support_kind="occurrence",
            role="definition",
            path="b.py",
        )
        old_relation = IRRelation(
            relation_id="rel:scip:self",
            src_unit_id="unit:sym:b.py:func:old",
            dst_unit_id="unit:sym:b.py:func:old",
            relation_type="definition",
            resolution_state="anchored",
            support_sources={"scip"},
            support_ids=["sup:scip:old"],
        )
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2"},
            extra_units=[old_func],
            supports=[old_support],
            relations=[old_relation],
        )
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new"},
            extra_units=[new_func],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"], unchanged=["a.py"])

        result = apply_incremental_update(
            old,
            new,
            cs,
            preserve_sources_for_modified_paths={"scip"},
        )

        assert any(
            support.source == "scip"
            and support.unit_id == "unit:sym:b.py:func:new"
            and support.metadata["relink_reason"] == "source_owned_incremental_preserve"
            for support in result.supports
        )
        assert any(
            relation.src_unit_id == "unit:sym:b.py:func:new"
            and relation.dst_unit_id == "unit:sym:b.py:func:new"
            and relation.support_sources == {"scip"}
            and relation.metadata["relink_reason"]
            == "source_owned_incremental_preserve"
            for relation in result.relations
        )

    def test_relinked_preserved_supports_do_not_round_trip_through_dicts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        old_func = _make_symbol_unit(
            "b.py",
            "func",
            unit_id="unit:sym:b.py:func:old",
        )
        new_func = _make_symbol_unit(
            "b.py",
            "func",
            unit_id="unit:sym:b.py:func:new",
        )
        old = _build_simple_snapshot(
            {"b.py": "h2"},
            extra_units=[old_func],
            supports=[
                IRUnitSupport(
                    support_id="sup:scip:old",
                    unit_id="unit:sym:b.py:func:old",
                    source="scip",
                    support_kind="occurrence",
                    role="definition",
                    path="b.py",
                )
            ],
        )
        new = _build_simple_snapshot(
            {"b.py": "h2_new"},
            extra_units=[new_func],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )

        def _boom(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise AssertionError("incremental relink must not call to_dict()")

        monkeypatch.setattr(IRUnitSupport, "to_dict", _boom)

        result = apply_incremental_update(
            old,
            new,
            FileChangeSet(modified=["b.py"]),
            preserve_sources_for_modified_paths={"scip"},
        )

        assert any(
            support.source == "scip" and support.unit_id == "unit:sym:b.py:func:new"
            for support in result.supports
        )

    def test_unlisted_source_owned_supports_are_not_preserved(self) -> None:
        old_func = _make_symbol_unit(
            "b.py",
            "func",
            unit_id="unit:sym:b.py:func:old",
        )
        new_func = _make_symbol_unit(
            "b.py",
            "func",
            unit_id="unit:sym:b.py:func:new",
        )
        old = _build_simple_snapshot(
            {"b.py": "h2"},
            extra_units=[old_func],
            supports=[
                IRUnitSupport(
                    support_id="sup:semantic:old",
                    unit_id="unit:sym:b.py:func:old",
                    source="semantic_resolver",
                    support_kind="occurrence",
                    path="b.py",
                )
            ],
        )
        new = _build_simple_snapshot(
            {"b.py": "h2_new"},
            extra_units=[new_func],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )
        cs = FileChangeSet(modified=["b.py"])

        result = apply_incremental_update(
            old,
            new,
            cs,
            preserve_sources_for_modified_paths={"scip"},
        )

        assert all(support.source != "semantic_resolver" for support in result.supports)


# ---------------------------------------------------------------------------
# Tests: integration — diff + apply together
# ---------------------------------------------------------------------------


class TestIncrementalIntegration:
    def test_full_incremental_cycle(self) -> None:
        """End-to-end: build old snapshot, create new extraction, diff, apply."""
        # Old snapshot: 3 files with symbols and embeddings
        old = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2", "c.py": "h3"},
            extra_units=[
                _make_symbol_unit("a.py", "stable_fn"),
                _make_symbol_unit("b.py", "changed_fn"),
                _make_symbol_unit("c.py", "deleted_fn"),
            ],
            relations=[
                _make_relation(
                    "unit:sym:a.py:stable_fn",
                    "unit:sym:b.py:changed_fn",
                    "call",
                ),
                _make_relation(
                    "unit:sym:c.py:deleted_fn",
                    "unit:sym:a.py:stable_fn",
                    "call",
                ),
                _make_relation(
                    "unit:file:c.py",
                    "unit:sym:c.py:deleted_fn",
                    "contain",
                ),
            ],
            embeddings=[
                _make_embedding("unit:sym:a.py:stable_fn", [1.0, 0.0]),
                _make_embedding("unit:sym:b.py:changed_fn", [0.5, 0.5]),
                _make_embedding("unit:sym:c.py:deleted_fn", [0.0, 1.0]),
            ],
        )

        # New extraction: b.py changed, c.py removed, d.py added
        new = _build_simple_snapshot(
            {"a.py": "h1", "b.py": "h2_new", "d.py": "h4"},
            extra_units=[
                _make_symbol_unit("a.py", "stable_fn"),
                _make_symbol_unit("b.py", "new_changed_fn"),
                _make_symbol_unit("d.py", "added_fn"),
            ],
            relations=[
                _make_relation(
                    "unit:sym:b.py:new_changed_fn",
                    "unit:sym:a.py:stable_fn",
                    "call",
                ),
            ],
            embeddings=[
                _make_embedding("unit:sym:b.py:new_changed_fn", [0.7, 0.3]),
                _make_embedding("unit:sym:d.py:added_fn", [0.2, 0.8]),
            ],
            snapshot_id="snap:test:new",
            commit_id="new123",
        )

        # Diff
        cs = diff_changed_files(old, new)
        assert cs.unchanged == ["a.py"]
        assert cs.modified == ["b.py"]
        assert cs.removed == ["c.py"]
        assert cs.added == ["d.py"]

        # Apply
        result = apply_incremental_update(old, new, cs)

        # Verify units
        unit_ids = {u.unit_id for u in result.units}
        assert "unit:sym:a.py:stable_fn" in unit_ids  # preserved
        assert "unit:sym:b.py:changed_fn" not in unit_ids  # tombstoned
        assert "unit:sym:b.py:new_changed_fn" in unit_ids  # new
        assert "unit:sym:c.py:deleted_fn" not in unit_ids  # removed
        assert "unit:sym:d.py:added_fn" in unit_ids  # added

        # Verify embeddings preserved for unchanged
        emb_ids = {e.unit_id for e in result.embeddings}
        assert "unit:sym:a.py:stable_fn" in emb_ids  # preserved
        assert "unit:sym:b.py:new_changed_fn" in emb_ids  # new
        assert "unit:sym:d.py:added_fn" in emb_ids  # new

        # Verify snapshot identity is from new
        assert result.snapshot_id == "snap:test:new"
        assert result.commit_id == "new123"

    def test_no_changes_returns_old_content_with_new_identity(self) -> None:
        """When blob_oids match, apply_incremental_update preserves old units."""
        units = [
            _make_file_unit("a.py", "h1"),
            _make_symbol_unit("a.py", "foo"),
        ]
        unchanged_support = IRUnitSupport(
            support_id="sup:stable",
            unit_id="unit:sym:a.py:foo",
            source="scip",
            support_kind="occurrence",
            role="definition",
        )
        unchanged_relation = _make_relation(
            "unit:file:a.py",
            "unit:sym:a.py:foo",
            "contain",
            rel_id="rel:stable",
        )
        unchanged_embedding = _make_embedding("unit:sym:a.py:foo")
        old = _make_snapshot(
            snapshot_id="snap:test:old",
            units=units,
            supports=[unchanged_support],
            relations=[unchanged_relation],
            embeddings=[unchanged_embedding],
            commit_id="old123",
        )
        new_units = [
            _make_file_unit("a.py", "h1"),
            _make_symbol_unit("a.py", "foo"),
        ]
        new = _make_snapshot(
            snapshot_id="snap:test:new",
            units=new_units,
            commit_id="new123",
        )
        cs = diff_changed_files(old, new)
        assert cs.unchanged == ["a.py"]
        assert cs.modified == []

        result = apply_incremental_update(old, new, cs)
        assert len(result.units) == 2
        assert result.units[0] is units[0]
        assert result.units[1] is units[1]
        assert result.supports[0] is unchanged_support
        assert result.relations[0] is unchanged_relation
        assert result.embeddings[0] is unchanged_embedding
        assert result.snapshot_id == "snap:test:new"
