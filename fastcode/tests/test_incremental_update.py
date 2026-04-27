"""
Tests for incremental update via blob_oid diffing.

Each test is self-contained with its own factory functions.
"""

from __future__ import annotations

from fastcode.incremental_update import (
    FileChangeSet,
    apply_incremental_update,
    diff_changed_files,
)
from fastcode.semantic_ir import (
    IRCodeUnit,
    IRRelation,
    IRSnapshot,
    IRUnitEmbedding,
    IRUnitSupport,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_file_unit(path: str, blob_oid: str, unit_id: str | None = None) -> IRCodeUnit:
    """Create a file-level IRCodeUnit with blob_oid in metadata."""
    uid = unit_id or f"unit:file:{path}"
    return IRCodeUnit(
        unit_id=uid,
        kind="file",
        path=path,
        language="python",
        display_name=path,
        source_set={"fc_structure"},
        metadata={"blob_oid": blob_oid},
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
        # Manually build with None blob_oid
        new_units = [_make_file_unit("a.py", None)]
        new = _make_snapshot(snapshot_id="snap:test:new", units=new_units)
        cs = diff_changed_files(old, new)
        assert cs.modified == ["a.py"]


# ---------------------------------------------------------------------------
# Tests: apply_incremental_update
# ---------------------------------------------------------------------------


class TestApplyIncrementalUpdate:
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
        old = _make_snapshot(
            snapshot_id="snap:test:old",
            units=units,
            embeddings=[_make_embedding("unit:sym:a.py:foo")],
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
        assert result.snapshot_id == "snap:test:new"
