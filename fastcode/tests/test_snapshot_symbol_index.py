"""Tests for snapshot_symbol_index module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.semantic_ir import IRSnapshot, IRSymbol
from fastcode.snapshot_symbol_index import SnapshotSymbolIndex, SnapshotSymbolMaps

# --- Strategies ---

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)


def _sym(
    symbol_id: str,
    display_name: str = "",
    qualified_name: str | None = None,
    path: str = "",
    metadata: dict[str, Any] | None = None,
) -> IRSymbol:
    return IRSymbol(
        symbol_id=symbol_id,
        external_symbol_id=None,
        path=path,
        display_name=display_name,
        kind="function",
        language="python",
        qualified_name=qualified_name,
        source_priority=10,
        source_set={"ast"},
        metadata=metadata or {},
    )


def _snapshot(
    snapshot_id: str = "snap:test",
    symbols: list[IRSymbol] | None = None,
) -> IRSnapshot:
    return IRSnapshot(
        repo_name="repo",
        snapshot_id=snapshot_id,
        symbols=symbols or [],
    )


# Alias-containing metadata strategy
alias_metadata_st = st.dictionaries(
    keys=st.just("aliases"),
    values=st.lists(st.one_of(st.none(), st.just(""), identifier), max_size=5),
)


# Symbol strategy with controlled fields
def _mk_sym(sid: str, name: str, qname: str, path: str, meta: dict[str, Any]) -> Any:
    return _sym(sid, name, qname, path, meta)


def _mk_snap(sid: str, syms: list[IRSymbol]) -> Any:
    return _snapshot(sid, syms)


def _mk_sym_id(x: Any) -> str:
    return f"sym:{x}"


def _mk_qname(x: Any) -> str:
    return f"pkg.{x}"


def _mk_path(a: Any, b: Any) -> str:
    return f"{a}/{b}.py"


def _mk_snap_id(x: Any) -> str:
    return f"snap:{x}"


symbol_st = st.builds(
    _mk_sym,
    sid=st.builds(_mk_sym_id, identifier),
    name=st.one_of(st.just(""), identifier),
    qname=st.one_of(st.none(), st.builds(_mk_qname, identifier)),
    path=st.one_of(st.just(""), st.builds(_mk_path, identifier, identifier)),
    meta=st.one_of(
        st.none(),
        st.dictionaries(st.text(min_size=1), st.integers()),
        alias_metadata_st,
    ),
)


snapshot_st = st.builds(
    _mk_snap,
    sid=st.builds(_mk_snap_id, identifier),
    syms=st.lists(symbol_st, max_size=10),
)


# --- Happy-path tests ---


def test_snapshot_symbol_index_canonical_and_alias_resolution():
    """HAPPY: canonicalize_symbol resolves aliases; resolve_symbol finds by name."""
    snap = IRSnapshot(
        repo_name="r",
        snapshot_id="snap:1",
        symbols=[
            IRSymbol(
                symbol_id="scip:snap:1:foo",
                external_symbol_id="foo",
                path="a.py",
                display_name="foo",
                kind="function",
                language="python",
                source_priority=100,
                source_set={"scip"},
                metadata={
                    "aliases": ["ast:snap:1:python:a.py:function:foo:1:2"],
                    "source": "scip",
                },
            )
        ],
    )
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    assert (
        idx.canonicalize_symbol("snap:1", "ast:snap:1:python:a.py:function:foo:1:2")
        == "scip:snap:1:foo"
    )
    assert idx.resolve_symbol("snap:1", name="foo") == "scip:snap:1:foo"


def test_register_snapshot_basic_property():
    """Registering a snapshot stores it so has_snapshot returns True."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:basic",
        [
            _sym("sym:foo", display_name="foo", path="a.py"),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.has_snapshot("snap:basic")


def test_register_snapshot_canonical_by_alias_self_property():
    """Each symbol's own ID is registered as a self-alias."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:self",
        [
            _sym("sym:alpha", display_name="alpha"),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.canonicalize_symbol("snap:self", "sym:alpha") == "sym:alpha"


def test_register_snapshot_display_name_indexed_property():
    """display_name is indexed in symbols_by_name."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:dn",
        [
            _sym("sym:myfunc", display_name="myfunc"),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:dn", name="myfunc") == "sym:myfunc"


def test_register_snapshot_qualified_name_indexed_property():
    """qualified_name is indexed in symbols_by_name."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:qn",
        [
            _sym("sym:myfunc", display_name="myfunc", qualified_name="pkg.myfunc"),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:qn", name="pkg.myfunc") == "sym:myfunc"


def test_register_snapshot_path_indexed_property():
    """symbol path is indexed in symbols_by_path."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:path",
        [
            _sym("sym:cls", display_name="cls", path="src/main.py"),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:path", path="src/main.py") == "sym:cls"


def test_register_snapshot_aliases_from_metadata_property():
    """Aliases in symbol metadata are registered."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:aliases",
        [
            _sym(
                "sym:real",
                display_name="real",
                metadata={"aliases": ["alias1", "alias2"]},
            ),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.canonicalize_symbol("snap:aliases", "alias1") == "sym:real"
    assert idx.canonicalize_symbol("snap:aliases", "alias2") == "sym:real"
    assert idx.get_aliases("snap:aliases", "sym:real") == [
        "alias1",
        "alias2",
        "sym:real",
    ]


def test_get_aliases_returns_sorted_property():
    """get_aliases returns a sorted list."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:sorted",
        [
            _sym(
                "sym:zeta",
                display_name="zeta",
                metadata={"aliases": ["beta_alias", "alpha_alias"]},
            ),
        ],
    )
    idx.register_snapshot(snap)
    aliases = idx.get_aliases("snap:sorted", "sym:zeta")
    assert aliases == sorted(aliases)


def test_resolve_symbol_by_symbol_id_property():
    """resolve_symbol with symbol_id canonicalizes via alias map."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:resolve_id",
        [
            _sym("sym:orig", display_name="orig", metadata={"aliases": ["aka"]}),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:resolve_id", symbol_id="aka") == "sym:orig"


def test_resolve_symbol_by_name_property():
    """resolve_symbol with name finds via symbols_by_name."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:resolve_name",
        [
            _sym("sym:foo", display_name="foo"),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:resolve_name", name="foo") == "sym:foo"


def test_resolve_symbol_by_path_property():
    """resolve_symbol with path finds via symbols_by_path."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:resolve_path",
        [
            _sym("sym:bar", display_name="bar", path="x/y.py"),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:resolve_path", path="x/y.py") == "sym:bar"


def test_resolve_symbol_priority_symbol_id_over_name_property():
    """symbol_id is checked before name."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:prio",
        [
            _sym(
                "sym:first", display_name="first", metadata={"aliases": ["shared_name"]}
            ),
            _sym("sym:second", display_name="shared_name"),
        ],
    )
    idx.register_snapshot(snap)
    # symbol_id lookup finds via alias map
    result = idx.resolve_symbol(
        "snap:prio", symbol_id="shared_name", name="shared_name"
    )
    assert result == "sym:first"


def test_multiple_symbols_same_path_property():
    """Multiple symbols on the same path are all indexed."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:mpath",
        [
            _sym("sym:a", display_name="a", path="shared.py"),
            _sym("sym:b", display_name="b", path="shared.py"),
        ],
    )
    idx.register_snapshot(snap)
    result = idx.resolve_symbol("snap:mpath", path="shared.py")
    assert result in ("sym:a", "sym:b")


def test_register_multiple_snapshots_property():
    """Multiple snapshots coexist independently."""
    idx = SnapshotSymbolIndex()
    snap1 = _snapshot("snap:one", [_sym("sym:a", display_name="a")])
    snap2 = _snapshot("snap:two", [_sym("sym:b", display_name="b")])
    idx.register_snapshot(snap1)
    idx.register_snapshot(snap2)
    assert idx.has_snapshot("snap:one")
    assert idx.has_snapshot("snap:two")
    assert idx.canonicalize_symbol("snap:one", "sym:a") == "sym:a"
    assert idx.canonicalize_symbol("snap:two", "sym:b") == "sym:b"


# --- Edge-case tests ---


@pytest.mark.edge
def test_has_snapshot_false_for_unknown_property():
    """has_snapshot returns False for unregistered snapshot."""
    idx = SnapshotSymbolIndex()
    assert not idx.has_snapshot("snap:nonexistent")


@pytest.mark.edge
def test_canonicalize_symbol_missing_snapshot_property():
    """canonicalize_symbol returns None for unknown snapshot_id."""
    idx = SnapshotSymbolIndex()
    assert idx.canonicalize_symbol("snap:nope", "sym:x") is None


@pytest.mark.edge
def test_canonicalize_symbol_missing_symbol_property():
    """canonicalize_symbol returns None for unknown symbol_id."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot("snap:ex", [_sym("sym:known", display_name="known")])
    idx.register_snapshot(snap)
    assert idx.canonicalize_symbol("snap:ex", "sym:unknown") is None


@pytest.mark.edge
def test_get_aliases_missing_snapshot_property():
    """get_aliases returns empty list for unknown snapshot_id."""
    idx = SnapshotSymbolIndex()
    assert idx.get_aliases("snap:missing", "sym:x") == []


@pytest.mark.edge
def test_get_aliases_missing_symbol_property():
    """get_aliases returns empty list for unknown canonical id."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot("snap:ex2", [_sym("sym:known", display_name="known")])
    idx.register_snapshot(snap)
    assert idx.get_aliases("snap:ex2", "sym:unknown") == []


@pytest.mark.edge
def test_resolve_symbol_missing_snapshot_property():
    """resolve_symbol returns None for unknown snapshot_id."""
    idx = SnapshotSymbolIndex()
    assert idx.resolve_symbol("snap:nope", symbol_id="sym:x") is None


@pytest.mark.edge
def test_resolve_symbol_no_params_property():
    """resolve_symbol returns None when no lookup params given."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot("snap:empty", [_sym("sym:x", display_name="x")])
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:empty") is None


@pytest.mark.edge
def test_resolve_symbol_all_params_none_match_property():
    """resolve_symbol returns None when nothing matches."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot("snap:nm", [_sym("sym:x", display_name="x")])
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:nm", symbol_id="sym:nope") is None
    assert idx.resolve_symbol("snap:nm", name="nope") is None
    assert idx.resolve_symbol("snap:nm", path="nope.py") is None


@pytest.mark.edge
def test_register_snapshot_empty_symbols_property():
    """Registering a snapshot with no symbols works."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot("snap:empty_sym")
    idx.register_snapshot(snap)
    assert idx.has_snapshot("snap:empty_sym")
    assert idx.canonicalize_symbol("snap:empty_sym", "sym:x") is None


@pytest.mark.edge
def test_register_snapshot_empty_alias_skipped_property():
    """Empty-string aliases are skipped during registration."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:empty_alias",
        [
            _sym("sym:x", display_name="x", metadata={"aliases": ["", "valid_alias"]}),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.canonicalize_symbol("snap:empty_alias", "") is None
    assert idx.canonicalize_symbol("snap:empty_alias", "valid_alias") == "sym:x"


@pytest.mark.edge
def test_register_snapshot_none_alias_skipped_property():
    """None values in aliases list are skipped."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:none_alias",
        [
            _sym("sym:y", display_name="y", metadata={"aliases": [None, "real_alias"]}),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.canonicalize_symbol("snap:none_alias", "real_alias") == "sym:y"


@pytest.mark.edge
def test_register_snapshot_none_metadata_property():
    """Symbol with metadata=None is handled gracefully."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:null_meta",
        [
            _sym("sym:z", display_name="z", metadata=None),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.canonicalize_symbol("snap:null_meta", "sym:z") == "sym:z"


@pytest.mark.edge
def test_register_snapshot_empty_display_name_not_indexed_property():
    """Empty display_name does not create an entry in symbols_by_name."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:empty_dn",
        [
            _sym("sym:empty", display_name="", qualified_name=None),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:empty_dn", name="") is None


@pytest.mark.edge
def test_register_snapshot_empty_path_not_indexed_property():
    """Empty path does not create an entry in symbols_by_path."""
    idx = SnapshotSymbolIndex()
    snap = _snapshot(
        "snap:empty_path",
        [
            _sym("sym:nopath", display_name="nopath", path=""),
        ],
    )
    idx.register_snapshot(snap)
    assert idx.resolve_symbol("snap:empty_path", path="") is None


@pytest.mark.edge
def test_register_overwrites_previous_snapshot_property():
    """Re-registering the same snapshot_id overwrites the previous data."""
    idx = SnapshotSymbolIndex()
    snap1 = _snapshot("snap:overwrite", [_sym("sym:old", display_name="old")])
    snap2 = _snapshot("snap:overwrite", [_sym("sym:new", display_name="new")])
    idx.register_snapshot(snap1)
    idx.register_snapshot(snap2)
    assert idx.canonicalize_symbol("snap:overwrite", "sym:old") is None
    assert idx.canonicalize_symbol("snap:overwrite", "sym:new") == "sym:new"


@pytest.mark.edge
def test_snapshot_symbol_maps_defaults_property():
    """SnapshotSymbolMaps dataclass has empty defaults."""
    maps = SnapshotSymbolMaps()
    assert maps.canonical_by_alias == {}
    assert maps.aliases_by_canonical == {}
    assert maps.symbols_by_name == {}
    assert maps.symbols_by_path == {}


# --- Property-based tests ---


@given(snap=snapshot_st)
@settings(max_examples=30)
def test_has_snapshot_after_register_property(snap: IRSnapshot):
    """After registering, has_snapshot returns True."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    assert idx.has_snapshot(snap.snapshot_id)


@given(snap=snapshot_st)
@settings(max_examples=30)
def test_canonicalize_self_alias_property(snap: IRSnapshot):
    """Every symbol's own ID canonicalizes to itself after registration."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    for sym in snap.symbols:
        result = idx.canonicalize_symbol(snap.snapshot_id, sym.symbol_id)
        assert result == sym.symbol_id


@given(snap=snapshot_st)
@settings(max_examples=30)
def test_get_aliases_contains_self_property(snap: IRSnapshot):
    """Every symbol's alias set includes its own ID."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    for sym in snap.symbols:
        aliases = idx.get_aliases(snap.snapshot_id, sym.symbol_id)
        assert sym.symbol_id in aliases


@given(snap=snapshot_st)
@settings(max_examples=30)
def test_get_aliases_sorted_property(snap: IRSnapshot):
    """get_aliases always returns a sorted list."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    for sym in snap.symbols:
        aliases = idx.get_aliases(snap.snapshot_id, sym.symbol_id)
        assert aliases == sorted(aliases)


@given(snap=snapshot_st)
@settings(max_examples=30)
def test_resolve_symbol_by_id_matches_canonicalize_property(snap: IRSnapshot):
    """resolve_symbol(symbol_id=x) matches canonicalize_symbol(x) when canonical exists."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    for sym in snap.symbols:
        canonical = idx.canonicalize_symbol(snap.snapshot_id, sym.symbol_id)
        resolved = idx.resolve_symbol(snap.snapshot_id, symbol_id=sym.symbol_id)
        assert canonical == resolved


@given(snap=snapshot_st)
@settings(max_examples=30)
@pytest.mark.edge
def test_canonicalize_unknown_returns_none_property(snap: IRSnapshot):
    """canonicalize_symbol with a completely unknown ID returns None."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    result = idx.canonicalize_symbol(snap.snapshot_id, "sym:does_not_exist_at_all_xyz")
    assert result is None


@given(snap=snapshot_st)
@settings(max_examples=30)
@pytest.mark.edge
def test_canonicalize_wrong_snapshot_returns_none_property(snap: IRSnapshot):
    """canonicalize_symbol with wrong snapshot_id returns None."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    result = idx.canonicalize_symbol(
        "snap:wrong_snapshot", snap.symbols[0].symbol_id if snap.symbols else "sym:x"
    )
    assert result is None


@given(snap=snapshot_st)
@settings(max_examples=30)
@pytest.mark.edge
def test_get_aliases_wrong_snapshot_empty_property(snap: IRSnapshot):
    """get_aliases with wrong snapshot_id returns empty list."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    result = idx.get_aliases("snap:wrong", "sym:x")
    assert result == []


@given(snap=snapshot_st)
@settings(max_examples=30)
@pytest.mark.edge
def test_resolve_symbol_wrong_snapshot_none_property(snap: IRSnapshot):
    """resolve_symbol with wrong snapshot_id returns None."""
    idx = SnapshotSymbolIndex()
    idx.register_snapshot(snap)
    result = idx.resolve_symbol("snap:wrong", symbol_id="sym:x")
    assert result is None
