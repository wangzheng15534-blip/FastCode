from fastcode.semantic_ir import IRSnapshot, IRSymbol
from fastcode.snapshot_symbol_index import SnapshotSymbolIndex


def test_snapshot_symbol_index_canonical_and_alias_resolution():
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
