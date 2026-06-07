from __future__ import annotations

import pytest

from fastcode.retrieval.ranking.fcx import parse_block, render_block, render_record


def test_fcx_render_and_parse_block_roundtrip() -> None:
    records = [
        render_record("F", "f1", fields={"scope": "turn", "refs": ("e1", "e2")}),
        render_record("Q", "q1", fields={"need": "evidence"}, tail="Need auth flow"),
        render_record("END", fields={"refs": 2}),
    ]

    block = render_block(
        mode="turn",
        header_fields={
            "v": 1,
            "sid": "session-1",
            "turn": 2,
            "snap": "snap:1",
            "art": "art:1",
            "fp": "fcx-v1",
        },
        records=records,
    )

    parsed = parse_block(block)

    assert parsed["header"]["mode"] == "turn"
    assert parsed["header"]["sid"] == "session-1"
    assert parsed["records"][0]["tag"] == "F"
    assert parsed["records"][0]["fields"]["refs"] == ["e1", "e2"]
    assert parsed["records"][1]["tail"] == "Need auth flow"
    assert parsed["records"][-1]["tag"] == "END"


def test_fcx_parse_block_rejects_duplicate_ids() -> None:
    block = "@fcx mode=turn v=1 sid=s1 turn=1 snap=snap:1 art=art:1 fp=fcx-v1\nF f1 scope=turn refs=e1,e2\nQ f1 need=evidence | duplicate id\nEND refs=2"

    with pytest.raises(ValueError, match="duplicate FCX record id"):
        parse_block(block)
