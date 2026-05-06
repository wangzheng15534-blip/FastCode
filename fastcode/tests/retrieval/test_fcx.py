from __future__ import annotations

import pytest

from fastcode.retrieval.core import fcx


def test_fcx_render_and_parse_block_roundtrip() -> None:
    records = [
        fcx.render_record("F", "f1", fields={"scope": "turn", "refs": ("e1", "e2")}),
        fcx.render_record(
            "Q", "q1", fields={"need": "evidence"}, tail="Need auth flow"
        ),
        fcx.render_record("END", fields={"refs": 2}),
    ]

    block = fcx.render_block(
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

    parsed = fcx.parse_block(block)

    assert parsed["header"]["mode"] == "turn"
    assert parsed["header"]["sid"] == "session-1"
    assert parsed["records"][0]["tag"] == "F"
    assert parsed["records"][0]["fields"]["refs"] == ["e1", "e2"]
    assert parsed["records"][1]["tail"] == "Need auth flow"
    assert parsed["records"][-1]["tag"] == "END"


def test_fcx_parse_block_rejects_duplicate_ids() -> None:
    block = "\n".join(
        [
            "@fcx mode=turn v=1 sid=s1 turn=1 snap=snap:1 art=art:1 fp=fcx-v1",
            "F f1 scope=turn refs=e1,e2",
            "Q f1 need=evidence | duplicate id",
            "END refs=2",
        ]
    )

    with pytest.raises(ValueError, match="duplicate FCX record id"):
        fcx.parse_block(block)
