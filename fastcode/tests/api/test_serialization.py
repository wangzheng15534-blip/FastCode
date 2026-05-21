from __future__ import annotations

from fastcode.api.contracts import QuerySourceRecord
from fastcode.api.serialization import (
    serialize_dialogue_history,
    serialize_query_source,
    serialize_query_source_record,
    serialize_query_source_record_payload,
)


class _NoDictPathSource:
    def __init__(self) -> None:
        self.repository = "repo"
        self.path = "src/only_path.py"
        self.name = "load_config"
        self.type = "function"
        self.lines = "11"
        self.score = 0.75

    def to_dict(self) -> dict[str, object]:
        raise AssertionError("serializer must not call to_dict()")


class _NoDictTurn:
    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.turn_number = 3
        self.timestamp = 12.5
        self.query = "Where?"
        self.answer = "There."
        self.summary = "Found it"
        self.retrieved_elements = [_NoDictPathSource()]
        self.metadata = {
            "intent": "where",
            "keywords": ("config",),
            "repo_filter": ("repo",),
            "multi_turn": True,
        }

    def to_dict(self) -> dict[str, object]:
        raise AssertionError("serializer must not call to_dict()")


def test_serialize_query_source_supports_path_only_and_single_line_lines() -> None:
    result = serialize_query_source(_NoDictPathSource())

    assert result == {
        "repository": "repo",
        "repo": "repo",
        "file": "src/only_path.py",
        "name": "load_config",
        "type": "function",
        "lines": "11-11",
        "start_line": 11,
        "end_line": 11,
        "score": 0.75,
    }


def test_serialize_query_source_builds_typed_record_before_payload() -> None:
    record = serialize_query_source_record(_NoDictPathSource())

    assert record == QuerySourceRecord(
        repository="repo",
        file="src/only_path.py",
        name="load_config",
        source_type="function",
        lines="11-11",
        start_line=11,
        end_line=11,
        score=0.75,
    )
    assert serialize_query_source_record_payload(record) == {
        "repository": "repo",
        "repo": "repo",
        "file": "src/only_path.py",
        "name": "load_config",
        "type": "function",
        "lines": "11-11",
        "start_line": 11,
        "end_line": 11,
        "score": 0.75,
    }


def test_serialize_dialogue_history_uses_nested_source_serializer() -> None:
    result = serialize_dialogue_history([_NoDictTurn()])

    assert result == [
        {
            "session_id": "sess-1",
            "turn_number": 3,
            "timestamp": 12.5,
            "query": "Where?",
            "answer": "There.",
            "summary": "Found it",
            "retrieved_elements": [
                {
                    "repository": "repo",
                    "repo": "repo",
                    "file": "src/only_path.py",
                    "name": "load_config",
                    "type": "function",
                    "lines": "11-11",
                    "start_line": 11,
                    "end_line": 11,
                    "score": 0.75,
                }
            ],
            "metadata": {
                "intent": "where",
                "keywords": ["config"],
                "repo_filter": ["repo"],
                "multi_turn": True,
            },
        }
    ]
