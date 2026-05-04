from __future__ import annotations

from pathlib import Path

from fastcode.store.cache import (
    _CACHE_EMBEDDING_KIND,
    _CACHE_JSON_KIND,
    _CACHE_RECORD_MAGIC,
    CacheManager,
)


def _cache_config(
    tmp_path: Path, *, cache_queries: bool = False
) -> dict[str, dict[str, object]]:
    return {
        "cache": {
            "enabled": True,
            "backend": "disk",
            "cache_directory": str(tmp_path),
            "cache_queries": cache_queries,
        }
    }


def test_save_dialogue_turn_uses_json_cache_envelope(tmp_path: Path) -> None:
    manager = CacheManager(_cache_config(tmp_path))

    try:
        saved = manager.save_dialogue_turn(
            session_id="session-1",
            turn_number=1,
            query="Where is config loaded?",
            answer="src/config.py",
            summary="Located config load path",
            retrieved_elements=[
                {
                    "file": "src/config.py",
                    "repo": "repo",
                    "type": "file",
                    "name": "config.py",
                    "start_line": 1,
                    "end_line": 20,
                }
            ],
            metadata={"intent": "where", "keywords": ["config"], "multi_turn": True},
        )

        assert saved is True

        raw_turn = manager.cache.get("dialogue_session-1_turn_1")
        raw_index = manager.cache.get("dialogue_session_session-1_index")

        assert isinstance(raw_turn, bytes)
        assert isinstance(raw_index, bytes)
        assert raw_turn.startswith(_CACHE_RECORD_MAGIC + _CACHE_JSON_KIND)
        assert raw_index.startswith(_CACHE_RECORD_MAGIC + _CACHE_JSON_KIND)

        turn = manager.get_dialogue_turn("session-1", 1)
        assert turn is not None
        assert turn["summary"] == "Located config load path"
        assert turn["retrieved_elements"][0]["file"] == "src/config.py"

        session_index = manager._get_session_index("session-1")
        assert session_index is not None
        assert session_index["session_id"] == "session-1"
        assert session_index["total_turns"] == 1
        assert session_index["multi_turn"] is True
        assert isinstance(session_index["created_at"], float)
        assert isinstance(session_index["last_updated"], float)
    finally:
        manager.cache.close()


def test_set_cached_embedding_payload_uses_buffer_cache_envelope(
    tmp_path: Path,
) -> None:
    manager = CacheManager(_cache_config(tmp_path))

    try:
        payload = {
            "embedding_format": "ndarray.float32.v1",
            "embedding_dtype": "float32",
            "embedding_shape": (2,),
            "embedding_bytes": b"\x00\x00\x80?\x00\x00\x00@",
            "provider": "sentence_transformers",
            "model": "test-model",
            "dimension": 2,
            "max_seq_length": 512,
            "normalize": True,
            "text_sha256": "abc123",
        }

        saved = manager.set_cached_embedding_payload("embedding_v2_abc", payload)

        assert saved is True

        raw_value = manager.cache.get("embedding_v2_abc")
        assert isinstance(raw_value, bytes)
        assert raw_value.startswith(_CACHE_RECORD_MAGIC + _CACHE_EMBEDDING_KIND)

        restored = manager.get_cached_embedding_payload("embedding_v2_abc")
        assert restored is not None
        assert restored["embedding_format"] == "ndarray.float32.v1"
        assert restored["embedding_shape"] == [2]
        assert restored["embedding_bytes"] == payload["embedding_bytes"]
    finally:
        manager.cache.close()


def test_query_result_cache_uses_json_cache_envelope(tmp_path: Path) -> None:
    manager = CacheManager(_cache_config(tmp_path, cache_queries=True))

    try:
        result = {
            "answer": "Use src/config.py",
            "sources": [
                {
                    "file": "src/config.py",
                    "repo": "repo",
                    "type": "file",
                    "name": "config.py",
                    "start_line": 1,
                    "end_line": 20,
                }
            ],
            "summary": "Config lives in src/config.py",
        }

        saved = manager.set_query_result("Where is config loaded?", "repo-hash", result)

        assert saved is True

        cache_key = manager._generate_key(
            "query", "Where is config loaded?", "repo-hash"
        )
        raw_value = manager.cache.get(cache_key)
        assert isinstance(raw_value, bytes)
        assert raw_value.startswith(_CACHE_RECORD_MAGIC + _CACHE_JSON_KIND)
        assert (
            manager.get_query_result("Where is config loaded?", "repo-hash") == result
        )
    finally:
        manager.cache.close()
