"""
Caching Module - Cache embeddings, queries, and results
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, cast

from diskcache import Cache as DiskCache

from .records import (
    DialogueSessionRecord,
    DialogueTurnRecord,
    HandoffArtifactRecord,
    TurnJournalRecord,
    WorkingMemoryRecord,
)

_CACHE_RECORD_MAGIC = b"fastcode-cache:v1:"
_CACHE_JSON_KIND = b"json:"
_CACHE_EMBEDDING_KIND = b"embedding:"
_CACHE_NOT_MARSHALLED = object()


class CacheManager:
    """Manage caching for FastCode"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cache_config = config.get("cache", {})
        self.logger = logging.getLogger(__name__)

        self.enabled = self.cache_config.get("enabled", True)
        self.backend = self.cache_config.get("backend", "disk")
        self.ttl = self.cache_config.get("ttl", 3600)
        self.max_size_mb = self.cache_config.get("max_size_mb", 1000)
        self.cache_directory = self.cache_config.get("cache_directory", "./data/cache")

        self.cache_embeddings = self.cache_config.get("cache_embeddings", True)
        self.cache_queries = self.cache_config.get("cache_queries", False)

        # Dialogue history TTL (default: 30 days for long-term conversation history)
        self.dialogue_ttl = self.cache_config.get(
            "dialogue_ttl", 2592000
        )  # 30 days in seconds

        self.cache: Any = None

        if self.enabled:
            self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize cache backend"""
        if self.backend == "disk":
            Path(self.cache_directory).mkdir(parents=True, exist_ok=True)
            max_size_bytes = self.max_size_mb * 1024 * 1024
            self.cache = DiskCache(self.cache_directory, size_limit=max_size_bytes)
            self.logger.info(f"Initialized disk cache at {self.cache_directory}")

        elif self.backend == "redis":
            try:
                import redis

                self.cache = redis.Redis(
                    host=self.cache_config.get("redis_host", "localhost"),
                    port=int(self.cache_config.get("redis_port", 6379)),
                    db=0,
                    decode_responses=False,
                )
                self.cache.ping()
                self.logger.info("Initialized Redis cache")
            except Exception as e:
                self.logger.error(f"Failed to initialize Redis cache: {e}")
                self.enabled = False

        else:
            self.logger.warning(f"Unknown cache backend: {self.backend}")
            self.enabled = False

    def _generate_key(self, prefix: str, *args: Any) -> str:
        """Generate cache key from arguments"""
        # Create a hash of all arguments
        content = "_".join(str(arg) for arg in args)
        hash_val = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}_{hash_val}"

    def _cache_get_raw(self, key: str) -> Any | None:
        if self.backend == "disk":
            return self.cache.get(key)
        if self.backend == "redis":
            return self.cache.get(key)
        return None

    def _cache_set_raw(self, key: str, value: Any, ttl: int) -> bool:
        if self.backend == "disk":
            self.cache.set(key, value, expire=ttl)
            return True
        if self.backend == "redis":
            self.cache.setex(key, ttl, value)  # type: ignore[arg-type]
            return True
        return False

    @staticmethod
    def _json_cache_payload(value: Any) -> bytes:
        json_bytes = json.dumps(value, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        return _CACHE_RECORD_MAGIC + _CACHE_JSON_KIND + json_bytes

    @staticmethod
    def _dialogue_turn_payload(record: DialogueTurnRecord) -> dict[str, Any]:
        return {
            "session_id": record.session_id,
            "turn_number": record.turn_number,
            "timestamp": record.timestamp,
            "query": record.query,
            "answer": record.answer,
            "summary": record.summary,
            "retrieved_elements": list(record.retrieved_elements),
            "metadata": dict(record.metadata),
        }

    @staticmethod
    def _dialogue_session_payload(record: DialogueSessionRecord) -> dict[str, Any]:
        return {
            "session_id": record.session_id,
            "created_at": record.created_at,
            "total_turns": record.total_turns,
            "last_updated": record.last_updated,
            "multi_turn": record.multi_turn,
        }

    @staticmethod
    def _turn_journal_payload(record: TurnJournalRecord) -> dict[str, Any]:
        return {
            "session_id": record.session_id,
            "turn_number": record.turn_number,
            "snapshot_id": record.snapshot_id,
            "artifact_key": record.artifact_key,
            "compiler_fingerprint": record.compiler_fingerprint,
            "payload_json": record.payload_json,
            "created_at": record.created_at,
        }

    @staticmethod
    def _working_memory_payload(record: WorkingMemoryRecord) -> dict[str, Any]:
        return {
            "session_id": record.session_id,
            "turn_number": record.turn_number,
            "snapshot_id": record.snapshot_id,
            "artifact_key": record.artifact_key,
            "compiler_fingerprint": record.compiler_fingerprint,
            "payload_json": record.payload_json,
            "stable_fcx": record.stable_fcx,
            "turn_fcx": record.turn_fcx,
            "obs_fcx": record.obs_fcx,
            "full_fcx": record.full_fcx,
            "created_at": record.created_at,
        }

    @staticmethod
    def _handoff_artifact_payload(record: HandoffArtifactRecord) -> dict[str, Any]:
        return {
            "artifact_id": record.artifact_id,
            "session_id": record.session_id,
            "turn_number": record.turn_number,
            "snapshot_id": record.snapshot_id,
            "compiler_fingerprint": record.compiler_fingerprint,
            "mode": record.mode,
            "payload_json": record.payload_json,
            "full_fcx": record.full_fcx,
            "created_at": record.created_at,
        }

    @staticmethod
    def _embedding_cache_payload(value: dict[str, Any]) -> bytes:
        raw_buffer = value.get("embedding_bytes")
        if not isinstance(raw_buffer, (bytes, bytearray, memoryview)):
            raise TypeError("embedding_bytes must be bytes-like")
        metadata = {k: v for k, v in value.items() if k != "embedding_bytes"}
        metadata_bytes = json.dumps(
            metadata, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return (
            _CACHE_RECORD_MAGIC
            + _CACHE_EMBEDDING_KIND
            + str(len(metadata_bytes)).encode("ascii")
            + b":"
            + metadata_bytes
            + bytes(cast(Any, raw_buffer))
        )

    @staticmethod
    def _decode_marshaled_value(value: Any) -> Any:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            return _CACHE_NOT_MARSHALLED

        raw_value = bytes(cast(Any, value))
        if not raw_value.startswith(_CACHE_RECORD_MAGIC):
            return _CACHE_NOT_MARSHALLED

        payload = raw_value[len(_CACHE_RECORD_MAGIC) :]
        if payload.startswith(_CACHE_JSON_KIND):
            json_bytes = payload[len(_CACHE_JSON_KIND) :]
            return json.loads(json_bytes.decode("utf-8"))

        if payload.startswith(_CACHE_EMBEDDING_KIND):
            remainder = payload[len(_CACHE_EMBEDDING_KIND) :]
            delimiter_index = remainder.find(b":")
            if delimiter_index <= 0:
                raise ValueError("invalid embedding cache payload header")
            metadata_len = int(remainder[:delimiter_index].decode("ascii"))
            metadata_start = delimiter_index + 1
            metadata_end = metadata_start + metadata_len
            metadata_obj = json.loads(
                remainder[metadata_start:metadata_end].decode("utf-8")
            )
            if not isinstance(metadata_obj, dict):
                raise ValueError("invalid embedding cache metadata")
            metadata = cast(dict[str, Any], metadata_obj)
            metadata["embedding_bytes"] = remainder[metadata_end:]
            return metadata

        raise ValueError("unsupported cache payload kind")

    def get(self, key: str) -> Any | None:
        """Get value from cache"""
        if not self.enabled or self.cache is None:
            return None

        try:
            value = self._cache_get_raw(key)
            if value is None:
                return None

            decoded = self._decode_marshaled_value(value)
            if decoded is not _CACHE_NOT_MARSHALLED:
                self.logger.debug(f"Cache hit: {key}")
                return decoded

            if self.backend == "redis" and isinstance(
                value, (bytes, bytearray, memoryview)
            ):
                self.logger.debug(f"Cache hit: {key}")
                return pickle.loads(bytes(cast(Any, value)))

            self.logger.debug(f"Cache hit: {key}")
            return value

        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache"""
        if not self.enabled or self.cache is None:
            return False

        if ttl is None:
            ttl = self.ttl

        try:
            if self.backend == "disk":
                self.cache.set(key, value, expire=ttl)
                return True

            if self.backend == "redis":
                self.cache.setex(key, ttl, pickle.dumps(value))  # type: ignore[arg-type]
                return True

        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
            return False

        return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or self.cache is None:
            return False

        try:
            if self.backend == "disk":
                return bool(self.cache.delete(key))
            if self.backend == "redis":
                return bool(self.cache.delete(key))
        except Exception as e:
            self.logger.warning(f"Cache delete error: {e}")
            return False

        return False

    def clear(self) -> bool:
        """Clear all cache"""
        if not self.enabled or self.cache is None:
            return False

        try:
            if self.backend == "disk":
                self.cache.clear()
                self.logger.info("Cleared disk cache")
                return True
            if self.backend == "redis":
                self.cache.flushdb()  # type: ignore[union-attr]
                self.logger.info("Cleared Redis cache")
                return True
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return False

        return False

    def get_embedding(self, text: str) -> Any | None:
        """Get cached embedding"""
        if not self.cache_embeddings:
            return None
        key = self._generate_key("embedding", text)
        return self.get(key)

    def set_embedding(self, text: str, embedding: Any) -> bool:
        """Cache embedding"""
        if not self.cache_embeddings:
            return False
        key = self._generate_key("embedding", text)
        return self.set(key, embedding)

    def get_cached_embedding_payload(self, key: str) -> dict[str, Any] | None:
        """Get embedding payload stored under an explicit cache key."""
        value = self.get(key)
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
        return None

    def set_cached_embedding_payload(
        self, key: str, payload: dict[str, Any], ttl: int | None = None
    ) -> bool:
        """Store embedding payload in a buffer-aware cache envelope."""
        if not self.enabled or self.cache is None:
            return False
        ttl_value = int(self.ttl if ttl is None else ttl)
        try:
            return self._cache_set_raw(
                key, self._embedding_cache_payload(payload), ttl_value
            )
        except Exception as e:
            self.logger.warning(f"Embedding cache set error: {e}")
            return False

    def get_query_result(self, query: str, repo_hash: str) -> Any | None:
        """Get cached query result"""
        if not self.cache_queries:
            return None
        key = self._generate_key("query", query, repo_hash)
        return self.get(key)

    def set_query_result(self, query: str, repo_hash: str, result: Any) -> bool:
        """Cache query result"""
        if not self.cache_queries:
            return False
        key = self._generate_key("query", query, repo_hash)
        if not self.enabled or self.cache is None:
            return False
        try:
            ttl = self.ttl
            return self._cache_set_raw(key, self._json_cache_payload(result), ttl)
        except Exception as e:
            self.logger.warning(f"Query cache set error: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled or self.cache is None:
            return {"enabled": False}

        try:
            if self.backend == "disk":
                return {
                    "enabled": True,
                    "backend": "disk",
                    "size": self.cache.volume(),
                    "items": len(self.cache),
                }
            if self.backend == "redis":
                info: Any = self.cache.info()  # type: ignore[union-attr]
                return {
                    "enabled": True,
                    "backend": "redis",
                    "size": info.get("used_memory", 0),
                    "items": self.cache.dbsize(),  # type: ignore[union-attr]
                }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}

        return {"enabled": False}

    # ===== Multi-turn Dialogue Session Cache Methods =====

    def save_dialogue_turn(
        self,
        session_id: str,
        turn_number: int,
        query: str,
        answer: str,
        summary: str,
        retrieved_elements: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Save a single dialogue turn to cache

        Args:
            session_id: Unique session identifier
            turn_number: Turn number (1-indexed)
            query: User query
            answer: Generated answer
            summary: Brief summary of the dialogue turn
            retrieved_elements: Retrieved code elements (optional)
            metadata: Additional metadata (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            turn_record = DialogueTurnRecord(
                session_id=session_id,
                turn_number=turn_number,
                timestamp=time.time(),
                query=query,
                answer=answer,
                summary=summary,
                retrieved_elements=list(retrieved_elements or []),
                metadata=dict(metadata or {}),
            )

            # Generate key
            key = f"dialogue_{session_id}_turn_{turn_number}"

            # Save to cache (with longer TTL for dialogue history)
            # Use configurable dialogue_ttl instead of hardcoded value
            self._cache_set_raw(
                key,
                self._json_cache_payload(self._dialogue_turn_payload(turn_record)),
                self.dialogue_ttl,
            )

            # Update session index (propagate multi_turn flag from metadata)
            multi_turn = (metadata or {}).get("multi_turn")
            self._update_session_index(session_id, turn_number, multi_turn=multi_turn)

            self.logger.debug(f"Saved dialogue turn: {session_id} turn {turn_number}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save dialogue turn: {e}")
            return False

    def get_dialogue_turn(
        self, session_id: str, turn_number: int
    ) -> dict[str, Any] | None:
        """
        Get a specific dialogue turn from cache

        Args:
            session_id: Session identifier
            turn_number: Turn number to retrieve

        Returns:
            Turn data dictionary or None
        """
        if not self.enabled:
            return None

        record = self.get_dialogue_turn_record(session_id, turn_number)
        return self._dialogue_turn_payload(record) if record is not None else None

    def get_dialogue_turn_record(
        self, session_id: str, turn_number: int
    ) -> DialogueTurnRecord | None:
        if not self.enabled:
            return None
        key = f"dialogue_{session_id}_turn_{turn_number}"
        value = self.get(key)
        if not isinstance(value, dict):
            return None
        return DialogueTurnRecord.from_dict(cast(dict[str, Any], value))

    def get_dialogue_history_records(
        self, session_id: str, max_turns: int | None = None
    ) -> list[DialogueTurnRecord]:
        """
        Get dialogue history for a session

        Args:
            session_id: Session identifier
            max_turns: Maximum number of recent turns to retrieve (None = all)

        Returns:
            List of turn data dictionaries, ordered from oldest to newest
        """
        if not self.enabled:
            return []

        try:
            # Get session index
            session_index = self.get_session_index_record(session_id)
            if not session_index:
                return []

            total_turns = session_index.total_turns
            if total_turns == 0:
                return []

            # Determine which turns to retrieve
            if max_turns is None or max_turns >= total_turns:
                start_turn = 1
            else:
                start_turn = total_turns - max_turns + 1

            # Retrieve turns
            history: list[DialogueTurnRecord] = []
            for turn_num in range(start_turn, total_turns + 1):
                turn_record = self.get_dialogue_turn_record(session_id, turn_num)
                if turn_record is not None:
                    history.append(turn_record)

            return history

        except Exception as e:
            self.logger.error(f"Failed to get dialogue history: {e}")
            return []

    def get_dialogue_history(
        self, session_id: str, max_turns: int | None = None
    ) -> list[dict[str, Any]]:
        return [
            self._dialogue_turn_payload(record)
            for record in self.get_dialogue_history_records(session_id, max_turns)
        ]

    def get_recent_summaries(
        self, session_id: str, num_rounds: int
    ) -> list[dict[str, Any]]:
        """
        Get recent dialogue summaries for context

        Args:
            session_id: Session identifier
            num_rounds: Number of recent rounds to retrieve

        Returns:
            List of summary data with turn_number, query, and summary
        """
        if not self.enabled:
            return []

        try:
            history = self.get_dialogue_history_records(
                session_id, max_turns=num_rounds
            )

            summaries: list[dict[str, Any]] = []
            for turn in history:
                summaries.append(
                    {
                        "turn_number": turn.turn_number,
                        "query": turn.query,
                        "summary": turn.summary,
                    }
                )

            return summaries

        except Exception as e:
            self.logger.error(f"Failed to get recent summaries: {e}")
            return []

    def save_turn_journal_record(self, record: TurnJournalRecord) -> bool:
        if not self.enabled:
            return False
        try:
            key = f"turn_journal_{record.session_id}_turn_{record.turn_number}"
            self._cache_set_raw(
                key,
                self._json_cache_payload(self._turn_journal_payload(record)),
                self.dialogue_ttl,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save turn journal record: {e}")
            return False

    def get_turn_journal_record(
        self, session_id: str, turn_number: int
    ) -> TurnJournalRecord | None:
        if not self.enabled:
            return None
        key = f"turn_journal_{session_id}_turn_{turn_number}"
        value = self.get(key)
        if not isinstance(value, dict):
            return None
        return TurnJournalRecord.from_dict(cast(dict[str, Any], value))

    def get_turn_journal_records(
        self, session_id: str, max_turns: int | None = None
    ) -> list[TurnJournalRecord]:
        if not self.enabled:
            return []
        session_index = self.get_session_index_record(session_id)
        if session_index is None:
            return []
        total_turns = session_index.total_turns
        if total_turns <= 0:
            return []
        start_turn = 1
        if max_turns is not None and max_turns < total_turns:
            start_turn = total_turns - max_turns + 1
        records: list[TurnJournalRecord] = []
        for turn_number in range(start_turn, total_turns + 1):
            record = self.get_turn_journal_record(session_id, turn_number)
            if record is not None:
                records.append(record)
        return records

    def save_working_memory_record(self, record: WorkingMemoryRecord) -> bool:
        if not self.enabled:
            return False
        try:
            key = f"working_memory_{record.session_id}_turn_{record.turn_number}"
            self._cache_set_raw(
                key,
                self._json_cache_payload(self._working_memory_payload(record)),
                self.dialogue_ttl,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save working memory record: {e}")
            return False

    def get_working_memory_record(
        self, session_id: str, turn_number: int
    ) -> WorkingMemoryRecord | None:
        if not self.enabled:
            return None
        key = f"working_memory_{session_id}_turn_{turn_number}"
        value = self.get(key)
        if not isinstance(value, dict):
            return None
        return WorkingMemoryRecord.from_dict(cast(dict[str, Any], value))

    def get_working_memory_records(
        self, session_id: str, max_turns: int | None = None
    ) -> list[WorkingMemoryRecord]:
        if not self.enabled:
            return []
        session_index = self.get_session_index_record(session_id)
        if session_index is None:
            return []
        total_turns = session_index.total_turns
        if total_turns <= 0:
            return []
        start_turn = 1
        if max_turns is not None and max_turns < total_turns:
            start_turn = total_turns - max_turns + 1
        records: list[WorkingMemoryRecord] = []
        for turn_number in range(start_turn, total_turns + 1):
            record = self.get_working_memory_record(session_id, turn_number)
            if record is not None:
                records.append(record)
        return records

    def get_latest_working_memory_record(
        self, session_id: str
    ) -> WorkingMemoryRecord | None:
        session_index = self.get_session_index_record(session_id)
        if session_index is None or session_index.total_turns <= 0:
            return None
        return self.get_working_memory_record(session_id, session_index.total_turns)

    def save_handoff_artifact_record(self, record: HandoffArtifactRecord) -> bool:
        if not self.enabled:
            return False
        try:
            key = f"handoff_{record.artifact_id}"
            self._cache_set_raw(
                key,
                self._json_cache_payload(self._handoff_artifact_payload(record)),
                self.dialogue_ttl,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save handoff artifact record: {e}")
            return False

    def get_handoff_artifact_record(
        self, artifact_id: str
    ) -> HandoffArtifactRecord | None:
        if not self.enabled:
            return None
        key = f"handoff_{artifact_id}"
        value = self.get(key)
        if not isinstance(value, dict):
            return None
        return HandoffArtifactRecord.from_dict(cast(dict[str, Any], value))

    def list_handoff_artifact_records(
        self, session_id: str | None = None
    ) -> list[HandoffArtifactRecord]:
        if not self.enabled or self.cache is None:
            return []
        records: list[HandoffArtifactRecord] = []
        try:
            if self.backend == "disk":
                iterable = self.cache.iterkeys()
            elif self.backend == "redis":
                iterable = self.cache.scan_iter(match="handoff_*")
            else:
                iterable = []

            for raw_key in iterable:
                key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                if not isinstance(key, str) or not key.startswith("handoff_"):
                    continue
                value = self.get(key)
                if not isinstance(value, dict):
                    continue
                record = HandoffArtifactRecord.from_dict(cast(dict[str, Any], value))
                if session_id is None or record.session_id == session_id:
                    records.append(record)
            records.sort(key=lambda record: record.created_at, reverse=True)
            return records
        except Exception as e:
            self.logger.error(f"Failed to list handoff artifact records: {e}")
            return []

    def _update_session_index(
        self, session_id: str, turn_number: int, multi_turn: bool | None = None
    ) -> bool:
        """Update session index with new turn"""
        try:
            key = f"dialogue_session_{session_id}_index"
            existing_record = self.get_session_index_record(session_id)
            now = time.time()
            session_index = DialogueSessionRecord(
                session_id=session_id,
                created_at=(
                    existing_record.created_at if existing_record is not None else now
                ),
                total_turns=max(
                    existing_record.total_turns if existing_record is not None else 0,
                    turn_number,
                ),
                last_updated=now,
                multi_turn=(
                    True
                    if multi_turn is True
                    else (
                        existing_record.multi_turn
                        if existing_record is not None
                        else False
                    )
                ),
            )

            # Once a session is marked as multi_turn, keep it that way
            # Use configurable dialogue_ttl instead of hardcoded value
            self._cache_set_raw(
                key,
                self._json_cache_payload(self._dialogue_session_payload(session_index)),
                self.dialogue_ttl,
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to update session index: {e}")
            return False

    def get_session_index_record(self, session_id: str) -> DialogueSessionRecord | None:
        key = f"dialogue_session_{session_id}_index"
        value = self.get(key)
        if not isinstance(value, dict):
            return None
        return DialogueSessionRecord.from_dict(cast(dict[str, Any], value))

    def _get_session_index(self, session_id: str) -> dict[str, Any] | None:
        """Get session index"""
        record = self.get_session_index_record(session_id)
        return self._dialogue_session_payload(record) if record is not None else None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete an entire dialogue session

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Get session index
            session_index = self.get_session_index_record(session_id)
            if not session_index:
                return False

            total_turns = session_index.total_turns

            # Delete all turns
            for turn_num in range(1, total_turns + 1):
                key = f"dialogue_{session_id}_turn_{turn_num}"
                self.delete(key)
                self.delete(f"turn_journal_{session_id}_turn_{turn_num}")
                self.delete(f"working_memory_{session_id}_turn_{turn_num}")

            # Delete session index
            index_key = f"dialogue_session_{session_id}_index"
            self.delete(index_key)

            for record in self.list_handoff_artifact_records(session_id):
                self.delete(f"handoff_{record.artifact_id}")

            self.logger.info(f"Deleted session {session_id} with {total_turns} turns")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete session: {e}")
            return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all dialogue sessions

        Returns:
            List of session metadata dictionaries
        """
        if not self.enabled or self.cache is None:
            return []

        try:
            return [
                self._dialogue_session_payload(record)
                for record in self.list_session_records()
            ]

        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return []

    def list_session_records(self) -> list[DialogueSessionRecord]:
        if not self.enabled or self.cache is None:
            return []

        try:
            sessions: list[DialogueSessionRecord] = []

            if self.backend == "disk":
                # Scan for session index keys
                for key in self.cache.iterkeys():
                    if (
                        isinstance(key, str)
                        and key.startswith("dialogue_session_")
                        and key.endswith("_index")
                    ):
                        session_data = self.get(key)
                        if isinstance(session_data, dict):
                            sessions.append(
                                DialogueSessionRecord.from_dict(
                                    cast(dict[str, Any], session_data)
                                )
                            )

            elif self.backend == "redis":
                # Scan for session index keys
                for key in self.cache.scan_iter(match="dialogue_session_*_index"):
                    session_data = self.get(
                        key.decode() if isinstance(key, bytes) else key
                    )
                    if isinstance(session_data, dict):
                        sessions.append(
                            DialogueSessionRecord.from_dict(
                                cast(dict[str, Any], session_data)
                            )
                        )

            # Sort by creation time descending (fallback to last_updated)
            sessions.sort(
                key=lambda record: (record.created_at, record.last_updated),
                reverse=True,
            )
            return sessions

        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return []
