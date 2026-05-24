"""Domain-facing artifact store capability ports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol


class FileArtifactRecordView(Protocol):
    """Read-only file artifact record shape used across app/runtime boundaries."""

    @property
    def repo_name(self) -> str: ...

    @property
    def relative_path(self) -> str: ...

    @property
    def identity_kind(self) -> str: ...

    @property
    def identity_value(self) -> str: ...

    @property
    def artifact_type(self) -> str: ...

    @property
    def schema_version(self) -> str: ...

    @property
    def payload_json(self) -> str: ...

    @property
    def unit_count(self) -> int: ...

    @property
    def support_count(self) -> int: ...

    @property
    def relation_count(self) -> int: ...

    @property
    def embedding_count(self) -> int: ...

    @property
    def metadata_json(self) -> str | None: ...

    @property
    def created_at(self) -> str: ...


class UnitArtifactStore(Protocol):
    """Persistent per-unit artifact capability used by indexing flows."""

    def refresh_units(
        self,
        snapshot_id: str,
        *,
        stable_unit_ids: list[str],
        elements: list[dict[str, Any]],
    ) -> None: ...

    def replace_snapshot_units(
        self,
        snapshot_id: str,
        *,
        elements: list[dict[str, Any]],
    ) -> None: ...

    def publish_snapshot_units_delta(
        self,
        snapshot_id: str,
        *,
        previous_snapshot_id: str,
        changed_paths: list[str],
        removed_paths: list[str],
        elements: list[dict[str, Any]],
    ) -> dict[str, int | str]: ...

    def replace_snapshot_file_ir_shards(
        self,
        snapshot_id: str,
        *,
        shards: Sequence[Mapping[str, Any]],
    ) -> dict[str, int | str]: ...

    def publish_snapshot_file_ir_shards_delta(
        self,
        snapshot_id: str,
        *,
        previous_snapshot_id: str,
        changed_paths: Sequence[str],
        removed_paths: Sequence[str],
        shards: Sequence[Mapping[str, Any]],
        reused_shards: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, int | str]: ...


class FileArtifactStore(Protocol):
    """Content-addressed file artifact capability used by indexing flows."""

    parsed_elements_artifact_type: ClassVar[str]

    def list_file_ir_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> Sequence[FileArtifactRecordView]: ...

    def list_parsed_element_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> Sequence[FileArtifactRecordView]: ...

    def list_embedding_ref_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> Sequence[FileArtifactRecordView]: ...

    def list_semantic_fact_records_for_file_infos(
        self,
        *,
        repo_name: str,
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str],
    ) -> Sequence[FileArtifactRecordView]: ...

    @classmethod
    def file_ir_payload_from_record(
        cls,
        record: FileArtifactRecordView,
        *,
        snapshot_id: str | None = None,
    ) -> dict[str, Any]: ...

    @classmethod
    def parsed_elements_payload_from_record(
        cls,
        record: FileArtifactRecordView,
    ) -> dict[str, Any]: ...

    @classmethod
    def embedding_refs_payload_from_record(
        cls,
        record: FileArtifactRecordView,
    ) -> dict[str, Any]: ...

    @classmethod
    def semantic_facts_payload_from_record(
        cls,
        record: FileArtifactRecordView,
    ) -> dict[str, Any]: ...

    def upsert_file_ir_shards(
        self,
        *,
        repo_name: str,
        shards: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, int | str], Sequence[FileArtifactRecordView]]: ...

    def upsert_parsed_elements(
        self,
        *,
        repo_name: str,
        elements: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> tuple[dict[str, int | str], Sequence[FileArtifactRecordView]]: ...

    def upsert_embedding_refs(
        self,
        *,
        repo_name: str,
        rows: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> tuple[dict[str, int | str], Sequence[FileArtifactRecordView]]: ...

    def upsert_semantic_fact_shards(
        self,
        *,
        repo_name: str,
        shards: Sequence[Mapping[str, Any]],
        file_infos: Sequence[Mapping[str, Any]],
        paths: Sequence[str] | None = None,
    ) -> tuple[dict[str, int | str], Sequence[FileArtifactRecordView]]: ...
