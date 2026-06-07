"""Contract tests for fastcode.common.error — meaning_core, ZERO test doubles.

KernelError is the shared validation error type. These tests verify that
its structure and factory methods produce correct values.
"""

from __future__ import annotations

import pytest

from fastcode.common.error import KernelError


class TestKernelErrorStructure:
    def test_is_value_error_subclass(self) -> None:
        assert issubclass(KernelError, ValueError)

    def test_stores_code_and_reason(self) -> None:
        err = KernelError("code_x", "reason y")
        assert err.code == "code_x"
        assert err.reason == "reason y"

    def test_str_includes_code_and_reason(self) -> None:
        err = KernelError("code_x", "reason y")
        assert "code_x" in str(err)
        assert "reason y" in str(err)


class TestKernelErrorFactories:
    def test_invalid_repo_name(self) -> None:
        err = KernelError.invalid_repo_name("bad chars")
        assert err.code == "invalid_repo_name"
        assert "bad chars" in err.reason

    def test_invalid_snapshot_id(self) -> None:
        err = KernelError.invalid_snapshot_id("missing prefix")
        assert err.code == "invalid_snapshot_id"
        assert "missing prefix" in err.reason

    def test_invalid_artifact_key(self) -> None:
        err = KernelError.invalid_artifact_key("bad pattern")
        assert err.code == "invalid_artifact_key"
        assert "bad pattern" in err.reason

    @pytest.mark.edge
    def test_code_always_string(self) -> None:
        err = KernelError.invalid_repo_name("x")
        assert isinstance(err.code, str)

    @pytest.mark.edge
    def test_reason_always_string(self) -> None:
        err = KernelError.invalid_repo_name("x")
        assert isinstance(err.reason, str)
