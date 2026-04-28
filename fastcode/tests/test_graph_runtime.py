"""Tests for graph_runtime module."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.graph_runtime import LadybugGraphRuntime, _esc

# --- Strategies ---

text_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-./:@?&=%+",
    min_size=0,
    max_size=50,
)

unsafe_text_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz\n\r\x00\"';\\",
    min_size=0,
    max_size=30,
)


# --- Basic tests ---


def test_sanitize_attach_dsn_rejects_sql_control_tokens():
    with pytest.raises(ValueError, match="unsafe"):
        LadybugGraphRuntime._sanitize_attach_dsn(
            "postgresql://u:p@localhost/db;DROP TABLE x"
        )
    with pytest.raises(ValueError, match="unsafe"):
        LadybugGraphRuntime._sanitize_attach_dsn(
            "postgresql://u:p@localhost/db -- comment"
        )


def test_sanitize_attach_dsn_allows_postgres_and_escapes_quotes():
    dsn = "postgresql://user:pa'ss@localhost:5432/fastcode"
    safe = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
    assert safe == "postgresql://user:pa''ss@localhost:5432/fastcode"


# --- Properties ---


class TestEscFunction:
    @given(val=text_st)
    @settings(max_examples=30)
    def test_esc_returns_string_property(self, val: Any):
        """HAPPY: _esc always returns a string."""
        result = _esc(val)
        assert isinstance(result, str)

    @given(val=text_st)
    @settings(max_examples=30)
    def test_esc_wraps_in_quotes_property(self, val: Any):
        """HAPPY: _esc wraps result in double quotes."""
        result = _esc(val)
        assert result.startswith('"')
        assert result.endswith('"')

    def test_esc_none_property(self):
        """HAPPY: _esc(None) returns 'NULL'."""
        assert _esc(None) == "NULL"

    def test_esc_string_property(self):
        """HAPPY: _esc('hello') returns quoted string."""
        assert _esc("hello") == '"hello"'

    def test_esc_empty_string_property(self):
        """HAPPY: _esc('') returns quoted empty."""
        assert _esc("") == '""'

    @pytest.mark.edge
    def test_esc_strips_null_bytes_property(self):
        """EDGE: null bytes stripped from output."""
        assert "\x00" not in _esc("hello\x00world")

    @pytest.mark.edge
    def test_esc_newlines_replaced_property(self):
        """EDGE: newlines replaced with spaces."""
        result = _esc("line1\nline2\rline3")
        assert "\n" not in result
        assert "\r" not in result

    @pytest.mark.edge
    def test_esc_escapes_backslash_property(self):
        """EDGE: backslashes escaped."""
        result = _esc("path\\to\\file")
        assert "\\\\" in result

    @pytest.mark.edge
    def test_esc_escapes_double_quote_property(self):
        """EDGE: double quotes escaped."""
        result = _esc('say "hello"')
        assert '\\"' in result

    @pytest.mark.edge
    def test_esc_escapes_single_quote_property(self):
        """EDGE: single quotes escaped."""
        result = _esc("it's")
        assert "\\'" in result

    @given(val=unsafe_text_st)
    @settings(max_examples=30)
    @pytest.mark.edge
    def test_esc_never_contains_unescaped_special_property(self, val: Any):
        """EDGE: no unescaped special chars in output."""
        result = _esc(val)
        # Strip outer quotes
        inner = result[1:-1]
        # No bare newlines, null bytes
        assert "\n" not in inner
        assert "\r" not in inner
        assert "\x00" not in inner

    @pytest.mark.edge
    def test_esc_integer_property(self):
        """EDGE: integer input converted to string."""
        result = _esc(42)
        assert result == '"42"'

    @pytest.mark.edge
    def test_esc_float_property(self):
        """EDGE: float input converted to string."""
        result = _esc(3.14)
        assert result == '"3.14"'

    @pytest.mark.edge
    def test_esc_boolean_property(self):
        """EDGE: boolean converted to string."""
        assert _esc(True) == '"True"'
        assert _esc(False) == '"False"'


class TestSanitizeAttachDsn:
    def test_valid_postgres_dsn_property(self):
        """HAPPY: valid postgres DSN passes sanitization."""
        dsn = "postgresql://user:pass@host:5432/db"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert result == dsn

    def test_valid_postgres_scheme_property(self):
        """HAPPY: 'postgres' scheme also accepted."""
        dsn = "postgres://user:pass@host:5432/db"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert result == dsn

    @pytest.mark.negative
    def test_empty_dsn_raises_property(self):
        """EDGE: empty DSN raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            LadybugGraphRuntime._sanitize_attach_dsn("")

    @pytest.mark.negative
    def test_whitespace_only_dsn_raises_property(self):
        """EDGE: whitespace-only DSN raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            LadybugGraphRuntime._sanitize_attach_dsn("   ")

    @pytest.mark.negative
    def test_semicolon_rejected_property(self):
        """EDGE: semicolons rejected (SQL injection)."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("postgres://host/db; DROP TABLE")

    @pytest.mark.negative
    def test_sql_comment_rejected_property(self):
        """EDGE: SQL comments rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host-- comment")

    @pytest.mark.negative
    def test_block_comment_rejected_property(self):
        """EDGE: block comments rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host/* comment */")

    @pytest.mark.negative
    def test_null_byte_rejected_property(self):
        """EDGE: null bytes rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host\x00db")

    @pytest.mark.negative
    def test_newline_rejected_property(self):
        """EDGE: newlines rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host\ndb")

    @pytest.mark.negative
    def test_unsupported_scheme_rejected_property(self):
        """EDGE: non-postgres scheme rejected."""
        with pytest.raises(ValueError, match="unsupported"):
            LadybugGraphRuntime._sanitize_attach_dsn("mysql://host/db")

    @pytest.mark.negative
    def test_http_scheme_rejected_property(self):
        """EDGE: http scheme rejected."""
        with pytest.raises(ValueError, match="unsupported"):
            LadybugGraphRuntime._sanitize_attach_dsn("http://host/db")

    @pytest.mark.edge
    def test_non_url_without_scheme_passes_property(self):
        """EDGE: non-URL DSN without scheme validated by regex."""
        dsn = "host.example.com:5432/mydb"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert result == dsn

    @pytest.mark.edge
    def test_single_quotes_escaped_property(self):
        """EDGE: single quotes in DSN are escaped."""
        dsn = "postgres://user:p'ass@host/db"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert "''" in result

    def test_disabled_by_default_property(self):
        """HAPPY: LadybugGraphRuntime disabled when not configured."""
        rt = LadybugGraphRuntime({"graph": {}})
        assert rt.enabled is False

    def test_disabled_no_config_property(self):
        """HAPPY: LadybugGraphRuntime disabled with empty config."""
        rt = LadybugGraphRuntime({})
        assert rt.enabled is False

    @pytest.mark.edge
    def test_sync_disabled_returns_false_property(self):
        """EDGE: sync_docs returns False when disabled."""
        rt = LadybugGraphRuntime({})
        assert rt.sync_docs(chunks=[], mentions=[]) is False

    @pytest.mark.edge
    def test_query_disabled_returns_empty_property(self):
        """EDGE: query_docs returns empty list when disabled."""
        rt = LadybugGraphRuntime({})
        assert rt.query_docs(snapshot_id="snap:1") == []

    @pytest.mark.edge
    def test_close_disabled_no_error_property(self):
        """EDGE: close() on disabled runtime doesn't crash."""
        rt = LadybugGraphRuntime({})
        rt.close()  # should not raise
