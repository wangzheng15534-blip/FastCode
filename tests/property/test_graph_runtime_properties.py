"""Property-based tests for graph_runtime module (pure functions only)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.graph_runtime import LadybugGraphRuntime, _esc


# --- Strategies ---

text_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-./:@?&=%+",
    min_size=0, max_size=50,
)

unsafe_text_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz\n\r\x00\"';\\",
    min_size=0, max_size=30,
)


# --- Properties ---


@pytest.mark.property
class TestEscFunction:

    @given(val=text_st)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_esc_returns_string(self, val):
        """HAPPY: _esc always returns a string."""
        result = _esc(val)
        assert isinstance(result, str)

    @given(val=text_st)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_esc_wraps_in_quotes(self, val):
        """HAPPY: _esc wraps result in double quotes."""
        result = _esc(val)
        assert result.startswith('"')
        assert result.endswith('"')

    @pytest.mark.happy
    def test_esc_none(self):
        """HAPPY: _esc(None) returns 'NULL'."""
        assert _esc(None) == "NULL"

    @pytest.mark.happy
    def test_esc_string(self):
        """HAPPY: _esc('hello') returns quoted string."""
        assert _esc("hello") == '"hello"'

    @pytest.mark.happy
    def test_esc_empty_string(self):
        """HAPPY: _esc('') returns quoted empty."""
        assert _esc("") == '""'

    @pytest.mark.edge
    def test_esc_strips_null_bytes(self):
        """EDGE: null bytes stripped from output."""
        assert "\x00" not in _esc("hello\x00world")

    @pytest.mark.edge
    def test_esc_newlines_replaced(self):
        """EDGE: newlines replaced with spaces."""
        result = _esc("line1\nline2\rline3")
        assert "\n" not in result
        assert "\r" not in result

    @pytest.mark.edge
    def test_esc_escapes_backslash(self):
        """EDGE: backslashes escaped."""
        result = _esc("path\\to\\file")
        assert "\\\\" in result

    @pytest.mark.edge
    def test_esc_escapes_double_quote(self):
        """EDGE: double quotes escaped."""
        result = _esc('say "hello"')
        assert '\\"' in result

    @pytest.mark.edge
    def test_esc_escapes_single_quote(self):
        """EDGE: single quotes escaped."""
        result = _esc("it's")
        assert "\\'" in result

    @given(val=unsafe_text_st)
    @settings(max_examples=30)
    @pytest.mark.edge
    def test_esc_never_contains_unescaped_special(self, val):
        """EDGE: no unescaped special chars in output."""
        result = _esc(val)
        # Strip outer quotes
        inner = result[1:-1]
        # No bare newlines, null bytes
        assert "\n" not in inner
        assert "\r" not in inner
        assert "\x00" not in inner

    @pytest.mark.edge
    def test_esc_integer(self):
        """EDGE: integer input converted to string."""
        result = _esc(42)
        assert result == '"42"'

    @pytest.mark.edge
    def test_esc_float(self):
        """EDGE: float input converted to string."""
        result = _esc(3.14)
        assert result == '"3.14"'

    @pytest.mark.edge
    def test_esc_boolean(self):
        """EDGE: boolean converted to string."""
        assert _esc(True) == '"True"'
        assert _esc(False) == '"False"'


@pytest.mark.property
class TestSanitizeAttachDsn:

    @pytest.mark.happy
    def test_valid_postgres_dsn(self):
        """HAPPY: valid postgres DSN passes sanitization."""
        dsn = "postgresql://user:pass@host:5432/db"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert result == dsn

    @pytest.mark.happy
    def test_valid_postgres_scheme(self):
        """HAPPY: 'postgres' scheme also accepted."""
        dsn = "postgres://user:pass@host:5432/db"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert result == dsn

    @pytest.mark.edge
    def test_empty_dsn_raises(self):
        """EDGE: empty DSN raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            LadybugGraphRuntime._sanitize_attach_dsn("")

    @pytest.mark.edge
    def test_whitespace_only_dsn_raises(self):
        """EDGE: whitespace-only DSN raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            LadybugGraphRuntime._sanitize_attach_dsn("   ")

    @pytest.mark.edge
    def test_semicolon_rejected(self):
        """EDGE: semicolons rejected (SQL injection)."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("postgres://host/db; DROP TABLE")

    @pytest.mark.edge
    def test_sql_comment_rejected(self):
        """EDGE: SQL comments rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host-- comment")

    @pytest.mark.edge
    def test_block_comment_rejected(self):
        """EDGE: block comments rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host/* comment */")

    @pytest.mark.edge
    def test_null_byte_rejected(self):
        """EDGE: null bytes rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host\x00db")

    @pytest.mark.edge
    def test_newline_rejected(self):
        """EDGE: newlines rejected."""
        with pytest.raises(ValueError, match="unsafe"):
            LadybugGraphRuntime._sanitize_attach_dsn("host\ndb")

    @pytest.mark.edge
    def test_unsupported_scheme_rejected(self):
        """EDGE: non-postgres scheme rejected."""
        with pytest.raises(ValueError, match="unsupported"):
            LadybugGraphRuntime._sanitize_attach_dsn("mysql://host/db")

    @pytest.mark.edge
    def test_http_scheme_rejected(self):
        """EDGE: http scheme rejected."""
        with pytest.raises(ValueError, match="unsupported"):
            LadybugGraphRuntime._sanitize_attach_dsn("http://host/db")

    @pytest.mark.edge
    def test_non_url_without_scheme_passes(self):
        """EDGE: non-URL DSN without scheme validated by regex."""
        dsn = "host.example.com:5432/mydb"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert result == dsn

    @pytest.mark.edge
    def test_single_quotes_escaped(self):
        """EDGE: single quotes in DSN are escaped."""
        dsn = "postgres://user:p'ass@host/db"
        result = LadybugGraphRuntime._sanitize_attach_dsn(dsn)
        assert "''" in result

    @pytest.mark.happy
    def test_disabled_by_default(self):
        """HAPPY: LadybugGraphRuntime disabled when not configured."""
        rt = LadybugGraphRuntime({"graph": {}})
        assert rt.enabled is False

    @pytest.mark.happy
    def test_disabled_no_config(self):
        """HAPPY: LadybugGraphRuntime disabled with empty config."""
        rt = LadybugGraphRuntime({})
        assert rt.enabled is False

    @pytest.mark.edge
    def test_sync_disabled_returns_false(self):
        """EDGE: sync_docs returns False when disabled."""
        rt = LadybugGraphRuntime({})
        assert rt.sync_docs(chunks=[], mentions=[]) is False

    @pytest.mark.edge
    def test_query_disabled_returns_empty(self):
        """EDGE: query_docs returns empty list when disabled."""
        rt = LadybugGraphRuntime({})
        assert rt.query_docs(snapshot_id="snap:1") == []

    @pytest.mark.edge
    def test_close_disabled_no_error(self):
        """EDGE: close() on disabled runtime doesn't crash."""
        rt = LadybugGraphRuntime({})
        rt.close()  # should not raise
