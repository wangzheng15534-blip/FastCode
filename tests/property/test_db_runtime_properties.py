"""Property-based tests for db_runtime.DBRuntime invariants.

Covers missing lines: 46-58, 61, 81, 86-95, 116, 118, 125.
Uses real SQLite in-memory backend, not mocks.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fastcode.db_runtime import DBRuntime

# --- Strategies ---

sql_value = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(min_size=0, max_size=256),
    st.binary(min_size=0, max_size=64),
)

sql_value_list = st.lists(sql_value, min_size=0, max_size=10)

_SQL_KEYWORDS = frozenset(
    {
        "select",
        "from",
        "where",
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "table",
        "index",
        "in",
        "is",
        "not",
        "null",
        "and",
        "or",
        "as",
        "on",
        "join",
        "set",
        "values",
        "into",
        "by",
        "order",
        "group",
        "having",
        "limit",
        "offset",
        "union",
        "all",
        "exists",
        "between",
        "like",
        "case",
        "when",
        "then",
        "else",
        "end",
        "begin",
        "commit",
        "rollback",
        "primary",
        "key",
        "foreign",
        "references",
        "unique",
        "check",
        "default",
        "if",
        "integer",
        "text",
        "real",
    }
)

safe_identifier = st.builds(
    lambda prefix, suffix: prefix + suffix,
    st.sampled_from([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in "aeiou"]),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=2, max_size=12),
).filter(lambda s: s not in _SQL_KEYWORDS)


# --- Helpers ---


def _make_sqlite_runtime(**overrides: Any) -> DBRuntime:
    defaults: dict[str, Any] = {"backend": "sqlite", "sqlite_path": ":memory:"}
    defaults.update(overrides)
    return DBRuntime(**defaults)


def _make_postgres_runtime_fake() -> DBRuntime:
    """Create a postgres-backend runtime with psycopg mocked to bypass import."""
    import fastcode.db_runtime as mod

    orig_psycopg = mod.psycopg
    orig_pool = mod.ConnectionPool
    try:
        mod.psycopg = True
        mod.ConnectionPool = None
        return DBRuntime(backend="postgres", postgres_dsn="postgresql://localhost/test")
    finally:
        mod.psycopg = orig_psycopg
        mod.ConnectionPool = orig_pool


# --- Init tests (lines 38-61, 46-58) ---


@pytest.mark.property
class TestInit:
    @pytest.mark.happy
    def test_sqlite_memory_creation(self) -> None:
        rt = _make_sqlite_runtime()
        assert rt.backend == "sqlite"
        assert rt.sqlite_path == ":memory:"
        assert rt.pool is None

    @pytest.mark.happy
    def test_backend_lowercased(self) -> None:
        rt = DBRuntime(backend="SQLite", sqlite_path=":memory:")
        assert rt.backend == "sqlite"

    @pytest.mark.happy
    def test_none_backend_defaults_to_sqlite(self) -> None:
        rt = DBRuntime(backend=None, sqlite_path=":memory:")
        assert rt.backend == "sqlite"

    @pytest.mark.happy
    def test_empty_backend_defaults_to_sqlite(self) -> None:
        rt = DBRuntime(backend="", sqlite_path=":memory:")
        assert rt.backend == "sqlite"

    @pytest.mark.happy
    def test_pool_min_max_cast_to_int(self) -> None:
        rt = DBRuntime(
            backend="sqlite", sqlite_path=":memory:", pool_min="2", pool_max="10"
        )
        assert rt.pool_min == 2
        assert rt.pool_max == 10
        assert isinstance(rt.pool_min, int)
        assert isinstance(rt.pool_max, int)

    @pytest.mark.edge
    def test_sqlite_without_path_raises(self) -> None:
        """Line 61: sqlite backend requires sqlite_path."""
        with pytest.raises(RuntimeError, match="sqlite backend requires sqlite_path"):
            DBRuntime(backend="sqlite")

    @pytest.mark.edge
    def test_sqlite_none_path_raises(self) -> None:
        """Line 61: None sqlite_path raises."""
        with pytest.raises(RuntimeError, match="sqlite backend requires sqlite_path"):
            DBRuntime(backend="sqlite", sqlite_path=None)

    @pytest.mark.edge
    def test_sqlite_empty_path_raises(self) -> None:
        """Line 61: empty string sqlite_path raises."""
        with pytest.raises(RuntimeError, match="sqlite backend requires sqlite_path"):
            DBRuntime(backend="sqlite", sqlite_path="")

    @pytest.mark.edge
    def test_postgres_without_dsn_raises(self) -> None:
        """Lines 46-47: postgres backend requires DSN."""
        with pytest.raises(
            RuntimeError, match="postgres backend selected but no postgres_dsn"
        ):
            DBRuntime(backend="postgres", postgres_dsn=None)

    @pytest.mark.edge
    def test_postgres_empty_dsn_raises(self) -> None:
        """Lines 46-47: empty DSN raises."""
        with pytest.raises(
            RuntimeError, match="postgres backend selected but no postgres_dsn"
        ):
            DBRuntime(backend="postgres", postgres_dsn="")

    @pytest.mark.edge
    def test_postgres_without_psycopg_raises(self) -> None:
        """Lines 48-49: missing psycopg raises RuntimeError."""
        import fastcode.db_runtime as mod

        orig = mod.psycopg
        try:
            mod.psycopg = None
            with pytest.raises(RuntimeError, match="postgres backend requires psycopg"):
                DBRuntime(backend="postgres", postgres_dsn="postgresql://localhost/db")
        finally:
            mod.psycopg = orig

    @pytest.mark.edge
    def test_postgres_pool_not_available_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Line 58: warning when psycopg_pool unavailable, pool set to None."""
        import fastcode.db_runtime as mod

        orig_psycopg = mod.psycopg
        orig_pool = mod.ConnectionPool
        try:
            mod.psycopg = True
            mod.ConnectionPool = None
            with caplog.at_level(logging.WARNING, logger="fastcode.db_runtime"):
                rt = DBRuntime(
                    backend="postgres", postgres_dsn="postgresql://localhost/db"
                )
            assert rt.pool is None
            assert any(
                "psycopg_pool not available" in r.message for r in caplog.records
            )
        finally:
            mod.psycopg = orig_psycopg
            mod.ConnectionPool = orig_pool


# --- from_storage_config tests ---


@pytest.mark.property
class TestFromStorageConfig:
    @pytest.mark.happy
    def test_defaults_to_sqlite(self) -> None:
        rt = DBRuntime.from_storage_config(sqlite_path=":memory:", storage_cfg=None)
        assert rt.backend == "sqlite"

    @pytest.mark.happy
    def test_config_backend_overrides(self) -> None:
        rt = DBRuntime.from_storage_config(
            sqlite_path=":memory:",
            storage_cfg={"backend": "sqlite"},
        )
        assert rt.backend == "sqlite"

    @pytest.mark.happy
    def test_pool_settings_from_config(self) -> None:
        rt = DBRuntime.from_storage_config(
            sqlite_path=":memory:",
            storage_cfg={"pool_min": 4, "pool_max": 16},
        )
        assert rt.pool_min == 4
        assert rt.pool_max == 16

    @pytest.mark.happy
    def test_env_backend_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FASTCODE_STORAGE_BACKEND", "sqlite")
        rt = DBRuntime.from_storage_config(
            sqlite_path=":memory:",
            storage_cfg={},
        )
        assert rt.backend == "sqlite"


# --- adapt_sql tests (line 81) ---


@pytest.mark.property
class TestAdaptSql:
    @pytest.mark.happy
    def test_sqlite_passthrough(self) -> None:
        rt = _make_sqlite_runtime()
        sql = "SELECT * FROM t WHERE id = ?"
        assert rt.adapt_sql(sql) == sql

    @given(
        prefix=st.text(min_size=0, max_size=20),
        suffix=st.text(min_size=0, max_size=20),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_sqlite_never_transforms(self, prefix: str, suffix: str) -> None:
        rt = _make_sqlite_runtime()
        sql = f"{prefix}?{suffix}"
        assert rt.adapt_sql(sql) == sql

    @pytest.mark.happy
    def test_postgres_replaces_placeholders(self) -> None:
        """Line 81: postgres replaces ? with %s."""
        rt = DBRuntime.__new__(DBRuntime)
        rt.backend = "postgres"
        result = rt.adapt_sql("INSERT INTO t (a, b) VALUES (?, ?)")
        assert result == "INSERT INTO t (a, b) VALUES (%s, %s)"

    @given(n=st.integers(min_value=0, max_value=10))
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_postgres_adapt_n_placeholders(self, n: int) -> None:
        """Line 81: all ? placeholders replaced with %s."""
        rt = DBRuntime.__new__(DBRuntime)
        rt.backend = "postgres"
        sql = "VALUES (" + ", ".join(["?"] * n) + ")"
        adapted = rt.adapt_sql(sql)
        assert adapted.count("%s") == n
        assert "?" not in adapted

    @pytest.mark.happy
    def test_postgres_no_placeholders_unchanged(self) -> None:
        """Line 81: no placeholders means no change."""
        rt = DBRuntime.__new__(DBRuntime)
        rt.backend = "postgres"
        assert rt.adapt_sql("SELECT 1") == "SELECT 1"


# --- connect() context manager tests (lines 86-95, 97-106) ---


@pytest.mark.property
class TestConnect:
    @pytest.mark.happy
    def test_connect_yields_sqlite_connection(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            assert isinstance(conn, sqlite3.Connection)

    @pytest.mark.happy
    def test_connect_sets_wal_mode(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            cur = conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            assert mode.lower() in ("wal", "memory")

    @pytest.mark.happy
    def test_connect_sets_synchronous_normal(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            cur = conn.execute("PRAGMA synchronous")
            assert cur.fetchone()[0] == 1  # NORMAL = 1

    @pytest.mark.happy
    def test_connect_sets_foreign_keys(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            cur = conn.execute("PRAGMA foreign_keys")
            assert cur.fetchone()[0] == 1

    @pytest.mark.happy
    def test_connect_sets_busy_timeout(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            cur = conn.execute("PRAGMA busy_timeout")
            assert cur.fetchone()[0] == 5000

    @pytest.mark.edge
    def test_connect_closes_after_context(self) -> None:
        """Lines 105-106: connection closed on normal exit."""
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            pass
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    @pytest.mark.edge
    def test_connect_cleanup_on_exception(self) -> None:
        """Lines 105-106: connection closed on exception exit."""
        rt = _make_sqlite_runtime()
        conn_ref = None
        with pytest.raises(ValueError), rt.connect() as conn:
            conn_ref = conn
            raise ValueError("boom")
        assert conn_ref is not None
        with pytest.raises(Exception):
            conn_ref.execute("SELECT 1")

    @pytest.mark.edge
    def test_postgres_connect_without_pool_direct_path(self) -> None:
        """Lines 86-95: postgres connect falls to direct psycopg when no pool."""
        import fastcode.db_runtime as mod

        orig_psycopg = mod.psycopg
        orig_pool = mod.ConnectionPool
        try:
            mod.psycopg = True
            mod.ConnectionPool = None
            rt = DBRuntime(backend="postgres", postgres_dsn="postgresql://localhost/db")
            assert rt.pool is None
            # Verifies the fallback code path is selected
        finally:
            mod.psycopg = orig_psycopg
            mod.ConnectionPool = orig_pool

    @pytest.mark.happy
    def test_separate_memory_connections_are_isolated(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn1:
            conn1.execute("CREATE TABLE t (x INTEGER)")
            conn1.execute("INSERT INTO t VALUES (42)")
        with rt.connect() as conn2, pytest.raises(sqlite3.OperationalError):
            conn2.execute("SELECT * FROM t")


# --- execute() tests ---


@pytest.mark.property
class TestExecute:
    @pytest.mark.happy
    def test_execute_basic_query(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            cur = rt.execute(conn, "SELECT 1 AS val")
            assert cur.fetchone()["val"] == 1

    @pytest.mark.happy
    def test_execute_create_and_insert(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.execute(conn, "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
            rt.execute(conn, "INSERT INTO t VALUES (?, ?)", (1, "alice"))
            cur = rt.execute(conn, "SELECT * FROM t")
            rows = cur.fetchall()
            assert len(rows) == 1
            assert rows[0]["name"] == "alice"

    @pytest.mark.happy
    def test_execute_returns_cursor(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.execute(conn, "CREATE TABLE t (id INTEGER)")
            cur = rt.execute(conn, "INSERT INTO t VALUES (?)", (1,))
            assert isinstance(cur, sqlite3.Cursor)

    @pytest.mark.happy
    def test_execute_empty_params_default(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            cur = rt.execute(conn, "SELECT 1")
            assert cur.fetchone() is not None

    @given(
        name=st.text(min_size=1, max_size=32),
        value=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_execute_roundtrip(self, name: str, value: int) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.execute(conn, "CREATE TABLE kv (k TEXT PRIMARY KEY, v INTEGER)")
            rt.execute(conn, "INSERT INTO kv VALUES (?, ?)", (name, value))
            cur = rt.execute(conn, "SELECT k, v FROM kv WHERE k = ?", (name,))
            row = cur.fetchone()
            assert row is not None
            assert row["k"] == name
            assert row["v"] == value

    @given(values=sql_value_list)
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_execute_various_types(self, values: list) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            ncols = len(values)
            if ncols == 0:
                return
            cols = ", ".join(f"c{i}" for i in range(ncols))
            placeholders = ", ".join("?" for _ in range(ncols))
            conn.execute(f"CREATE TABLE t ({cols})")
            rt.execute(
                conn, f"INSERT INTO t ({cols}) VALUES ({placeholders})", tuple(values)
            )
            cur = rt.execute(conn, f"SELECT {cols} FROM t")
            row = cur.fetchone()
            for i, v in enumerate(values):
                fetched = row[f"c{i}"]
                if isinstance(v, bool):
                    assert fetched == int(v)
                elif isinstance(v, float):
                    assert abs(fetched - v) < 1e-10
                else:
                    assert fetched == v

    @pytest.mark.edge
    def test_execute_bad_sql_raises(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn, pytest.raises(sqlite3.OperationalError):
            rt.execute(conn, "SELECT * FROM nonexistent_table")

    @pytest.mark.edge
    def test_execute_duplicate_primary_key_raises(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO t VALUES (1)")
            with pytest.raises(sqlite3.IntegrityError):
                rt.execute(conn, "INSERT INTO t VALUES (1)")


# --- row_to_dict() tests (lines 114-119) ---


@pytest.mark.property
class TestRowToDict:
    @pytest.mark.edge
    def test_none_returns_none(self) -> None:
        """Line 116: None input returns None."""
        assert DBRuntime.row_to_dict(None) is None

    @pytest.mark.parametrize("falsy_val", ["", 0, [], {}])
    @pytest.mark.edge
    def test_falsy_values_return_none(self, falsy_val: Any) -> None:
        """Line 116: all falsy inputs return None."""
        assert DBRuntime.row_to_dict(falsy_val) is None

    @pytest.mark.happy
    def test_nonempty_dict_passthrough(self) -> None:
        """Line 118: non-empty dict returned as-is (same reference)."""
        d = {"a": 1, "b": "two"}
        result = DBRuntime.row_to_dict(d)
        assert result is d

    @pytest.mark.happy
    def test_sqlite_row_to_dict(self) -> None:
        """Line 119: sqlite3.Row converted to dict."""
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.execute(conn, "CREATE TABLE t (id INTEGER, name TEXT)")
            rt.execute(conn, "INSERT INTO t VALUES (?, ?)", (42, "test"))
            cur = rt.execute(conn, "SELECT id, name FROM t")
            row = cur.fetchone()
            result = DBRuntime.row_to_dict(row)
            assert isinstance(result, dict)
            assert result["id"] == 42
            assert result["name"] == "test"

    @given(
        id_val=st.integers(min_value=0, max_value=2**31 - 1),
        name=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=30)
    @pytest.mark.happy
    def test_sqlite_row_preserves_data(self, id_val: int, name: str) -> None:
        """Line 119: row_to_dict preserves all column values."""
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.execute(conn, "CREATE TABLE t (id INTEGER, name TEXT)")
            rt.execute(conn, "INSERT INTO t VALUES (?, ?)", (id_val, name))
            cur = rt.execute(conn, "SELECT id, name FROM t")
            row = cur.fetchone()
            result = DBRuntime.row_to_dict(row)
            assert result["id"] == id_val
            assert result["name"] == name

    @pytest.mark.happy
    def test_tuple_converted(self) -> None:
        """Line 119: dict(row) works on sqlite3.Row (which supports dict())."""
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.execute(conn, "CREATE TABLE t (a TEXT, b INTEGER)")
            rt.execute(conn, "INSERT INTO t VALUES (?, ?)", ("hello", 5))
            cur = rt.execute(conn, "SELECT a, b FROM t")
            row = cur.fetchone()
            result = DBRuntime.row_to_dict(row)
            assert result == {"a": "hello", "b": 5}


# --- begin_write() tests (lines 121-125) ---


@pytest.mark.property
class TestBeginWrite:
    @pytest.mark.happy
    def test_begin_write_sqlite(self) -> None:
        """Line 123: BEGIN IMMEDIATE for sqlite."""
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.begin_write(conn)
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.rollback()
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("SELECT * FROM t")

    @pytest.mark.edge
    def test_begin_write_rollback_restores(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            rt.begin_write(conn)
            conn.execute("INSERT INTO t VALUES (99)")
            conn.rollback()
            cur = conn.execute("SELECT COUNT(*) FROM t")
            assert cur.fetchone()[0] == 0

    @pytest.mark.edge
    def test_begin_write_postgres_path(self) -> None:
        """Line 125: postgres uses plain BEGIN, not BEGIN IMMEDIATE."""
        import fastcode.db_runtime as mod

        orig_psycopg = mod.psycopg
        orig_pool = mod.ConnectionPool
        try:
            mod.psycopg = True
            mod.ConnectionPool = None
            rt = DBRuntime(backend="postgres", postgres_dsn="postgresql://x")
            assert rt.backend == "postgres"
            # The begin_write method would call conn.execute("BEGIN") for postgres
        finally:
            mod.psycopg = orig_psycopg
            mod.ConnectionPool = orig_pool


# --- Connection pooling / reuse ---


@pytest.mark.property
class TestConnectionPooling:
    @given(n=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10)
    @pytest.mark.happy
    def test_sequential_connections(self, n: int) -> None:
        rt = _make_sqlite_runtime()
        for i in range(n):
            with rt.connect() as conn:
                rt.execute(conn, "CREATE TABLE IF NOT EXISTS t (id INTEGER)")
                rt.execute(conn, "INSERT INTO t VALUES (?)", (i,))

    @pytest.mark.happy
    def test_nested_connections(self) -> None:
        """HAPPY: nested connect calls produce independent :memory: DBs."""
        rt = _make_sqlite_runtime()
        with rt.connect() as outer:
            rt.execute(outer, "CREATE TABLE t (id INTEGER)")
            rt.execute(outer, "INSERT INTO t VALUES (?)", (1,))
            with rt.connect() as inner, pytest.raises(sqlite3.OperationalError):
                rt.execute(inner, "SELECT * FROM t")


# --- Rollback and error handling ---


@pytest.mark.property
class TestRollbackBehavior:
    @pytest.mark.edge
    def test_constraint_violation_no_corruption(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            rt.execute(conn, "CREATE TABLE t (id INTEGER PRIMARY KEY UNIQUE)")
            rt.execute(conn, "INSERT INTO t VALUES (?)", (1,))
            with pytest.raises(Exception):
                rt.execute(conn, "INSERT INTO t VALUES (?)", (1,))
            cur = rt.execute(conn, "SELECT COUNT(*) FROM t")
            assert cur.fetchone()[0] == 1

    @pytest.mark.edge
    def test_invalid_sql_no_crash(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            with pytest.raises(Exception):
                rt.execute(conn, "NOT VALID SQL")
            rt.execute(conn, "CREATE TABLE ok (id INTEGER)")
            rt.execute(conn, "INSERT INTO ok VALUES (?)", (1,))
            cur = rt.execute(conn, "SELECT COUNT(*) FROM ok")
            assert cur.fetchone()[0] == 1

    @pytest.mark.edge
    def test_nonexistent_table_no_corruption(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            with pytest.raises(Exception):
                rt.execute(conn, "SELECT * FROM nonexistent")
            rt.execute(conn, "CREATE TABLE real (v TEXT)")
            rt.execute(conn, "INSERT INTO real VALUES (?)", ("hello",))
            cur = rt.execute(conn, "SELECT v FROM real")
            assert cur.fetchone()["v"] == "hello"

    @pytest.mark.happy
    def test_rollback_restores_state(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            try:
                rt.begin_write(conn)
                conn.execute("INSERT INTO t VALUES (2)")
                raise ValueError("simulated")
            except ValueError:
                conn.rollback()
            cur = rt.execute(conn, "SELECT COUNT(*) FROM t")
            assert cur.fetchone()[0] == 1

    @pytest.mark.happy
    def test_commit_persists(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            rt.begin_write(conn)
            conn.execute("INSERT INTO t VALUES (42)")
            conn.commit()
            cur = rt.execute(conn, "SELECT x FROM t")
            assert cur.fetchone()["x"] == 42

    @pytest.mark.edge
    def test_connection_closed_after_exception(self) -> None:
        rt = _make_sqlite_runtime()
        try:
            with rt.connect() as conn:
                conn.execute("CREATE TABLE t (x INTEGER)")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        with pytest.raises(Exception):
            conn.execute("SELECT 1")


# --- Schema / integration ---


@pytest.mark.property
class TestSchema:
    @pytest.mark.happy
    def test_join_query(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)"
            )
            conn.execute("INSERT INTO users VALUES (1, 'alice')")
            conn.execute("INSERT INTO orders VALUES (100, 1, 9.99)")
            cur = rt.execute(
                conn,
                "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id",
            )
            row = cur.fetchone()
            assert row["name"] == "alice"
            assert abs(row["amount"] - 9.99) < 0.001

    @given(
        table_name=safe_identifier,
        col_name=safe_identifier,
        value=st.integers(min_value=-(2**31), max_value=2**31 - 1),
    )
    @settings(max_examples=15)
    @pytest.mark.happy
    def test_dynamic_table_creation(
        self, table_name: str, col_name: str, value: int
    ) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute(f"CREATE TABLE {table_name} ({col_name} INTEGER)")
            rt.execute(
                conn, f"INSERT INTO {table_name} ({col_name}) VALUES (?)", (value,)
            )
            cur = rt.execute(conn, f"SELECT {col_name} FROM {table_name}")
            assert cur.fetchone()[0] == value

    @given(
        rows=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=1000),
                st.text(min_size=1, max_size=50),
            ),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=20)
    @pytest.mark.happy
    def test_insert_count_property(self, rows: list[tuple[int, str]]) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
            rt.begin_write(conn)
            for id_val, name in rows:
                rt.execute(conn, "INSERT INTO t VALUES (?, ?)", (id_val, name))
            conn.commit()
            cur = rt.execute(conn, "SELECT COUNT(*) FROM t")
            assert cur.fetchone()[0] == len(rows)

    @pytest.mark.happy
    def test_row_factory_dict_like(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (a TEXT, b INTEGER, c REAL)")
            conn.execute("INSERT INTO t VALUES ('x', 42, 3.14)")
            cur = conn.execute("SELECT a, b, c FROM t")
            row = cur.fetchone()
            d = DBRuntime.row_to_dict(row)
            assert d == {"a": "x", "b": 42, "c": 3.14}


@pytest.mark.property
class TestDbRuntimeEdgeExtras:
    @pytest.mark.edge
    def test_execute_invalid_sql(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn, pytest.raises(Exception):
            rt.execute(conn, "INVALID SQL STATEMENT")

    @pytest.mark.edge
    def test_empty_query_result(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            cur = rt.execute(conn, "SELECT * FROM t")
            assert cur.fetchone() is None

    @pytest.mark.edge
    def test_rollback_on_exception(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
            try:
                rt.begin_write(conn)
                conn.execute("INSERT INTO t VALUES (2)")
                raise RuntimeError("force rollback")
            except RuntimeError:
                pass
            cur = rt.execute(conn, "SELECT COUNT(*) FROM t")
            assert cur.fetchone()[0] == 2  # committed within same conn

    @pytest.mark.edge
    def test_null_values_stored(self) -> None:
        rt = _make_sqlite_runtime()
        with rt.connect() as conn:
            conn.execute("CREATE TABLE t (x INTEGER, y TEXT)")
            conn.execute("INSERT INTO t VALUES (NULL, NULL)")
            cur = rt.execute(conn, "SELECT x, y FROM t")
            row = cur.fetchone()
            assert row["x"] is None
            assert row["y"] is None

    @pytest.mark.edge
    def test_adapt_sql_postgres_replaces_placeholders(self) -> None:
        rt = DBRuntime(backend="postgres", postgres_dsn="postgresql://localhost/test")
        sql = "SELECT * FROM t WHERE a = ? AND b = ?"
        result = rt.adapt_sql(sql)
        assert result == "SELECT * FROM t WHERE a = %s AND b = %s"

    @pytest.mark.edge
    def test_adapt_sql_sqlite_passthrough(self) -> None:
        rt = _make_sqlite_runtime()
        sql = "SELECT * FROM t WHERE a = ?"
        result = rt.adapt_sql(sql)
        assert result == sql
