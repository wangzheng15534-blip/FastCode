import pytest

from fastcode.graph_runtime import LadybugGraphRuntime


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
