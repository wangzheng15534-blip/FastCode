"""Schemathesis API fuzz tests — auto-generated from FastAPI schema.

Run: uv run pytest fastcode/tests/test_schemathesis_api.py -v -p schemathesis
Do NOT run with default pytest config (-p no:schemathesis is in addopts).
"""
from __future__ import annotations

import schemathesis
from fastapi.testclient import TestClient

from fastcode import api

schema = schemathesis.openapi.from_asgi("/openapi.json", api.app)


@schema.parametrize()
def test_api_fuzz(case: schemathesis.Case) -> None:
    """Fuzz all API endpoints with auto-generated inputs."""
    client = TestClient(api.app)
    response = case.call(client=client)
    case.validate_response(response)
