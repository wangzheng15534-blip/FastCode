"""TerminusDB HTTP publication sender."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit


class TerminusHttpSender:
    """Concrete HTTP POST sender for TerminusDB lineage payloads."""

    def __init__(
        self,
        *,
        endpoint: str | None,
        api_key: str | None,
        timeout: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)

    def post(
        self,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> None:
        """Execute the HTTP POST to TerminusDB."""
        if not self.endpoint:
            msg = "Terminus endpoint is not configured"
            raise RuntimeError(msg)
        endpoint = self._validated_endpoint()

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"X-Idempotency-Key": idempotency_key} if idempotency_key else {}),
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status >= 300:
                    msg = f"Terminus publish failed: HTTP {response.status}"
                    raise RuntimeError(msg)
                self.logger.info("Published snapshot lineage to Terminus")
        except urllib.error.URLError as e:
            msg = f"Terminus publish error: {e}"
            raise RuntimeError(msg) from e

    def _validated_endpoint(self) -> str:
        endpoint = self.endpoint or ""
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            msg = "Terminus endpoint must be an HTTP(S) URL"
            raise RuntimeError(msg)
        return endpoint
