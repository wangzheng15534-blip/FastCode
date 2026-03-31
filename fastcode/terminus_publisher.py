"""
TerminusDB lineage publisher.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict


class TerminusPublisher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        terminus_cfg = config.get("terminus", {})
        self.endpoint = terminus_cfg.get("endpoint")
        self.api_key = terminus_cfg.get("api_key")
        self.timeout = int(terminus_cfg.get("timeout_seconds", 15))

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def publish_snapshot_lineage(
        self,
        snapshot: Dict[str, Any],
        manifest: Dict[str, Any],
        git_meta: Dict[str, Any],
    ) -> None:
        if not self.endpoint:
            raise RuntimeError("Terminus endpoint is not configured")

        payload = {
            "snapshot": snapshot,
            "manifest": manifest,
            "git_meta": git_meta,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 300:
                    raise RuntimeError(f"Terminus publish failed: HTTP {resp.status}")
                self.logger.info("Published snapshot lineage to Terminus")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Terminus publish error: {e}") from e

