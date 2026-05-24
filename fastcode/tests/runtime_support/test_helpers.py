from __future__ import annotations

import logging
from pathlib import Path

from fastcode.runtime_support.health import readiness_health
from fastcode.runtime_support.observability import configure_logging
from fastcode.runtime_support.retry import RetryPolicy, exponential_backoff_seconds


def test_readiness_health_degraded_when_loaded_but_not_indexed() -> None:
    result = readiness_health(repo_loaded=True, repo_indexed=False).to_dict()
    assert result["healthy"] is True
    assert result["status"] == "degraded"


def test_exponential_backoff_respects_policy_maximum() -> None:
    policy = RetryPolicy(minimum_seconds=1, base=2, maximum_seconds=5)
    assert exponential_backoff_seconds(10, policy=policy) == 5


def test_configure_logging_returns_named_logger(tmp_path: Path) -> None:
    logger = configure_logging(
        level=logging.INFO,
        log_file=str(tmp_path / "logs" / "runtime-support.log"),
        console=False,
        logger_name="fastcode.test.runtime_support",
    )
    assert logger.name == "fastcode.test.runtime_support"
