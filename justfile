set shell := ["bash", "-c"]

# FastCode quality gates.
default:
    @just --list

help:
    @just --list

# Tier 1: quick feedback.
qa: fmt-check check check-deps
    @echo "Tier 1 QA passed"

# Tier 2: lint + architecture.
qa-lint: qa lint arch-check type-check
    @echo "Tier 2 QA passed"

# Tier 3: full repository verification.
qa-full: qa-lint test
    @echo "Tier 3 QA passed"

fmt:
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

type-check:
    uv run pyright

# Dependency direction + hexagonal role enforcement.
check:
    python3 scripts/fcis_project.py check --root .

check-deps:
    uv run lint-imports --config pyproject.toml

# Full architecture policy suite.
arch-check:
    uv run pytest -q fastcode/tests/architecture

test:
    uv run pytest -q
