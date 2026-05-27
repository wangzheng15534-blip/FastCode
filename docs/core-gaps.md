# Core Gaps Status

This file tracks the short core-only list that was split out of the larger TODO docs.

## Status

The three execution-model gaps from the original short list are now closed on the active primary paths:

- Incremental structural indexing no longer carries the stale whole-element fallback helper in `fastcode/src/fastcode/indexing/pipeline.py`.
- Shard-native lexical load/reload now stays shard-native on the active paths in `fastcode/src/fastcode/query/retriever.py` and `fastcode/src/fastcode/main/fastcode.py`.
- Semantic patch application now preserves untouched collection containers through deferred copy-on-write sequences in `fastcode/src/fastcode/semantic/resolvers/patching.py`.

## Verification

- `uv run pytest fastcode/tests/retrieval/test_retriever.py fastcode/tests/main/test_main.py fastcode/tests/semantic/test_semantic_resolvers.py -q`
- `uv run ruff check fastcode/src/fastcode/query/retriever.py fastcode/src/fastcode/main/fastcode.py fastcode/src/fastcode/semantic/resolvers/patching.py fastcode/src/fastcode/indexing/pipeline.py fastcode/tests/retrieval/test_retriever.py fastcode/tests/main/test_main.py fastcode/tests/semantic/test_semantic_resolvers.py`
- `uv run pyright fastcode/src/fastcode/query/retriever.py fastcode/src/fastcode/main/fastcode.py fastcode/src/fastcode/semantic/resolvers/patching.py fastcode/src/fastcode/indexing/pipeline.py`

Pyright still reports existing warnings in `semantic/resolvers/patching.py`, but no errors on the touched files.

## Not In Scope

This does not close packaging, release-gate, dependency, operator-runbook, or benchmark-budget work from the longer trackers.
