# Open Core Gaps — May 29, 2026

Temporary tracker. Verified against source, not just docs.
Delete or merge into PERFORMANCE_TODOS / IMPLEMENTATION_TODOS when resolved.

## Performance / Architecture

### 1. NetworkX compatibility surfaces still materialize

Review: real compatibility surface, not a primary-path blocker.
Implemented here: MCP graph-tool snapshot fallback now routes through compact
`IRGraphView` traversal instead of composing NetworkX graphs. `IRGraphView.copy()`,
`IRGraphView.to_undirected()`, `build_combined_graph()`, and legacy
`graph/build.py` remain explicit compatibility/materialization boundaries.

- `ir/graph.py:338-342`: `copy()` and `to_undirected()` call `to_networkx()`
- `graph/build.py`: entire module is NetworkX-backed (call/dependency/inheritance graphs)
- MCP graph tools: fallback to NetworkX when compact handles unavailable

Impact: any caller of `copy()`/`to_undirected()` pays full graph materialization.

### 2. BM25 full-rebuild fallback path

Review: real gap on filtered/multi-repo reload fallbacks.
Implemented here: repository-specific reload and multi-repo cache loading now
adapt legacy BM25 payloads into the shard-runtime scorer instead of rebuilding a
full `BM25Okapi` object. Single-artifact legacy `load_bm25()` remains an
explicit compatibility path.

- `retriever.py:2390`: `BM25Okapi(all_bm25_corpus)` when shard-native loading fails
- `_load_bm25_payload()` materializes full corpus/element lists from shards (`retriever.py:3171-3220`)

Impact: shard-native is primary path; fallback is rare but unbounded.

### 3. Temporary artifact builders lack shard-handle passthrough

Review: mostly false positive for the current primary path. Temporary artifacts
are built from changed elements only, and publish APIs receive previous artifact
keys plus reusable path sets and hardlink/copy unchanged shards. The deeper
P0.2 wish for builders to carry handles directly remains a performance TODO, not
a correctness gap in the active delta publication path.

- `service.py:4593-4650`: `temp_store`, `temp_graph`, `temp_retriever` built from changed elements
  only, but builders don't accept previous shard handles for fallback scenarios
- One remaining unchecked TODO in PERFORMANCE_TODOS P0.2

Impact: fallback paths may still need wider reconstruction than necessary.

### 4. safe_jsonable() on hot IR payload save path

Review: real gap.
Implemented here: `store/snapshots/ir_payloads.py` no longer imports or calls
generic `safe_jsonable()`; it uses explicit bounded JSON normalization for IR
metadata, vector lists, and opaque values.

- `store/snapshots/ir_payloads.py:163,174`: `_json_mapping_payload()` and `_json_list_payload()`
  use `safe_jsonable()` for every code unit, support, relation, and embedding metadata save
- Not covered by materialization boundary guard

Impact: recursive normalization on every snapshot save for every IR object.

### 5. File inventory not yet canonical single-pass

Review: partially real. The pipeline already has a typed single inventory path,
with compatibility fallbacks for older tests/integrations. Implemented here:
inventory metrics now include `scanned_bytes` and `hashed_bytes`.

- Multiple `scan_files()` calls across pipeline stages (`service.py:354,375,3176`)
- Missing `scanned_bytes` and `hashed_bytes` metrics

Impact: redundant file stats/hashes on incremental runs.

## Store / Boundary

### 6. Compatibility dict-return APIs still exist

Review: intentional compatibility surface, not an active hot-path gap by itself.
Typed record APIs exist alongside these methods; callers on active paths should
prefer typed records.

- SnapshotStore: `get_scip_artifact_ref()` returns `dict[str, Any]` (not typed record)
- `load_snapshot_metadata()` returns `dict[str, Any]`
- VectorStore, PG retrieval, cache: legacy dict/tuple-return methods alongside typed record APIs

Impact: callers may bypass typed records accidentally.

## Release Readiness

### 7. Supported Python matrix not automated

Review: real release-readiness gap.
Implemented here: CI test job now runs on Python 3.11, 3.12, and 3.13.

- Gate only runs on active host interpreter (Python 3.13)
- Declared `>=3.11` but no multi-version CI evidence

### 8. External SCIP/tool integration matrix unverified

Review: real release-readiness gap.
Implemented here: availability-gated stable SCIP tool command smokes were added
under `fastcode/tests/scip/test_scip_tool_smoke.py`. Full per-language indexing
evidence is still required for stable release notes.

- Resolver contracts are unit-tested, but real command smoke coverage missing
- Zig, Fortran, Julia remain experimental without CI validation

### 9. Release gate automation incomplete

Review: partially real.
Implemented here: CI now exposes a manual `release-gate` workflow job for
`scripts/release_gate.py`, including an input for heavy extras. Threshold policy
and captured benchmark outputs remain release-management work.

- Backend and external-tool gates documented but not automated
- No release-threshold policy or captured expected outputs
- Artifact-family compatibility tests missing

### 10. Production runbooks missing

Review: real documentation gap.
Implemented here: `docs/deployment-guide.md` now includes operator runbooks for
backup/restore, migration/rollback, cache invalidation, lock recovery, and
failed upload/index remediation.

- No backup/restore, migration/rollback, cache invalidation, or lock recovery runbooks
- No operator playbook for failed upload/index remediation
