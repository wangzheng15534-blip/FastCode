# Test Mirror Layout Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `fastcode/tests/` to mirror `src/fastcode/` directory structure with one test file per source module, named `test_<module>.py`.

**Architecture:** Tests for subpackage modules (`adapters/`, `core/`, `infrastructure/`, `schemas/`, `utils/`) move into matching test subdirectories. Root-level modules keep tests flat in `tests/`. Multiple test files for the same source module are consolidated into one file. Test functions use technique postfixes (`_property`, `_edge`, `_negative`, `_double`, `_snapshot`) from commit 2a164f9 — these are preserved and verified during consolidation.

**Parametrize is a methodology, not a technique.** `@pytest.mark.parametrize` is table-driven test generation — it applies ACROSS technique categories. When consolidating `_parametrize` files, each parametrized test is reclassified into its actual technique (edge, negative, integration, or basic) based on what it tests. The `@pytest.mark.parametrize` decorator is always preserved. No `_parametrize` postfix on function names — instead, the technique postfix (`_edge`, `_negative`, etc.) is applied.

**Tech Stack:** Python 3.11, pytest, hypothesis, syrupy (snapshots)

---

## Current vs Target Layout

### Already Correct (no change needed)
- `tests/core/test_*.py` — 14 files mirroring `src/fastcode/core/`
- `tests/infrastructure/test_db.py`, `test_fs.py`, `test_llm.py` — mirroring `src/fastcode/infrastructure/`
- `tests/schemas/test_core_types.py`, `test_ir.py` — mirroring `src/fastcode/schemas/`
- `tests/utils/test_hashing.py`, `test_json.py`, `test_paths.py` — mirroring `src/fastcode/utils/`
- `tests/conftest.py` — stays at root
- `tests/bench_*.py` — 3 benchmark files stay at root

### Migration Map

Below is every flat test file mapped to its target location. Lines are approximate.

| Current File | Lines | Source Module | Target File |
|---|---|---|---|
| `test_ast_to_ir_parametrize.py` | 474 | `adapters/ast_to_ir.py` | `tests/adapters/test_ast_to_ir.py` |
| `test_ast_to_ir_properties.py` | 670 | `adapters/ast_to_ir.py` | merge ↑ |
| `test_scip_to_ir_parametrize.py` | 490 | `adapters/scip_to_ir.py` | `tests/adapters/test_scip_to_ir.py` |
| `test_scip_to_ir_properties.py` | 550 | `adapters/scip_to_ir.py` | merge ↑ |
| `test_scip_parsing_properties.py` | — | `adapters/scip_to_ir.py` | merge ↑ |
| `core/test_module_resolver_properties.py` | — | `module_resolver.py` | `tests/test_module_resolver.py` |
| `infrastructure/test_manifest_store_db_properties.py` | — | `manifest_store.py` | merge into `tests/test_manifest_store.py` |
| `test_utils_properties.py` | 481 | `utils/__init__.py` | `tests/utils/test_utils.py` |
| `test_db_runtime_properties.py` | — | `db_runtime.py` | `tests/test_db_runtime.py` |
| `test_global_index_builder_properties.py` | — | `global_index_builder.py` | `tests/test_global_index_builder.py` |
| `test_ir_graph_builder_properties.py` | — | `ir_graph_builder.py` | `tests/test_ir_graph_builder.py` |
| `test_ir_validators_properties.py` | 600 | `ir_validators.py` | `tests/test_ir_validators.py` |
| `test_manifest_store_properties.py` | — | `manifest_store.py` | `tests/test_manifest_store.py` |
| `test_path_utils_properties.py` | — | `path_utils.py` | `tests/test_path_utils.py` |
| `test_pg_retrieval_fallback_filters.py` | — | `pg_retrieval.py` | `tests/test_pg_retrieval.py` |
| `test_repo_selector_properties.py` | — | `repo_selector.py` | `tests/test_repo_selector.py` |
| `test_scip_loader_properties.py` | — | `scip_loader.py` | `tests/test_scip_loader.py` |
| `test_symbol_resolver_properties.py` | 272 | `symbol_resolver.py` | `tests/test_symbol_resolver.py` |
| `test_adaptive_fusion.py` | 97 | `retriever.py` | `tests/test_retriever.py` |
| `test_doc_channel_projection.py` | 168 | `retriever.py` | merge ↑ |
| `test_graph_api.py` | 150 | `api.py` | `tests/test_api.py` |
| `test_main_doc_pipeline.py` | — | `main.py` | `tests/test_main.py` |
| `test_session_prefix.py` | 378 | `main.py` | merge ↑ |
| `test_doc_ingester.py` | 72 | `doc_ingester.py` | `tests/test_doc_ingester.py` |
| `test_doc_ingester_properties.py` | 283 | `doc_ingester.py` | merge ↑ |
| `test_chonkie_chunking.py` | 302 | `doc_ingester.py` | merge ↑ |
| `test_graph_runtime.py` | 20 | `graph_runtime.py` | `tests/test_graph_runtime.py` |
| `test_graph_runtime_properties.py` | 228 | `graph_runtime.py` | merge ↑ |
| `test_ir_merge_properties.py` | 599 | `ir_merge.py` | `tests/test_ir_merge.py` |
| `test_ir_alignment_algorithm.py` | 177 | `ir_merge.py` | merge ↑ |
| `test_projection_models_properties.py` | 130 | `projection_models.py` | `tests/test_projection_models.py` |
| `test_projection_contract.py` | 220 | `projection_transform.py` | `tests/test_projection_transform.py` |
| `test_projection_properties.py` | 478 | `projection_transform.py` | merge ↑ |
| `test_projection_pipeline.py` | 169 | `projection_transform.py` | merge ↑ |
| `test_projection_supporting_docs.py` | 196 | `projection_transform.py` | merge ↑ |
| `test_projection_v2_schema.py` | 147 | `projection_transform.py` | merge ↑ |
| `test_redo_worker.py` | 102 | `redo_worker.py` | `tests/test_redo_worker.py` |
| `test_redo_worker_properties.py` | 286 | `redo_worker.py` | merge ↑ |
| `test_scip_indexers.py` | 213 | `scip_indexers.py` | `tests/test_scip_indexers.py` |
| `test_scip_indexers_properties.py` | 171 | `scip_indexers.py` | merge ↑ |
| `test_scip_binary.py` | 194 | `scip_loader.py` + `scip_pb2.py` | merge into `tests/test_scip_loader.py` |
| `test_scip_models.py` | 182 | `scip_models.py` | `tests/test_scip_models.py` |
| `test_scip_models_properties.py` | 465 | `scip_models.py` | merge ↑ |
| `test_semantic_ir_properties.py` | 1000 | `semantic_ir.py` | `tests/test_semantic_ir.py` |
| `test_ir_snapshot_contract.py` | 163 | `semantic_ir.py` | merge ↑ |
| `test_snapshot_store_properties.py` | 866 | `snapshot_store.py` | `tests/test_snapshot_store.py` |
| `test_snapshot_store_db_properties.py` | 792 | `snapshot_store.py` | merge ↑ |
| `test_snapshot_store_lifecycle_properties.py` | 516 | `snapshot_store.py` | merge ↑ |
| `test_snapshot_store_stateful.py` | 396 | `snapshot_store.py` | merge ↑ |
| `test_snapshot_symbol_index.py` | 32 | `snapshot_symbol_index.py` | `tests/test_snapshot_symbol_index.py` |
| `test_snapshot_symbol_index_properties.py` | 576 | `snapshot_symbol_index.py` | merge ↑ |
| `test_terminus_publisher_properties.py` | 444 | `terminus_publisher.py` | `tests/test_terminus_publisher.py` |
| `test_terminus_code_graph.py` | 303 | `terminus_publisher.py` | merge ↑ |
| `test_terminus_lineage_edges.py` | 48 | `terminus_publisher.py` | merge ↑ |
| `test_terminus_payload.py` | 40 | `terminus_publisher.py` | merge ↑ |
| `test_mcp_graph_tools.py` | 580 | `mcp_graph_tools.py` | `tests/test_mcp_graph_tools.py` |
| `test_mcp_directed_path.py` | 375 | `mcp_graph_tools.py` | merge ↑ |

### Keep As-Is (cross-module / integration / e2e)
- `test_outbox_pattern.py` — tests terminus_publisher + snapshot_store + redo_worker
- `test_snapshot_pipeline.py` — tests snapshot_store + manifest_store + index_run
- `test_incremental_update.py` — tests incremental_update (already correct name)
- `test_scip_resolution_bridge.py` — already correct name
- `test_e2e_indexing.py` — e2e test
- `test_e2e_semantic_pipeline.py` — e2e test

### Snapshot Files (rename to match new test file names)
- `tests/__snapshots__/test_ir_snapshot_contract.ambr` → `tests/__snapshots__/test_semantic_ir.ambr`
- `tests/__snapshots__/test_projection_contract.ambr` → `tests/__snapshots__/test_projection_transform.ambr`

---

## Consolidation Rules

When merging N test files into one:

1. **Imports:** Merge all import blocks, deduplicate. Keep `from __future__ import annotations` first.
2. **Module docstring:** Use a single docstring naming the source module being tested.
3. **pytestmark:** Merge module-level marks into one list (e.g., `pytestmark = [pytest.mark.hypothesis, pytest.mark.test_double]`).
4. **Helper classes/functions:** Keep all. Prefix with `_` if not already. Deduplicate identical ones.
5. **Test functions — technique postfixes:** Every non-basic test function MUST have a technique postfix (`_property`, `_edge`, `_negative`, `_double`, `_perf`, `_snapshot`). Verify this during consolidation. If a function lacks a postfix but tests a boundary/edge case, add `_edge`. If it tests invalid input/expected failure, add `_negative`. If it uses mocks/fakes, add `_double`.
6. **Parametrize reclassification:** Tests from `_parametrize` files are NOT kept as a "parametrize" section. Each parametrized test is classified by its actual technique:
   - Boundary values / off-by-one / clamping → `_edge` postfix
   - Invalid input / expected failures → `_negative` postfix
   - Multi-component interaction → `_integration` postfix or no postfix (basic)
   - Positive behavior across valid inputs → no postfix (basic)
   - **Always preserve** `@pytest.mark.parametrize` decorator on every parametrized test
   - If a parametrized test already has a technique mark (e.g., `@pytest.mark.edge`), add the matching postfix to the function name
7. **Order:** Group by test class (if any), then top-level test functions grouped by technique: basic → edge → negative → property → double → snapshot.
8. **Target file line limit:** If a consolidated file would exceed 1200 lines, split into `test_<module>.py` (main) + `test_<module>_extended.py` (overflow). This is only expected for `snapshot_store` (~2570 lines).
9. **`_properties` single files:** Renaming a `_properties` file to `test_<module>.py` is NOT a "simple rename" — it requires a verification pass over every test function to confirm the `_property` postfix is present. Add missing postfixes before committing.

---

## Tasks

### Task 1: Create `tests/adapters/` and migrate AST adapter tests

**Files:**
- Create: `tests/adapters/test_ast_to_ir.py`
- Delete: `tests/test_ast_to_ir_parametrize.py`, `tests/test_ast_to_ir_properties.py`

- [ ] **Step 1: Create adapters directory**

```bash
mkdir -p tests/adapters
```

- [ ] **Step 2: Consolidate `test_ast_to_ir_parametrize.py` + `test_ast_to_ir_properties.py` into `tests/adapters/test_ast_to_ir.py`**

Merge the two files with reclassification of parametrized tests:
- Combined imports: `from fastcode.adapters.ast_to_ir import build_ir_from_ast`, `from fastcode.indexer import CodeElement`, `from fastcode.semantic_ir import IRSnapshot`, hypothesis imports, pytest
- Keep all helper functions and test classes from both files
- Property tests from `test_ast_to_ir_properties.py`: keep `_property` postfixes as-is
- Parametrized tests from `test_ast_to_ir_parametrize.py`: reclassify each by technique:

| Current Name | Parametrized? | Technique | Rename To |
|---|---|---|---|
| `test_type_produces_symbol` | Yes (`_SYMBOL_TYPES`) | basic (positive) | keep as-is |
| `test_type_skips_symbol` | Yes (`_SKIP_TYPES`) | negative (skipped types) | `test_type_skips_symbol_negative` |
| `test_type_skips_symbol_but_still_creates_document` | Yes (`_SKIP_TYPES`) | edge (partial behavior) | `test_type_skips_symbol_but_still_creates_document_edge` |
| `test_qualified_name_construction` | Yes (`_QUALIFIED_NAME_CASES`) | edge (already `@pytest.mark.edge`) | `test_qualified_name_construction_edge` |
| `test_qualified_name_with_class` | No | basic | keep as-is |
| `test_qualified_name_without_class` | No | edge (missing class) | `test_qualified_name_without_class_edge` |
| `test_source_set_on_symbols` | No | basic | keep as-is |
| `test_source_set_on_documents` | No | basic | keep as-is |
| `test_start_line_clamping` | Yes (`_LINE_CLAMP_CASES`) | edge (already `@pytest.mark.edge`) | `test_start_line_clamping_edge` |
| `test_source_priority_constant` | No | basic | keep as-is |
| `test_symbol_metadata_contains_ast_fields` | No | basic | keep as-is |
| `test_symbol_metadata_excludes_embedding_keys` | No | edge (exclusion) | `test_symbol_metadata_excludes_embedding_keys_edge` |
| `test_embedding_attachment_created_from_metadata` | No | basic | keep as-is |
| `test_summary_attachment_created_from_element_summary` | No | basic | keep as-is |
| `test_occurrence_metadata_kind` | No | basic | keep as-is |
| `test_snapshot_metadata_source_modes` | No | basic | keep as-is |
| `test_contain_edge_metadata` | No | basic | keep as-is |
| `test_empty_elements_no_symbols_or_docs_edge` | No | edge (already has postfix) | keep as-is |
| `test_empty_elements_identity_fields` | No | edge (empty input) | `test_empty_elements_identity_fields_edge` |
| `test_multiple_elements_same_file_one_document` | No | integration | keep as-is |
| `test_multiple_files_multiple_documents` | No | integration | keep as-is |
| `test_contain_edge_per_symbol` | No | basic | keep as-is |
| `test_occurrence_role_always_definition` | No | basic | keep as-is |
| `test_occurrence_source_always_ast` | No | basic | keep as-is |
| `test_class_element_creates_inheritance_edge_from_bases` | No | basic | keep as-is |
| `test_file_element_with_imports_creates_import_edge` | No | basic | keep as-is |
| `test_no_self_import_edge` | No | edge (already has postfix) | keep as-is |
| `test_language_fallback_to_unknown_edge` | No | edge (already has postfix) | keep as-is |
| `test_symbol_id_is_deterministic` | No | basic | keep as-is |

- **Every parametrized test keeps `@pytest.mark.parametrize`.** No `_parametrize` postfix on any function name.
- Module docstring: `"""Tests for adapters.ast_to_ir module."""`

- [ ] **Step 3: Run tests to verify**

```bash
uv run pytest tests/adapters/test_ast_to_ir.py -v
```
Expected: All tests pass (same count as before consolidation)

- [ ] **Step 4: Delete old files**

```bash
rm tests/test_ast_to_ir_parametrize.py tests/test_ast_to_ir_properties.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/adapters/ && git add -u tests/test_ast_to_ir_parametrize.py tests/test_ast_to_ir_properties.py
git commit -m "refactor: migrate ast_to_ir tests to adapters mirror layout"
```

---

### Task 2: Migrate SCIP adapter tests

**Files:**
- Create: `tests/adapters/test_scip_to_ir.py`
- Delete: `tests/test_scip_to_ir_parametrize.py`, `tests/test_scip_to_ir_properties.py`, `tests/test_scip_parsing_properties.py`

- [ ] **Step 1: Consolidate the 3 files into `tests/adapters/test_scip_to_ir.py`**

Merge: `test_scip_to_ir_parametrize.py` + `test_scip_to_ir_properties.py` + `test_scip_parsing_properties.py`
- Combined imports: `from fastcode.adapters.scip_to_ir import build_ir_from_scip`, `from fastcode.scip_models import ...`, `from fastcode.semantic_ir import IRSnapshot`, hypothesis, pytest
- Keep all helpers and test functions
- Property tests from `test_scip_to_ir_properties.py` and `test_scip_parsing_properties.py`: keep `_property` postfixes as-is
- Parametrized tests from `test_scip_to_ir_parametrize.py`: reclassify each by technique:

| Current Name | Parametrized? | Technique | Rename To |
|---|---|---|---|
| `test_role_produces_ref_edge` | Yes (`_REF_ROLES`) | edge (boundary roles) | keep (already has `_edge`) |
| `test_role_does_not_produce_ref_edge` | Yes (`_NO_REF_ROLES`) | negative (excluded roles) | `test_role_does_not_produce_ref_edge_negative` |
| `test_occurrence_range_normalization` | Yes (`_RANGE_CASES`) | edge (boundary ranges) | `test_occurrence_range_normalization_edge` |
| `test_occurrence_none_cols_become_zero` | Yes | edge (None handling) | `test_occurrence_none_cols_become_zero_edge` |
| `test_empty_variants_no_symbols_or_occurrences` | Yes (`_EMPTY_DOC_VARIANTS`) | edge (empty inputs) | `test_empty_variants_no_symbols_or_occurrences_edge` |
| `test_empty_symbols_still_creates_document_edge` | No | edge (already has postfix) | keep as-is |
| `test_symbol_without_symbol_field_skipped_edge` | No | edge (already has postfix) | keep as-is |
| `test_occurrence_without_symbol_field_skipped_edge` | No | edge (already has postfix) | keep as-is |
| `test_language_fallback_chain` | Yes (`_LANGUAGE_CASES`) | edge (fallback boundary) | `test_language_fallback_chain_edge` |
| `test_display_name_fallback` | Yes (`_DISPLAY_NAME_CASES`) | edge (fallback boundary) | `test_display_name_fallback_edge` |
| `test_all_scip_symbols_have_priority_100` | No | basic | keep as-is |
| `test_symbol_metadata_fields` | No | basic | keep as-is |
| `test_indexer_name_version_propagated_to_symbol_metadata` | No | basic | keep as-is |
| `test_occurrence_metadata_fields` | No | basic | keep as-is |
| `test_snapshot_metadata_source_modes` | No | basic | keep as-is |
| `test_contain_edge_metadata` | No | basic | keep as-is |
| `test_dict_input_produces_snapshot` | No | basic | keep as-is |
| `test_scip_index_input_produces_snapshot` | No | basic | keep as-is |
| `test_dict_and_scip_index_produce_equivalent_symbols` | No | integration | keep as-is |
| `test_multiple_documents_each_scoped` | No | integration | keep as-is |
| `test_ref_edge_carries_occurrence_id` | No | basic | keep as-is |
| `test_source_set_always_scip` | No | basic | keep as-is |
| `test_contain_edge_per_symbol` | No | basic | keep as-is |
| `test_snapshot_identity_fields` | No | basic | keep as-is |

- **Every parametrized test keeps `@pytest.mark.parametrize`.** No `_parametrize` postfix on any function name.
- Module docstring: `"""Tests for adapters.scip_to_ir module."""`

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/adapters/test_scip_to_ir.py -v
```

- [ ] **Step 3: Delete old files**

```bash
rm tests/test_scip_to_ir_parametrize.py tests/test_scip_to_ir_properties.py tests/test_scip_parsing_properties.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/adapters/ && git add -u tests/test_scip_to_ir_parametrize.py tests/test_scip_to_ir_properties.py tests/test_scip_parsing_properties.py
git commit -m "refactor: migrate scip_to_ir tests to adapters mirror layout"
```

---

### Task 3: Fix misplacement — module_resolver

`module_resolver.py` is root-level, but its test is in `tests/core/`.

**Files:**
- Move: `tests/core/test_module_resolver_properties.py` → `tests/test_module_resolver.py`

- [ ] **Step 1: Move and rename**

```bash
mv tests/core/test_module_resolver_properties.py tests/test_module_resolver.py
```

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_module_resolver.py -v
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: move module_resolver test to mirror root-level source"
```

---

### Task 4: Fix misplacement — manifest_store_db

`manifest_store.py` is root-level, but one test is in `tests/infrastructure/`.

**Files:**
- Delete: `tests/infrastructure/test_manifest_store_db_properties.py`
- Modify: `tests/test_manifest_store.py` (merge contents)

- [ ] **Step 1: Move the file to root and merge**

Move `tests/infrastructure/test_manifest_store_db_properties.py` content into `tests/test_manifest_store.py` (which will be created from `test_manifest_store_properties.py` in Task 10). If Task 10 hasn't run yet, just move the file:

```bash
mv tests/infrastructure/test_manifest_store_db_properties.py tests/test_manifest_store_db.py
```

Then in Task 10, this file will be merged into `test_manifest_store.py`.

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_manifest_store_db.py -v
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: move manifest_store_db test out of infrastructure to root"
```

---

### Task 5: Migrate `test_utils_properties.py` to `tests/utils/`

**Files:**
- Create: `tests/utils/test_utils.py`
- Delete: `tests/test_utils_properties.py`

- [ ] **Step 1: Move and rename**

```bash
mv tests/test_utils_properties.py tests/utils/test_utils.py
```

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/utils/test_utils.py -v
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: move utils test to mirrored subdirectory"
```

---

### Task 6: Rename `_properties` single files with postfix verification

These files are the sole test files for their modules. Renaming requires verifying every test function has the `_property` postfix.

**Files to rename (all are sole test files for their module):**

| Old Name | New Name |
|----------|----------|
| `test_db_runtime_properties.py` | `test_db_runtime.py` |
| `test_global_index_builder_properties.py` | `test_global_index_builder.py` |
| `test_ir_graph_builder_properties.py` | `test_ir_graph_builder.py` |
| `test_ir_validators_properties.py` | `test_ir_validators.py` |
| `test_path_utils_properties.py` | `test_path_utils.py` |
| `test_repo_selector_properties.py` | `test_repo_selector.py` |
| `test_scip_loader_properties.py` | `test_scip_loader.py` |
| `test_symbol_resolver_properties.py` | `test_symbol_resolver.py` |
| `test_manifest_store_properties.py` | `test_manifest_store.py` |

- [ ] **Step 1: Rename all 9 files**

```bash
cd tests
mv test_db_runtime_properties.py test_db_runtime.py
mv test_global_index_builder_properties.py test_global_index_builder.py
mv test_ir_graph_builder_properties.py test_ir_graph_builder.py
mv test_ir_validators_properties.py test_ir_validators.py
mv test_path_utils_properties.py test_path_utils.py
mv test_repo_selector_properties.py test_repo_selector.py
mv test_scip_loader_properties.py test_scip_loader.py
mv test_symbol_resolver_properties.py test_symbol_resolver.py
mv test_manifest_store_properties.py test_manifest_store.py
```

- [ ] **Step 2: Verify every test function has `_property` postfix**

For each renamed file, grep for test functions WITHOUT technique postfixes:

```bash
for f in tests/test_db_runtime.py tests/test_global_index_builder.py tests/test_ir_graph_builder.py tests/test_ir_validators.py tests/test_path_utils.py tests/test_repo_selector.py tests/test_scip_loader.py tests/test_symbol_resolver.py tests/test_manifest_store.py; do
  echo "=== $f ==="
  grep 'def test_' "$f" | grep -v '_property\b' | grep -v '_edge\b' | grep -v '_negative\b' | grep -v '_double\b' | grep -v '_perf\b' | grep -v '_snapshot\b'
done
```

If any functions appear in the output, they need a `_property` postfix added (since these are all hypothesis-based files). Fix each one, then re-run the grep to confirm zero output.

- [ ] **Step 3: Run tests to verify**

```bash
uv run pytest tests/test_db_runtime.py tests/test_global_index_builder.py tests/test_ir_graph_builder.py tests/test_ir_validators.py tests/test_path_utils.py tests/test_repo_selector.py tests/test_scip_loader.py tests/test_symbol_resolver.py tests/test_manifest_store.py -v
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: rename property test files to match module names, verify function postfixes"
```

---

### Task 7: Rename non-`_properties` files to match module names

These are simple renames where the filename doesn't match the source module.

**Files to rename:**

| Old Name | New Name | Source Module |
|----------|----------|---------------|
| `test_pg_retrieval_fallback_filters.py` | `test_pg_retrieval.py` | `pg_retrieval.py` |
| `test_graph_api.py` | `test_api.py` | `api.py` |

- [ ] **Step 1: Rename both files**

```bash
cd tests
mv test_pg_retrieval_fallback_filters.py test_pg_retrieval.py
mv test_graph_api.py test_api.py
```

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_pg_retrieval.py tests/test_api.py -v
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: rename tests to match source module names"
```

---

### Task 8: Merge manifest_store_db into manifest_store

**Files:**
- Delete: `tests/test_manifest_store_db.py` (moved from infrastructure in Task 4)
- Modify: `tests/test_manifest_store.py`

- [ ] **Step 1: Merge `test_manifest_store_db.py` into `test_manifest_store.py`**

Append the test functions from `test_manifest_store_db.py` to `test_manifest_store.py`. Deduplicate imports. Keep all function names as-is.

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_manifest_store.py -v
```

- [ ] **Step 3: Delete merged file**

```bash
rm tests/test_manifest_store_db.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate manifest_store tests into single file"
```

---

### Task 9: Consolidate retriever tests

**Files:**
- Create: `tests/test_retriever.py`
- Delete: `tests/test_adaptive_fusion.py`, `tests/test_doc_channel_projection.py`

- [ ] **Step 1: Consolidate into `tests/test_retriever.py`**

Merge `test_adaptive_fusion.py` (97 lines) + `test_doc_channel_projection.py` (168 lines).
- Combined imports: `from fastcode.retriever import HybridRetriever`, `from fastcode.indexer import CodeElement`
- Module docstring: `"""Tests for retriever module."""`

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_retriever.py -v
```

- [ ] **Step 3: Delete old files**

```bash
rm tests/test_adaptive_fusion.py tests/test_doc_channel_projection.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate retriever tests into single file"
```

---

### Task 10: Consolidate main tests

**Files:**
- Create: `tests/test_main.py`
- Delete: `tests/test_main_doc_pipeline.py`, `tests/test_session_prefix.py`

- [ ] **Step 1: Consolidate into `tests/test_main.py`**

Merge `test_main_doc_pipeline.py` + `test_session_prefix.py` (378 lines).
- Combined imports: `from fastcode.main import FastCode`, mock/patch imports
- Module docstring: `"""Tests for main FastCode class."""`

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_main.py -v
```

- [ ] **Step 3: Delete old files**

```bash
rm tests/test_main_doc_pipeline.py tests/test_session_prefix.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate main tests into single file"
```

---

### Task 11: Consolidate doc_ingester tests

**Files:**
- Modify: `tests/test_doc_ingester.py`
- Delete: `tests/test_doc_ingester_properties.py`, `tests/test_chonkie_chunking.py`

- [ ] **Step 1: Consolidate into `tests/test_doc_ingester.py`**

Merge `test_doc_ingester.py` (72) + `test_doc_ingester_properties.py` (283) + `test_chonkie_chunking.py` (302).
- Combined imports: `from fastcode.doc_ingester import KeyDocIngester`, hypothesis, mock
- Module docstring: `"""Tests for doc_ingester module."""`

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_doc_ingester.py -v
```

- [ ] **Step 3: Delete old files**

```bash
rm tests/test_doc_ingester_properties.py tests/test_chonkie_chunking.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate doc_ingester tests into single file"
```

---

### Task 12: Consolidate graph_runtime tests

**Files:**
- Modify: `tests/test_graph_runtime.py`
- Delete: `tests/test_graph_runtime_properties.py`

- [ ] **Step 1: Consolidate**

Merge `test_graph_runtime.py` (20) + `test_graph_runtime_properties.py` (228) into `tests/test_graph_runtime.py`.

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_graph_runtime.py -v
```

- [ ] **Step 3: Delete old file**

```bash
rm tests/test_graph_runtime_properties.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate graph_runtime tests into single file"
```

---

### Task 13: Consolidate ir_merge tests

**Files:**
- Create: `tests/test_ir_merge.py`
- Delete: `tests/test_ir_merge_properties.py`, `tests/test_ir_alignment_algorithm.py`

- [ ] **Step 1: Consolidate**

Merge `test_ir_merge_properties.py` (599) + `test_ir_alignment_algorithm.py` (177).
- Combined imports: `from fastcode.ir_merge import merge_ir`, `from fastcode.semantic_ir import ...`
- Module docstring: `"""Tests for ir_merge module."""`

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_ir_merge.py -v
```

- [ ] **Step 3: Delete old files**

```bash
rm tests/test_ir_merge_properties.py tests/test_ir_alignment_algorithm.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate ir_merge tests into single file"
```

---

### Task 14: Consolidate mcp_graph_tools tests

**Files:**
- Modify: `tests/test_mcp_graph_tools.py`
- Delete: `tests/test_mcp_directed_path.py`

- [ ] **Step 1: Consolidate**

Merge `test_mcp_graph_tools.py` (580) + `test_mcp_directed_path.py` (375) into `tests/test_mcp_graph_tools.py`.
- Combined imports: `from fastcode.mcp_graph_tools import ...`, `from fastcode.semantic_ir import ...`, `from fastcode.ir_graph_builder import ...`
- Module docstring: `"""Tests for mcp_graph_tools module."""`

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_mcp_graph_tools.py -v
```

- [ ] **Step 3: Delete old file**

```bash
rm tests/test_mcp_directed_path.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate mcp_graph_tools tests into single file"
```

---

### Task 15: Consolidate projection_models test

**Files:**
- Rename: `tests/test_projection_models_properties.py` → `tests/test_projection_models.py`

- [ ] **Step 1: Rename**

```bash
mv tests/test_projection_models_properties.py tests/test_projection_models.py
```

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_projection_models.py -v
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: rename projection_models test to match module"
```

---

### Task 16: Consolidate projection_transform tests

**Files:**
- Create: `tests/test_projection_transform.py`
- Delete: `tests/test_projection_contract.py`, `tests/test_projection_properties.py`, `tests/test_projection_pipeline.py`, `tests/test_projection_supporting_docs.py`, `tests/test_projection_v2_schema.py`
- Rename snapshot: `tests/__snapshots__/test_projection_contract.ambr` → `tests/__snapshots__/test_projection_transform.ambr`

- [ ] **Step 1: Rename snapshot file**

```bash
mv tests/__snapshots__/test_projection_contract.ambr tests/__snapshots__/test_projection_transform.ambr
```

- [ ] **Step 2: Consolidate all 5 files**

Merge in order:
1. `test_projection_properties.py` (478) — largest, property-based
2. `test_projection_pipeline.py` (169) — pipeline integration
3. `test_projection_contract.py` (220) — snapshot/contract tests
4. `test_projection_supporting_docs.py` (196)
5. `test_projection_v2_schema.py` (147)

Combined imports: `from fastcode.projection_transform import ProjectionTransformer`, `from fastcode.projection_models import ...`, `from fastcode.semantic_ir import ...`, `from fastcode.ir_graph_builder import ...`, hypothesis, networkx

Update syrupy snapshot reference: the `snapshot` fixture in `test_projection_contract.py` test functions will automatically use the new snapshot file name `test_projection_transform.ambr`.

Module docstring: `"""Tests for projection_transform module."""`

- [ ] **Step 3: Run tests to verify**

```bash
uv run pytest tests/test_projection_transform.py -v
```

- [ ] **Step 4: Delete old files**

```bash
rm tests/test_projection_contract.py tests/test_projection_properties.py tests/test_projection_pipeline.py tests/test_projection_supporting_docs.py tests/test_projection_v2_schema.py
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: consolidate projection_transform tests into single file"
```

---

### Task 17: Consolidate redo_worker tests

**Files:**
- Modify: `tests/test_redo_worker.py`
- Delete: `tests/test_redo_worker_properties.py`

- [ ] **Step 1: Consolidate**

Merge `test_redo_worker.py` (102) + `test_redo_worker_properties.py` (286) into `tests/test_redo_worker.py`.

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_redo_worker.py -v
```

- [ ] **Step 3: Delete old file**

```bash
rm tests/test_redo_worker_properties.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate redo_worker tests into single file"
```

---

### Task 18: Consolidate scip_indexers + scip_binary tests

**Files:**
- Create: `tests/test_scip_indexers.py` (overwrite with consolidation)
- Modify: `tests/test_scip_loader.py` (merge scip_binary)
- Delete: `tests/test_scip_indexers_properties.py`, `tests/test_scip_binary.py`

- [ ] **Step 1: Consolidate scip_indexers**

Merge `test_scip_indexers.py` (213) + `test_scip_indexers_properties.py` (171) into `tests/test_scip_indexers.py`.

- [ ] **Step 2: Merge scip_binary into scip_loader**

Merge `test_scip_binary.py` (194) into `tests/test_scip_loader.py` (renamed in Task 7).

- [ ] **Step 3: Run tests to verify**

```bash
uv run pytest tests/test_scip_indexers.py tests/test_scip_loader.py -v
```

- [ ] **Step 4: Delete old files**

```bash
rm tests/test_scip_indexers_properties.py tests/test_scip_binary.py
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: consolidate scip_indexers and scip_loader tests"
```

---

### Task 19: Consolidate scip_models tests

**Files:**
- Modify: `tests/test_scip_models.py`
- Delete: `tests/test_scip_models_properties.py`

- [ ] **Step 1: Consolidate**

Merge `test_scip_models.py` (182) + `test_scip_models_properties.py` (465) into `tests/test_scip_models.py`.

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_scip_models.py -v
```

- [ ] **Step 3: Delete old file**

```bash
rm tests/test_scip_models_properties.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate scip_models tests into single file"
```

---

### Task 20: Consolidate semantic_ir tests

**Files:**
- Create: `tests/test_semantic_ir.py`
- Delete: `tests/test_semantic_ir_properties.py`, `tests/test_ir_snapshot_contract.py`
- Rename snapshot: `tests/__snapshots__/test_ir_snapshot_contract.ambr` → `tests/__snapshots__/test_semantic_ir.ambr`

- [ ] **Step 1: Rename snapshot file**

```bash
mv tests/__snapshots__/test_ir_snapshot_contract.ambr tests/__snapshots__/test_semantic_ir.ambr
```

- [ ] **Step 2: Consolidate**

Merge `test_semantic_ir_properties.py` (1000) + `test_ir_snapshot_contract.py` (163) into `tests/test_semantic_ir.py`.

Update snapshot reference in syrupy tests: the `snapshot` fixture will automatically look for `test_semantic_ir.ambr`.

- [ ] **Step 3: Run tests to verify**

```bash
uv run pytest tests/test_semantic_ir.py -v
```

- [ ] **Step 4: Delete old files**

```bash
rm tests/test_semantic_ir_properties.py tests/test_ir_snapshot_contract.py
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: consolidate semantic_ir tests into single file"
```

---

### Task 21: Consolidate snapshot_store tests

This is the largest consolidation (~2570 lines). Split into two files to stay under 1200 lines each.

**Files:**
- Create: `tests/test_snapshot_store.py` (core + properties ~1400 lines)
- Create: `tests/test_snapshot_store_extended.py` (db + lifecycle + stateful ~1200 lines)
- Delete: `tests/test_snapshot_store_properties.py`, `tests/test_snapshot_store_db_properties.py`, `tests/test_snapshot_store_lifecycle_properties.py`, `tests/test_snapshot_store_stateful.py`

- [ ] **Step 1: Create `tests/test_snapshot_store.py`**

Merge `test_snapshot_store_properties.py` (866 lines) content. This is the primary test file covering basic CRUD, query, and property-based invariants.

- [ ] **Step 2: Create `tests/test_snapshot_store_extended.py`**

Merge:
- `test_snapshot_store_db_properties.py` (792 lines) — database-level invariants
- `test_snapshot_store_lifecycle_properties.py` (516 lines) — lifecycle state machine
- `test_snapshot_store_stateful.py` (396 lines) — hypothesis stateful testing

Combined ~1700 lines. If needed, split the lifecycle into a third file `test_snapshot_store_lifecycle.py`. Prefer keeping in two files.

- [ ] **Step 3: Run tests to verify**

```bash
uv run pytest tests/test_snapshot_store.py tests/test_snapshot_store_extended.py -v
```

- [ ] **Step 4: Delete old files**

```bash
rm tests/test_snapshot_store_properties.py tests/test_snapshot_store_db_properties.py tests/test_snapshot_store_lifecycle_properties.py tests/test_snapshot_store_stateful.py
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: consolidate snapshot_store tests into two files"
```

---

### Task 22: Consolidate snapshot_symbol_index tests

**Files:**
- Modify: `tests/test_snapshot_symbol_index.py`
- Delete: `tests/test_snapshot_symbol_index_properties.py`

- [ ] **Step 1: Consolidate**

Merge `test_snapshot_symbol_index.py` (32) + `test_snapshot_symbol_index_properties.py` (576) into `tests/test_snapshot_symbol_index.py`.

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_snapshot_symbol_index.py -v
```

- [ ] **Step 3: Delete old file**

```bash
rm tests/test_snapshot_symbol_index_properties.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate snapshot_symbol_index tests into single file"
```

---

### Task 23: Consolidate terminus_publisher tests

**Files:**
- Create: `tests/test_terminus_publisher.py`
- Delete: `tests/test_terminus_publisher_properties.py`, `tests/test_terminus_code_graph.py`, `tests/test_terminus_lineage_edges.py`, `tests/test_terminus_payload.py`

- [ ] **Step 1: Consolidate**

Merge all 4 files (~835 lines total):
1. `test_terminus_publisher_properties.py` (444) — property-based
2. `test_terminus_code_graph.py` (303) — code graph operations
3. `test_terminus_lineage_edges.py` (48) — lineage edge tests
4. `test_terminus_payload.py` (40) — payload serialization

Module docstring: `"""Tests for terminus_publisher module."""`

- [ ] **Step 2: Run tests to verify**

```bash
uv run pytest tests/test_terminus_publisher.py -v
```

- [ ] **Step 3: Delete old files**

```bash
rm tests/test_terminus_publisher_properties.py tests/test_terminus_code_graph.py tests/test_terminus_lineage_edges.py tests/test_terminus_payload.py
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: consolidate terminus_publisher tests into single file"
```

---

### Task 24: Final verification — run full test suite

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: All tests pass. Test count should be identical to pre-migration (no tests lost or duplicated).

- [ ] **Step 2: Verify no orphaned flat test files remain**

```bash
find tests/ -maxdepth 1 -name 'test_*' -type f | sort
```

Expected: Only root-level module tests + cross-module tests remain:
- `test_api.py`
- `test_db_runtime.py`
- `test_doc_ingester.py`
- `test_e2e_indexing.py`
- `test_e2e_semantic_pipeline.py`
- `test_global_index_builder.py`
- `test_graph_runtime.py`
- `test_incremental_update.py`
- `test_ir_graph_builder.py`
- `test_ir_merge.py`
- `test_ir_validators.py`
- `test_main.py`
- `test_manifest_store.py`
- `test_mcp_graph_tools.py`
- `test_module_resolver.py`
- `test_outbox_pattern.py`
- `test_path_utils.py`
- `test_pg_retrieval.py`
- `test_projection_models.py`
- `test_projection_transform.py`
- `test_redo_worker.py`
- `test_retriever.py`
- `test_scip_indexers.py`
- `test_scip_loader.py`
- `test_scip_models.py`
- `test_scip_resolution_bridge.py`
- `test_semantic_ir.py`
- `test_snapshot_pipeline.py`
- `test_snapshot_store.py`
- `test_snapshot_store_extended.py`
- `test_snapshot_symbol_index.py`
- `test_symbol_resolver.py`
- `test_terminus_publisher.py`

- [ ] **Step 3: Verify subdirectory structure**

```bash
find tests/ -type d | sort
```

Expected:
```
tests/
tests/__snapshots__
tests/adapters
tests/core
tests/infrastructure
tests/schemas
tests/utils
```

- [ ] **Step 4: Verify snapshot files renamed correctly**

```bash
ls tests/__snapshots__/
```

Expected:
```
test_projection_transform.ambr
test_semantic_ir.ambr
```

- [ ] **Step 5: Final commit if any cleanup needed**

```bash
git add -A && git commit -m "refactor: test mirror layout migration complete"
```

---

## Dependency Order

Tasks can be partially parallelized:

**Batch A (independent, can run in parallel):**
- Task 1 (adapters/ast_to_ir — includes parametrize reclassification)
- Task 2 (adapters/scip_to_ir — includes parametrize reclassification)
- Task 3 (module_resolver misplacement)
- Task 4 (manifest_store_db misplacement)
- Task 5 (utils)
- Task 6 (`_properties` renames with postfix verification)

**Batch B (depends on Task 4, Task 6):**
- Task 7 (non-`_properties` simple renames)
- Task 8 (merge manifest_store_db → manifest_store, depends on Task 4 + Task 6)

**Batch C (independent, can run in parallel):**
- Task 9 (retriever)
- Task 10 (main)
- Task 11 (doc_ingester)
- Task 12 (graph_runtime)
- Task 13 (ir_merge)
- Task 14 (mcp_graph_tools)
- Task 15 (projection_models)
- Task 16 (projection_transform)
- Task 17 (redo_worker)
- Task 18 (scip_indexers + scip_binary — depends on Task 6 for test_scip_loader.py rename)
- Task 19 (scip_models)
- Task 20 (semantic_ir)
- Task 21 (snapshot_store)
- Task 22 (snapshot_symbol_index)
- Task 23 (terminus_publisher)

**Final:**
- Task 24 (verification)

## Execution Notes

- All work is done in the `fastcode/` working directory (the subproject root containing `tests/`)
- `conftest.py` stays at `tests/` root — all subdirectory tests inherit its fixtures and strategies automatically
- No `__init__.py` files needed in test directories (pytest doesn't require them)
- `pythonpath = [".."]` in pyproject.toml means all imports (`from fastcode.xxx`) work regardless of test file location
- Technique postfixes go on **function names** (`test_foo_property`, `test_bar_edge`), not filenames
- `@pytest.mark.parametrize` is always preserved — it's a methodology (table-driven), not a technique category
- `_properties` files are not "simple renames" — always verify function postfixes match the technique
