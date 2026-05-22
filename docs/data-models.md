# FastCode Data Models

## Canonical IR (`fastcode/semantic_ir.py`)

The canonical IR is the deepest truth layer. All extraction (SCIP, AST) produces IR before storage.

### IRSnapshot

Top-level container for a single repository snapshot.

| Field | Type | Description |
|-------|------|-------------|
| `repo_name` | `str` | Repository name |
| `snapshot_id` | `str` | `snap:{repo_name}:{commit_id}` format |
| `commit_id` | `str` | Git commit SHA |
| `branch` | `str?` | Branch name |
| `tree_id` | `str?` | Git tree SHA |
| `documents` | `list[IRDocument]` | File-level documents |
| `symbols` | `list[IRSymbol]` | Symbol definitions |
| `occurrences` | `list[IROccurrence]` | Symbol occurrences in documents |
| `edges` | `list[IREdge]` | Relationships between nodes |
| `attachments` | `list[IRAttachment]` | Derived data (summaries, embeddings) |

### IRDocument

File-level representation.

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Content-addressable ID |
| `path` | `str` | Relative file path |
| `language` | `str` | Programming language |
| `blob_oid` | `str?` | Git blob object ID |
| `source_set` | `set[str]` | Extraction sources (`{"scip"}`, `{"fc_structure"}`) |

### IRSymbol

Symbol definition (function, class, method, variable, etc.).

| Field | Type | Description |
|-------|------|-------------|
| `symbol_id` | `str` | Content-addressable ID |
| `display_name` | `str` | Human-readable name |
| `kind` | `str` | Symbol type (function, class, method, variable, etc.) |
| `qualified_name` | `str?` | Fully qualified name |
| `source_priority` | `int` | Priority (100=SCIP, 50=AST) |
| `source_set` | `set[str]` | Extraction sources |
| `confidence` | `str` | `"precise"` (SCIP) or `"resolved"` (AST) |
| `external_symbol_id` | `str?` | Original SCIP symbol string |
| `signature` | `str?` | Function/class signature |
| `metadata` | `dict` | Additional metadata |

### IROccurrence

Symbol occurrence within a document.

| Field | Type | Description |
|-------|------|-------------|
| `occurrence_id` | `str` | Content-addressable ID |
| `symbol_id` | `str` | Reference to IRSymbol |
| `doc_id` | `str` | Reference to IRDocument |
| `role` | `str` | `definition`, `reference`, `import`, `write_access`, etc. |
| `range` | `tuple` | `(start_line, start_col, end_line, end_col)` |
| `source` | `str?` | Extraction source |

### IREdge

Relationship between nodes.

| Field | Type | Description |
|-------|------|-------------|
| `edge_id` | `str` | Content-addressable ID |
| `edge_type` | `str` | `call`, `import`, `inherit`, `ref`, `contain` |
| `src_id` | `str` | Source node ID |
| `dst_id` | `str` | Target node ID |
| `confidence` | `str?` | `"precise"`, `"resolved"`, `"heuristic"` |
| `source_set` | `set[str]` | Extraction sources |
| `metadata` | `dict` | Additional metadata |

### IRAttachment

Derived data attached to symbols/documents.

| Field | Type | Description |
|-------|------|-------------|
| `attachment_id` | `str` | Content-addressable ID |
| `target_type` | `str` | `"symbol"` or `"document"` |
| `target_id` | `str` | Reference to IRSymbol or IRDocument |
| `attachment_type` | `str` | `"summary"`, `"embedding"` |
| `source` | `str` | `"fc_structure"`, `"fc_embedding"` |
| `confidence` | `str?` | Confidence level |
| `payload` | `any` | Attachment data (text, vector, etc.) |
| `metadata` | `dict` | Additional metadata |

---

## SCIP Models (`fastcode/scip_models.py`)

Pydantic models for SCIP protocol buffer payloads.

| Model | Purpose |
|-------|---------|
| `SCIPIndex` | Top-level SCIP output (indexer name/version, documents) |
| `SCIPDocument` | Per-file SCIP data (path, language, symbols, occurrences) |
| `SCIPSymbol` | Symbol metadata (symbol string, name, kind, range, signature) |
| `SCIPOccurrence` | Symbol occurrence (symbol, role, range, override_docs) |

---

## Projection Models (`fastcode/projection_models.py`)

Graph algorithm output models.

| Model | Purpose |
|-------|---------|
| `ClusterInfo` | Leiden cluster (id, label, members, metadata) |
| `ProjectionResult` | Full projection (L0 summary, L1 hierarchy, L2 chunks) |
| `ProjectionMeta` | Projection metadata (snapshot_id, algorithm params, timing) |

---

## ID Schemes

| Source | Algorithm | Format |
|--------|-----------|--------|
| SCIP adapter | MD5 (24 hex chars) | `scip:{snapshot_id}:{external_symbol}` |
| AST adapter | Blake2b (20 hex chars) | `ast:{snapshot_id}:{composite_key}` |
| Documents (SCIP) | MD5 | `doc:{md5(snapshot_id:path)}` |
| Documents (AST) | Blake2b | `doc:{blake2b(snapshot_id:rel_path)}` |
| Occurrences | MD5 or Blake2b | `occ:{hash(...)}` |
| Edges | MD5 or Blake2b | `edge:{hash(...)}` |
| Attachments | Blake2b | `att:{hash(...)}` |

---

## Storage Backends

| Layer | Backend | Storage Type |
|-------|---------|-------------|
| Canonical facts | PostgreSQL | IR snapshots, provenance, vectors, FTS |
| Derived graph | TerminusDB | Branch-aware graph views |
| Retrieval | PostgreSQL + ripgrep | pgvector HNSW + GIN FTS + agent-side rg |
| Cache | SQLite / Redis / disk | Query cache, embedding cache |
| Projections | PostgreSQL | L0/L1/L2 projection data |
| Sessions | JSONL | Multi-turn dialogue history |
| Documents | LadybugDB (optional) | Architecture docs with MENTIONS edges |

---

## Merge Strategy (Precision-Anchored)

Tree-sitter builds unit skeletons, SCIP anchors precision onto them. NOT "SCIP wins on overlap."

**Alignment scoring** (replaces hard overwrite):
- Span overlap + name match + kind compat + container compat
- High score → canonical anchor (precise identity)
- Medium score → candidate alias
- Low score → unanchored AST unit

| Rule | Name | Behavior |
|------|------|----------|
| A | SCIP anchors | Alignment scoring matches SCIP symbols to tree-sitter units. Anchored units get precise identity. |
| B | AST fills gaps | Tree-sitter units without SCIP match stay as structural units |
| C | Edges coexist | Edges deduped by `(src, dst, type)`, first writer wins |
| D | SCIP refs first | Occurrences deduped, SCIP processed first |

**Key principle:** SCIP anchors, not overwrites. Tree-sitter provides the skeleton (always, all files). SCIP sharpens the skeleton when available (8 languages). Embeddings/LLM are derived attachments, not graph facts.
