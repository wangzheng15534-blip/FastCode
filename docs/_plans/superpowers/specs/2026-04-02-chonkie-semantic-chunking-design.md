# Chonkie Semantic Chunking Integration

**Date:** 2026-04-02
**Status:** Approved
**Scope:** Replace hand-rolled word-based chunking with Chonkie SemanticChunker

## Problem

The document ingestion pipeline uses a hand-rolled chunking implementation that splits text by words with a sliding window. This produces poor-quality chunks for RAG retrieval: sentences split mid-way, no semantic boundary awareness, and no token-level size control.

Two locations:
1. `KeyDocIngester._chunk_document()` in `doc_ingester.py:168-209` — active, used by index pipeline
2. `chunk_text()` in `utils.py:263-281` — dead code, no callers

## Solution

Replace word-based chunking with [Chonkie](https://github.com/chunking/chonkie) `SemanticChunker`, which uses embedding similarity to detect semantic boundaries between sentences.

**Approach: Heading pre-split + SemanticChunker within sections.** Keep the existing markdown heading detection to preserve structural metadata (heading name, line ranges), then apply SemanticChunker within each section for high-quality chunk boundaries.

## Architecture

### What changes

| Component | Change |
|-----------|--------|
| `KeyDocIngester.__init__()` | Create lazy `SemanticChunker` instance |
| `KeyDocIngester._chunk_document()` | Replace word-split with `SemanticChunker.chunk()` per section |
| `KeyDocIngester.__init__()` config | Read `chunk_token_size`, `similarity_threshold` instead of `chunk_size`, `chunk_overlap`, `max_chunk_chars` |
| `utils.chunk_text()` | Remove entirely (dead code) |
| `pyproject.toml` | Add `chonkie[semantic]>=1.6` dependency |
| `config/config.yaml` | Update `docs_integration` section keys |

### What stays the same

- `DocChunk` dataclass
- `_discover_files()`, `_matches_any()`, `_is_denied()`, `_read_text()`
- `_detect_doc_type()`, `_extract_mentions()`
- `ingest()` method signature and flow
- All downstream consumers (`to_element()`, mentions, embedding)

## Chunking Flow

```
text
  → splitlines → detect headings (#{1,6} regex)
  → [(heading, start_line, end_line, section_text), ...]
  → for each section: SemanticChunker.chunk(section_text)
  → map character offsets back to line numbers
  → [{"heading", "start_line", "end_line", "text"}, ...]
```

### Line number mapping

Chonkie's `Chunk` provides `start_index` and `end_index` as character offsets within the section text. Map these to line numbers:

```python
def _char_offset_to_line(text: str, offset: int, base_line: int) -> int:
    return base_line + text[:offset].count("\n")
```

## SemanticChunker Configuration

| Parameter | Config key | Default | Description |
|-----------|-----------|---------|-------------|
| `embedding_model` | (hardcoded) | `"minishlab/potion-base-32M"` | Model2Vec, ~32MB, no GPU needed |
| `chunk_size` | `chunk_token_size` | `512` | Max tokens per chunk |
| `threshold` | `similarity_threshold` | `0.5` | Cosine similarity boundary (lower = larger chunks) |

Lazy initialization — the chunker is created on first call to `_chunk_document()`, not in `__init__()`, to avoid loading the model when doc integration is disabled.

## Config Changes

### `config/config.yaml`

```yaml
docs_integration:
  enabled: true
  curated_paths: ["README*", "docs/design/**", "docs/research/**", "docs/adr/**", "docs/rfc/**"]
  allow_paths: []
  deny_paths: []
  chunk_token_size: 512
  similarity_threshold: 0.5
```

Removed keys: `chunk_size`, `chunk_overlap`, `max_chunk_chars`.

### `pyproject.toml`

Add to `dependencies`:
```
"chonkie[semantic]>=1.6",
```

## Error Handling & Fallback

Three-level graceful degradation:

1. **Import failure** — if `chonkie` is not installed, log warning, fall back to original word-based chunking (extracted to `_chunk_document_fallback()`)

2. **Model load failure** — if SemanticChunker initialization fails (model download, OOM), log warning, fall back to word-based chunking

3. **Per-section failure** — if `chunker.chunk()` throws on a specific section, catch exception, log, fall back to word-split for that section only. Other sections continue with semantic chunking.

Fallback method preserves the old `_chunk_document()` logic as `_chunk_document_fallback()` private method.

## Test Updates

### `test_doc_ingester.py`

- Update config fixture to use `chunk_token_size` and `similarity_threshold` instead of `chunk_size`/`chunk_overlap`
- Verify chunks are produced (structural test, not exact-match, since semantic boundaries vary)
- Add test for empty section handling
- Add test for fallback when chonkie is not available (mock import error)

## Files Modified

| File | Change |
|------|--------|
| `fastcode/doc_ingester.py` | Replace `_chunk_document()`, add lazy chunker init, add fallback |
| `fastcode/utils.py` | Remove `chunk_text()` function |
| `config/config.yaml` | Update `docs_integration` section |
| `pyproject.toml` | Add `chonkie[semantic]` dependency |
| `tests/test_doc_ingester.py` | Update config fixture, add fallback test |
