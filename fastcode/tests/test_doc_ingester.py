from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from fastcode.doc_ingester import KeyDocIngester
from fastcode.semantic_ir import IRSnapshot, IRSymbol

pytestmark = [pytest.mark.slow]


class _DummyEmbedder:
    def embed_text(self, text: str) -> Any:
        if not text:
            return None
        return [0.1, 0.2, 0.3]


def test_doc_ingester_selects_curated_files_and_extracts_mentions():
    with tempfile.TemporaryDirectory(prefix="fc_doc_ingest_") as tmp:
        root = Path(tmp)
        (root / "docs" / "design").mkdir(parents=True, exist_ok=True)
        (root / "docs" / "private").mkdir(parents=True, exist_ok=True)
        (root / "docs" / "design" / "pipeline.md").write_text(
            "# Pipeline Design\nThe BranchIndexer coordinates indexing.\n",
            encoding="utf-8",
        )
        (root / "docs" / "private" / "secret.md").write_text(
            "# Secret\nShould not be ingested.\n", encoding="utf-8"
        )

        cfg = {
            "docs_integration": {
                "enabled": True,
                "curated_paths": ["docs/design/**"],
                "allow_paths": [],
                "deny_paths": ["docs/private/**"],
                "chunk_size": 64,
                "chunk_overlap": 8,
            }
        }
        ingester = KeyDocIngester(cfg, _DummyEmbedder())
        snapshot = IRSnapshot(
            repo_name="repo",
            snapshot_id="snap:repo:1",
            symbols=[
                IRSymbol(
                    symbol_id="sym:branchindexer",
                    external_symbol_id=None,
                    path="fastcode/main.py",
                    display_name="BranchIndexer",
                    kind="class",
                    language="python",
                    source_set={"ast"},
                    metadata={"source": "ast"},
                )
            ],
        )
        result = ingester.ingest(
            repo_path=str(root),
            repo_name="repo",
            snapshot_id="snap:repo:1",
            snapshot=snapshot,
        )

        assert result["chunks"]
        assert result["elements"]
        assert all(e["type"] == "design_document" for e in result["elements"])
        assert all("docs/design/" in e["relative_path"] for e in result["elements"])
        assert all(
            "docs/private/" not in e["relative_path"] for e in result["elements"]
        )
        assert any(m["symbol_id"] == "sym:branchindexer" for m in result["mentions"])
