"""
Lightweight key-document ingestion for architecture/design/research docs.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .embedder import CodeEmbedder
from .semantic_ir import IRSnapshot


@dataclass
class DocChunk:
    chunk_id: str
    repo_name: str
    snapshot_id: str
    path: str
    title: str
    heading: Optional[str]
    doc_type: str
    text: str
    start_line: int
    end_line: int
    embedding: Optional[List[float]]

    def to_element(self) -> Dict[str, Any]:
        name = self.title if not self.heading else f"{self.title} - {self.heading}"
        summary = (self.text[:240] + "...") if len(self.text) > 240 else self.text
        return {
            "id": self.chunk_id,
            "type": "design_document",
            "name": name,
            "file_path": self.path,
            "relative_path": self.path,
            "language": "markdown",
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.text,
            "signature": None,
            "docstring": self.text[:1200],
            "summary": summary,
            "metadata": {
                "snapshot_id": self.snapshot_id,
                "source": "key_docs",
                "doc_type": self.doc_type,
                "heading": self.heading,
                "is_design_doc": True,
                "embedding": self.embedding,
            },
            "repo_name": self.repo_name,
            "repo_url": None,
        }


class KeyDocIngester:
    def __init__(self, config: Dict[str, Any], embedder: CodeEmbedder):
        self.config = config
        self.embedder = embedder
        cfg = config.get("docs_integration", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.curated_paths = list(
            cfg.get("curated_paths")
            or [
                "README*",
                "docs/design/**",
                "docs/research/**",
                "docs/adr/**",
                "docs/rfc/**",
            ]
        )
        self.allow_paths = list(cfg.get("allow_paths") or [])
        self.deny_paths = list(cfg.get("deny_paths") or [])
        self.chunk_size = int(cfg.get("chunk_size", 420))
        self.chunk_overlap = int(cfg.get("chunk_overlap", 80))
        self.max_chunk_chars = int(cfg.get("max_chunk_chars", 2400))

    def ingest(
        self,
        *,
        repo_path: str,
        repo_name: str,
        snapshot_id: str,
        snapshot: IRSnapshot,
    ) -> Dict[str, Any]:
        if not self.enabled or not repo_path or not os.path.isdir(repo_path):
            return {"chunks": [], "mentions": [], "elements": []}

        selected_files = self._discover_files(repo_path)
        chunks: List[DocChunk] = []
        for rel_path in selected_files:
            abs_path = os.path.join(repo_path, rel_path)
            text = self._read_text(abs_path)
            if not text.strip():
                continue
            title = os.path.basename(rel_path)
            doc_type = self._detect_doc_type(rel_path)
            for i, piece in enumerate(self._chunk_document(text)):
                chunk_id = self._chunk_id(snapshot_id, rel_path, i)
                emb = self._embed(piece["text"])
                chunks.append(
                    DocChunk(
                        chunk_id=chunk_id,
                        repo_name=repo_name,
                        snapshot_id=snapshot_id,
                        path=rel_path,
                        title=title,
                        heading=piece.get("heading"),
                        doc_type=doc_type,
                        text=piece["text"],
                        start_line=piece.get("start_line", 1),
                        end_line=piece.get("end_line", piece.get("start_line", 1)),
                        embedding=emb,
                    )
                )

        mentions = self._extract_mentions(snapshot=snapshot, chunks=chunks)
        elements = [c.to_element() for c in chunks]
        return {"chunks": chunks, "mentions": mentions, "elements": elements}

    def _discover_files(self, repo_path: str) -> List[str]:
        out: List[str] = []
        patterns = list(self.curated_paths)
        patterns.extend(self.allow_paths)
        for p in Path(repo_path).rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(repo_path)).replace("\\", "/")
            if self._is_denied(rel):
                continue
            if self._matches_any(rel, patterns):
                out.append(rel)
        out.sort()
        return out

    def _matches_any(self, rel_path: str, patterns: Sequence[str]) -> bool:
        path = rel_path.lstrip("./")
        base = os.path.basename(path)
        for pattern in patterns:
            pat = (pattern or "").strip()
            if not pat:
                continue
            if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(base, pat):
                return True
        return False

    def _is_denied(self, rel_path: str) -> bool:
        return self._matches_any(rel_path, self.deny_paths)

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""

    def _chunk_document(self, text: str) -> List[Dict[str, Any]]:
        lines = text.splitlines()
        sections: List[Tuple[Optional[str], int, int, str]] = []
        current_heading: Optional[str] = None
        start = 1
        buf: List[str] = []
        for idx, line in enumerate(lines, start=1):
            if re.match(r"^\s*#{1,6}\s+", line):
                if buf:
                    sections.append((current_heading, start, idx - 1, "\n".join(buf)))
                current_heading = re.sub(r"^\s*#{1,6}\s+", "", line).strip()
                start = idx
                buf = [line]
            else:
                buf.append(line)
        if buf:
            sections.append((current_heading, start, len(lines), "\n".join(buf)))

        chunks: List[Dict[str, Any]] = []
        for heading, s_line, e_line, sec_text in sections:
            words = sec_text.split()
            if not words:
                continue
            i = 0
            while i < len(words):
                j = min(len(words), i + self.chunk_size)
                token_slice = words[i:j]
                piece = " ".join(token_slice)
                if len(piece) > self.max_chunk_chars:
                    piece = piece[: self.max_chunk_chars]
                chunks.append(
                    {
                        "heading": heading,
                        "start_line": s_line,
                        "end_line": e_line,
                        "text": piece,
                    }
                )
                if j >= len(words):
                    break
                i = max(0, j - self.chunk_overlap)
        return chunks

    @staticmethod
    def _chunk_id(snapshot_id: str, rel_path: str, idx: int) -> str:
        payload = f"{snapshot_id}:{rel_path}:{idx}"
        return f"docchunk:{hashlib.md5(payload.encode('utf-8')).hexdigest()[:24]}"

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            v = self.embedder.embed_text(text)
            if v is None:
                return None
            return [float(x) for x in v]
        except Exception:
            return None

    @staticmethod
    def _detect_doc_type(rel_path: str) -> str:
        path = rel_path.lower()
        if path.startswith("docs/design/") or "/design/" in path:
            return "design"
        if path.startswith("docs/research/") or "/research/" in path:
            return "research"
        if path.startswith("docs/adr/") or "/adr/" in path or "/decisions/" in path:
            return "adr"
        if path.startswith("docs/rfc/") or "/rfc/" in path:
            return "rfc"
        if os.path.basename(path).startswith("readme"):
            return "readme"
        return "doc"

    def _extract_mentions(self, snapshot: IRSnapshot, chunks: Iterable[DocChunk]) -> List[Dict[str, Any]]:
        symbols = []
        for sym in snapshot.symbols:
            name = (sym.display_name or "").strip()
            if len(name) < 3:
                continue
            symbols.append((sym.symbol_id, name))

        mentions: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = chunk.text
            for symbol_id, name in symbols:
                if re.search(rf"\b{re.escape(name)}\b", text):
                    mentions.append(
                        {
                            "snapshot_id": chunk.snapshot_id,
                            "chunk_id": chunk.chunk_id,
                            "symbol_id": symbol_id,
                            "symbol_name": name,
                            "confidence": "heuristic",
                        }
                    )
        # dedupe
        dedup = {}
        for m in mentions:
            dedup[(m["chunk_id"], m["symbol_id"])] = m
        return list(dedup.values())
