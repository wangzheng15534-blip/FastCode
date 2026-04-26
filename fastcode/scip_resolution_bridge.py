"""
SCIP Resolution Bridge: 3-strategy cascade for mapping doc entity mentions
to code SCIP symbol IDs.

Strategies (evaluated in order, first match wins):
1. Lexical exact  -- high confidence (0.95), low recall
2. Namespace contextual -- medium confidence (0.70)
3. Vector semantic -- low confidence (clamped [0.3, 0.7]), high recall
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from .semantic_ir import IRSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionResult:
    symbol_id: str
    confidence: float
    match_strategy: str  # "lexical" | "namespace" | "semantic"
    match_text: str


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class SCIPResolutionBridge:
    """Resolve free-text doc mentions to code SCIP symbols via a 3-strategy cascade."""

    def __init__(
        self,
        snapshot: IRSnapshot,
        embedder: Any | None = None,
        semantic_threshold: float = 0.7,
        semantic_confidence_floor: float = 0.3,
        semantic_confidence_ceil: float = 0.7,
    ) -> None:
        self.snapshot = snapshot
        self.embedder = embedder
        self.semantic_threshold = semantic_threshold
        self.semantic_confidence_floor = semantic_confidence_floor
        self.semantic_confidence_ceil = semantic_confidence_ceil
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build lookup structures from snapshot units."""
        # display_name -> [unit_id, ...]
        self._name_index: dict[str, list[str]] = {}
        # qualified_name -> unit_id
        self._qualified_index: dict[str, str] = {}
        # directory -> [unit_id, ...]  (namespace grouping)
        self._dir_index: dict[str, list[str]] = {}
        # unit_id -> display_name  (reverse lookup for strategy 2)
        self._unit_name: dict[str, str] = {}

        for unit in self.snapshot.units:
            if unit.kind in {"file", "doc"}:
                continue

            name = (unit.display_name or "").strip()
            if len(name) < 3:
                continue

            uid = unit.unit_id
            self._name_index.setdefault(name, []).append(uid)
            self._unit_name[uid] = name

            if unit.qualified_name:
                self._qualified_index[unit.qualified_name] = uid

            # Namespace: directory of the file path
            parent_dir = os.path.dirname(unit.path)
            if parent_dir:
                self._dir_index.setdefault(parent_dir, []).append(uid)

        # Pre-compute embedding matrix if embedder available
        self._embedding_texts: list[str] = []
        self._embedding_unit_ids: list[str] = []
        self._embedding_vectors: Any | None = None

        if self.embedder is not None:
            self._precompute_embeddings()

    def _precompute_embeddings(self) -> None:
        """Pre-compute embedding vectors for all symbol display names."""
        # Deduplicate display names for embedding efficiency
        seen_names: set[str] = set()
        for uid, name in self._unit_name.items():
            if name not in seen_names:
                seen_names.add(name)
                self._embedding_texts.append(name)
                self._embedding_unit_ids.append(uid)

        if not self._embedding_texts:
            return

        try:
            batch = self.embedder.embed_batch(self._embedding_texts)
            # Normalise to list-of-lists so we stay serialisable
            self._embedding_vectors = [[float(v) for v in vec] for vec in batch]
        except Exception as exc:
            logger.warning(
                "SCIPResolutionBridge: embedding pre-computation failed: %s", exc
            )
            self._embedding_vectors = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        mention_text: str,
        context_path: str | None = None,
    ) -> ResolutionResult | None:
        """Resolve a doc mention to a code symbol using 3-strategy cascade.

        Parameters
        ----------
        mention_text:
            The free-text mention found in a document (e.g. "BranchIndexer").
        context_path:
            Optional file path of the document containing the mention, used
            for namespace contextual matching (strategy 2).
        """
        mention = (mention_text or "").strip()
        if len(mention) < 3:
            return None

        # Strategy 1: Lexical exact
        result = self._resolve_lexical(mention)
        if result is not None:
            return result

        # Strategy 2: Namespace contextual
        result = self._resolve_namespace(mention, context_path)
        if result is not None:
            return result

        # Strategy 3: Vector semantic
        if self._embedding_vectors is not None:
            result = self._resolve_semantic(mention)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Strategy 1 -- Lexical exact (confidence 0.95)
    # ------------------------------------------------------------------

    def _resolve_lexical(self, mention: str) -> ResolutionResult | None:
        # Exact display_name match
        uids = self._name_index.get(mention)
        if uids:
            return ResolutionResult(
                symbol_id=uids[0],
                confidence=0.95,
                match_strategy="lexical",
                match_text=mention,
            )
        # Exact qualified_name match
        uid = self._qualified_index.get(mention)
        if uid:
            return ResolutionResult(
                symbol_id=uid,
                confidence=0.95,
                match_strategy="lexical",
                match_text=mention,
            )
        return None

    # ------------------------------------------------------------------
    # Strategy 2 -- Namespace contextual (confidence 0.70)
    # ------------------------------------------------------------------

    def _resolve_namespace(
        self,
        mention: str,
        context_path: str | None,
    ) -> ResolutionResult | None:
        if not context_path:
            return None

        parent_dir = os.path.dirname(context_path)
        if not parent_dir:
            return None

        # Collect units in the same directory
        candidate_uids = self._dir_index.get(parent_dir)
        if not candidate_uids:
            return None

        for uid in candidate_uids:
            if self._unit_name.get(uid) == mention:
                return ResolutionResult(
                    symbol_id=uid,
                    confidence=0.70,
                    match_strategy="namespace",
                    match_text=mention,
                )
        return None

    # ------------------------------------------------------------------
    # Strategy 3 -- Vector semantic (confidence clamped [floor, ceil])
    # ------------------------------------------------------------------

    def _resolve_semantic(self, mention: str) -> ResolutionResult | None:
        if self._embedding_vectors is None or self.embedder is None:
            return None

        try:
            query_vec = self.embedder.embed_text(mention)
            if query_vec is None:
                return None
            query_list = [float(v) for v in query_vec]
        except Exception:
            return None

        best_sim = -1.0
        best_idx = -1

        for i, stored_vec in enumerate(self._embedding_vectors):
            sim = _cosine_similarity(query_list, stored_vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx < 0 or best_sim < self.semantic_threshold:
            return None

        confidence = max(
            self.semantic_confidence_floor, min(self.semantic_confidence_ceil, best_sim)
        )
        uid = self._embedding_unit_ids[best_idx]
        return ResolutionResult(
            symbol_id=uid,
            confidence=confidence,
            match_strategy="semantic",
            match_text=mention,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
