"""
Deterministic transform layer for L0/L1/L2 projection generation.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, cast

import networkx as nx

from .ir_graph_builder import IRGraphs
from .projection_models import ProjectionBuildResult, ProjectionScope
from .semantic_ir import IRSnapshot
from .utils import utc_now

try:
    import igraph as ig  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ig = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


def _stable_hash(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _clean_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(w) > 1]


class ProjectionTransformer:
    ALGO_VERSION = "algo_v2_hierarchical"

    def __init__(self, config: dict[str, Any]) -> None:
        proj_cfg = config.get("projection", {})
        self.max_entity_hops = int(proj_cfg.get("max_entity_hops", 2))
        self.max_query_hops = int(proj_cfg.get("max_query_hops", 2))
        self.max_chunk_count = int(proj_cfg.get("max_chunk_count", 64))
        self.enable_leiden = bool(proj_cfg.get("enable_leiden", True))
        self.hierarchical_leiden_enabled = bool(
            proj_cfg.get("hierarchical_leiden_enabled", False)
        )
        self.leiden_resolutions = [
            float(x) for x in (proj_cfg.get("leiden_resolutions") or [1.0])
        ]
        self.hierarchy_max_levels = int(proj_cfg.get("hierarchy_max_levels", 4))
        self.hierarchy_max_nodes = int(proj_cfg.get("hierarchy_max_nodes", 12000))
        self.steiner_prune = bool(proj_cfg.get("steiner_prune", True))
        self.aggregation_top_members = int(proj_cfg.get("aggregation_top_members", 8))
        self.max_supporting_docs_per_cluster = int(
            proj_cfg.get("max_supporting_docs_per_cluster", 5)
        )
        self.llm_enabled = bool(proj_cfg.get("llm_enabled", True))
        self.llm_timeout_seconds = int(proj_cfg.get("llm_timeout_seconds", 8))
        self.llm_max_tokens = int(proj_cfg.get("llm_max_tokens", 180))
        self.llm_temperature = float(proj_cfg.get("llm_temperature", 0.2))
        self._llm_client = None
        self._llm_model = os.getenv("MODEL")
        self._llm_base_url = os.getenv("BASE_URL")
        self._llm_api_key = os.getenv("OPENAI_API_KEY")
        if self.llm_enabled and OpenAI is not None and self._llm_model:
            try:
                self._llm_client = OpenAI(
                    api_key=self._llm_api_key, base_url=self._llm_base_url
                )
            except Exception:
                self._llm_client = None
        self.edge_weights = dict(
            {
                "contain": 4.0,
                "defines": 4.0,
                "owns": 4.0,
                "call": 2.0,
                "import": 2.0,
                "inherit": 2.0,
                "ref": 2.0,
                "reference": 2.0,
            },
            **cast(dict[str, float], proj_cfg.get("edge_weights", {}) or {}),
        )

    def build(
        self,
        scope: ProjectionScope,
        snapshot: IRSnapshot,
        ir_graphs: IRGraphs | None = None,
        doc_mentions: list[dict[str, Any]] | None = None,
    ) -> ProjectionBuildResult:
        warnings: list[str] = []
        g = self._build_weighted_graph(snapshot, ir_graphs)
        dg = self._build_directed_weighted_graph(snapshot, ir_graphs)
        if g.number_of_nodes() == 0:
            raise RuntimeError("projection generation failed: empty graph")

        scoped_nodes, focus_nodes = self._scope_nodes(scope, snapshot, g)
        if not scoped_nodes:
            scoped_nodes = set(g.nodes())
            warnings.append("scope_empty_fallback_to_full_graph")

        sg: nx.Graph[str] = g.subgraph(scoped_nodes).copy()
        sdg: Any = dg.subgraph(scoped_nodes).copy()
        hidden_edge_count = self._compress_hubs(sg)
        hierarchy_levels, cluster_method = self._cluster_hierarchy(sg)
        clusters, selected_level = self._select_cluster_level(hierarchy_levels)
        representatives, centrality = self._pick_representatives(sg, clusters)
        tree_edges, root_cluster = self._build_backbone_arborescence(
            sdg, clusters, focus_nodes
        )
        xrefs = self._cross_cluster_xrefs(sdg, clusters, limit=32)
        hierarchy_links = self._hierarchy_parent_links(hierarchy_levels, selected_level)

        projection_id = self._projection_id(scope)
        labels = self._cluster_labels(snapshot, clusters, representatives)
        l0_summary = self._build_l0_summary(
            scope, labels, sg.number_of_nodes(), sg.number_of_edges()
        )
        l1_summary = self._build_l1_summary(scope, labels, tree_edges, xrefs)
        llm_l0 = self._llm_rewrite_summary("L0", scope, labels, l0_summary)
        llm_l1 = self._llm_rewrite_summary("L1", scope, labels, l1_summary)
        if llm_l0:
            l0_summary = llm_l0
        elif self.llm_enabled:
            warnings.append("llm_l0_unavailable_fallback")
        if llm_l1:
            l1_summary = llm_l1
        elif self.llm_enabled:
            warnings.append("llm_l1_unavailable_fallback")

        l0 = self._envelope(
            layer="L0",
            kind="summary",
            node_id=f"proj:{projection_id}:l0",
            path=f"/projection/{projection_id}/l0",
            title=f"Projection {scope.scope_kind} L0",
            summary=l0_summary,
            source_refs=self._source_refs(snapshot, scope),
            content_extra={
                "tags": [scope.scope_kind, "projection", "overview"],
                "importance": min(1.0, 0.3 + (len(clusters) / 20.0)),
            },
            projection_meta=self._projection_meta(
                sg,
                xrefs,
                hidden_edge_count,
                cluster_method,
                parent_reason="community_coarsening",
                extra={
                    "algo_version": self.ALGO_VERSION,
                    "hierarchy_level": selected_level,
                    "root_cluster": root_cluster,
                },
            ),
        )

        sections: list[dict[str, Any]] = []
        navigation: list[dict[str, Any]] = []
        for cid, members in sorted(
            clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            label = labels.get(cid, f"Cluster {cid}")
            sections.append({"name": label, "text": f"{len(members)} nodes"})
            rep = representatives.get(cid)
            if rep:
                navigation.append(
                    {"label": label, "ref": self._source_ref_for_node(rep, snapshot)}
                )
        source_refs = self._source_refs(snapshot, scope)

        l1 = self._envelope(
            layer="L1",
            kind="summary",
            node_id=f"proj:{projection_id}:l1",
            path=f"/projection/{projection_id}/l1",
            title=f"Projection {scope.scope_kind} L1",
            summary=l1_summary,
            source_refs=source_refs,
            content_extra={
                "sections": sections,
                "relations": {
                    "xref": [
                        {
                            "id": f"{src}->{dst}",
                            "title": f"{src} -> {dst}",
                            "type": "xref",
                            "confidence": min(1.0, float(weight) / 4.0),
                        }
                        for src, dst, weight in xrefs
                    ],
                    "hierarchy": [
                        {
                            "child": child,
                            "parent": parent,
                            "level": level,
                        }
                        for child, parent, level in hierarchy_links
                    ],
                    "backbone": [
                        {"src": src, "dst": dst, "type": "tree"}
                        for src, dst in tree_edges
                    ],
                },
                "navigation": navigation,
                "decisions": [
                    f"cluster_method={cluster_method}",
                    f"backbone_edges={len(tree_edges)}",
                    f"hierarchy_level={selected_level}",
                    f"root_cluster={root_cluster}",
                ],
                "related_code": source_refs,
                "related_memory": [],
            },
            projection_meta=self._projection_meta(
                sg,
                xrefs,
                hidden_edge_count,
                cluster_method,
                parent_reason="backbone_arborescence",
                extra={
                    "algo_version": self.ALGO_VERSION,
                    "hierarchy_level": selected_level,
                    "root_cluster": root_cluster,
                },
            ),
        )

        chunks = self._build_l2_chunks(
            snapshot=snapshot,
            sg=sg,
            clusters=clusters,
            representatives=representatives,
            labels=labels,
            centrality=centrality,
            doc_mentions=doc_mentions,
        )
        l2_index = self._build_l2_index(
            projection_id=projection_id,
            snapshot=snapshot,
            scope=scope,
            chunks=chunks,
            sg=sg,
            xrefs=xrefs,
            hidden_edge_count=hidden_edge_count,
            projection_method=cluster_method,
        )

        if ig is None and self.enable_leiden:
            warnings.append("python_igraph_not_available_using_networkx_fallback")
        return ProjectionBuildResult(
            projection_id=projection_id,
            snapshot_id=scope.snapshot_id,
            scope_kind=scope.scope_kind,
            scope_key=scope.scope_key,
            l0=l0,
            l1=l1,
            l2_index=l2_index,
            chunks=chunks,
            warnings=warnings,
        )

    def _projection_id(self, scope: ProjectionScope) -> str:
        payload = (
            f"{scope.snapshot_id}|{scope.scope_kind}|{scope.scope_key}|"
            f"{scope.target_id or ''}|{scope.query or ''}|{sorted(scope.filters.items())}|{self.ALGO_VERSION}"
        )
        return f"proj_{_stable_hash(payload)[:20]}"

    def _build_weighted_graph(
        self, snapshot: IRSnapshot, ir_graphs: IRGraphs | None
    ) -> nx.Graph[str]:
        g: nx.Graph[str] = nx.Graph()
        docs_by_id = {d.doc_id: d for d in snapshot.documents}
        symbols_by_id = {s.symbol_id: s for s in snapshot.symbols}

        for d in snapshot.documents:
            g.add_node(
                d.doc_id,
                node_kind="document",
                title=d.path,
                path=d.path,
                language=d.language,
            )
        for s in snapshot.symbols:
            g.add_node(
                s.symbol_id,
                node_kind="symbol",
                title=s.display_name,
                path=s.path,
                language=s.language,
                symbol_kind=s.kind,
            )

        for e in snapshot.edges:
            if e.src_id not in g.nodes or e.dst_id not in g.nodes:
                continue
            wt = float(self.edge_weights.get(e.edge_type, 1.0))
            if g.has_edge(e.src_id, e.dst_id):
                g[e.src_id][e.dst_id]["weight"] += wt
                g[e.src_id][e.dst_id]["edge_types"].add(e.edge_type)
            else:
                g.add_edge(
                    e.src_id,
                    e.dst_id,
                    weight=wt,
                    edge_types={e.edge_type},
                    source=e.source,
                )

        if ir_graphs:
            for edge_type, graph in [
                ("import", ir_graphs.dependency_graph),
                ("call", ir_graphs.call_graph),
                ("inherit", ir_graphs.inheritance_graph),
                ("ref", ir_graphs.reference_graph),
                ("contain", ir_graphs.containment_graph),
            ]:
                for src, dst in graph.edges():
                    if src not in g.nodes or dst not in g.nodes:
                        continue
                    wt = float(self.edge_weights.get(edge_type, 1.0))
                    if g.has_edge(src, dst):
                        g[src][dst]["weight"] += wt
                        g[src][dst]["edge_types"].add(edge_type)
                    else:
                        g.add_edge(
                            src,
                            dst,
                            weight=wt,
                            edge_types={edge_type},
                            source="ir_graph",
                        )

        if g.number_of_edges() == 0:
            docs_by_path = {d.path: d.doc_id for d in docs_by_id.values()}
            for sym in symbols_by_id.values():
                doc_id = docs_by_path.get(sym.path)
                if doc_id and doc_id in g.nodes:
                    g.add_edge(
                        sym.symbol_id,
                        doc_id,
                        weight=1.0,
                        edge_types={"contain"},
                        source="fallback",
                    )
        return g

    def _build_directed_weighted_graph(
        self, snapshot: IRSnapshot, ir_graphs: IRGraphs | None
    ) -> nx.DiGraph[str]:
        g: nx.DiGraph[str] = nx.DiGraph()
        for d in snapshot.documents:
            g.add_node(d.doc_id)
        for s in snapshot.symbols:
            g.add_node(s.symbol_id)

        def add_edge(src: str, dst: str, edge_type: str, source: str) -> None:
            if src not in g.nodes or dst not in g.nodes:
                return
            wt = float(self.edge_weights.get(edge_type, 1.0))
            if g.has_edge(src, dst):
                g[src][dst]["weight"] += wt
                g[src][dst]["edge_types"].add(edge_type)
            else:
                g.add_edge(src, dst, weight=wt, edge_types={edge_type}, source=source)

        for e in snapshot.edges:
            add_edge(e.src_id, e.dst_id, e.edge_type, e.source)
        if ir_graphs:
            for edge_type, graph in [
                ("import", ir_graphs.dependency_graph),
                ("call", ir_graphs.call_graph),
                ("inherit", ir_graphs.inheritance_graph),
                ("ref", ir_graphs.reference_graph),
                ("contain", ir_graphs.containment_graph),
            ]:
                for src, dst in graph.edges():
                    add_edge(src, dst, edge_type, "ir_graph")

        if g.number_of_edges() == 0:
            docs_by_path = {d.path: d.doc_id for d in snapshot.documents}
            for s in snapshot.symbols:
                doc_id = docs_by_path.get(s.path)
                if doc_id and doc_id in g.nodes:
                    g.add_edge(
                        s.symbol_id,
                        doc_id,
                        weight=1.0,
                        edge_types={"contain"},
                        source="fallback",
                    )
        return g

    def _scope_nodes(
        self, scope: ProjectionScope, snapshot: IRSnapshot, g: nx.Graph[str]
    ) -> tuple[set[str], set[str]]:
        all_nodes = set(g.nodes())
        if scope.scope_kind == "snapshot":
            return all_nodes, set()
        if scope.scope_kind == "entity":
            focus = self._resolve_entity_node(scope.target_id, snapshot, g)
            if not focus:
                return set(), set()
            nodes = set(
                nx.single_source_shortest_path_length(
                    g, focus, cutoff=self.max_entity_hops
                ).keys()
            )
            return nodes, {focus}
        if scope.scope_kind == "query":
            terminals = self._query_terminals(scope.query or "", snapshot, g)
            if not terminals:
                return set(), set()
            if len(terminals) == 1:
                focus = next(iter(terminals))
                nodes = set(
                    nx.single_source_shortest_path_length(
                        g, focus, cutoff=self.max_query_hops
                    ).keys()
                )
                return nodes, terminals
            try:
                weighted = g.copy()
                for _src, _dst, data in weighted.edges(data=True):
                    data["distance"] = 1.0 / max(0.1, float(data.get("weight", 1.0)))
                _steiner = getattr(nx.approximation, "steiner_tree")
                _steiner_result: Any = _steiner(weighted, terminals, weight="distance")
                tree: nx.Graph[str] = cast(nx.Graph[str], _steiner_result)
                if self.steiner_prune:
                    tree = self._prune_steiner_leaves(tree, terminals)
                nodes = set(tree.nodes())
                for t in list(terminals):
                    nodes.update(
                        nx.single_source_shortest_path_length(g, t, cutoff=1).keys()
                    )
                return nodes, terminals
            except Exception:
                nodes: set[str] = set()
                for t in terminals:
                    nodes.update(
                        nx.single_source_shortest_path_length(
                            g, t, cutoff=self.max_query_hops
                        ).keys()
                    )
                return nodes, terminals
        return all_nodes, set()

    @staticmethod
    def _prune_steiner_leaves(
        tree: nx.Graph[str], terminals: set[str]
    ) -> nx.Graph[str]:
        pruned = tree.copy()
        changed = True
        while changed:
            changed = False
            for node in list(pruned.nodes()):
                if node in terminals:
                    continue
                if pruned.degree(node) == 1:
                    pruned.remove_node(node)
                    changed = True
        return pruned

    def _resolve_entity_node(
        self, target_id: str | None, snapshot: IRSnapshot, g: nx.Graph[str]
    ) -> str | None:
        if not target_id:
            return None
        if target_id in g.nodes:
            return target_id
        for sym in snapshot.symbols:
            if (
                target_id in {sym.symbol_id, sym.display_name, sym.path}
            ) and sym.symbol_id in g.nodes:
                return sym.symbol_id
        for doc in snapshot.documents:
            if target_id in (doc.doc_id, doc.path) and doc.doc_id in g.nodes:
                return doc.doc_id
        return None

    def _query_terminals(
        self, query: str, snapshot: IRSnapshot, g: nx.Graph[str]
    ) -> set[str]:
        del snapshot
        tokens = set(_clean_words(query))
        if not tokens:
            return set()
        scored: list[tuple[float, str]] = []
        for n, attrs in g.nodes(data=True):
            text = " ".join(
                [
                    str(attrs.get("title", "")),
                    str(attrs.get("path", "")),
                    str(attrs.get("symbol_kind", "")),
                ]
            ).lower()
            score = sum(1 for t in tokens if t in text)
            if score > 0:
                score = score + math.log1p(g.degree(n))
                scored.append((score, n))
        scored.sort(reverse=True)
        return {n for _, n in scored[:12]}

    def _compress_hubs(self, g: nx.Graph[str]) -> int:
        if g.number_of_nodes() < 20:
            return 0
        degrees = sorted([d for _, d in g.degree()])
        if not degrees:
            return 0
        idx = max(0, min(len(degrees) - 1, int(0.99 * (len(degrees) - 1))))
        threshold = max(30, degrees[idx])
        to_prune = [n for n, d in g.degree() if d > threshold]
        hidden = 0
        for n in to_prune:
            nbrs = list(g.neighbors(n))
            hidden += len(nbrs)
            for nb in nbrs:
                g[n][nb]["compressed_by_hub"] = True
                g[n][nb]["weight"] = min(float(g[n][nb].get("weight", 1.0)), 0.5)
        return hidden

    def _cluster_hierarchy(
        self, g: nx.Graph[str]
    ) -> tuple[dict[int, dict[str, set[str]]], str]:
        if g.number_of_nodes() == 1:
            node = next(iter(g.nodes()))
            return {0: {"c0": {node}}}, "single"
        if g.number_of_nodes() > self.hierarchy_max_nodes:
            return {0: self._cluster_nodes_fallback(g)}, "greedy_modularity_large_graph"
        if self.enable_leiden and ig is not None and g.number_of_edges() > 0:
            try:
                nodes = list(g.nodes())
                idx = {n: i for i, n in enumerate(nodes)}
                ig_g: Any = ig.Graph()
                ig_g.add_vertices(len(nodes))
                ig_edges = [(idx[u], idx[v]) for u, v in g.edges()]
                ig_g.add_edges(ig_edges)
                if (
                    not self.hierarchical_leiden_enabled
                    and len(self.leiden_resolutions) == 1
                ):
                    resolution = self.leiden_resolutions[0]
                    part: Any = ig_g.community_leiden(resolution_parameter=resolution)
                    clusters = {
                        f"c{ci}": {nodes[int(m)] for m in members}
                        for ci, members in enumerate(part)
                    }
                    return {0: clusters}, "leiden"
                if self.hierarchical_leiden_enabled:
                    levels: dict[int, dict[str, set[str]]] = {}
                    for level, resolution in enumerate(
                        self.leiden_resolutions[: self.hierarchy_max_levels]
                    ):
                        part = ig_g.community_leiden(resolution_parameter=resolution)
                        clusters = {
                            f"c{ci}": {nodes[int(m)] for m in members}
                            for ci, members in enumerate(part)
                        }
                        levels[level] = clusters
                        if len(clusters) <= 1:
                            break
                    if levels:
                        return levels, "hierarchical_leiden"
            except Exception:
                pass
        return {0: self._cluster_nodes_fallback(g)}, "greedy_modularity"

    @staticmethod
    def _cluster_nodes_fallback(g: nx.Graph[str]) -> dict[str, set[str]]:
        communities = list(nx.algorithms.community.greedy_modularity_communities(g))
        return (
            {f"c{i}": set(c) for i, c in enumerate(communities)}
            if communities
            else {"c0": set(g.nodes())}
        )

    @staticmethod
    def _select_cluster_level(
        hierarchy_levels: dict[int, dict[str, set[str]]],
    ) -> tuple[dict[str, set[str]], int]:
        target, min_count, max_count = 12, 6, 24
        selected_level = min(hierarchy_levels.keys())
        best_score = float("inf")
        for level, clusters in hierarchy_levels.items():
            c = len(clusters)
            if min_count <= c <= max_count:
                score = abs(c - target)
            elif c < min_count:
                score = (min_count - c) + 100
            else:
                score = (c - max_count) + 100
            if score < best_score:
                best_score = score
                selected_level = level
        return hierarchy_levels[selected_level], selected_level

    @staticmethod
    def _hierarchy_parent_links(
        hierarchy_levels: dict[int, dict[str, set[str]]], selected_level: int
    ) -> list[tuple[str, str, int]]:
        if (selected_level + 1) not in hierarchy_levels:
            return []
        parent_level = hierarchy_levels[selected_level]
        child_level = hierarchy_levels[selected_level + 1]
        parent_links: list[tuple[str, str, int]] = []
        for child_id, child_members in child_level.items():
            parent_id = None
            for pid, pmembers in parent_level.items():
                if child_members.issubset(pmembers):
                    parent_id = pid
                    break
            if parent_id:
                parent_links.append((child_id, parent_id, selected_level + 1))
        return parent_links

    def _pick_representatives(
        self, g: nx.Graph[str], clusters: dict[str, set[str]]
    ) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        reps: dict[str, str] = {}
        centrality: dict[str, dict[str, float]] = {}
        if g.number_of_nodes() == 0:
            return reps, centrality
        pr = (
            nx.pagerank(g, alpha=0.85)
            if g.number_of_edges()
            else dict.fromkeys(g.nodes(), 1.0)
        )
        degree_cent = (
            nx.degree_centrality(g)
            if g.number_of_nodes() > 1
            else dict.fromkeys(g.nodes(), 0.0)
        )
        for cid, nodes in clusters.items():
            rep = max(
                nodes,
                key=lambda n: (
                    pr.get(n, 0.0),
                    degree_cent.get(n, 0.0),
                    g.degree(n),
                ),
            )
            reps[cid] = rep
            centrality[cid] = {
                "pagerank": float(pr.get(rep, 0.0)),
                "degree_centrality": float(degree_cent.get(rep, 0.0)),
                "degree": float(g.degree(rep)),
            }
        return reps, centrality

    def _build_backbone_arborescence(
        self,
        g: nx.DiGraph[str],
        clusters: dict[str, set[str]],
        focus_nodes: set[str],
    ) -> tuple[list[tuple[str, str]], str]:
        if len(clusters) <= 1:
            root = next(iter(clusters.keys())) if clusters else "c0"
            return [], root
        by_node: dict[str, str] = {}
        for cid, members in clusters.items():
            for m in members:
                by_node[m] = cid
        cg: nx.DiGraph[str] = nx.DiGraph()
        for cid, members in clusters.items():
            cg.add_node(cid, size=len(members))
        for u, v, data in g.edges(data=True):
            cu = by_node.get(u)
            cv = by_node.get(v)
            if not cu or not cv or cu == cv:
                continue
            w = float(data.get("weight", 1.0))
            if cg.has_edge(cu, cv):
                cg[cu][cv]["weight"] += w
            else:
                cg.add_edge(cu, cv, weight=w)
        root = self._find_root_cluster(clusters, focus_nodes)
        if cg.number_of_edges() == 0:
            return [], root
        tree_edges: list[tuple[str, str]] = []
        visited: set[str] = {root}
        remaining: set[str] = set(cg.nodes()) - visited
        while remaining:
            best: tuple[str, str] | None = None
            best_weight = -1.0
            for u in list(visited):
                for _, v, data in cg.out_edges(u, data=True):
                    if v in visited:
                        continue
                    w = float(data.get("weight", 1.0))
                    if w > best_weight:
                        best = (u, v)
                        best_weight = w
                for v, _, data in cg.in_edges(u, data=True):
                    if v in visited:
                        continue
                    w = float(data.get("weight", 1.0))
                    if w > best_weight:
                        best = (u, v)
                        best_weight = w
            if not best:
                next_cluster: str = max(
                    remaining, key=lambda c: cg.out_degree(c) + cg.in_degree(c)
                )
                tree_edges.append((root, next_cluster))
                visited.add(next_cluster)
                remaining.remove(next_cluster)
                continue
            tree_edges.append(best)
            visited.add(best[1])
            remaining = set(cg.nodes()) - visited
        return tree_edges, root

    @staticmethod
    def _find_root_cluster(clusters: dict[str, set[str]], focus_nodes: set[str]) -> str:
        if focus_nodes:
            focus = next(iter(focus_nodes))
            for cid, members in clusters.items():
                if focus in members:
                    return cid
        return max(clusters.keys(), key=lambda c: len(clusters[c]))

    def _cross_cluster_xrefs(
        self,
        g: nx.DiGraph[str],
        clusters: dict[str, set[str]],
        limit: int = 32,
    ) -> list[tuple[str, str, float]]:
        by_node: dict[str, str] = {}
        for cid, members in clusters.items():
            for m in members:
                by_node[m] = cid
        edge_weights: dict[tuple[str, str], float] = defaultdict(float)
        for u, v, data in g.edges(data=True):
            cu = by_node.get(u)
            cv = by_node.get(v)
            if not cu or not cv or cu == cv:
                continue
            edge_weights[(cu, cv)] += float(data.get("weight", 1.0))
        rows = [(k[0], k[1], w) for k, w in edge_weights.items()]
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:limit]

    def _cluster_labels(
        self,
        snapshot: IRSnapshot,
        clusters: dict[str, set[str]],
        representatives: dict[str, str],
    ) -> dict[str, str]:
        labels: dict[str, str] = {}
        sym_map = {s.symbol_id: s for s in snapshot.symbols}
        doc_map = {d.doc_id: d for d in snapshot.documents}
        for cid, members in clusters.items():
            rep = representatives.get(cid)
            if not rep:
                labels[cid] = f"Cluster {cid}"
                continue
            if rep in sym_map:
                labels[cid] = f"{sym_map[rep].display_name} cluster"
            elif rep in doc_map:
                labels[cid] = f"{doc_map[rep].path} cluster"
            else:
                labels[cid] = f"Cluster {cid}"
            if len(members) > 1:
                labels[cid] = f"{labels[cid]} ({len(members)})"
        return labels

    def _build_l0_summary(
        self, scope: ProjectionScope, labels: dict[str, str], nodes: int, edges: int
    ) -> str:
        top = ", ".join(list(labels.values())[:3]) or "No clusters"
        return (
            f"{scope.scope_kind.capitalize()} projection over {nodes} nodes and {edges} edges. "
            f"Key communities: {top}."
        )

    def _build_l1_summary(
        self,
        scope: ProjectionScope,
        labels: dict[str, str],
        tree_edges: list[tuple[str, str]],
        xrefs: list[tuple[str, str, float]],
    ) -> str:
        return (
            f"{scope.scope_kind.capitalize()} hierarchy with {len(labels)} clusters, "
            f"{len(tree_edges)} directed backbone links, and {len(xrefs)} cross-links."
        )

    def _llm_rewrite_summary(
        self,
        layer: str,
        scope: ProjectionScope,
        labels: dict[str, str],
        fallback: str,
    ) -> str | None:
        if not self.llm_enabled:
            return None
        if self._llm_client is None or not self._llm_model:
            return None
        top_labels = ", ".join(list(labels.values())[:6])
        prompt = (
            f"Write a concise {layer} projection summary for code navigation. "
            f"Scope={scope.scope_kind}. Use 1-2 sentences. "
            f"Communities: {top_labels}. "
            f"Fallback summary: {fallback}"
        )
        try:
            resp = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                timeout=self.llm_timeout_seconds,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            return None
        except Exception:
            return None

    def _source_refs(
        self, snapshot: IRSnapshot, scope: ProjectionScope
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "repository",
                "id": snapshot.repo_name,
                "repo": snapshot.repo_name,
                "branch": snapshot.branch,
                "commit": snapshot.commit_id,
            },
            {
                "type": "commit",
                "id": snapshot.commit_id or snapshot.tree_id or snapshot.snapshot_id,
                "repo": snapshot.repo_name,
                "branch": snapshot.branch,
                "commit": snapshot.commit_id,
            },
            {
                "type": "document",
                "id": scope.snapshot_id,
                "repo": snapshot.repo_name,
                "branch": snapshot.branch,
                "commit": snapshot.commit_id,
                "label": f"scope:{scope.scope_kind}:{scope.scope_key}",
            },
        ]

    def _source_ref_for_node(
        self, node_id: str, snapshot: IRSnapshot
    ) -> dict[str, Any]:
        for s in snapshot.symbols:
            if s.symbol_id == node_id:
                return {
                    "type": "symbol",
                    "id": s.symbol_id,
                    "repo": snapshot.repo_name,
                    "branch": snapshot.branch,
                    "commit": snapshot.commit_id,
                    "path": s.path,
                    "label": s.display_name,
                }
        for d in snapshot.documents:
            if d.doc_id == node_id:
                return {
                    "type": "file",
                    "id": d.doc_id,
                    "repo": snapshot.repo_name,
                    "branch": snapshot.branch,
                    "commit": snapshot.commit_id,
                    "path": d.path,
                    "label": d.path,
                }
        return {"type": "document", "id": node_id}

    def _projection_meta(
        self,
        sg: nx.Graph[str],
        xrefs: Sequence[tuple[str, str, float]],
        hidden_edge_count: int,
        projection_method: str,
        parent_reason: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "updated_at": utc_now(),
            "covers_nodes": sorted(sg.nodes()),
            "covers_edges": sorted([f"{u}->{v}" for u, v in sg.edges()]),
            "xrefs": [{"src": s, "dst": d, "weight": w} for s, d, w in xrefs],
            "hidden_edge_count": int(hidden_edge_count),
            "projection_method": projection_method,
            "parent_reason": parent_reason,
        }
        if extra:
            payload.update(extra)
        return payload

    def _envelope(
        self,
        layer: str,
        kind: str,
        node_id: str,
        path: str,
        title: str,
        summary: str,
        source_refs: list[dict[str, Any]],
        content_extra: dict[str, Any],
        projection_meta: dict[str, Any],
    ) -> dict[str, Any]:
        content = {"summary": summary}
        content.update(content_extra)
        return {
            "version": "v1",
            "kind": kind,
            "layer": layer,
            "id": node_id,
            "path": path,
            "title": title,
            "source": {"domain": "code", "refs": source_refs},
            "content": content,
            "render": {"text": summary},
            "meta": projection_meta,
        }

    def _aggregate_cluster_attributes(
        self,
        snapshot: IRSnapshot,
        members: set[str],
        sg: nx.Graph[str],
        representative: str,
    ) -> dict[str, Any]:
        sym_map = {s.symbol_id: s for s in snapshot.symbols}
        doc_map = {d.doc_id: d for d in snapshot.documents}
        symbols = [sym_map[m] for m in members if m in sym_map]
        docs = [doc_map[m] for m in members if m in doc_map]

        lang_counter = Counter((s.language or "") for s in symbols if s.language)
        kind_counter = Counter((s.kind or "") for s in symbols if s.kind)
        paths = [p for p in [*(s.path for s in symbols), *(d.path for d in docs)] if p]
        common_prefix = os.path.commonpath(paths) if paths else ""

        edge_type_counter: Counter[str] = Counter()
        for u, v, data in sg.edges(data=True):
            if u not in members or v not in members:
                continue
            for t in sorted(data.get("edge_types", set())):
                edge_type_counter[t] += 1

        top_members = sorted(
            members,
            key=lambda n: (sg.degree(n), str(n)),
            reverse=True,
        )[: self.aggregation_top_members]

        return {
            "member_count": len(members),
            "symbol_count": len(symbols),
            "doc_count": len(docs),
            "dominant_language": lang_counter.most_common(1)[0][0]
            if lang_counter
            else None,
            "dominant_kind": kind_counter.most_common(1)[0][0]
            if kind_counter
            else None,
            "kind_distribution": dict(kind_counter),
            "edge_type_distribution": dict(edge_type_counter),
            "common_path_prefix": common_prefix,
            "representative": representative,
            "top_members": top_members,
        }

    def _build_l2_chunks(
        self,
        snapshot: IRSnapshot,
        sg: nx.Graph[str],
        clusters: dict[str, set[str]],
        representatives: dict[str, str],
        labels: dict[str, str],
        centrality: dict[str, dict[str, float]],
        doc_mentions: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        sym_map = {s.symbol_id: s for s in snapshot.symbols}
        doc_map = {d.doc_id: d for d in snapshot.documents}
        max_supporting = self.max_supporting_docs_per_cluster

        # Pre-index doc mentions by symbol_id for fast cluster lookup
        mentions_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if doc_mentions:
            for m in doc_mentions:
                sid = m.get("symbol_id")
                if sid:
                    mentions_by_symbol[sid].append(m)

        for cid, members in sorted(
            clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            if len(chunks) >= self.max_chunk_count:
                break
            rep = representatives.get(cid)
            if not rep:
                continue
            chunk_id = f"c_{cid}"
            refs = [self._source_ref_for_node(rep, snapshot)]
            agg = self._aggregate_cluster_attributes(
                snapshot, members, sg, representative=rep
            )
            facts = [
                {"type": "cluster_size", "value": len(members)},
                {"type": "cluster_label", "value": labels.get(cid)},
                {"type": "dominant_kind", "value": agg.get("dominant_kind")},
                {"type": "dominant_language", "value": agg.get("dominant_language")},
            ]
            supporting_docs = self._supporting_docs_for_cluster(
                members,
                mentions_by_symbol,
                max_supporting,
            )
            if rep in sym_map:
                sym = sym_map[rep]
                snippet = f"{sym.kind} {sym.display_name} in {sym.path}"
                signature = sym.signature or sym.qualified_name or sym.display_name
                content = {
                    "file": sym.path,
                    "range": {
                        "start_line": max(1, int(sym.start_line or 1)),
                        "start_col": max(0, int(sym.start_col or 0)),
                        "end_line": max(1, int(sym.end_line or sym.start_line or 1)),
                        "end_col": max(0, int(sym.end_col or 0)),
                    },
                    "symbol": sym.display_name,
                    "signature": signature,
                    "snippet": snippet,
                    "facts": facts,
                    "refs": refs,
                }
                if supporting_docs:
                    content["supporting_docs"] = supporting_docs
            elif rep in doc_map:
                doc = doc_map[rep]
                content = {
                    "file": doc.path,
                    "snippet": f"Document cluster for {doc.path}",
                    "facts": facts,
                    "refs": refs,
                }
                if supporting_docs:
                    content["supporting_docs"] = supporting_docs
            else:
                content = {
                    "snippet": f"Cluster representative {rep}",
                    "facts": facts,
                    "refs": refs,
                }
                if supporting_docs:
                    content["supporting_docs"] = supporting_docs
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "kind": "cluster_evidence",
                    "content": content,
                    "version": "v1",
                    "layer": "L2",
                    "id": f"chunk:{chunk_id}",
                    "path": f"./chunks/{chunk_id}.json",
                    "title": labels.get(cid) or chunk_id,
                    "source": {"domain": "code", "refs": refs},
                    "render": {"text": content.get("snippet") or ""},
                    "meta": {
                        "cluster_id": cid,
                        "centrality": centrality.get(cid, {}),
                        "aggregation": agg,
                    },
                }
            )
        return chunks

    @staticmethod
    def _supporting_docs_for_cluster(
        members: set[str],
        mentions_by_symbol: dict[str, list[dict[str, Any]]],
        max_docs: int,
    ) -> list[dict[str, str | list[str]]]:
        """Build supporting_docs list for a cluster from doc mentions."""
        chunk_mentions: dict[str, list[str]] = defaultdict(list)
        for sid in members:
            for m in mentions_by_symbol.get(sid, []):
                cid = m.get("chunk_id")
                if cid:
                    chunk_mentions[cid].append(m.get("symbol_name", ""))
        # Sort by mention count descending, take top max_docs
        ranked = sorted(chunk_mentions.items(), key=lambda kv: -len(kv[1]))[:max_docs]
        return [
            {"chunk_id": cid, "mentioned_symbols": list(dict.fromkeys(names))}
            for cid, names in ranked
        ]

    def _build_l2_index(
        self,
        projection_id: str,
        snapshot: IRSnapshot,
        scope: ProjectionScope,
        chunks: list[dict[str, Any]],
        sg: nx.Graph[str],
        xrefs: Sequence[tuple[str, str, float]],
        hidden_edge_count: int,
        projection_method: str,
    ) -> dict[str, Any]:
        chunk_rows: list[dict[str, Any]] = []
        for c in chunks:
            c_content: dict[str, Any] = c.get("content", {})
            c_range: dict[str, Any] = c_content.get("range", {}) or {}
            chunk_rows.append(
                {
                    "chunk_id": c["chunk_id"],
                    "kind": c.get("kind", "evidence"),
                    "path": f"./chunks/{c['chunk_id']}.json",
                    "file": c_content.get("file"),
                    "start_line": c_range.get("start_line"),
                    "end_line": c_range.get("end_line"),
                    "label": c_content.get("symbol")
                    or c_content.get("file")
                    or c["chunk_id"],
                }
            )
        summary = f"Detailed evidence available in {len(chunk_rows)} chunks."
        return {
            "version": "v1",
            "kind": "summary",
            "layer": "L2",
            "id": f"proj:{projection_id}:l2",
            "path": f"/projection/{projection_id}/l2",
            "title": f"Projection {scope.scope_kind} L2",
            "source": {"domain": "code", "refs": self._source_refs(snapshot, scope)},
            "content": {"chunks": chunk_rows},
            "render": {"text": summary},
            "meta": self._projection_meta(
                sg=sg,
                xrefs=xrefs,
                hidden_edge_count=hidden_edge_count,
                projection_method=projection_method,
                parent_reason="chunked_evidence",
                extra={"algo_version": self.ALGO_VERSION},
            ),
        }
