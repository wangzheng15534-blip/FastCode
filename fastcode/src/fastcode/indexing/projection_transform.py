"""
Deterministic transform layer for L0/L1/L2 projection generation.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from typing import Any, cast

from ..ir.graph import IRGraphs
from ..ir.projection import ProjectionBuildResult, ProjectionScope
from ..ir.types import IRSnapshot
from ..utils.clock import utc_now

try:
    import igraph as ig  # type: ignore[import-untyped]
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


class _ProjectionGraph:
    """Projection-native graph with compact side tables for attrs and weights."""

    def __init__(self, *, directed: bool = False) -> None:
        self.directed = directed
        self._node_attrs: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str], dict[str, Any]] = {}
        self._out: dict[str, set[str]] = defaultdict(set)
        self._in: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node: str, **attrs: Any) -> None:
        node_id = str(node)
        existing = self._node_attrs.setdefault(node_id, {})
        existing.update(
            {key: value for key, value in attrs.items() if value is not None}
        )
        self._out.setdefault(node_id, set())
        self._in.setdefault(node_id, set())

    def _edge_key(self, src: str, dst: str) -> tuple[str, str]:
        if self.directed or src <= dst:
            return src, dst
        return dst, src

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        src_id = str(src)
        dst_id = str(dst)
        self.add_node(src_id)
        self.add_node(dst_id)
        key = self._edge_key(src_id, dst_id)
        if key not in self._edges:
            self._edges[key] = dict(attrs)
        else:
            self._edges[key].update(attrs)
        self._out[src_id].add(dst_id)
        self._in[dst_id].add(src_id)
        if not self.directed:
            self._out[dst_id].add(src_id)
            self._in[src_id].add(dst_id)

    def add_weighted_edge(
        self, src: str, dst: str, *, weight: float, edge_type: str, source: str
    ) -> None:
        if src not in self._node_attrs:
            self.add_node(src)
        if dst not in self._node_attrs:
            self.add_node(dst)
        attrs = self.edge_attrs(src, dst)
        if attrs is None:
            self.add_edge(
                src,
                dst,
                weight=float(weight),
                edge_types={edge_type},
                source=source,
            )
            return
        attrs["weight"] = float(attrs.get("weight", 0.0)) + float(weight)
        raw_types = attrs.setdefault("edge_types", set())
        if isinstance(raw_types, set):
            raw_types.add(edge_type)
        else:
            attrs["edge_types"] = {str(raw_types), edge_type}

    def __contains__(self, node: object) -> bool:
        return isinstance(node, str) and node in self._node_attrs

    def nodes(self, data: bool = False) -> list[Any]:
        if data:
            return [
                (node, dict(self._node_attrs[node]))
                for node in sorted(self._node_attrs)
            ]
        return sorted(self._node_attrs)

    def edges(self, data: bool = False) -> list[Any]:
        rows: list[Any] = []
        for src, dst in sorted(self._edges):
            if data:
                rows.append((src, dst, dict(self._edges[(src, dst)])))
            else:
                rows.append((src, dst))
        return rows

    def number_of_nodes(self) -> int:
        return len(self._node_attrs)

    def number_of_edges(self) -> int:
        return len(self._edges)

    def has_edge(self, src: str, dst: str) -> bool:
        return self._edge_key(str(src), str(dst)) in self._edges

    def edge_attrs(self, src: str, dst: str) -> dict[str, Any] | None:
        return self._edges.get(self._edge_key(str(src), str(dst)))

    def successors(self, node: str) -> list[str]:
        return sorted(self._out.get(str(node), set()))

    def predecessors(self, node: str) -> list[str]:
        return sorted(self._in.get(str(node), set()))

    def neighbors(self, node: str, *, mode: str = "all") -> list[str]:
        node_id = str(node)
        if mode == "out":
            return self.successors(node_id)
        if mode == "in":
            return self.predecessors(node_id)
        return sorted(self._out.get(node_id, set()) | self._in.get(node_id, set()))

    def degree(self, node: str | None = None) -> Any:
        if node is not None:
            return len(self.neighbors(str(node), mode="all"))
        return [
            (node_id, len(self.neighbors(node_id, mode="all")))
            for node_id in self.nodes()
        ]

    def out_degree(self, node: str) -> int:
        return len(self._out.get(str(node), set()))

    def in_degree(self, node: str) -> int:
        return len(self._in.get(str(node), set()))

    def subgraph(self, scoped_nodes: set[str]) -> _ProjectionGraph:
        scoped = {str(node) for node in scoped_nodes}
        graph = _ProjectionGraph(directed=self.directed)
        for node in sorted(scoped):
            if node in self._node_attrs:
                graph.add_node(node, **self._node_attrs[node])
        for src, dst, attrs in self.edges(data=True):
            if src in scoped and dst in scoped:
                graph.add_edge(src, dst, **attrs)
        return graph

    def copy(self) -> _ProjectionGraph:
        return self.subgraph(set(self._node_attrs))

    def connected_components(self) -> list[set[str]]:
        remaining = set(self._node_attrs)
        components: list[set[str]] = []
        while remaining:
            seed = min(remaining)
            component = {seed}
            queue: deque[str] = deque([seed])
            remaining.remove(seed)
            while queue:
                node = queue.popleft()
                for neighbor in self.neighbors(node, mode="all"):
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
            components.append(component)
        components.sort(
            key=lambda items: (-len(items), sorted(items)[0] if items else "")
        )
        return components

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        max_hops: int | None = None,
        undirected: bool = False,
    ) -> list[str]:
        source_id = str(source)
        target_id = str(target)
        if source_id not in self._node_attrs or target_id not in self._node_attrs:
            return []
        if source_id == target_id:
            return [source_id]
        queue: deque[list[str]] = deque([[source_id]])
        visited = {source_id}
        while queue:
            path = queue.popleft()
            if max_hops is not None and len(path) - 1 >= max_hops:
                continue
            node = path[-1]
            neighbors = self.neighbors(node, mode="all" if undirected else "out")
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                next_path = [*path, neighbor]
                if neighbor == target_id:
                    return next_path
                visited.add(neighbor)
                queue.append(next_path)
        return []

    def distances_within(self, seed: str, max_hops: int) -> dict[str, int]:
        seed_id = str(seed)
        if seed_id not in self._node_attrs:
            return {}
        distances: dict[str, int] = {seed_id: 0}
        queue: deque[str] = deque([seed_id])
        while queue:
            node = queue.popleft()
            distance = distances[node]
            if distance >= max_hops:
                continue
            for neighbor in self.neighbors(node, mode="all"):
                if neighbor in distances:
                    continue
                distances[neighbor] = distance + 1
                queue.append(neighbor)
        distances.pop(seed_id, None)
        return distances

    def pagerank(self) -> dict[str, float]:
        nodes = self.nodes()
        if not nodes:
            return {}
        if self.number_of_edges() == 0:
            value = 1.0 / len(nodes)
            return dict.fromkeys(nodes, value)
        if ig is not None:
            graph, names = self.to_igraph(directed=self.directed)
            values = graph.pagerank(directed=self.directed)
            return {names[index]: float(value) for index, value in enumerate(values)}
        value = 1.0 / len(nodes)
        return dict.fromkeys(nodes, value)

    def to_igraph(self, *, directed: bool | None = None) -> tuple[Any, list[str]]:
        graph_directed = self.directed if directed is None else directed
        names = self.nodes()
        index = {node: pos for pos, node in enumerate(names)}
        graph: Any = ig.Graph(directed=graph_directed) if ig is not None else None
        if graph is None:
            raise RuntimeError(
                "python-igraph is required for projection graph conversion"
            )
        graph.add_vertices(len(names))
        graph.vs["name"] = names
        edges = [(index[src], index[dst]) for src, dst in self.edges()]
        if edges:
            graph.add_edges(edges)
            graph.es["weight"] = [
                float(attrs.get("weight", 1.0))
                for _src, _dst, attrs in self.edges(data=True)
            ]
        return graph, names


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
        gen_cfg = config.get("generation", {})
        self._llm_client = None
        self._llm_model = gen_cfg.get("model")
        self._llm_base_url = gen_cfg.get("base_url")
        self._llm_api_key = gen_cfg.get("openai_api_key")
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

        sg = g.subgraph(scoped_nodes)
        sdg = dg.subgraph(scoped_nodes)
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
            warnings.append("python_igraph_not_available_using_native_fallback")
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
    ) -> _ProjectionGraph:
        g = _ProjectionGraph(directed=False)
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
            if e.src_id not in g or e.dst_id not in g:
                continue
            wt = float(self.edge_weights.get(e.edge_type, 1.0))
            g.add_weighted_edge(
                e.src_id,
                e.dst_id,
                weight=wt,
                edge_type=e.edge_type,
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
                    wt = float(self.edge_weights.get(edge_type, 1.0))
                    g.add_weighted_edge(
                        str(src),
                        str(dst),
                        weight=wt,
                        edge_type=edge_type,
                        source="ir_graph",
                    )

        if g.number_of_edges() == 0:
            docs_by_path = {d.path: d.doc_id for d in docs_by_id.values()}
            for sym in symbols_by_id.values():
                doc_id = docs_by_path.get(sym.path)
                if doc_id and doc_id in g:
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
    ) -> _ProjectionGraph:
        g = _ProjectionGraph(directed=True)
        for d in snapshot.documents:
            g.add_node(d.doc_id)
        for s in snapshot.symbols:
            g.add_node(s.symbol_id)

        def add_edge(src: str, dst: str, edge_type: str, source: str) -> None:
            wt = float(self.edge_weights.get(edge_type, 1.0))
            g.add_weighted_edge(
                src,
                dst,
                weight=wt,
                edge_type=edge_type,
                source=source,
            )

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
                if doc_id and doc_id in g:
                    g.add_edge(
                        s.symbol_id,
                        doc_id,
                        weight=1.0,
                        edge_types={"contain"},
                        source="fallback",
                    )
        return g

    def _scope_nodes(
        self, scope: ProjectionScope, snapshot: IRSnapshot, g: _ProjectionGraph
    ) -> tuple[set[str], set[str]]:
        all_nodes = set(g.nodes())
        if scope.scope_kind == "snapshot":
            return all_nodes, set()
        if scope.scope_kind == "entity":
            focus = self._resolve_entity_node(scope.target_id, snapshot, g)
            if not focus:
                return set(), set()
            nodes = {focus, *g.distances_within(focus, self.max_entity_hops)}
            return nodes, {focus}
        if scope.scope_kind == "query":
            terminals = self._query_terminals(scope.query or "", snapshot, g)
            if not terminals:
                return set(), set()
            if len(terminals) == 1:
                focus = next(iter(terminals))
                nodes = {focus, *g.distances_within(focus, self.max_query_hops)}
                return nodes, terminals
            nodes = self._bounded_terminal_connector(g, terminals)
            for terminal in list(terminals):
                nodes.add(terminal)
                nodes.update(g.distances_within(terminal, 1))
            if nodes:
                return nodes, terminals
            fallback_nodes: set[str] = set()
            for terminal in terminals:
                fallback_nodes.add(terminal)
                fallback_nodes.update(g.distances_within(terminal, self.max_query_hops))
            return fallback_nodes, terminals
        return all_nodes, set()

    @staticmethod
    def _prune_steiner_leaves(tree: Any, terminals: set[str]) -> Any:
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

    def _bounded_terminal_connector(
        self, g: _ProjectionGraph, terminals: set[str]
    ) -> set[str]:
        ordered = sorted(terminals)
        if not ordered:
            return set()
        tree_nodes: set[str] = {ordered[0]}
        for terminal in ordered[1:]:
            best_path: list[str] = []
            for existing in sorted(tree_nodes):
                path = g.shortest_path(
                    existing,
                    terminal,
                    max_hops=self.max_query_hops * max(1, len(ordered)),
                    undirected=True,
                )
                if path and (not best_path or len(path) < len(best_path)):
                    best_path = path
            if best_path:
                tree_nodes.update(best_path)
        return tree_nodes

    def _resolve_entity_node(
        self, target_id: str | None, snapshot: IRSnapshot, g: _ProjectionGraph
    ) -> str | None:
        if not target_id:
            return None
        if target_id in g:
            return target_id
        for sym in snapshot.symbols:
            if (
                target_id in {sym.symbol_id, sym.display_name, sym.path}
            ) and sym.symbol_id in g:
                return sym.symbol_id
        for doc in snapshot.documents:
            if target_id in (doc.doc_id, doc.path) and doc.doc_id in g:
                return doc.doc_id
        return None

    def _query_terminals(
        self, query: str, snapshot: IRSnapshot, g: _ProjectionGraph
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

    def _compress_hubs(self, g: _ProjectionGraph) -> int:
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
                attrs = g.edge_attrs(n, nb)
                if attrs is None:
                    continue
                attrs["compressed_by_hub"] = True
                attrs["weight"] = min(float(attrs.get("weight", 1.0)), 0.5)
        return hidden

    def _cluster_hierarchy(
        self, g: _ProjectionGraph
    ) -> tuple[dict[int, dict[str, set[str]]], str]:
        if g.number_of_nodes() == 1:
            node = next(iter(g.nodes()))
            return {0: {"c0": {node}}}, "single"
        if g.number_of_nodes() > self.hierarchy_max_nodes:
            return {0: self._cluster_nodes_fallback(g)}, "native_components_large_graph"
        if self.enable_leiden and ig is not None and g.number_of_edges() > 0:
            try:
                ig_g, nodes = g.to_igraph(directed=False)
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
        return {0: self._cluster_nodes_fallback(g)}, "native_components"

    @staticmethod
    def _cluster_nodes_fallback(g: _ProjectionGraph) -> dict[str, set[str]]:
        communities = g.connected_components()
        return {f"c{i}": set(c) for i, c in enumerate(communities)} or {
            "c0": set(g.nodes())
        }

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
        self, g: _ProjectionGraph, clusters: dict[str, set[str]]
    ) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
        reps: dict[str, str] = {}
        centrality: dict[str, dict[str, float]] = {}
        if g.number_of_nodes() == 0:
            return reps, centrality
        pr = g.pagerank()
        degree_scale = max(1, g.number_of_nodes() - 1)
        degree_cent = {node: float(g.degree(node)) / degree_scale for node in g.nodes()}
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
        g: _ProjectionGraph,
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
        cluster_nodes = set(clusters)
        edge_weights: dict[tuple[str, str], float] = defaultdict(float)
        for u, v, data in g.edges(data=True):
            cu = by_node.get(u)
            cv = by_node.get(v)
            if not cu or not cv or cu == cv:
                continue
            edge_weights[(cu, cv)] += float(data.get("weight", 1.0))
        root = self._find_root_cluster(clusters, focus_nodes)
        if not edge_weights:
            return [], root
        tree_edges: list[tuple[str, str]] = []
        visited: set[str] = {root}
        remaining: set[str] = cluster_nodes - visited
        while remaining:
            best: tuple[str, str] | None = None
            best_weight = -1.0
            for source, target in sorted(edge_weights):
                weight = edge_weights[(source, target)]
                if source in visited and target not in visited:
                    if weight > best_weight:
                        best = (source, target)
                        best_weight = weight
                elif (
                    target in visited and source not in visited and weight > best_weight
                ):
                    best = (target, source)
                    best_weight = weight
            if not best:
                weighted_degree: dict[str, float] = defaultdict(float)
                for (source, target), weight in edge_weights.items():
                    weighted_degree[source] += weight
                    weighted_degree[target] += weight
                next_cluster = max(
                    remaining,
                    key=lambda cluster_id: (
                        weighted_degree.get(cluster_id, 0.0),
                        len(clusters[cluster_id]),
                        cluster_id,
                    ),
                )
                tree_edges.append((root, next_cluster))
                visited.add(next_cluster)
                remaining.remove(next_cluster)
                continue
            tree_edges.append(best)
            visited.add(best[1])
            remaining = cluster_nodes - visited
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
        g: _ProjectionGraph,
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
        sg: _ProjectionGraph,
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
        sg: _ProjectionGraph,
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
        sg: _ProjectionGraph,
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
        sg: _ProjectionGraph,
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
