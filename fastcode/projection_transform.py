"""
Deterministic transform layer for L0/L1/L2 projection generation.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from .ir_graph_builder import IRGraphs
from .projection_models import ProjectionBuildResult, ProjectionScope
from .semantic_ir import IRSnapshot

try:
    import igraph as ig  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ig = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _clean_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(w) > 1]


class ProjectionTransformer:
    def __init__(self, config: Dict[str, Any]):
        proj_cfg = config.get("projection", {})
        self.max_entity_hops = int(proj_cfg.get("max_entity_hops", 2))
        self.max_query_hops = int(proj_cfg.get("max_query_hops", 2))
        self.max_chunk_count = int(proj_cfg.get("max_chunk_count", 64))
        self.enable_leiden = bool(proj_cfg.get("enable_leiden", True))
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
                self._llm_client = OpenAI(api_key=self._llm_api_key, base_url=self._llm_base_url)
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
            **(proj_cfg.get("edge_weights", {}) or {}),
        )

    def build(
        self,
        scope: ProjectionScope,
        snapshot: IRSnapshot,
        ir_graphs: Optional[IRGraphs] = None,
    ) -> ProjectionBuildResult:
        warnings: List[str] = []
        g = self._build_weighted_graph(snapshot, ir_graphs)
        if g.number_of_nodes() == 0:
            raise RuntimeError("projection generation failed: empty graph")

        scoped_nodes, focus_nodes = self._scope_nodes(scope, snapshot, g)
        if not scoped_nodes:
            scoped_nodes = set(g.nodes())
            warnings.append("scope_empty_fallback_to_full_graph")

        sg = g.subgraph(scoped_nodes).copy()
        hidden_edge_count = self._compress_hubs(sg)
        clusters, cluster_method = self._cluster_nodes(sg)
        representatives = self._pick_representatives(sg, clusters)
        tree_edges = self._build_backbone_tree(sg, clusters, focus_nodes)
        xrefs = self._cross_cluster_xrefs(sg, clusters, limit=32)

        projection_id = self._projection_id(scope)
        labels = self._cluster_labels(snapshot, clusters, representatives)
        l0_summary = self._build_l0_summary(scope, labels, sg.number_of_nodes(), sg.number_of_edges())
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
            ),
        )

        sections = []
        navigation = []
        for cid, members in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            label = labels.get(cid, f"Cluster {cid}")
            sections.append({"name": label, "text": f"{len(members)} nodes"})
            rep = representatives.get(cid)
            if rep:
                navigation.append(
                    {
                        "label": label,
                        "ref": self._source_ref_for_node(rep, snapshot),
                    }
                )

        l1_relations: Dict[str, Any] = {
            "cross_links": [
                {
                    "id": f"{src}->{dst}",
                    "title": f"{src} -> {dst}",
                    "type": "xref",
                    "confidence": min(1.0, float(weight) / 4.0),
                }
                for src, dst, weight in xrefs
            ]
        }
        l1_relations_v2: Dict[str, Any] = {
            "xref": [
                {
                    "id": f"{src}->{dst}",
                    "title": f"{src} -> {dst}",
                    "type": "xref",
                    "confidence": min(1.0, float(weight) / 4.0),
                }
                for src, dst, weight in xrefs
            ]
        }
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
                "relations": l1_relations,
                "relations_v2": l1_relations_v2,
                "navigation": navigation,
                "decisions": [
                    f"cluster_method={cluster_method}",
                    f"backbone_edges={len(tree_edges)}",
                ],
                "related_code": source_refs,
                "related_memory": [],
            },
            projection_meta=self._projection_meta(
                sg,
                xrefs,
                hidden_edge_count,
                cluster_method,
                parent_reason="backbone_tree",
            ),
        )

        chunks = self._build_l2_chunks(
            snapshot=snapshot,
            sg=sg,
            clusters=clusters,
            representatives=representatives,
            labels=labels,
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
            f"{scope.target_id or ''}|{scope.query or ''}|{sorted(scope.filters.items())}"
        )
        return f"proj_{_stable_hash(payload)[:20]}"

    def _build_weighted_graph(self, snapshot: IRSnapshot, ir_graphs: Optional[IRGraphs]) -> nx.Graph:
        g = nx.Graph()
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
                        g.add_edge(src, dst, weight=wt, edge_types={edge_type}, source="ir_graph")

        if g.number_of_edges() == 0:
            # Minimal connectivity fallback: connect each symbol to its file document.
            docs_by_path = {d.path: d.doc_id for d in docs_by_id.values()}
            for sym in symbols_by_id.values():
                doc_id = docs_by_path.get(sym.path)
                if doc_id and doc_id in g.nodes:
                    g.add_edge(sym.symbol_id, doc_id, weight=1.0, edge_types={"contain"}, source="fallback")
        return g

    def _scope_nodes(self, scope: ProjectionScope, snapshot: IRSnapshot, g: nx.Graph) -> Tuple[Set[str], Set[str]]:
        all_nodes = set(g.nodes())
        if scope.scope_kind == "snapshot":
            return all_nodes, set()

        if scope.scope_kind == "entity":
            focus = self._resolve_entity_node(scope.target_id, snapshot, g)
            if not focus:
                return set(), set()
            nodes = set(nx.single_source_shortest_path_length(g, focus, cutoff=self.max_entity_hops).keys())
            return nodes, {focus}

        if scope.scope_kind == "query":
            terminals = self._query_terminals(scope.query or "", snapshot, g)
            if not terminals:
                return set(), set()
            if len(terminals) == 1:
                focus = next(iter(terminals))
                nodes = set(nx.single_source_shortest_path_length(g, focus, cutoff=self.max_query_hops).keys())
                return nodes, terminals

            try:
                for src, dst, data in g.edges(data=True):
                    data["distance"] = 1.0 / max(0.1, float(data.get("weight", 1.0)))
                tree = nx.approximation.steiner_tree(g, terminals, weight="distance")
                nodes = set(tree.nodes())
                for t in list(terminals):
                    nodes.update(nx.single_source_shortest_path_length(g, t, cutoff=1).keys())
                return nodes, terminals
            except Exception:
                nodes = set()
                for t in terminals:
                    nodes.update(nx.single_source_shortest_path_length(g, t, cutoff=self.max_query_hops).keys())
                return nodes, terminals

        return all_nodes, set()

    def _resolve_entity_node(self, target_id: Optional[str], snapshot: IRSnapshot, g: nx.Graph) -> Optional[str]:
        if not target_id:
            return None
        if target_id in g.nodes:
            return target_id

        for sym in snapshot.symbols:
            if sym.symbol_id == target_id or sym.display_name == target_id or sym.path == target_id:
                if sym.symbol_id in g.nodes:
                    return sym.symbol_id
        for doc in snapshot.documents:
            if doc.doc_id == target_id or doc.path == target_id:
                if doc.doc_id in g.nodes:
                    return doc.doc_id
        return None

    def _query_terminals(self, query: str, snapshot: IRSnapshot, g: nx.Graph) -> Set[str]:
        tokens = set(_clean_words(query))
        if not tokens:
            return set()
        scored: List[Tuple[float, str]] = []
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

    def _compress_hubs(self, g: nx.Graph) -> int:
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

    def _cluster_nodes(self, g: nx.Graph) -> Tuple[Dict[str, Set[str]], str]:
        if g.number_of_nodes() == 1:
            node = next(iter(g.nodes()))
            return {"c0": {node}}, "single"

        if self.enable_leiden and ig is not None and g.number_of_edges() > 0:
            try:
                nodes = list(g.nodes())
                idx = {n: i for i, n in enumerate(nodes)}
                ig_g = ig.Graph()
                ig_g.add_vertices(len(nodes))
                ig_edges = [(idx[u], idx[v]) for u, v in g.edges()]
                ig_g.add_edges(ig_edges)
                part = ig_g.community_leiden()
                clusters: Dict[str, Set[str]] = {}
                for ci, members in enumerate(part):
                    clusters[f"c{ci}"] = {nodes[m] for m in members}
                return clusters, "leiden"
            except Exception:
                pass

        communities = list(nx.algorithms.community.greedy_modularity_communities(g))
        clusters = {f"c{i}": set(c) for i, c in enumerate(communities)} if communities else {"c0": set(g.nodes())}
        return clusters, "greedy_modularity"

    def _pick_representatives(self, g: nx.Graph, clusters: Dict[str, Set[str]]) -> Dict[str, str]:
        reps: Dict[str, str] = {}
        if g.number_of_nodes() == 0:
            return reps
        pr = nx.pagerank(g, alpha=0.85) if g.number_of_edges() else {n: 1.0 for n in g.nodes()}
        for cid, nodes in clusters.items():
            reps[cid] = max(nodes, key=lambda n: (pr.get(n, 0.0), g.degree(n)))
        return reps

    def _build_backbone_tree(
        self,
        g: nx.Graph,
        clusters: Dict[str, Set[str]],
        focus_nodes: Set[str],
    ) -> List[Tuple[str, str]]:
        if len(clusters) <= 1:
            return []
        by_node = {}
        for cid, members in clusters.items():
            for m in members:
                by_node[m] = cid

        cg = nx.Graph()
        for cid in clusters.keys():
            cg.add_node(cid, size=len(clusters[cid]))
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

        if cg.number_of_edges() == 0:
            return []
        mst = nx.maximum_spanning_tree(cg, weight="weight")

        root = None
        if focus_nodes:
            focus = next(iter(focus_nodes))
            for cid, members in clusters.items():
                if focus in members:
                    root = cid
                    break
        if not root:
            root = max(clusters.keys(), key=lambda c: len(clusters[c]))

        edges: List[Tuple[str, str]] = []
        visited = {root}
        queue = [root]
        while queue:
            cur = queue.pop(0)
            for nb in mst.neighbors(cur):
                if nb in visited:
                    continue
                visited.add(nb)
                queue.append(nb)
                edges.append((cur, nb))
        return edges

    def _cross_cluster_xrefs(
        self,
        g: nx.Graph,
        clusters: Dict[str, Set[str]],
        limit: int = 32,
    ) -> List[Tuple[str, str, float]]:
        by_node = {}
        for cid, members in clusters.items():
            for m in members:
                by_node[m] = cid
        edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        for u, v, data in g.edges(data=True):
            cu = by_node.get(u)
            cv = by_node.get(v)
            if not cu or not cv or cu == cv:
                continue
            key = tuple(sorted((cu, cv)))
            edge_weights[key] += float(data.get("weight", 1.0))
        rows = [(k[0], k[1], w) for k, w in edge_weights.items()]
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:limit]

    def _cluster_labels(
        self,
        snapshot: IRSnapshot,
        clusters: Dict[str, Set[str]],
        representatives: Dict[str, str],
    ) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        sym_map = {s.symbol_id: s for s in snapshot.symbols}
        doc_map = {d.doc_id: d for d in snapshot.documents}
        for cid, members in clusters.items():
            rep = representatives.get(cid)
            if not rep:
                labels[cid] = f"Cluster {cid}"
                continue
            if rep in sym_map:
                sym = sym_map[rep]
                labels[cid] = f"{sym.display_name} cluster"
            elif rep in doc_map:
                labels[cid] = f"{doc_map[rep].path} cluster"
            else:
                labels[cid] = f"Cluster {cid}"
        return labels

    def _build_l0_summary(self, scope: ProjectionScope, labels: Dict[str, str], nodes: int, edges: int) -> str:
        top = ", ".join(list(labels.values())[:3]) or "No clusters"
        return (
            f"{scope.scope_kind.capitalize()} projection over {nodes} nodes and {edges} edges. "
            f"Key communities: {top}."
        )

    def _build_l1_summary(
        self,
        scope: ProjectionScope,
        labels: Dict[str, str],
        tree_edges: List[Tuple[str, str]],
        xrefs: List[Tuple[str, str, float]],
    ) -> str:
        return (
            f"{scope.scope_kind.capitalize()} hierarchy with {len(labels)} clusters, "
            f"{len(tree_edges)} backbone links, and {len(xrefs)} cross-links."
        )

    def _llm_rewrite_summary(
        self,
        layer: str,
        scope: ProjectionScope,
        labels: Dict[str, str],
        fallback: str,
    ) -> Optional[str]:
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

    def _source_refs(self, snapshot: IRSnapshot, scope: ProjectionScope) -> List[Dict[str, Any]]:
        refs = [
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
        return refs

    def _source_ref_for_node(self, node_id: str, snapshot: IRSnapshot) -> Dict[str, Any]:
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
        sg: nx.Graph,
        xrefs: Sequence[Tuple[str, str, float]],
        hidden_edge_count: int,
        projection_method: str,
        parent_reason: str,
    ) -> Dict[str, Any]:
        return {
            "updated_at": _utc_now(),
            "covers_nodes": sorted(list(sg.nodes())),
            "covers_edges": sorted([f"{u}->{v}" for u, v in sg.edges()]),
            "xrefs": [{"src": s, "dst": d, "weight": w} for s, d, w in xrefs],
            "hidden_edge_count": int(hidden_edge_count),
            "projection_method": projection_method,
            "parent_reason": parent_reason,
        }

    def _envelope(
        self,
        layer: str,
        kind: str,
        node_id: str,
        path: str,
        title: str,
        summary: str,
        source_refs: List[Dict[str, Any]],
        content_extra: Dict[str, Any],
        projection_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
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

    def _build_l2_chunks(
        self,
        snapshot: IRSnapshot,
        sg: nx.Graph,
        clusters: Dict[str, Set[str]],
        representatives: Dict[str, str],
        labels: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        sym_map = {s.symbol_id: s for s in snapshot.symbols}
        doc_map = {d.doc_id: d for d in snapshot.documents}

        for cid, members in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(chunks) >= self.max_chunk_count:
                break
            rep = representatives.get(cid)
            if not rep:
                continue
            chunk_id = f"c_{cid}"
            refs = [self._source_ref_for_node(rep, snapshot)]
            facts = [
                {"type": "cluster_size", "value": len(members)},
                {"type": "cluster_label", "value": labels.get(cid)},
            ]
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
            elif rep in doc_map:
                doc = doc_map[rep]
                content = {
                    "file": doc.path,
                    "snippet": f"Document cluster for {doc.path}",
                    "facts": facts,
                    "refs": refs,
                }
            else:
                content = {
                    "snippet": f"Cluster representative {rep}",
                    "facts": facts,
                    "refs": refs,
                }
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
                    "meta": {"cluster_id": cid, "members": len(members)},
                }
            )
        return chunks

    def _build_l2_index(
        self,
        projection_id: str,
        snapshot: IRSnapshot,
        scope: ProjectionScope,
        chunks: List[Dict[str, Any]],
        sg: nx.Graph,
        xrefs: Sequence[Tuple[str, str, float]],
        hidden_edge_count: int,
        projection_method: str,
    ) -> Dict[str, Any]:
        chunk_rows = []
        for c in chunks:
            c_content = c.get("content", {})
            c_range = c_content.get("range", {}) or {}
            chunk_rows.append(
                {
                    "chunk_id": c["chunk_id"],
                    "kind": c.get("kind", "evidence"),
                    "path": f"./chunks/{c['chunk_id']}.json",
                    "file": c_content.get("file"),
                    "start_line": c_range.get("start_line"),
                    "end_line": c_range.get("end_line"),
                    "label": c_content.get("symbol") or c_content.get("file") or c["chunk_id"],
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
            "source": {
                "domain": "code",
                "refs": self._source_refs(snapshot, scope),
            },
            "content": {"chunks": chunk_rows},
            "render": {"text": summary},
            "meta": self._projection_meta(
                sg=sg,
                xrefs=xrefs,
                hidden_edge_count=hidden_edge_count,
                projection_method=projection_method,
                parent_reason="chunked_evidence",
            ),
        }
