# Design: Graph Algorithm Gaps & Implementation Plan

**Date:** 2026-04-01
**Status:** Draft
**Scope:** `fastcode/projection_transform.py` graph-to-tree pipeline

---

## 0. Current State

The projection pipeline in `fastcode/projection_transform.py` converts a weighted
NetworkX graph into a navigable L0/L1/L2 artifact. The pipeline runs these steps:

```
1. Build weighted graph          (_build_weighted_graph, :228-292)
2. Scope nodes                   (_scope_nodes, :294-329)
3. Compress hubs                 (_compress_hubs, :367-383)
4. Cluster nodes                 (_cluster_nodes, :385-408)
5. Pick representatives          (_pick_representatives, :410-417)
6. Build backbone tree           (_build_backbone_tree, :419-471)
7. Cross-cluster xrefs           (_cross_cluster_xrefs, :473-493)
8. Generate L0 / L1 / L2        (:97-204)
```

### Target algorithms vs implementation

| Algorithm | Spec role | Status | Location |
|-----------|-----------|--------|----------|
| Leiden clustering | Group messy graphs into coherent clusters | **Partial** — flat only, no resolution tuning | `:390-402` `ig.Graph.community_leiden()` |
| Hierarchical Leiden | Multi-scale cluster hierarchy (dendrogram) | **Missing** | — |
| Maximum spanning tree | Readable backbone connecting clusters | **Partial** — undirected MST, not arborescence | `:448` `nx.maximum_spanning_tree()` |
| Arborescence | Directed rooted tree for true hierarchy | **Missing** | — |
| Steiner tree | Query-scoped minimal explanation subgraph | **Partial** — approximation only, no pruning | `:318` `nx.approximation.steiner_tree()` |
| Steiner post-pruning | Strip non-essential intermediate nodes | **Missing** | — |
| SNAP aggregation | Attribute-preserving cluster summaries | **Missing** | — |
| PageRank centrality | Cluster representative selection | **Done** | `:414` `nx.pagerank()` |
| Betweenness centrality | Bridge detection, flow analysis | **Missing** | — |
| Degree centrality | Hub detection, explicit scoring | **Partial** — implicit tiebreaker only | `:416` `g.degree(n)` |
| Closeness centrality | Information flow analysis | **Missing** | — |

### Existing helpers already in place

- **Hub compression** (`:367-383`): removes >99th percentile nodes, downweights edges
- **Query terminal detection** (`:347-365`): token-based matching with log(degree) boost
- **Cross-cluster xrefs** (`:473-493`): top-32 inter-cluster edges by weight
- **Config flag**: `enable_leiden` (default `True`), falls back to `nx.greedy_modularity_communities()`

---

## 1. Hierarchical Leiden

### Gap

`_cluster_nodes()` produces a single flat partition. For large codebases, a multi-scale
hierarchy is needed: top-level modules → submodules → clusters. This enables drill-down
navigation in L1 and progressive detail in L2.

### Approach

Run `ig.Graph.community_leiden()` at multiple resolution parameters to produce a
dendrogram. Store the hierarchy as `Dict[int, Dict[str, Set[str]]]` (level → clusters).

```python
# Pseudocode
def _cluster_hierarchical(self, g: nx.Graph) -> Tuple[Dict[int, Dict[str, Set[str]]], str]:
    levels = {}
    for level, resolution in enumerate([0.5, 1.0, 2.0, 4.0]):
        part = ig_g.community_leiden(resolution_parameter=resolution)
        clusters = {f"c{ci}": {nodes[m] for m in members} for ci, members in enumerate(part)}
        levels[level] = clusters
        if len(clusters) <= 1:
            break
    return levels, "hierarchical_leiden"
```

### Integration points

- `_cluster_nodes()` — return the hierarchy dict instead of a flat dict
- `_build_backbone_tree()` — operate on the coarsest level, store fine-level membership
- L1 `relations` — add `hierarchy` key with parent-child cluster links
- L2 chunks — group by leaf clusters within coarse parents

### Files to modify

| File | Change |
|------|--------|
| `fastcode/projection_transform.py` | `_cluster_nodes()` returns hierarchy; `_build_backbone_tree()` selects level |
| `fastcode/projection_models.py` | Add `hierarchy_levels: int` config field |
| `tests/test_projection_v2_schema.py` | Test that hierarchy metadata exists in L1 output |

---

## 2. True Arborescence (Directed Maximum Branching)

### Gap

`_build_backbone_tree()` builds an undirected MST via `nx.maximum_spanning_tree()` then
flattens it with BFS. This loses directionality — a "depends on" edge and a "is used by"
edge are treated symmetrically. For code, direction matters: imports flow downward,
callers are above callees.

### Approach

Replace undirected MST + BFS with `nx.algorithms.tree.branchings.maximum_branching()`
on a directed cluster graph. Root at the cluster containing the focus node (or the
largest cluster).

```python
# Pseudocode
def _build_backbone_arborescence(self, dg: nx.DiGraph, clusters, focus_nodes):
    # Build directed cluster graph
    dcg = nx.DiGraph()
    for u, v, data in dg.edges(data=True):
        cu, cv = by_node[u], by_node[v]
        if cu != cv:
            w = float(data.get("weight", 1.0))
            dcg.add_edge(cu, cv, weight=w)  # directed

    # Maximum branching (arborescence) rooted at focus cluster
    root = self._find_root_cluster(clusters, focus_nodes)
    arbo = nx.algorithms.tree.branchings.maximum_branching(dcg, root)

    # Topological sort for level assignment
    levels = {}
    for i, node in enumerate(nx.topological_sort(arbo)):
        levels[node] = i
    return list(arbo.edges()), levels
```

### Integration points

- `_build_backbone_tree()` — accept directed graph, use `maximum_branching()`
- `_build_weighted_graph()` — build a `nx.DiGraph` variant (already mostly directed)
- L1 `navigation` — use topological levels for parent-child ordering
- L1 `decisions` — log `backbone_edges` and `root_cluster`

### Files to modify

| File | Change |
|------|--------|
| `fastcode/projection_transform.py` | `_build_backbone_tree()` → arborescence; add `_build_directed_cluster_graph()` |
| `tests/test_projection_v2_schema.py` | Verify edges have direction, root is correct |

---

## 3. Steiner Tree Post-Pruning

### Gap

`_scope_nodes()` calls `nx.approximation.steiner_tree()` for multi-terminal queries
but includes all nodes on paths between terminals, including non-essential intermediaries.
For "minimal explanation" views, leaf nodes that are not terminals should be pruned.

### Approach

After computing the Steiner tree, iteratively remove leaf nodes not in the terminal set
until no more leaves can be removed. This produces the minimum Steiner tree approximation.

```python
def _prune_steiner_leaves(tree: nx.Graph, terminals: Set[str]) -> nx.Graph:
    pruned = tree.copy()
    changed = True
    while changed:
        changed = False
        for node in list(pruned.nodes()):
            if node in terminals:
                continue
            if pruned.degree(node) == 1:  # leaf, not terminal
                pruned.remove_node(node)
                changed = True
    return pruned
```

### Integration points

- `_scope_nodes()` — call `_prune_steiner_leaves()` after `steiner_tree()`
- Config — add `steiner_prune: bool = True`

### Files to modify

| File | Change |
|------|--------|
| `fastcode/projection_transform.py` | Add `_prune_steiner_leaves()`, call in `_scope_nodes()` |
| `tests/test_projection_pipeline.py` | Test that query scope excludes non-essential intermediaries |

---

## 4. SNAP-Style Aggregation (Attribute-Preserving Summaries)

### Gap

L2 chunks currently pick a single representative per cluster and emit a flat content dict.
No attribute-preserving aggregation exists — cluster summaries lose structural information
about member types, edge patterns, and shared properties.

### Approach

Borrow from MemOS's hierarchical summarization pattern (`MemOS/src/memos/memories/textual/tree_text_memory/organize/reorganizer.py`):
cluster members → extract shared attributes → generate summary node with preserved attributes.

For code clusters, the "attributes" are: dominant language, common prefix/path, edge type
distribution, kind distribution (function/class/method), average complexity.

```python
def _aggregate_cluster_attributes(self, snapshot, cluster_id, members, g):
    symbols = [s for s in snapshot.symbols if s.symbol_id in members]
    languages = Counter(s.language for s in symbols if s.language)
    kinds = Counter(s.kind for s in symbols if s.kind)
    paths = set(s.path for s in symbols if s.path)
    common_prefix = os.path.commonprefix(list(paths)) if paths else ""

    # Edge type distribution within cluster
    intra_edges = [e for e in snapshot.edges if e.src_id in members and e.dst_id in members]
    edge_types = Counter(e.edge_type for e in intra_edges)

    return {
        "cluster_id": cluster_id,
        "member_count": len(members),
        "dominant_language": languages.most_common(1)[0][0] if languages else None,
        "dominant_kind": kinds.most_common(1)[0][0] if kinds else None,
        "kind_distribution": dict(kinds),
        "common_path_prefix": common_prefix,
        "edge_type_distribution": dict(edge_types),
        "representative": self._pick_representative(g, {cluster_id: members}).get(cluster_id),
    }
```

### Integration points

- `_build_l2_chunks()` — replace flat representative with aggregated attributes
- L2 chunk `meta` — include `kind_distribution`, `dominant_language`, `common_path_prefix`
- L1 cluster descriptions — use aggregated attributes for human-readable labels

### Files to modify

| File | Change |
|------|--------|
| `fastcode/projection_transform.py` | Add `_aggregate_cluster_attributes()`, integrate into `_build_l2_chunks()` |
| `tests/test_projection_v2_schema.py` | Verify L2 chunks contain aggregated attributes |

---

## 5. Betweenness + Closeness Centrality

### Gap

`_pick_representatives()` uses only PageRank + degree. Betweenness centrality would
identify bridge nodes (symbols that connect otherwise disconnected modules). Closeness
centrality identifies symbols that are "close to everything" — good navigation hubs.

### Approach

Compute centrality scores once, use composite ranking for representative selection.

```python
def _pick_representatives(self, g: nx.Graph, clusters):
    if g.number_of_nodes() == 0:
        return {}

    pr = nx.pagerank(g, alpha=0.85) if g.number_of_edges() else {n: 1.0 for n in g.nodes()}
    bt = nx.betweenness_centrality(g) if g.number_of_nodes() < 5000 else {}
    cl = nx.closeness_centrality(g) if g.number_of_nodes() < 5000 else {}

    reps = {}
    for cid, nodes in clusters.items():
        reps[cid] = max(nodes, key=lambda n: (
            pr.get(n, 0.0),        # global importance
            bt.get(n, 0.0),         # bridge detection
            cl.get(n, 0.0),         # navigation hub
            g.degree(n),            # connectivity
        ))
    return reps
```

### Performance guard

Betweenness and closeness are O(VE) and O(V*(V+E)) respectively. Guard with a node count
threshold (5000) — fall back to PageRank-only for large graphs.

### Integration points

- `_pick_representatives()` — add betweenness + closeness to composite score
- L2 chunk `meta` — store representative's centrality scores
- Config — add `centrality_max_nodes: int = 5000`

### Files to modify

| File | Change |
|------|--------|
| `fastcode/projection_transform.py` | `_pick_representatives()` — composite centrality scoring |
| `tests/test_projection_pipeline.py` | Test that bridge nodes are preferred as representatives |

---

## 6. Implementation Priority

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Betweenness centrality | Small (1 method change) | Better cluster representatives |
| P0 | Steiner post-pruning | Small (1 new method) | Tighter query-scoped views |
| P1 | SNAP aggregation | Medium (new method + L2 integration) | Richer L2 chunks |
| P1 | Arborescence | Medium (refactor backbone) | Directed hierarchy in L1 |
| P2 | Hierarchical Leiden | Medium (multi-resolution loop) | Multi-scale navigation |

---

## 7. Cross-Reference: What graphiti and MemOS Actually Provide

Neither repo implements these classical graph algorithms:

| Algorithm | graphiti | MemOS | FastCode (current) |
|-----------|----------|-------|--------------------|
| Leiden | No (label propagation instead) | No (K-Means instead) | Partial |
| Spanning tree | No | No | Partial (undirected) |
| Steiner tree | No | No | Partial (no pruning) |
| SNAP aggregation | No | LLM-based summarization | No |
| PageRank | No | No | Done |
| Betweenness | No | No | Missing |

**Conclusion:** These gaps must be filled with NetworkX/igraph — no code to borrow from
sibling repos for the algorithm implementations themselves. graphiti and MemOS provide
architectural patterns (driver abstraction, hierarchical summarization) but not the
graph algorithms.
