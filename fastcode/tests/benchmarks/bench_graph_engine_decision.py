"""Graph engine benchmark and decision-record generator."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import resource
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import igraph as ig
import networkx as nx
import pytest

pytestmark = [pytest.mark.perf]

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "fastcode"


def _synthetic_edges(size: int) -> tuple[list[str], list[tuple[str, str]]]:
    nodes = [f"n:{index}" for index in range(size)]
    edges: list[tuple[str, str]] = []
    for index, node in enumerate(nodes):
        edges.append((node, nodes[(index + 1) % size]))
        if index + 7 < size:
            edges.append((node, nodes[index + 7]))
        if index % 11 == 0 and index + 23 < size:
            edges.append((node, nodes[index + 23]))
    return nodes, edges


def _repo_like_edges(limit: int = 240) -> tuple[list[str], list[tuple[str, str]]]:
    paths = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix() for path in PACKAGE_ROOT.rglob("*.py")
    )[:limit]
    if not paths:
        return _synthetic_edges(64)
    nodes = [f"file:{path}" for path in paths]
    by_package: dict[str, list[str]] = {}
    for node, rel_path in zip(nodes, paths, strict=True):
        by_package.setdefault(rel_path.split("/", 1)[0], []).append(node)
    edges: list[tuple[str, str]] = []
    for package_nodes in by_package.values():
        for source, target in pairwise(package_nodes):
            edges.append((source, target))
    for source, target in pairwise(nodes):
        edges.append((source, target))
    return nodes, edges


def _build_networkx(nodes: list[str], edges: list[tuple[str, str]]) -> nx.DiGraph[str]:
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def _build_igraph(
    nodes: list[str], edges: list[tuple[str, str]]
) -> tuple[Any, dict[str, int]]:
    index = {node: pos for pos, node in enumerate(nodes)}
    graph: Any = ig.Graph(directed=True)
    graph.add_vertices(len(nodes))
    graph.vs["name"] = nodes
    graph.add_edges([(index[source], index[target]) for source, target in edges])
    return graph, index


def _time_call(func: Any, *args: Any) -> tuple[Any, float, int]:
    started = time.perf_counter()
    value = func(*args)
    elapsed_ms = (time.perf_counter() - started) * 1000
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return value, round(elapsed_ms, 3), rss


def _measure_case(
    name: str, nodes: list[str], edges: list[tuple[str, str]]
) -> dict[str, Any]:
    nx_graph, nx_build_ms, nx_build_rss = _time_call(_build_networkx, nodes, edges)
    (ig_graph, ig_index), ig_build_ms, ig_build_rss = _time_call(
        _build_igraph, nodes, edges
    )
    seed = nodes[0]
    target = nodes[min(len(nodes) - 1, max(0, len(nodes) // 2))]
    seed_index = ig_index[seed]
    target_index = ig_index[target]

    _, nx_reach_ms, nx_reach_rss = _time_call(
        nx.single_source_shortest_path_length,
        nx_graph,
        seed,
        3,
    )
    _, ig_reach_ms, ig_reach_rss = _time_call(
        ig_graph.neighborhood,
        seed_index,
        3,
        "out",
    )
    _, nx_path_ms, nx_path_rss = _time_call(nx.shortest_path, nx_graph, seed, target)
    _, ig_path_ms, ig_path_rss = _time_call(
        lambda: ig_graph.get_shortest_paths(seed_index, to=target_index, mode="out")
    )
    undirected_nx = nx_graph.to_undirected()
    undirected_ig = ig_graph.as_undirected()
    _, nx_cluster_ms, nx_cluster_rss = _time_call(
        nx.algorithms.community.greedy_modularity_communities,
        undirected_nx,
    )
    _, ig_cluster_ms, ig_cluster_rss = _time_call(
        lambda: undirected_ig.community_leiden(objective_function="modularity")
    )
    _, nx_pr_ms, nx_pr_rss = _time_call(nx.pagerank, nx_graph)
    _, ig_pr_ms, ig_pr_rss = _time_call(ig_graph.pagerank)
    query_terms = {seed.rsplit(":", 1)[-1], target.rsplit(":", 1)[-1]}
    _, nx_scope_ms, nx_scope_rss = _time_call(
        lambda: [
            node
            for node in nx_graph.nodes
            if any(term in str(node) for term in query_terms)
        ]
    )
    _, ig_scope_ms, ig_scope_rss = _time_call(
        lambda: [
            name
            for name in ig_graph.vs["name"]
            if any(term in str(name) for term in query_terms)
        ]
    )
    _, nx_merge_ms, nx_merge_rss = _time_call(lambda: set(nx_graph.nodes) & set(nodes))
    _, ig_merge_ms, ig_merge_rss = _time_call(
        lambda: set(ig_graph.vs["name"]) & set(nodes)
    )

    return {
        "case": name,
        "nodes": len(nodes),
        "edges": len(edges),
        "networkx": {
            "build_ms": nx_build_ms,
            "cutoff_reachability_ms": nx_reach_ms,
            "shortest_path_ms": nx_path_ms,
            "projection_scope_ms": nx_scope_ms,
            "clustering_ms": nx_cluster_ms,
            "pagerank_centrality_ms": nx_pr_ms,
            "merge_matching_ms": nx_merge_ms,
            "max_rss_bytes": max(
                nx_build_rss,
                nx_reach_rss,
                nx_path_rss,
                nx_scope_rss,
                nx_cluster_rss,
                nx_pr_rss,
                nx_merge_rss,
            ),
        },
        "igraph": {
            "build_ms": ig_build_ms,
            "cutoff_reachability_ms": ig_reach_ms,
            "shortest_path_ms": ig_path_ms,
            "projection_scope_ms": ig_scope_ms,
            "clustering_ms": ig_cluster_ms,
            "pagerank_centrality_ms": ig_pr_ms,
            "merge_matching_ms": ig_merge_ms,
            "max_rss_bytes": max(
                ig_build_rss,
                ig_reach_rss,
                ig_path_rss,
                ig_scope_rss,
                ig_cluster_rss,
                ig_pr_rss,
                ig_merge_rss,
            ),
        },
    }


def write_graph_engine_decision_report(output_path: Path) -> dict[str, Any]:
    cases = [
        ("synthetic_small", *_synthetic_edges(64)),
        ("synthetic_medium", *_synthetic_edges(512)),
        ("synthetic_large", *_synthetic_edges(1536)),
        ("repo_fastcode", *_repo_like_edges()),
    ]
    report = {
        "schema_version": "fastcode.graph_engine_decision.v1",
        "workloads": [
            "ir_graph_build",
            "cutoff_reachability",
            "shortest_path",
            "projection_scope",
            "clustering",
            "pagerank_centrality",
            "merge_matching",
        ],
        "cases": [
            _measure_case(case_name, nodes, edges) for case_name, nodes, edges in cases
        ],
        "decision": {
            "canonical_hot_path_engine": "igraph_via_IRGraphView",
            "compatibility_engine": "networkx_export_only",
            "rationale": (
                "Projection and query graph hot paths use igraph-backed compact "
                "handles; NetworkX remains isolated to legacy compatibility and "
                "export/debug boundaries."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def test_graph_engine_decision_report_perf(tmp_path: Path, benchmark: Any) -> None:
    report_path = tmp_path / "reports" / "graph_engine_decision.json"

    report = benchmark(write_graph_engine_decision_report, report_path)

    assert report["schema_version"] == "fastcode.graph_engine_decision.v1"
    assert report["decision"]["canonical_hot_path_engine"] == "igraph_via_IRGraphView"
    assert {case["case"] for case in report["cases"]} >= {
        "synthetic_small",
        "synthetic_medium",
        "synthetic_large",
        "repo_fastcode",
    }
    assert report_path.exists()
