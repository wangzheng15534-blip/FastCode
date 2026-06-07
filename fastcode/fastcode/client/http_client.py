"""HTTP client for the FastCode API server.

Thin `entry_frame` wrapper — translates CLI commands into HTTP calls.
No FastCode internals, no env/config reading, no heavy initialization.
"""

from __future__ import annotations

from typing import Any

import httpx


class FastCodeClient:
    """Stateless HTTP client for a running FastCode API server."""

    def __init__(
        self, base_url: str = "http://localhost:8000", timeout: float = 300.0
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base}{path}"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(self._url(path), params=params)
            r.raise_for_status()
            return r.json()

    def _post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(self._url(path), json=json, params=params)
            r.raise_for_status()
            return r.json()

    def _delete(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.delete(self._url(path))
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------
    # Health & status
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def status(self, full_scan: bool = False) -> dict[str, Any]:
        return self._get("/status", params={"full_scan": full_scan})

    # ------------------------------------------------------------------
    # Repository management
    # ------------------------------------------------------------------

    def list_repositories(self, full_scan: bool = False) -> dict[str, Any]:
        return self._get("/repositories", params={"full_scan": full_scan})

    def load_repository(
        self,
        source: str,
        is_url: bool = False,
        is_zip: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/load",
            json={"source": source, "is_url": is_url, "is_zip": is_zip},
        )

    def load_and_index(
        self,
        source: str,
        is_url: bool = False,
        force: bool = False,
        is_zip: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/load-and-index",
            json={"source": source, "is_url": is_url, "is_zip": is_zip},
            params={"force": force},
        )

    def load_cached_repos(self, repo_names: list[str] | None = None) -> dict[str, Any]:
        return self._post("/load-repositories", json={"repo_names": repo_names})

    def index_repository(self, force: bool = False) -> dict[str, Any]:
        return self._post("/index", params={"force": force})

    def index_multiple(self, sources: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post("/index-multiple", json={"sources": sources})

    def delete_repository(self) -> dict[str, Any]:
        return self._delete("/repository")

    # ------------------------------------------------------------------
    # Snapshot-based indexing
    # ------------------------------------------------------------------

    def run_index_pipeline(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/index/run", json=kwargs)

    def get_index_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/index/runs/{run_id}")

    def publish_index_run(
        self, run_id: str, ref_name: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if ref_name:
            params["ref_name"] = ref_name
        return self._post(f"/index/publish/{run_id}", params=params)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, **kwargs}
        return self._post("/query", json=payload)

    def query_snapshot(self, question: str, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, **kwargs}
        return self._post("/query-snapshot", json=payload)

    # ------------------------------------------------------------------
    # Summary & stats
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return self._get("/summary")

    def repo_stats(self) -> dict[str, Any]:
        return self._get("/status")

    def repo_refs(self, repo_name: str) -> dict[str, Any]:
        return self._get(f"/repos/{repo_name}/refs")

    # ------------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------------

    def branch_manifest(self, repo_name: str, ref_name: str) -> dict[str, Any]:
        return self._get(f"/manifests/{repo_name}/{ref_name}")

    def snapshot_manifest(self, snapshot_id: str) -> dict[str, Any]:
        return self._get(f"/manifests/snapshot/{snapshot_id}")

    # ------------------------------------------------------------------
    # Symbols & graph
    # ------------------------------------------------------------------

    def find_symbol(self, snapshot_id: str, **kwargs: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"snapshot_id": snapshot_id, **kwargs}
        return self._get("/symbols/find", params=params)

    def graph_callees(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> dict[str, Any]:
        return self._get(
            "/graph/callees",
            params={
                "snapshot_id": snapshot_id,
                "symbol_id": symbol_id,
                "max_hops": max_hops,
            },
        )

    def graph_callers(
        self, snapshot_id: str, symbol_id: str, max_hops: int = 1
    ) -> dict[str, Any]:
        return self._get(
            "/graph/callers",
            params={
                "snapshot_id": snapshot_id,
                "symbol_id": symbol_id,
                "max_hops": max_hops,
            },
        )

    def graph_dependencies(
        self, snapshot_id: str, doc_id: str, max_hops: int = 1
    ) -> dict[str, Any]:
        return self._get(
            "/graph/dependencies",
            params={"snapshot_id": snapshot_id, "doc_id": doc_id, "max_hops": max_hops},
        )

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def build_projection(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/projection/build", json=kwargs)

    def get_projection_layer(self, projection_id: str, layer: str) -> dict[str, Any]:
        return self._get(f"/projection/{projection_id}/{layer}")

    def get_projection_prefix(self, snapshot_id: str) -> dict[str, Any]:
        return self._get(f"/projection/snapshot/{snapshot_id}/prefix")

    # ------------------------------------------------------------------
    # Sessions & dialogue
    # ------------------------------------------------------------------

    def list_sessions(self) -> dict[str, Any]:
        return self._get("/sessions")

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._get(f"/session/{session_id}")

    def delete_session(self, session_id: str) -> dict[str, Any]:
        return self._delete(f"/session/{session_id}")

    def new_session(self, clear_session_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if clear_session_id:
            params["clear_session_id"] = clear_session_id
        return self._post("/new-session", params=params)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def clear_cache(self) -> dict[str, Any]:
        return self._post("/clear-cache")

    def cache_stats(self) -> dict[str, Any]:
        return self._get("/cache-stats")

    def refresh_index_cache(self) -> dict[str, Any]:
        return self._post("/refresh-index-cache")

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def get_turn_context(
        self,
        session_id: str,
        turn_number: int | None = None,
        output_format: str = "json",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"format": output_format}
        if turn_number is not None:
            params["turn_number"] = turn_number
        path = (
            f"/agent-context/session/{session_id}/latest"
            if turn_number is None
            else f"/agent-context/session/{session_id}/turn/{turn_number}"
        )
        return self._get(path, params=params)

    def get_context_bundle(
        self, session_id: str, turn_number: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        params: dict[str, Any] = kwargs
        if turn_number is not None:
            path = f"/agent-context/session/{session_id}/bundle/{turn_number}"
        else:
            path = f"/agent-context/session/{session_id}/bundle/latest"
        return self._get(path, params=params)

    def create_activation(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/agent-context/bundle/activation", json=kwargs)

    def create_handoff(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/agent-context/handoff", json=kwargs)

    # ------------------------------------------------------------------
    # SCIP & code status
    # ------------------------------------------------------------------

    def code_status_pack(
        self, snapshot_id: str, include_graph_facts: bool = True
    ) -> dict[str, Any]:
        return self._get(
            f"/code-status/{snapshot_id}",
            params={"include_graph_facts": include_graph_facts},
        )

    def scip_artifact(self, snapshot_id: str) -> dict[str, Any]:
        return self._get(f"/scip/artifacts/{snapshot_id}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return self._get("/diagnostics")
