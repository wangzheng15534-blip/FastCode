"""
FastCode 2.0 - REST API
Complete API with all features from web_app.py
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

import os
import platform

if platform.system() == "Darwin":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
import logging
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fastcode.api.cors import cors_middleware_options
from fastcode.api.inbound import (
    AgentContextHandoffRequest,
    ContextActivationRequest,
    DeleteReposRequest,
    ExpandContextBundleRefRequest,
    ExpandContextRefRequest,
    IndexMultipleRequest,
    IndexRunRequest,
    LoadRepositoriesRequest,
    LoadRepositoryRequest,
    ProjectionBuildRequest,
    QueryRequest,
    QuerySnapshotRequest,
    map_index_run_request,
    map_load_repository_request,
    map_snapshot_query_request,
)
from fastcode.api.outbound import (
    ApiStatus,
    DiagnosticBundleResponse,
    IndexRunResponse,
    NewSessionRecord,
    NewSessionResponse,
    QueryResponse,
    StatusResponse,
)
from fastcode.api.serialization import (
    serialize_diagnostic_bundle_record,
    serialize_diagnostic_bundle_response,
    serialize_dialogue_history,
    serialize_index_run_response,
    serialize_index_run_response_record,
    serialize_new_session_response,
    serialize_query_response,
    serialize_query_response_record,
    serialize_status_response,
    serialize_status_response_record,
)
from fastcode.main.fastcode import FastCode
from fastcode.runtime_support.health import readiness_health
from fastcode.runtime_support.observability import configure_logging
from fastcode.utils.archive import (
    UnsafeArchiveError,
)

# HTTP schema boundaries live in fastcode.api.inbound and fastcode.api.outbound.

# Initialize FastAPI app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("FastCode API started - system will initialize on first request")
    yield
    # Shutdown
    global fastcode_instance
    if fastcode_instance is not None:
        try:
            fastcode_instance.shutdown()
        except Exception as e:
            logger.warning(f"FastCode shutdown hook failed: {e}")
    logger.info("FastCode API shutting down")


app = FastAPI(
    title="FastCode API",
    description="Repository-Level Code Understanding System API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    **cors_middleware_options(),
)

# Global FastCode instance
fastcode_instance: FastCode | None = None

# Setup logging
log_dir = Path("./logs")
logger = configure_logging(
    level=logging.INFO,
    format_str="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file=str(log_dir / "api.log"),
    console=True,
    logger_name=__name__,
)


def _ensure_fastcode_initialized():
    """Ensure FastCode is initialized (lazy initialization)"""
    global fastcode_instance
    if fastcode_instance is None:
        logger.info("Initializing FastCode system (lazy initialization)")
        fastcode_instance = FastCode()
    return fastcode_instance


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "FastCode API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Lightweight health check endpoint"""
    if fastcode_instance is None:
        return {
            "status": "initializing",
            "message": "FastCode system will initialize on first use",
            "repo_loaded": False,
            "repo_indexed": False,
        }

    health = readiness_health(
        repo_loaded=fastcode_instance.repo_loaded,
        repo_indexed=fastcode_instance.repo_indexed,
        details={
            "multi_repo_mode": fastcode_instance.multi_repo_mode,
            "storage_backend": fastcode_instance.snapshot_store.db_runtime.backend,
            "retrieval_backend": fastcode_instance.config.get("retrieval", {}).get(
                "retrieval_backend", "local"
            ),
        },
    )
    return {
        "status": health.status,
        **health.details,
    }


@app.get("/status", response_model=StatusResponse)
async def get_status(full_scan: bool = False):
    """Get system status."""
    fastcode = _ensure_fastcode_initialized()
    info = fastcode.get_status_info(full_scan=full_scan)
    return serialize_status_response(
        serialize_status_response_record(
            status=ApiStatus.READY if info["repo_indexed"] else ApiStatus.NOT_READY,
            repo_loaded=info["repo_loaded"],
            repo_indexed=info["repo_indexed"],
            repo_info=info["repo_info"],
            graph_expansion_backend=info["graph_expansion_backend"],
            storage_backend=info["storage_backend"],
            retrieval_backend=info["retrieval_backend"],
            available_repositories=info["available_repositories"],
            loaded_repositories=info["loaded_repositories"],
        )
    )


@app.get("/diagnostics", response_model=DiagnosticBundleResponse)
async def get_diagnostics():
    """Return a support-safe diagnostic bundle for operators."""
    fastcode = _ensure_fastcode_initialized()
    try:
        bundle = await asyncio.to_thread(fastcode.build_diagnostic_bundle)
        return serialize_diagnostic_bundle_response(
            serialize_diagnostic_bundle_record(bundle)
        )
    except Exception as e:
        logger.error(f"Failed to build diagnostic bundle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repositories")
async def list_repositories(full_scan: bool = False):
    """
    List available (indexed on disk) and loaded repositories

    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    fastcode = _ensure_fastcode_initialized()

    try:
        available_repos = fastcode.vector_store.scan_available_indexes(
            use_cache=not full_scan
        )
        loaded_repos = fastcode.list_repositories()

        return {
            "status": "success",
            "available": available_repos,
            "loaded": loaded_repos,
        }
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load_repository(request: LoadRepositoryRequest):
    """Load a repository"""
    fastcode = _ensure_fastcode_initialized()
    command = map_load_repository_request(request)

    try:
        logger.info(f"Loading repository: {command.source}")
        await asyncio.to_thread(
            fastcode.load_repository, command.source, command.is_url
        )

        return {
            "status": "success",
            "message": "Repository loaded successfully",
            "repo_info": fastcode.repo_info,
        }

    except Exception as e:
        logger.error(f"Failed to load repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_repository(force: bool = False):
    """Index the loaded repository"""
    fastcode = _ensure_fastcode_initialized()

    if not fastcode.repo_loaded:
        raise HTTPException(status_code=400, detail="No repository loaded")

    try:
        logger.info("Indexing repository")
        await asyncio.to_thread(fastcode.index_repository, force=force)

        return {
            "status": "success",
            "message": "Repository indexed successfully",
            "summary": fastcode.get_repository_summary(),
            "deprecated": True,
            "deprecation_note": "Use /index/run for IR-first snapshot indexing.",
        }

    except Exception as e:
        logger.error(f"Failed to index repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/run", response_model=IndexRunResponse)
async def run_index_pipeline(request: IndexRunRequest):
    """Run snapshot-based indexing pipeline."""
    fastcode = _ensure_fastcode_initialized()
    command = map_index_run_request(request)
    try:
        result = await asyncio.to_thread(
            fastcode.run_index_pipeline,
            source=command.source,
            is_url=command.is_url,
            ref=command.ref,
            commit=command.commit,
            force=command.force,
            publish=command.publish,
            scip_artifact_path=command.scip_artifact_path,
            enable_scip=command.enable_scip,
        )
        await asyncio.to_thread(fastcode.vector_store.invalidate_scan_cache)
        return serialize_index_run_response(serialize_index_run_response_record(result))
    except Exception as e:
        logger.error(f"Index run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index/runs/{run_id}")
async def get_index_run(run_id: str):
    """Get index run status/details."""
    fastcode = _ensure_fastcode_initialized()
    run = fastcode.get_index_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"status": "success", "run": run}


@app.post("/index/publish/{run_id}")
async def publish_index_run(run_id: str, ref_name: str | None = None):
    """Publish an existing index run into manifest and lineage."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(fastcode.publish_index_run, run_id, ref_name)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Publish run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/publish/retry")
async def retry_publish_tasks(limit: int = 10):
    """Retry pending Terminus lineage publish tasks."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(fastcode.retry_pending_publishes, limit)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Retry publish failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redo/process")
async def process_redo_tasks(limit: int = 10):
    """Admin endpoint to process pending redo tasks immediately."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(fastcode.process_redo_tasks, limit)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Redo process failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-and-index")
async def load_and_index(request: LoadRepositoryRequest, force: bool = False):
    """Load and index repository in one call"""
    fastcode = _ensure_fastcode_initialized()
    command = map_load_repository_request(request)

    try:
        logger.info(f"Loading and indexing repository: {command.source}")
        return await asyncio.to_thread(
            fastcode.load_and_index, command.source, command.is_url, force=force
        )

    except Exception as e:
        logger.error(f"Failed to load and index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-repositories")
async def load_repositories(request: LoadRepositoriesRequest):
    """Load existing indexed repositories from cache"""
    fastcode = _ensure_fastcode_initialized()

    if not request.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        logger.info(f"Loading repositories from cache: {request.repo_names}")
        success = await asyncio.to_thread(
            fastcode._load_multi_repo_cache, repo_names=request.repo_names
        )

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to load repositories from cache"
            )

        return {
            "status": "success",
            "loaded": fastcode.list_repositories(),
            "stats": fastcode.get_repository_stats(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-multiple")
async def index_multiple(request: IndexMultipleRequest):
    """Load and index multiple repositories"""
    fastcode = _ensure_fastcode_initialized()

    if not request.sources:
        raise HTTPException(status_code=400, detail="No repositories provided")

    try:
        logger.info(f"Indexing {len(request.sources)} repositories")
        await asyncio.to_thread(
            fastcode.load_multiple_repositories,
            [
                {
                    "source": command.source,
                    "is_url": command.is_url,
                }
                for command in (
                    map_load_repository_request(source) for source in request.sources
                )
            ],
        )

        await asyncio.to_thread(fastcode.vector_store.invalidate_scan_cache)

        return {
            "status": "success",
            "message": "Repositories indexed successfully",
            "stats": fastcode.get_repository_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to index multiple repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-zip")
async def upload_repository_zip(file: UploadFile = File(...)):
    """Upload and extract repository ZIP file"""
    fastcode = _ensure_fastcode_initialized()

    filename = file.filename
    if not filename or not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    try:
        file_bytes = await file.read()
        return await asyncio.to_thread(
            fastcode.upload_repository_zip, file_bytes, filename
        )
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except UnsafeArchiveError as e:
        logger.warning(f"Rejected unsafe ZIP file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upload and extract ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-index")
async def upload_and_index(file: UploadFile = File(...), force: bool = False):
    """Upload ZIP and index in one call"""
    fastcode = _ensure_fastcode_initialized()

    filename = file.filename
    if not filename or not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    try:
        logger.info("Indexing uploaded repository")
        file_bytes = await file.read()
        return await asyncio.to_thread(
            fastcode.upload_and_index, file_bytes, filename, force=force
        )
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except UnsafeArchiveError as e:
        logger.warning(f"Rejected unsafe ZIP file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to index uploaded repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query by snapshot scope (snapshot_id or repo_name+ref_name)."""
    fastcode = _ensure_fastcode_initialized()
    query_request = map_snapshot_query_request(request)

    if not query_request.snapshot_id and not (
        query_request.repo_name and query_request.ref_name
    ):
        raise HTTPException(
            status_code=400,
            detail="Query requires snapshot_id or repo_name+ref_name",
        )

    try:
        session_id = query_request.session_id or str(uuid.uuid4())[:8]
        if query_request.multi_turn and not query_request.session_id:
            logger.info(f"Generated new multi-turn session: {session_id}")
        elif not query_request.session_id:
            logger.info(f"Generated session for single-turn request: {session_id}")

        logger.info(f"Processing query: {query_request.question}")
        result = await asyncio.to_thread(
            fastcode.query_snapshot,
            question=query_request.question,
            snapshot_id=query_request.snapshot_id,
            repo_name=query_request.repo_name,
            ref_name=query_request.ref_name,
            filters=dict(query_request.filters) if query_request.filters else None,
            session_id=session_id,
            enable_multi_turn=query_request.multi_turn,
        )

        return serialize_query_response(
            serialize_query_response_record(
                result,
                session_id=session_id,
            )
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-snapshot", response_model=QueryResponse)
async def query_snapshot(request: QuerySnapshotRequest):
    """Query a specific snapshot or repo/ref manifest head."""
    fastcode = _ensure_fastcode_initialized()
    query_request = map_snapshot_query_request(request)
    try:
        session_id = query_request.session_id or str(uuid.uuid4())[:8]
        result = await asyncio.to_thread(
            fastcode.query_snapshot,
            question=query_request.question,
            repo_name=query_request.repo_name,
            ref_name=query_request.ref_name,
            snapshot_id=query_request.snapshot_id,
            filters=dict(query_request.filters) if query_request.filters else None,
            session_id=session_id,
            enable_multi_turn=query_request.multi_turn,
        )
        return serialize_query_response(
            serialize_query_response_record(
                result,
                session_id=session_id,
            )
        )
    except Exception as e:
        logger.error(f"Snapshot query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-stream")
async def query_repository_stream(request: QueryRequest):
    """Deprecated in snapshot-cutover mode."""
    raise HTTPException(
        status_code=501,
        detail="query-stream is disabled in snapshot-cutover mode. Use POST /query with snapshot scope.",
    )


@app.get("/summary")
async def get_repository_summary():
    """Get repository summary"""
    fastcode = _ensure_fastcode_initialized()

    if not fastcode.repo_loaded:
        raise HTTPException(status_code=400, detail="No repository loaded")

    summary_payload: dict[str, Any] = {
        "status": "success",
    }

    try:
        if fastcode.multi_repo_mode:
            summary_payload["summary"] = fastcode.get_repository_stats()
        else:
            summary_payload["summary"] = fastcode.get_repository_summary()
    except Exception as e:
        logger.error(f"Failed to build summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return summary_payload


@app.get("/manifests/{repo_name}/{ref_name}")
async def get_branch_manifest(repo_name: str, ref_name: str):
    """Get latest published manifest for repo/ref."""
    fastcode = _ensure_fastcode_initialized()
    manifest = fastcode.get_branch_manifest(repo_name, ref_name)
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {"status": "success", "manifest": manifest}


@app.get("/repos/{repo_name}/refs")
async def get_repo_refs(repo_name: str):
    fastcode = _ensure_fastcode_initialized()
    try:
        refs = await asyncio.to_thread(fastcode.list_repo_refs, repo_name)
        return {"status": "success", "repo_name": repo_name, "refs": refs}
    except Exception as e:
        logger.error(f"Repo refs lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/manifests/snapshot/{snapshot_id}")
async def get_snapshot_manifest(snapshot_id: str):
    """Get latest published manifest for snapshot ID."""
    fastcode = _ensure_fastcode_initialized()
    manifest = fastcode.get_snapshot_manifest(snapshot_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {"status": "success", "manifest": manifest}


@app.get("/code-status/{snapshot_id}")
async def get_code_status_pack(
    snapshot_id: str,
    include_graph_facts: bool = True,
):
    """Get a DocKG-facing code_status_pack.v0 artifact for a snapshot."""
    fastcode = _ensure_fastcode_initialized()
    try:
        pack = await asyncio.to_thread(
            fastcode.get_code_status_pack,
            snapshot_id,
            include_graph_facts=include_graph_facts,
        )
        return {"status": "success", "pack": pack}
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Code status pack export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scip/artifacts/{snapshot_id}")
async def get_scip_artifact(snapshot_id: str):
    """Get preserved SCIP artifact metadata for a snapshot."""
    fastcode = _ensure_fastcode_initialized()
    artifact = fastcode.get_scip_artifact_ref(snapshot_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="SCIP artifact not found")
    artifacts = fastcode.list_scip_artifact_refs(snapshot_id)
    return {"status": "success", "artifact": artifact, "artifacts": artifacts}


@app.get("/symbols/find")
async def find_symbol(
    snapshot_id: str,
    symbol_id: str | None = None,
    name: str | None = None,
    path: str | None = None,
):
    fastcode = _ensure_fastcode_initialized()
    try:
        symbol = await asyncio.to_thread(
            fastcode.find_symbol,
            snapshot_id,
            symbol_id=symbol_id,
            name=name,
            path=path,
        )
        if not symbol:
            raise HTTPException(status_code=404, detail="Symbol not found")
        return {"status": "success", "symbol": symbol}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Symbol lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/callees")
async def get_graph_callees(snapshot_id: str, symbol_id: str, max_hops: int = 1):
    fastcode = _ensure_fastcode_initialized()
    try:
        callees = await asyncio.to_thread(
            fastcode.get_graph_callees, snapshot_id, symbol_id, max_hops
        )
        return {
            "status": "success",
            "snapshot_id": snapshot_id,
            "symbol_id": symbol_id,
            "callees": callees,
        }
    except Exception as e:
        logger.error(f"Graph callees lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/callers")
async def get_graph_callers(snapshot_id: str, symbol_id: str, max_hops: int = 1):
    fastcode = _ensure_fastcode_initialized()
    try:
        callers = await asyncio.to_thread(
            fastcode.get_graph_callers, snapshot_id, symbol_id, max_hops
        )
        return {
            "status": "success",
            "snapshot_id": snapshot_id,
            "symbol_id": symbol_id,
            "callers": callers,
        }
    except Exception as e:
        logger.error(f"Graph callers lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/dependencies")
async def get_graph_dependencies(snapshot_id: str, doc_id: str, max_hops: int = 1):
    fastcode = _ensure_fastcode_initialized()
    try:
        deps = await asyncio.to_thread(
            fastcode.get_graph_dependencies, snapshot_id, doc_id, max_hops
        )
        return {
            "status": "success",
            "snapshot_id": snapshot_id,
            "doc_id": doc_id,
            "dependencies": deps,
        }
    except Exception as e:
        logger.error(f"Graph dependencies lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projection/build")
async def build_projection(request: ProjectionBuildRequest):
    """Build or reuse a cached projection artifact."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.build_projection,
            scope_kind=request.scope_kind,
            snapshot_id=request.snapshot_id,
            repo_name=request.repo_name,
            ref_name=request.ref_name,
            query=request.query,
            target_id=request.target_id,
            filters=request.filters,
            force=request.force,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Projection build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projection/{projection_id}/{layer}")
async def get_projection_layer(projection_id: str, layer: str):
    """Fetch a specific projection layer (L0/L1/L2)."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.get_projection_layer, projection_id, layer
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Projection layer fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projection/{projection_id}/chunks/{chunk_id}")
async def get_projection_chunk(projection_id: str, chunk_id: str):
    """Fetch a single projection L2 chunk payload."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.get_projection_chunk, projection_id, chunk_id
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Projection chunk fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projection/snapshot/{snapshot_id}/prefix")
async def get_projection_prefix(snapshot_id: str):
    """Get L0+L1 projection as compact JSON for system prompt injection.

    Returns the architectural overview (L0) and navigation structure (L1)
    combined into a single response suitable for injecting into an AI
    agent's system prompt at session start.
    """
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(fastcode.get_session_prefix, snapshot_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session prefix fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent-context/session/{session_id}/latest")
async def get_latest_turn_context(session_id: str, format: str = "fcx"):
    """Fetch the latest typed working-memory artifact for a session."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.get_turn_context,
            session_id,
            None,
            format,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Latest turn context fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/agent-context/session/{session_id}/turn/{turn_number}")
async def get_turn_context(
    session_id: str,
    turn_number: int,
    format: str = "fcx",
):
    """Fetch a specific typed working-memory artifact for a session turn."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.get_turn_context,
            session_id,
            turn_number,
            format,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Turn context fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/agent-context/session/{session_id}/bundle/latest")
async def get_latest_context_bundle(
    session_id: str,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch the latest durable context bundle for a session."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.get_context_bundle,
            session_id,
            None,
            format,
            token_budget,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Latest context bundle fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/agent-context/session/{session_id}/bundle/{turn_number}")
async def get_context_bundle(
    session_id: str,
    turn_number: int,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch a durable context bundle for a session turn."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.get_context_bundle,
            session_id,
            turn_number,
            format,
            token_budget,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context bundle fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/agent-context/bundle/{bundle_id}")
async def get_context_bundle_by_id(
    bundle_id: str,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch a durable context bundle by bundle ID."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.get_context_bundle_by_id,
            bundle_id,
            format,
            token_budget,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context bundle fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agent-context/bundle/expand")
async def expand_agent_context_bundle_ref(request: ExpandContextBundleRefRequest):
    """Expand a single source ref from a durable context bundle."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.expand_context_bundle_ref,
            request.ref_id,
            session_id=request.session_id,
            turn_number=request.turn_number,
            bundle_id=request.bundle_id,
            depth=request.depth,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context bundle ref expansion failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agent-context/bundle/activation")
async def create_agent_context_activation(request: ContextActivationRequest):
    """Create and persist an activation record for a context bundle."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.create_context_activation,
            session_id=request.session_id,
            turn_number=request.turn_number,
            bundle_id=request.bundle_id,
            active_ref_ids=request.active_ref_ids,
            active_fact_ids=request.active_fact_ids,
            active_hypothesis_ids=request.active_hypothesis_ids,
            reason=request.reason,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context activation creation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agent-context/handoff")
async def create_agent_context_handoff(request: AgentContextHandoffRequest):
    """Create and persist a handoff artifact from a session turn."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.create_handoff,
            request.session_id,
            request.turn_number,
            request.mode,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Handoff creation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/agent-context/handoff/{artifact_id}")
async def get_agent_context_handoff(artifact_id: str):
    """Fetch a persisted handoff artifact."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(fastcode.get_handoff_artifact, artifact_id)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Handoff fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agent-context/expand")
async def expand_agent_context_ref(request: ExpandContextRefRequest):
    """Expand a single evidence ref from working memory."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.expand_context_ref,
            request.session_id,
            request.turn_number,
            request.ref_id,
            request.depth,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context ref expansion failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/new-session", response_model=NewSessionResponse)
async def new_session(clear_session_id: str | None = None):
    """Start a new conversation session"""
    fastcode = _ensure_fastcode_initialized()

    if clear_session_id:
        fastcode.delete_session(clear_session_id)

    session_id = str(uuid.uuid4())[:8]
    return serialize_new_session_response(NewSessionRecord(session_id=session_id))


@app.get("/sessions")
async def list_sessions():
    """List all dialogue sessions with titles (sorted by last update time)"""
    fastcode = _ensure_fastcode_initialized()
    try:
        sessions = fastcode.list_sessions()

        formatted_sessions = []
        for session in sessions:
            formatted_session = {
                "session_id": session.get("session_id", ""),
                "title": session.get(
                    "title", f"Session {session.get('session_id', '')}"
                ),
                "total_turns": session.get("total_turns", 0),
                "created": session.get("created", 0),
                "last_updated": session.get("last_updated", 0),
            }
            formatted_sessions.append(formatted_session)

        return {"status": "success", "sessions": formatted_sessions}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get full dialogue history for a session"""
    fastcode = _ensure_fastcode_initialized()
    try:
        history = fastcode.get_session_history(session_id) or []
        return {
            "status": "success",
            "session_id": session_id,
            "history": serialize_dialogue_history(history),
        }
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a single dialogue session"""
    fastcode = _ensure_fastcode_initialized()

    try:
        history = fastcode.get_session_history(session_id)
        if not history:
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found"
            )

        success = fastcode.delete_session(session_id)
        if success:
            return {
                "status": "success",
                "message": f"Session '{session_id}' deleted ({len(history)} turns)",
            }
        raise HTTPException(status_code=500, detail="Failed to delete session")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-repos")
async def delete_repositories(request: DeleteReposRequest):
    """Delete one or more repositories and all associated data"""
    fastcode = _ensure_fastcode_initialized()

    if not request.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        results = []
        for repo_name in request.repo_names:
            result = await asyncio.to_thread(
                fastcode.remove_repository,
                repo_name,
                delete_source=request.delete_source,
            )
            results.append(result)
            logger.info(
                f"Deleted repository '{repo_name}': "
                f"{len(result['deleted_files'])} files, {result['freed_mb']} MB freed"
            )

        total_freed = sum(r["freed_mb"] for r in results)
        return {
            "status": "success",
            "message": f"Deleted {len(results)} repository(ies), freed {total_freed:.2f} MB",
            "results": results,
        }
    except Exception as e:
        logger.error(f"Failed to delete repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-cache")
async def clear_cache():
    """Clear cache"""
    fastcode = _ensure_fastcode_initialized()

    success = await asyncio.to_thread(fastcode.clear_cache)

    if success:
        return {"status": "success", "message": "Cache cleared"}
    return {
        "status": "failed",
        "message": "Failed to clear cache or cache disabled",
    }


@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    fastcode = _ensure_fastcode_initialized()

    return await asyncio.to_thread(fastcode.get_cache_stats)


@app.post("/refresh-index-cache")
async def refresh_index_cache():
    """Force refresh the index scan cache"""
    fastcode = _ensure_fastcode_initialized()

    try:
        available_repos = await asyncio.to_thread(fastcode.refresh_index_cache)

        return {
            "status": "success",
            "message": "Index cache refreshed",
            "repository_count": len(available_repos),
        }
    except Exception as e:
        logger.error(f"Failed to refresh index cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/repository")
async def unload_repository():
    """Unload current repository"""
    global fastcode_instance

    if fastcode_instance:
        await asyncio.to_thread(fastcode_instance.cleanup)
        fastcode_instance = FastCode()

    return {"status": "success", "message": "Repository unloaded"}


def start_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the API server"""
    logger.info(f"Starting FastCode API at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    uvicorn.run("fastcode.api.routes:app", host=host, port=port, reload=reload)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FastCode API Server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    start_api(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
