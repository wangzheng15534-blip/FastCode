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
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
    logger.info("FastCode API started — initializing system")
    if not hasattr(app.state, "fastcode"):
        app.state.fastcode = FastCode()
    yield
    # Shutdown
    fc = getattr(app, "state", None) and getattr(app.state, "fastcode", None)
    if fc is not None:
        try:
            fc.shutdown()
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

# Setup logging
log_dir = Path("./logs")
logger = configure_logging(
    level=logging.INFO,
    format_str="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file=str(log_dir / "api.log"),
    console=True,
    logger_name=__name__,
)


def _fc(request: Request) -> FastCode:
    """Get FastCode instance from app state."""
    fc = getattr(request.app.state, "fastcode", None)
    if fc is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    return fc


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
async def health_check(request: Request):
    """Lightweight health check endpoint"""
    fc = getattr(request.app.state, "fastcode", None)
    if fc is None:
        return {
            "status": "initializing",
            "message": "FastCode system will initialize on first use",
            "repo_loaded": False,
            "repo_indexed": False,
        }

    info = fc.get_status_info()
    health = readiness_health(
        repo_loaded=info["repo_loaded"],
        repo_indexed=info["repo_indexed"],
        details={
            "multi_repo_mode": info["multi_repo_mode"],
            "storage_backend": info["storage_backend"],
            "retrieval_backend": info["retrieval_backend"],
        },
    )
    return {
        "status": health.status,
        **health.details,
    }


@app.get("/status", response_model=StatusResponse)
async def get_status(request: Request, full_scan: bool = False):
    """Get system status."""
    fastcode = _fc(request)
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
async def get_diagnostics(request: Request):
    """Return a support-safe diagnostic bundle for operators."""
    fastcode = _fc(request)
    try:
        bundle = await asyncio.to_thread(fastcode.build_diagnostic_bundle)
        return serialize_diagnostic_bundle_response(
            serialize_diagnostic_bundle_record(bundle)
        )
    except Exception as e:
        logger.error(f"Failed to build diagnostic bundle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repositories")
async def list_repositories(request: Request, full_scan: bool = False):
    """
    List available (indexed on disk) and loaded repositories

    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    fastcode = _fc(request)

    try:
        info = fastcode.get_status_info(full_scan=full_scan)

        return {
            "status": "success",
            "available": info["available_repositories"],
            "loaded": info["loaded_repositories"],
        }
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load_repository(request: Request, req: LoadRepositoryRequest):
    """Load a repository"""
    fastcode = _fc(request)
    command = map_load_repository_request(req)

    try:
        logger.info(f"Loading repository: {command.source}")
        await asyncio.to_thread(
            fastcode.load_repository, command.source, command.is_url
        )

        info = fastcode.get_status_info()
        return {
            "status": "success",
            "message": "Repository loaded successfully",
            "repo_info": info["repo_info"],
        }

    except Exception as e:
        logger.error(f"Failed to load repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_repository(request: Request, force: bool = False):
    """Index the loaded repository"""
    fastcode = _fc(request)

    if not fastcode.get_status_info()["repo_loaded"]:
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
async def run_index_pipeline(request: Request, req: IndexRunRequest):
    """Run snapshot-based indexing pipeline."""
    fastcode = _fc(request)
    command = map_index_run_request(req)
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
        await asyncio.to_thread(fastcode.invalidate_scan_cache)
        return serialize_index_run_response(serialize_index_run_response_record(result))
    except Exception as e:
        logger.error(f"Index run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index/runs/{run_id}")
async def get_index_run(request: Request, run_id: str):
    """Get index run status/details."""
    fastcode = _fc(request)
    run = fastcode.publishing.get_index_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"status": "success", "run": run}


@app.post("/index/publish/{run_id}")
async def publish_index_run(request: Request, run_id: str, ref_name: str | None = None):
    """Publish an existing index run into manifest and lineage."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.publishing.publish_index_run, run_id, ref_name
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Publish run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/publish/retry")
async def retry_publish_tasks(request: Request, limit: int = 10):
    """Retry pending Terminus lineage publish tasks."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.publishing.retry_pending_publishes, limit
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Retry publish failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redo/process")
async def process_redo_tasks(request: Request, limit: int = 10):
    """Admin endpoint to process pending redo tasks immediately."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(fastcode.publishing.process_redo_tasks, limit)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Redo process failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-and-index")
async def load_and_index(
    request: Request, req: LoadRepositoryRequest, force: bool = False
):
    """Load and index repository in one call"""
    fastcode = _fc(request)
    command = map_load_repository_request(req)

    try:
        logger.info(f"Loading and indexing repository: {command.source}")
        return await asyncio.to_thread(
            fastcode.load_and_index, command.source, command.is_url, force=force
        )

    except Exception as e:
        logger.error(f"Failed to load and index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-repositories")
async def load_repositories(request: Request, req: LoadRepositoriesRequest):
    """Load existing indexed repositories from cache"""
    fastcode = _fc(request)

    if not req.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        logger.info(f"Loading repositories from cache: {req.repo_names}")
        success = await asyncio.to_thread(
            fastcode.load_cached_repos, repo_names=req.repo_names
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
async def index_multiple(request: Request, req: IndexMultipleRequest):
    """Load and index multiple repositories"""
    fastcode = _fc(request)

    if not req.sources:
        raise HTTPException(status_code=400, detail="No repositories provided")

    try:
        logger.info(f"Indexing {len(req.sources)} repositories")
        await asyncio.to_thread(
            fastcode.load_multiple_repositories,
            [
                {
                    "source": command.source,
                    "is_url": command.is_url,
                }
                for command in (
                    map_load_repository_request(source) for source in req.sources
                )
            ],
        )

        await asyncio.to_thread(fastcode.invalidate_scan_cache)

        return {
            "status": "success",
            "message": "Repositories indexed successfully",
            "stats": fastcode.get_repository_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to index multiple repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-zip")
async def upload_repository_zip(request: Request, file: UploadFile = File(...)):
    """Upload and extract repository ZIP file"""
    fastcode = _fc(request)

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
async def upload_and_index(
    request: Request, file: UploadFile = File(...), force: bool = False
):
    """Upload ZIP and index in one call"""
    fastcode = _fc(request)

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
async def query_repository(request: Request, req: QueryRequest):
    """Query by snapshot scope (snapshot_id or repo_name+ref_name)."""
    fastcode = _fc(request)
    query_request = map_snapshot_query_request(req)

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
            fastcode.query.query_snapshot,
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
async def query_snapshot(request: Request, req: QuerySnapshotRequest):
    """Query a specific snapshot or repo/ref manifest head."""
    fastcode = _fc(request)
    query_request = map_snapshot_query_request(req)
    try:
        session_id = query_request.session_id or str(uuid.uuid4())[:8]
        result = await asyncio.to_thread(
            fastcode.query.query_snapshot,
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
async def query_repository_stream(request: Request, _req: QueryRequest):
    """Deprecated in snapshot-cutover mode."""
    raise HTTPException(
        status_code=501,
        detail="query-stream is disabled in snapshot-cutover mode. Use POST /query with snapshot scope.",
    )


@app.get("/summary")
async def get_repository_summary(request: Request):
    """Get repository summary"""
    fastcode = _fc(request)

    info = fastcode.get_status_info()
    if not info["repo_loaded"]:
        raise HTTPException(status_code=400, detail="No repository loaded")

    summary_payload: dict[str, Any] = {
        "status": "success",
    }

    try:
        if info["multi_repo_mode"]:
            summary_payload["summary"] = fastcode.get_repository_stats()
        else:
            summary_payload["summary"] = fastcode.get_repository_summary()
    except Exception as e:
        logger.error(f"Failed to build summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return summary_payload


@app.get("/manifests/{repo_name}/{ref_name}")
async def get_branch_manifest(request: Request, repo_name: str, ref_name: str):
    """Get latest published manifest for repo/ref."""
    fastcode = _fc(request)
    manifest = fastcode.get_branch_manifest(repo_name, ref_name)
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {"status": "success", "manifest": manifest}


@app.get("/repos/{repo_name}/refs")
async def get_repo_refs(request: Request, repo_name: str):
    fastcode = _fc(request)
    try:
        refs = await asyncio.to_thread(fastcode.list_repo_refs, repo_name)
        return {"status": "success", "repo_name": repo_name, "refs": refs}
    except Exception as e:
        logger.error(f"Repo refs lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/manifests/snapshot/{snapshot_id}")
async def get_snapshot_manifest(request: Request, snapshot_id: str):
    """Get latest published manifest for snapshot ID."""
    fastcode = _fc(request)
    manifest = fastcode.get_snapshot_manifest(snapshot_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {"status": "success", "manifest": manifest}


@app.get("/code-status/{snapshot_id}")
async def get_code_status_pack(
    request: Request,
    snapshot_id: str,
    include_graph_facts: bool = True,
):
    """Get a DocKG-facing code_status_pack.v0 artifact for a snapshot."""
    fastcode = _fc(request)
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
async def get_scip_artifact(request: Request, snapshot_id: str):
    """Get preserved SCIP artifact metadata for a snapshot."""
    fastcode = _fc(request)
    artifact = fastcode.get_scip_artifact_ref(snapshot_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="SCIP artifact not found")
    artifacts = fastcode.list_scip_artifact_refs(snapshot_id)
    return {"status": "success", "artifact": artifact, "artifacts": artifacts}


@app.get("/symbols/find")
async def find_symbol(
    request: Request,
    snapshot_id: str,
    symbol_id: str | None = None,
    name: str | None = None,
    path: str | None = None,
):
    fastcode = _fc(request)
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
async def get_graph_callees(
    request: Request, snapshot_id: str, symbol_id: str, max_hops: int = 1
):
    fastcode = _fc(request)
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
async def get_graph_callers(
    request: Request, snapshot_id: str, symbol_id: str, max_hops: int = 1
):
    fastcode = _fc(request)
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
async def get_graph_dependencies(
    request: Request, snapshot_id: str, doc_id: str, max_hops: int = 1
):
    fastcode = _fc(request)
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
async def build_projection(request: Request, req: ProjectionBuildRequest):
    """Build or reuse a cached projection artifact."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.projection.build_projection,
            scope_kind=req.scope_kind,
            snapshot_id=req.snapshot_id,
            repo_name=req.repo_name,
            ref_name=req.ref_name,
            query=req.query,
            target_id=req.target_id,
            filters=req.filters,
            force=req.force,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Projection build failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projection/{projection_id}/{layer}")
async def get_projection_layer(request: Request, projection_id: str, layer: str):
    """Fetch a specific projection layer (L0/L1/L2)."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.projection.get_projection_layer, projection_id, layer
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Projection layer fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projection/{projection_id}/chunks/{chunk_id}")
async def get_projection_chunk(request: Request, projection_id: str, chunk_id: str):
    """Fetch a single projection L2 chunk payload."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.projection.get_projection_chunk, projection_id, chunk_id
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Projection chunk fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projection/snapshot/{snapshot_id}/prefix")
async def get_projection_prefix(request: Request, snapshot_id: str):
    """Get L0+L1 projection as compact JSON for system prompt injection.

    Returns the architectural overview (L0) and navigation structure (L1)
    combined into a single response suitable for injecting into an AI
    agent's system prompt at session start.
    """
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.projection.get_session_prefix, snapshot_id
        )
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session prefix fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent-context/session/{session_id}/latest")
async def get_latest_turn_context(
    request: Request, session_id: str, format: str = "fcx"
):
    """Fetch the latest typed working-memory artifact for a session."""
    fastcode = _fc(request)
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
    request: Request,
    session_id: str,
    turn_number: int,
    format: str = "fcx",
):
    """Fetch a specific typed working-memory artifact for a session turn."""
    fastcode = _fc(request)
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
    request: Request,
    session_id: str,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch the latest durable context bundle for a session."""
    fastcode = _fc(request)
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
    request: Request,
    session_id: str,
    turn_number: int,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch a durable context bundle for a session turn."""
    fastcode = _fc(request)
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
    request: Request,
    bundle_id: str,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch a durable context bundle by bundle ID."""
    fastcode = _fc(request)
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
async def expand_agent_context_bundle_ref(
    request: Request, req: ExpandContextBundleRefRequest
):
    """Expand a single source ref from a durable context bundle."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.expand_context_bundle_ref,
            req.ref_id,
            session_id=req.session_id,
            turn_number=req.turn_number,
            bundle_id=req.bundle_id,
            depth=req.depth,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context bundle ref expansion failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agent-context/bundle/activation")
async def create_agent_context_activation(
    request: Request, req: ContextActivationRequest
):
    """Create and persist an activation record for a context bundle."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.create_context_activation,
            session_id=req.session_id,
            turn_number=req.turn_number,
            bundle_id=req.bundle_id,
            active_ref_ids=req.active_ref_ids,
            active_fact_ids=req.active_fact_ids,
            active_hypothesis_ids=req.active_hypothesis_ids,
            reason=req.reason,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context activation creation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agent-context/handoff")
async def create_agent_context_handoff(
    request: Request, req: AgentContextHandoffRequest
):
    """Create and persist a handoff artifact from a session turn."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.create_handoff,
            req.session_id,
            req.turn_number,
            req.mode,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Handoff creation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/agent-context/handoff/{artifact_id}")
async def get_agent_context_handoff(request: Request, artifact_id: str):
    """Fetch a persisted handoff artifact."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(fastcode.get_handoff_artifact, artifact_id)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Handoff fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agent-context/expand")
async def expand_agent_context_ref(request: Request, req: ExpandContextRefRequest):
    """Expand a single evidence ref from working memory."""
    fastcode = _fc(request)
    try:
        result = await asyncio.to_thread(
            fastcode.expand_context_ref,
            req.session_id,
            req.turn_number,
            req.ref_id,
            req.depth,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context ref expansion failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/new-session", response_model=NewSessionResponse)
async def new_session(request: Request, clear_session_id: str | None = None):
    """Start a new conversation session"""
    fastcode = _fc(request)

    if clear_session_id:
        fastcode.delete_session(clear_session_id)

    session_id = str(uuid.uuid4())[:8]
    return serialize_new_session_response(NewSessionRecord(session_id=session_id))


@app.get("/sessions")
async def list_sessions(request: Request):
    """List all dialogue sessions with titles (sorted by last update time)"""
    fastcode = _fc(request)
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
async def get_session(request: Request, session_id: str):
    """Get full dialogue history for a session"""
    fastcode = _fc(request)
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
async def delete_session(request: Request, session_id: str):
    """Delete a single dialogue session"""
    fastcode = _fc(request)

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
async def delete_repositories(request: Request, req: DeleteReposRequest):
    """Delete one or more repositories and all associated data"""
    fastcode = _fc(request)

    if not req.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        results = []
        for repo_name in req.repo_names:
            result = await asyncio.to_thread(
                fastcode.remove_repository,
                repo_name,
                delete_source=req.delete_source,
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
async def clear_cache(request: Request):
    """Clear cache"""
    fastcode = _fc(request)

    success = await asyncio.to_thread(fastcode.clear_cache)

    if success:
        return {"status": "success", "message": "Cache cleared"}
    return {
        "status": "failed",
        "message": "Failed to clear cache or cache disabled",
    }


@app.get("/cache-stats")
async def get_cache_stats(request: Request):
    """Get cache statistics"""
    fastcode = _fc(request)

    return await asyncio.to_thread(fastcode.get_cache_stats)


@app.post("/refresh-index-cache")
async def refresh_index_cache(request: Request):
    """Force refresh the index scan cache"""
    fastcode = _fc(request)

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
async def unload_repository(request: Request):
    """Unload current repository"""
    fc = getattr(request.app.state, "fastcode", None)
    if fc is not None:
        await asyncio.to_thread(fc.cleanup)
        request.app.state.fastcode = FastCode()

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
