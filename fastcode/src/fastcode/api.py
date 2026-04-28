"""
FastCode 2.0 - REST API
Complete API with all features from web_app.py
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import os
import platform

if platform.system() == "Darwin":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
import logging
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastcode import FastCode
from fastcode.utils import safe_jsonable


# Pydantic models
class LoadRepositoryRequest(BaseModel):
    source: str = Field(..., description="Repository URL or local path")
    is_url: bool | None = Field(
        None,
        description="True if source is URL, False if local path. If omitted, auto-detect.",
    )


class IndexRunRequest(BaseModel):
    source: str = Field(..., description="Repository URL or local path")
    is_url: bool | None = Field(None, description="Explicit source type override")
    ref: str | None = Field(None, description="Branch/tag/ref to index")
    commit: str | None = Field(None, description="Commit hash to index")
    force: bool = Field(
        False, description="Force re-index even if snapshot already exists"
    )
    publish: bool = Field(True, description="Publish manifest after indexing")
    enable_scip: bool = Field(True, description="Enable SCIP extraction path")
    scip_artifact_path: str | None = Field(
        None, description="Optional pre-built SCIP artifact path"
    )


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the repository")
    snapshot_id: str | None = Field(None, description="Direct snapshot ID")
    repo_name: str | None = Field(
        None, description="Repository name (for ref resolution)"
    )
    ref_name: str | None = Field(
        None, description="Branch/ref name (for ref resolution)"
    )
    filters: dict[str, Any] | None = Field(None, description="Optional filters")
    multi_turn: bool = Field(False, description="Enable multi-turn mode")
    session_id: str | None = Field(
        None, description="Session ID for multi-turn dialogue"
    )


class QuerySnapshotRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    snapshot_id: str | None = Field(None, description="Direct snapshot ID")
    repo_name: str | None = Field(
        None, description="Repository name (when resolving by ref)"
    )
    ref_name: str | None = Field(
        None, description="Branch/ref name (when resolving by ref)"
    )
    filters: dict[str, Any] | None = Field(
        None, description="Optional retrieval filters"
    )
    multi_turn: bool = Field(False, description="Enable multi-turn mode")
    session_id: str | None = Field(
        None, description="Session ID for multi-turn dialogue"
    )


class ProjectionBuildRequest(BaseModel):
    scope_kind: str = Field(
        ..., description="Projection scope: snapshot | query | entity"
    )
    snapshot_id: str | None = Field(None, description="Direct snapshot ID")
    repo_name: str | None = Field(
        None, description="Repository name (for ref resolution)"
    )
    ref_name: str | None = Field(
        None, description="Branch/ref name (for ref resolution)"
    )
    query: str | None = Field(
        None, description="Query text for query-scoped projection"
    )
    target_id: str | None = Field(
        None, description="Entity ID/path for entity-scoped projection"
    )
    filters: dict[str, Any] | None = Field(None, description="Optional scope filters")
    force: bool = Field(False, description="Force regeneration even when cached")


class QueryResponse(BaseModel):
    answer: str
    query: str
    context_elements: int
    sources: list[dict[str, Any]]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    session_id: str | None = None


class LoadRepositoriesRequest(BaseModel):
    repo_names: list[str] = Field(
        ..., description="Repository names to load from existing indexes"
    )


class IndexMultipleRequest(BaseModel):
    sources: list[LoadRepositoryRequest] = Field(
        ..., description="Multiple repositories to load and index"
    )


class NewSessionResponse(BaseModel):
    session_id: str


class DeleteReposRequest(BaseModel):
    repo_names: list[str] = Field(..., description="Repository names to delete")
    delete_source: bool = Field(
        True, description="Also delete cloned source code in repos/"
    )


class StatusResponse(BaseModel):
    status: str
    repo_loaded: bool
    repo_indexed: bool
    repo_info: dict[str, Any]
    graph_backend: str | None = None
    storage_backend: str | None = None
    retrieval_backend: str | None = None
    available_repositories: list[dict[str, Any]] = Field(default_factory=list)
    loaded_repositories: list[dict[str, Any]] = Field(default_factory=list)


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global FastCode instance
fastcode_instance: FastCode | None = None

# Setup logging
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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

    return {
        "status": "healthy",
        "repo_loaded": fastcode_instance.repo_loaded,
        "repo_indexed": fastcode_instance.repo_indexed,
        "multi_repo_mode": fastcode_instance.multi_repo_mode,
        "storage_backend": fastcode_instance.snapshot_store.db_runtime.backend,
        "retrieval_backend": fastcode_instance.config.get("retrieval", {}).get(
            "backend", "legacy"
        ),
    }


@app.get("/status", response_model=StatusResponse)
async def get_status(full_scan: bool = False):
    """
    Get system status

    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    fastcode = _ensure_fastcode_initialized()

    available_repos = fastcode.vector_store.scan_available_indexes(
        use_cache=not full_scan
    )
    loaded_repos = fastcode.list_repositories()

    return StatusResponse(
        status="ready" if fastcode.repo_indexed else "not_ready",
        repo_loaded=fastcode.repo_loaded,
        repo_indexed=fastcode.repo_indexed,
        repo_info=fastcode.repo_info,
        graph_backend=fastcode.config.get("retrieval", {}).get(
            "graph_backend", "legacy"
        ),
        storage_backend=fastcode.snapshot_store.db_runtime.backend,
        retrieval_backend=fastcode.config.get("retrieval", {}).get("backend", "legacy"),
        available_repositories=available_repos,
        loaded_repositories=loaded_repos,
    )


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

    try:
        logger.info(f"Loading repository: {request.source}")
        fastcode.load_repository(request.source, request.is_url)

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
        fastcode.index_repository(force=force)

        fastcode.vector_store.invalidate_scan_cache()

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


@app.post("/index/run")
async def run_index_pipeline(request: IndexRunRequest):
    """Run snapshot-based indexing pipeline."""
    fastcode = _ensure_fastcode_initialized()
    try:
        result = await asyncio.to_thread(
            fastcode.run_index_pipeline,
            source=request.source,
            is_url=request.is_url,
            ref=request.ref,
            commit=request.commit,
            force=request.force,
            publish=request.publish,
            scip_artifact_path=request.scip_artifact_path,
            enable_scip=request.enable_scip,
        )
        fastcode.vector_store.invalidate_scan_cache()
        return {"status": "success", "result": result}
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

    try:
        logger.info(f"Loading repository: {request.source}")
        await asyncio.to_thread(
            fastcode.load_repository, request.source, request.is_url
        )

        logger.info("Indexing repository")
        await asyncio.to_thread(fastcode.index_repository, force=force)

        fastcode.vector_store.invalidate_scan_cache()

        return {
            "status": "success",
            "message": "Repository loaded and indexed successfully",
            "summary": fastcode.get_repository_summary(),
        }

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
        success = fastcode._load_multi_repo_cache(repo_names=request.repo_names)

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
        fastcode.load_multiple_repositories([s.model_dump() for s in request.sources])

        fastcode.vector_store.invalidate_scan_cache()

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
    if not filename or not filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {max_size / (1024 * 1024)}MB",
        )

    try:
        repo_name = filename.rsplit(".", 1)[0]
        for suffix in ["-main", "-master", "_main", "_master"]:
            if repo_name.endswith(suffix):
                repo_name = repo_name[: -len(suffix)]
                break

        repo_workspace = getattr(fastcode.loader, "safe_repo_root", "./repos")
        repos_dir = Path(repo_workspace)
        repos_dir.mkdir(parents=True, exist_ok=True)
        repo_path = repos_dir / repo_name

        if repo_path.exists():
            fastcode.loader._backup_existing_repo(str(repo_path))

        temp_dir = tempfile.mkdtemp(prefix="fastcode_upload_")
        zip_path = Path(temp_dir) / filename

        logger.info(f"Saving uploaded ZIP file: {filename} ({file_size} bytes)")

        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir(exist_ok=True)

        logger.info(f"Extracting ZIP file to temporary directory: {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        extracted_items = list(extract_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_repo_path = extracted_items[0]
        else:
            source_repo_path = extract_dir

        logger.info(f"Moving repository to: {repo_path}")
        shutil.move(str(source_repo_path), str(repo_path))

        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp directory: {cleanup_error}")

        logger.info(f"Loading repository from: {repo_path}")
        fastcode.load_repository(str(repo_path), is_url=False)

        return {
            "status": "success",
            "message": f"ZIP file '{file.filename}' uploaded and extracted to repos/{repo_name}",
            "repo_info": fastcode.repo_info,
            "repo_path": str(repo_path),
        }

    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"Failed to upload and extract ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-index")
async def upload_and_index(file: UploadFile = File(...), force: bool = False):
    """Upload ZIP and index in one call"""
    fastcode = _ensure_fastcode_initialized()

    upload_result = await upload_repository_zip(file)

    if upload_result["status"] != "success":
        return upload_result

    try:
        logger.info("Indexing uploaded repository")
        fastcode.index_repository(force=force)

        fastcode.vector_store.invalidate_scan_cache()

        return {
            "status": "success",
            "message": "Repository uploaded and indexed successfully",
            "summary": fastcode.get_repository_summary(),
        }

    except Exception as e:
        logger.error(f"Failed to index uploaded repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query by snapshot scope (snapshot_id or repo_name+ref_name)."""
    fastcode = _ensure_fastcode_initialized()

    if not request.snapshot_id and not (request.repo_name and request.ref_name):
        raise HTTPException(
            status_code=400,
            detail="Query requires snapshot_id or repo_name+ref_name",
        )

    try:
        session_id = request.session_id or str(uuid.uuid4())[:8]
        if request.multi_turn and not request.session_id:
            logger.info(f"Generated new multi-turn session: {session_id}")
        elif not request.session_id:
            logger.info(f"Generated session for single-turn request: {session_id}")

        logger.info(f"Processing query: {request.question}")
        result = await asyncio.to_thread(
            fastcode.query_snapshot,
            question=request.question,
            snapshot_id=request.snapshot_id,
            repo_name=request.repo_name,
            ref_name=request.ref_name,
            filters=request.filters,
            session_id=session_id,
            enable_multi_turn=request.multi_turn,
        )

        prompt_tokens = result.get("prompt_tokens")
        completion_tokens = result.get("completion_tokens")
        total_tokens = result.get("total_tokens")

        if total_tokens is None and prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens

        sources = result.get("sources", [])
        serialized_sources = [safe_jsonable(source) for source in sources]

        return QueryResponse(
            answer=result.get("answer", ""),
            query=result.get("query", ""),
            context_elements=result.get("context_elements", 0),
            sources=serialized_sources,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-snapshot", response_model=QueryResponse)
async def query_snapshot(request: QuerySnapshotRequest):
    """Query a specific snapshot or repo/ref manifest head."""
    fastcode = _ensure_fastcode_initialized()
    try:
        session_id = request.session_id or str(uuid.uuid4())[:8]
        result = await asyncio.to_thread(
            fastcode.query_snapshot,
            question=request.question,
            repo_name=request.repo_name,
            ref_name=request.ref_name,
            snapshot_id=request.snapshot_id,
            filters=request.filters,
            session_id=session_id,
            enable_multi_turn=request.multi_turn,
        )

        prompt_tokens = result.get("prompt_tokens")
        completion_tokens = result.get("completion_tokens")
        total_tokens = result.get("total_tokens")
        if total_tokens is None and prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens

        return QueryResponse(
            answer=result.get("answer", ""),
            query=result.get("query", ""),
            context_elements=result.get("context_elements", 0),
            sources=[safe_jsonable(source) for source in result.get("sources", [])],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            session_id=session_id,
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


@app.get("/scip/artifacts/{snapshot_id}")
async def get_scip_artifact(snapshot_id: str):
    """Get preserved SCIP artifact metadata for a snapshot."""
    fastcode = _ensure_fastcode_initialized()
    artifact = fastcode.get_scip_artifact_ref(snapshot_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="SCIP artifact not found")
    return {"status": "success", "artifact": artifact}


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


@app.post("/new-session", response_model=NewSessionResponse)
async def new_session(clear_session_id: str | None = None):
    """Start a new conversation session"""
    fastcode = _ensure_fastcode_initialized()

    if clear_session_id:
        fastcode.delete_session(clear_session_id)

    session_id = str(uuid.uuid4())[:8]
    return NewSessionResponse(session_id=session_id)


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
        safe_history = [safe_jsonable(turn) for turn in history]
        return {"status": "success", "session_id": session_id, "history": safe_history}
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
            result = fastcode.remove_repository(
                repo_name, delete_source=request.delete_source
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

    success = fastcode.cache_manager.clear()

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

    stats = fastcode.cache_manager.get_stats()
    return stats


@app.post("/refresh-index-cache")
async def refresh_index_cache():
    """Force refresh the index scan cache"""
    fastcode = _ensure_fastcode_initialized()

    try:
        fastcode.vector_store.invalidate_scan_cache()
        available_repos = fastcode.vector_store.scan_available_indexes(use_cache=False)

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
        fastcode_instance.cleanup()
        fastcode_instance = FastCode()

    return {"status": "success", "message": "Repository unloaded"}


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    logger.info(f"Starting FastCode API at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    uvicorn.run("fastcode.api:app", host=host, port=port, reload=reload)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FastCode API Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    start_api(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
