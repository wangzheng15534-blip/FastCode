#!/usr/bin/env python3
"""
FastCode 2.0 - Web Interface routes.

Route handlers only — the FastAPI app, lifespan, CORS, static files,
and logging are wired by main/serve.py.  This module exports an APIRouter.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

import asyncio
import json as json_module
import logging
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from fastcode.api.inbound import (
    AgentContextHandoffRequest,
    ContextActivationRequest,
    DeleteReposRequest,
    ExpandContextBundleRefRequest,
    ExpandContextRefRequest,
    IndexMultipleRequest,
    LoadRepositoriesRequest,
    LoadRepositoryRequest,
    QueryRequest,
    map_load_repository_request,
    map_repository_query_request,
)
from fastcode.api.outbound import (
    ApiStatus,
    NewSessionRecord,
    NewSessionResponse,
    QueryResponse,
    StatusResponse,
)
from fastcode.api.serialization import (
    serialize_dialogue_history,
    serialize_new_session_response,
    serialize_query_response,
    serialize_query_response_record,
    serialize_query_sources,
    serialize_status_response,
    serialize_status_response_record,
)
from fastcode.runtime_support.health import readiness_health
from fastcode.utils.archive import UnsafeArchiveError

logger = logging.getLogger(__name__)

router = APIRouter()


def _facades(request: Request) -> Any:
    """Get injected FacadeContainer from app state (duck-typed)."""
    facades = getattr(request.app.state, "facades", None)
    if facades is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    return facades


@router.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the main web interface"""
    html_file = Path(__file__).parent.parent / "web_interface.html"
    if html_file.exists():
        return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="Web interface not found")


@router.get("/api/status", response_model=StatusResponse)
async def get_status(request: Request, full_scan: bool = False):
    """
    Get system status

    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    facades = _facades(request)

    info = facades.store.get_status_info(full_scan=full_scan)

    return serialize_status_response(
        serialize_status_response_record(
            status=ApiStatus.READY if info["repo_indexed"] else ApiStatus.NOT_READY,
            repo_loaded=info["repo_loaded"],
            repo_indexed=info["repo_indexed"],
            repo_info=info["repo_info"],
            available_repositories=info["available_repositories"],
            loaded_repositories=info["loaded_repositories"],
        )
    )


@router.get("/api/health")
async def health_check(request: Request):
    """Lightweight health check endpoint (no expensive operations)"""
    facades = _facades(request)

    info = facades.store.get_status_info()
    health = readiness_health(
        repo_loaded=info["repo_loaded"],
        repo_indexed=info["repo_indexed"],
        details={"multi_repo_mode": info["multi_repo_mode"]},
    )
    return {
        "status": health.status,
        **health.details,
    }


@router.get("/api/repositories")
async def list_repositories(request: Request, full_scan: bool = False):
    """
    List available (indexed on disk) and loaded repositories

    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    facades = _facades(request)

    try:
        info = facades.store.get_status_info(full_scan=full_scan)

        return {
            "status": "success",
            "available": info["available_repositories"],
            "loaded": info["loaded_repositories"],
        }
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/load")
async def load_repository(request: Request, req: LoadRepositoryRequest):
    """Load a repository"""
    facades = _facades(request)
    command = map_load_repository_request(req)

    try:
        logger.info(f"Loading repository: {command.source}")
        await asyncio.to_thread(
            facades.indexing.load_repository, command.source, command.is_url
        )

        info = facades.store.get_status_info()
        return {
            "status": "success",
            "message": "Repository loaded successfully",
            "repo_info": info["repo_info"],
        }

    except Exception as e:
        logger.error(f"Failed to load repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/index")
async def index_repository(request: Request, force: bool = False):
    """Index the loaded repository"""
    facades = _facades(request)

    if not facades.store.get_status_info()["repo_loaded"]:
        raise HTTPException(status_code=400, detail="No repository loaded")

    try:
        logger.info("Indexing repository")
        await asyncio.to_thread(facades.indexing.index_repository, force=force)

        return {
            "status": "success",
            "message": "Repository indexed successfully",
            "summary": facades.store.get_repository_summary(),
        }

    except Exception as e:
        logger.error(f"Failed to index repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/index-multiple")
async def index_multiple(request: Request, req: IndexMultipleRequest):
    """Load and index multiple repositories"""
    facades = _facades(request)

    if not req.sources:
        raise HTTPException(status_code=400, detail="No repositories provided")

    try:
        logger.info(f"Indexing {len(req.sources)} repositories")
        await asyncio.to_thread(
            facades.indexing.load_multiple_repositories,
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

        # Invalidate scan cache since we just added/updated indexes
        await asyncio.to_thread(
            facades.cache.invalidate_scan_cache,
        )

        return {
            "status": "success",
            "message": "Repositories indexed successfully",
            "stats": facades.store.get_repository_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to index multiple repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/load-and-index")
async def load_and_index(
    request: Request, req: LoadRepositoryRequest, force: bool = False
):
    """Load and index repository in one call"""
    facades = _facades(request)
    command = map_load_repository_request(req)

    try:
        logger.info(f"Loading and indexing repository: {command.source}")
        return await asyncio.to_thread(
            facades.indexing.load_and_index,
            command.source,
            command.is_url,
            force=force,
        )

    except Exception as e:
        logger.error(f"Failed to load and index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/load-repositories")
async def load_repositories(request: Request, req: LoadRepositoriesRequest):
    """Load existing indexed repositories from cache"""
    facades = _facades(request)

    if not req.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        logger.info(f"Loading repositories from cache: {req.repo_names}")
        success = await asyncio.to_thread(
            facades.cache.load_cached_repos, repo_names=req.repo_names
        )

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to load repositories from cache"
            )

        return {
            "status": "success",
            "loaded": facades.store.list_repositories(),
            "stats": facades.store.get_repository_stats(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/upload-zip")
async def upload_repository_zip(request: Request, file: UploadFile = File(...)):
    """Upload and extract repository ZIP file to repos directory (same as URL download)"""
    facades = _facades(request)

    # Validate file type
    filename = file.filename
    if not filename or not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    try:
        file_bytes = await file.read()
        return await asyncio.to_thread(
            facades.indexing.upload_repository_zip, file_bytes, filename
        )
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except UnsafeArchiveError as e:
        logger.warning(f"Rejected unsafe ZIP file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upload and extract ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/upload-and-index")
async def upload_and_index(
    request: Request, file: UploadFile = File(...), force: bool = False
):
    """Upload ZIP and index in one call"""
    facades = _facades(request)

    filename = file.filename
    if not filename or not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    try:
        logger.info("Indexing uploaded repository")
        file_bytes = await file.read()
        return await asyncio.to_thread(
            facades.indexing.upload_and_index,
            file_bytes,
            filename,
            force=force,
        )
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except UnsafeArchiveError as e:
        logger.warning(f"Rejected unsafe ZIP file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to index uploaded repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/query", response_model=QueryResponse)
async def query_repository(request: Request, req: QueryRequest):
    """Query the repository"""
    facades = _facades(request)
    query_request = map_repository_query_request(req)

    if not facades.store.get_status_info()["repo_indexed"]:
        raise HTTPException(status_code=400, detail="Repository not indexed")

    try:
        # Derive session handling for both modes (single-turn keeps a session for history)
        session_id = query_request.session_id or str(uuid.uuid4())[:8]
        if query_request.multi_turn and not query_request.session_id:
            logger.info(f"Generated new multi-turn session: {session_id}")
        elif not query_request.session_id:
            logger.info(f"Generated session for single-turn request: {session_id}")

        logger.info(f"Processing query: {query_request.question}")
        result = await asyncio.to_thread(
            facades.query.query,
            query_request.question,
            dict(query_request.filters) if query_request.filters else None,
            repo_filter=list(query_request.repo_filter) or None,
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


@router.post("/api/query-stream")
async def query_repository_stream(request: Request, req: QueryRequest):
    """Query the repository with streaming response (SSE)"""
    facades = _facades(request)

    # Derive session handling
    session_id = req.session_id or str(uuid.uuid4())[:8]
    if req.multi_turn and not req.session_id:
        logger.info(f"Generated new multi-turn session: {session_id}")
    elif not req.session_id:
        logger.info(f"Generated session for single-turn request: {session_id}")

    logger.info(f"Processing streaming query: {req.question}")

    if not facades.store.get_status_info()["repo_indexed"]:
        raise HTTPException(status_code=400, detail="Repository not indexed")

    async def event_generator():
        """Generate SSE events from query_stream"""
        try:
            for chunk, metadata in facades.query.query_stream(
                req.question,
                req.filters,
                repo_filter=req.repo_filter,
                session_id=session_id,
                enable_multi_turn=req.multi_turn,
            ):
                if metadata:
                    # Send metadata as JSON event
                    status = metadata.get("status", "")
                    if status == "retrieving":
                        event_data = {"type": "status", "status": "retrieving"}
                    elif status == "generating":
                        event_data = {
                            "type": "status",
                            "status": "generating",
                            "sources": serialize_query_sources(metadata.get("sources")),
                            "session_id": session_id,
                        }
                    elif status == "complete":
                        event_data = {
                            "type": "done",
                            "sources": serialize_query_sources(metadata.get("sources")),
                            "context_elements": metadata.get("context_elements", 0),
                            "session_id": session_id,
                        }
                    elif "error" in metadata:
                        event_data = {"type": "error", "error": metadata["error"]}
                    else:
                        continue
                    yield f"data: {json_module.dumps(event_data)}\n\n"
                elif chunk:
                    # Send text chunk
                    event_data = {"type": "chunk", "content": chunk}
                    yield f"data: {json_module.dumps(event_data)}\n\n"

                # Small delay to allow browser to process
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json_module.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/summary")
async def get_repository_summary(request: Request):
    """Get repository summary"""
    facades = _facades(request)

    info = facades.store.get_status_info()
    if not info["repo_loaded"]:
        raise HTTPException(status_code=400, detail="No repository loaded")

    summary_payload: dict[str, Any] = {
        "status": "success",
    }

    try:
        if info["multi_repo_mode"]:
            summary_payload["summary"] = facades.store.get_repository_stats()
        else:
            summary_payload["summary"] = (
                facades.store.get_repository_summary()
            )
    except Exception as e:
        logger.error(f"Failed to build summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return summary_payload


@router.post("/api/clear-cache")
async def clear_cache(request: Request):
    """Clear cache"""
    facades = _facades(request)

    success = await asyncio.to_thread(facades.cache.clear_cache)

    if success:
        return {"status": "success", "message": "Cache cleared"}
    return {
        "status": "failed",
        "message": "Failed to clear cache or cache disabled",
    }


@router.post("/api/refresh-index-cache")
async def refresh_index_cache(request: Request):
    """Force refresh the index scan cache"""
    facades = _facades(request)

    try:
        available_repos = await asyncio.to_thread(
            facades.cache.refresh_index_cache,
        )

        return {
            "status": "success",
            "message": "Index cache refreshed",
            "repository_count": len(available_repos),
        }
    except Exception as e:
        logger.error(f"Failed to refresh index cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/new-session", response_model=NewSessionResponse)
async def new_session(request: Request, clear_session_id: str | None = None):
    """Start a new conversation session"""
    facades = _facades(request)

    if clear_session_id:
        facades.context.delete_session(clear_session_id)

    session_id = str(uuid.uuid4())[:8]
    return serialize_new_session_response(NewSessionRecord(session_id=session_id))


@router.get("/api/agent-context/session/{session_id}/latest")
async def get_latest_turn_context(
    request: Request, session_id: str, format: str = "fcx"
):
    """Fetch the latest typed working-memory artifact for a session."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.get_turn_context,
            session_id,
            None,
            format,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Latest turn context fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/agent-context/session/{session_id}/turn/{turn_number}")
async def get_turn_context(
    request: Request,
    session_id: str,
    turn_number: int,
    format: str = "fcx",
):
    """Fetch a specific typed working-memory artifact for a session turn."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.get_turn_context,
            session_id,
            turn_number,
            format,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Turn context fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/agent-context/session/{session_id}/bundle/latest")
async def get_latest_context_bundle(
    request: Request,
    session_id: str,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch the latest durable context bundle for a session."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.get_context_bundle,
            session_id,
            None,
            format,
            token_budget,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Latest context bundle fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/agent-context/session/{session_id}/bundle/{turn_number}")
async def get_context_bundle(
    request: Request,
    session_id: str,
    turn_number: int,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch a durable context bundle for a session turn."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.get_context_bundle,
            session_id,
            turn_number,
            format,
            token_budget,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context bundle fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/agent-context/bundle/{bundle_id}")
async def get_context_bundle_by_id(
    request: Request,
    bundle_id: str,
    format: str = "json",
    token_budget: int = 2048,
):
    """Fetch a durable context bundle by bundle ID."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.get_context_bundle_by_id,
            bundle_id,
            format,
            token_budget,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context bundle fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/agent-context/bundle/expand")
async def expand_agent_context_bundle_ref(
    request: Request, req: ExpandContextBundleRefRequest
):
    """Expand a single source ref from a durable context bundle."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.expand_context_bundle_ref,
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


@router.post("/api/agent-context/bundle/activation")
async def create_agent_context_activation(
    request: Request, req: ContextActivationRequest
):
    """Create and persist an activation record for a context bundle."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.create_context_activation,
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


@router.post("/api/agent-context/handoff")
async def create_agent_context_handoff(
    request: Request, req: AgentContextHandoffRequest
):
    """Create and persist a handoff artifact from a session turn."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.create_handoff,
            req.session_id,
            req.turn_number,
            req.mode,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Handoff creation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/agent-context/handoff/{artifact_id}")
async def get_agent_context_handoff(request: Request, artifact_id: str):
    """Fetch a persisted handoff artifact."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.get_handoff_artifact,
            artifact_id,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Handoff fetch failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/agent-context/expand")
async def expand_agent_context_ref(request: Request, req: ExpandContextRefRequest):
    """Expand a single evidence ref from working memory."""
    facades = _facades(request)
    try:
        result = await asyncio.to_thread(
            facades.context.expand_context_ref,
            req.session_id,
            req.turn_number,
            req.ref_id,
            req.depth,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Context ref expansion failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/sessions")
async def list_sessions(request: Request):
    """List all dialogue sessions with titles (sorted by last update time)"""
    facades = _facades(request)
    try:
        sessions = facades.context.list_sessions()

        # Format sessions for better display
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
                "multi_turn": session.get("multi_turn", False),
            }
            formatted_sessions.append(formatted_session)

        return {"status": "success", "sessions": formatted_sessions}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/session/{session_id}")
async def get_session(request: Request, session_id: str):
    """Get full dialogue history for a session"""
    facades = _facades(request)
    try:
        history = facades.context.get_session_history(session_id) or []

        multi_turn = facades.context.get_session_multi_turn(session_id)

        return {
            "status": "success",
            "session_id": session_id,
            "history": serialize_dialogue_history(history),
            "multi_turn": multi_turn,
        }
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/delete-repos")
async def delete_repositories(request: Request, req: DeleteReposRequest):
    """Delete one or more repositories and all associated data"""
    facades = _facades(request)

    if not req.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        results = []
        for repo_name in req.repo_names:
            result = await asyncio.to_thread(
                facades.remove_repository,
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


@router.delete("/api/session/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a single dialogue session"""
    facades = _facades(request)

    try:
        history = facades.context.get_session_history(session_id)
        if not history:
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found"
            )

        success = facades.context.delete_session(session_id)
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
