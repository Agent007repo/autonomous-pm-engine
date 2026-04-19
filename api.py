"""
api.py

FastAPI REST interface for the Autonomous PM Engine.

Endpoints:
  POST /analyze        — Upload documents and run the full pipeline
  GET  /health         — Health check
  GET  /status/{job_id} — Poll job status (async jobs)

Run with:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from src.config.settings import get_settings
from src.orchestration.workflow import run_pipeline

app = FastAPI(
    title="Autonomous PM Engine",
    description="Multi-agent PRD generation from raw customer feedback.",
    version="1.0.0",
)

# In-memory job store (replace with Redis for production)
_jobs: dict[str, dict[str, Any]] = {}


# ── Request / Response schemas ────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str                        # 'pending' | 'running' | 'complete' | 'failed'
    created_at: str
    completed_at: str | None
    product_name: str
    prd_path: str | None
    roadmap_path: str | None
    matrix_path: str | None
    errors: list[str]
    stats: dict[str, Any]


# ── Background job runner ─────────────────────────────────────────────────────

async def _run_pipeline_job(
    job_id: str,
    input_dir: str,
    product_name: str,
    product_context: str,
) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        # Run the synchronous pipeline in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                input_dir=input_dir,
                product_name=product_name,
                product_context=product_context,
            ),
        )
        _jobs[job_id].update(
            {
                "status": "complete" if final_state.get("completed") else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "prd_path": final_state.get("final_prd_path"),
                "roadmap_path": final_state.get("final_roadmap_path"),
                "matrix_path": final_state.get("final_matrix_path"),
                "errors": final_state.get("errors", []),
                "stats": {
                    "raw_documents": final_state.get("raw_document_count", 0),
                    "chunks": final_state.get("chunk_count", 0),
                    "pain_points": final_state.get("pain_point_count", 0),
                    "critique_rounds": final_state.get("critique_rounds_completed", 0),
                    "final_score": (
                        final_state["critique_history"][-1]["score"]
                        if final_state.get("critique_history")
                        else None
                    ),
                },
            }
        )
    except Exception as e:
        logger.exception(f"Pipeline job {job_id} failed")
        _jobs[job_id].update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "errors": [str(e)],
            }
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="One or more feedback documents"),
    product_name: str = Form(default="Product", description="Product name"),
    product_context: str = Form(
        default="A software product used by business customers.",
        description="One-sentence product description",
    ),
) -> AnalyzeResponse:
    """
    Upload customer feedback documents and start an async pipeline run.
    Returns a job_id you can use to poll /status/{job_id}.
    """
    cfg = get_settings()
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded files to a temporary job-specific directory
    job_input_dir = Path(cfg.output_dir) / "jobs" / job_id / "input"
    job_input_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for upload in files:
        dest = job_input_dir / (upload.filename or f"file_{len(saved_files)}")
        dest.write_bytes(await upload.read())
        saved_files.append(str(dest))
        logger.info(f"Saved upload: {dest}")

    if not saved_files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "product_name": product_name,
        "prd_path": None,
        "roadmap_path": None,
        "matrix_path": None,
        "errors": [],
        "stats": {},
    }

    background_tasks.add_task(
        _run_pipeline_job,
        job_id=job_id,
        input_dir=str(job_input_dir),
        product_name=product_name,
        product_context=product_context,
    )

    return AnalyzeResponse(
        job_id=job_id,
        status="pending",
        message=f"Pipeline started. Poll GET /status/{job_id} for updates.",
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    """Poll the status of a running or completed pipeline job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    job = _jobs[job_id]
    return JobStatus(**job)


@app.get("/download/{job_id}/{doc_type}")
async def download_output(job_id: str, doc_type: str) -> FileResponse:
    """
    Download a generated output file.
    doc_type: 'prd' | 'roadmap' | 'matrix'
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = _jobs[job_id]
    path_key = {"prd": "prd_path", "roadmap": "roadmap_path", "matrix": "matrix_path"}.get(
        doc_type
    )
    if not path_key:
        raise HTTPException(status_code=400, detail=f"Unknown doc_type: {doc_type}")

    file_path = job.get(path_key)
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Output '{doc_type}' not yet available for job '{job_id}'.",
        )

    return FileResponse(
        path=file_path,
        media_type="text/markdown",
        filename=Path(file_path).name,
    )


@app.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """List all pipeline jobs and their statuses."""
    return list(_jobs.values())
