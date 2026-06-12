"""Phase A — Detection + clustering (synchronous compute callback).

This is the Compute-service endpoint the orchestrator (Airflow DAG) calls. It
blocks until the pipeline finishes (or fails) and returns the review payload. An
internal Job row carries per-stage progress for logs/audit.

Idempotency (v4 §9.4): pass an ``Idempotency-Key`` header (the orchestrator's
dag_run_id). A repeat call with a key whose run already SUCCEEDED replays the
result without recomputing; a key whose run is still in flight returns 409
CONFLICT_BUSY.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.deps import get_project, require_service_token
from app.api.v1.clustering import build_clustering_payload
from app.db import models
from app.db.session import get_db
from app.workers.tasks import job_a_analyze

router = APIRouter()

# Includes ANALYZING so the trigger (which already moved the project into the
# in-progress state) can hand off to this compute callback.
_ANALYZE_OK = {"UPLOADED", "ANALYZING", "AWAITING_LABELS", "FAILED"}


@router.post("/projects/{project_id}/analyze")
def start_analyze(
    request: Request,
    project=Depends(get_project),
    db: Session = Depends(get_db),
    _svc: str = Depends(require_service_token),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    # Idempotent replay / in-flight guard.
    if idempotency_key:
        prior = (
            db.query(models.Job)
            .filter_by(project_id=project.id, celery_task_id=idempotency_key)
            .order_by(models.Job.started_at.desc())
            .first()
        )
        if prior and prior.state == "SUCCEEDED":
            db.refresh(project)
            return build_clustering_payload(request, project)
        if prior and prior.state in ("QUEUED", "RUNNING"):
            raise HTTPException(409, {
                "code": "CONFLICT_BUSY",
                "message": "A run with this Idempotency-Key is already in progress",
                "project_id": project.id,
            })

    if project.state not in _ANALYZE_OK:
        raise HTTPException(409, {
            "code": "INVALID_STATE",
            "message": f"Cannot analyze from state {project.state}",
            "project_id": project.id,
        })
    if not project.orthos:
        raise HTTPException(400, {
            "code": "BAD_REQUEST",
            "message": "Upload at least one orthomosaic first",
            "project_id": project.id,
        })

    previous_state = project.state
    job = models.Job(
        project_id=project.id, type="analyze", state="RUNNING",
        started_at=datetime.utcnow(), celery_task_id=idempotency_key,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    project.state = "ANALYZING"
    project.error = None
    db.add(project)
    db.commit()

    try:
        job_a_analyze.apply(args=[project.id, job.id]).get(propagate=True)
    except Exception as exc:
        # The task's _fail() already wrote FAILED state + the error tail.
        db.refresh(project)
        db.refresh(job)
        stage = job.current_stage or "unknown"
        if project.state == "ANALYZING":
            project.state = previous_state
            db.add(project)
            db.commit()
        raise HTTPException(500, {
            "code": "COMPUTE_FAILED",
            "message": project.error or str(exc),
            "project_id": project.id,
            "stage": stage,
        }) from exc

    db.refresh(project)
    return build_clustering_payload(request, project)
