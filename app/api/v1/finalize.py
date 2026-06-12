"""Phase B — Species assignment + validation + export (synchronous compute callback).

Compute-service endpoint the orchestrator calls. Blocks until the KMZ is built
and returns the final results payload. Supports the same ``Idempotency-Key``
replay/in-flight semantics as ``/analyze`` (v4 §9.4).
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_project, require_service_token
from app.api.v1.results import build_results_payload
from app.db import models
from app.db.session import get_db
from app.workers.tasks import job_b_finalize

router = APIRouter()

# Includes FINALIZING so the trigger can hand off to this compute callback.
_FINALIZE_OK = {"LABELS_SUBMITTED", "FINALIZING", "COMPLETED", "FAILED"}


@router.post("/projects/{project_id}/finalize")
def start_finalize(
    project=Depends(get_project),
    db: Session = Depends(get_db),
    _svc: str = Depends(require_service_token),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if idempotency_key:
        prior = (
            db.query(models.Job)
            .filter_by(project_id=project.id, celery_task_id=idempotency_key)
            .order_by(models.Job.started_at.desc())
            .first()
        )
        if prior and prior.state == "SUCCEEDED":
            db.refresh(project)
            return build_results_payload(project)
        if prior and prior.state in ("QUEUED", "RUNNING"):
            raise HTTPException(409, {
                "code": "CONFLICT_BUSY",
                "message": "A run with this Idempotency-Key is already in progress",
                "project_id": project.id,
            })

    if project.state not in _FINALIZE_OK:
        raise HTTPException(409, {
            "code": "INVALID_STATE",
            "message": f"Cannot finalize from state {project.state}",
            "project_id": project.id,
        })

    n_labels = db.query(models.ClusterLabel).filter_by(project_id=project.id).count()
    if n_labels == 0:
        raise HTTPException(400, {
            "code": "BAD_REQUEST",
            "message": "Submit labels before finalizing",
            "project_id": project.id,
        })

    previous_state = project.state
    job = models.Job(
        project_id=project.id, type="finalize", state="RUNNING",
        started_at=datetime.utcnow(), celery_task_id=idempotency_key,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    project.state = "FINALIZING"
    project.error = None
    db.add(project)
    db.commit()

    try:
        job_b_finalize.apply(args=[project.id, job.id]).get(propagate=True)
    except Exception as exc:
        db.refresh(project)
        db.refresh(job)
        stage = job.current_stage or "unknown"
        if project.state == "FINALIZING":
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
    return build_results_payload(project)
