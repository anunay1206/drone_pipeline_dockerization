"""Phase A — Detection + clustering (synchronous).

The endpoint blocks until the pipeline finishes (or fails) and returns the
two-DAG review payload directly. There is no separate job-polling endpoint in
the synchronous contract; an internal Job row is still created for audit and
to carry per-stage progress for logs.
"""
import traceback
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.deps import get_project
from app.api.v1.clustering import build_clustering_payload
from app.db import models
from app.db.session import get_db
from app.workers.tasks import job_a_analyze

router = APIRouter()

_ANALYZE_OK = {"UPLOADED", "AWAITING_LABELS", "FAILED"}


@router.post("/projects/{project_id}/analyze")
def start_analyze(
    request: Request,
    project=Depends(get_project),
    db: Session = Depends(get_db),
):
    if project.state not in _ANALYZE_OK:
        raise HTTPException(409, f"Cannot analyze from state {project.state}")
    if not project.orthos:
        raise HTTPException(400, "Upload at least one orthomosaic first")

    previous_state = project.state
    job = models.Job(
        project_id=project.id,
        type="analyze",
        state="RUNNING",
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    project.state = "ANALYZING"
    project.error = None
    db.add(project)
    db.commit()

    try:
        # apply() runs the task body in-process with no broker required.
        # .get(propagate=True) re-raises if the task failed.
        job_a_analyze.apply(args=[project.id, job.id]).get(propagate=True)
    except Exception as exc:
        # The task itself has already written FAILED state and an error tail
        # into the DB via _fail(); just translate that to an HTTP error.
        db.refresh(project)
        db.refresh(job)
        detail = (
            f"Analyze failed at stage '{job.current_stage or 'unknown'}': "
            f"{project.error or str(exc)}"
        )
        # belt + braces — restore the prior state only if the task didn't
        # already write a definitive one
        if project.state == "ANALYZING":
            project.state = previous_state
            db.add(project)
            db.commit()
        # job_a_analyze sets finished_at via _fail; nothing more to do here
        raise HTTPException(status_code=500, detail=detail) from exc

    # success → project is now AWAITING_LABELS, job is SUCCEEDED
    db.refresh(project)
    return build_clustering_payload(request, project)
