"""Asynchronous run triggers.

These are the endpoints the *frontend* calls. Unlike the synchronous
``POST /analyze`` and ``POST /finalize`` (which block until the compute finishes
and are now the callbacks the Airflow DAG hits), these return immediately:

* set the project to the in-progress state,
* dispatch the work (Airflow DAG run, or a local background thread when Airflow
  is not configured),
* return a small payload with the run id.

The frontend then polls ``GET /projects/{id}`` until the state reaches
``AWAITING_LABELS`` (analyze) or ``COMPLETED`` (finalize), or ``FAILED``.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_project
from app.db import models
from app.db.session import get_db
from app.services.airflow_client import airflow_enabled
from app.services.run_dispatch import dispatch_analyze, dispatch_finalize

router = APIRouter()

_ANALYZE_OK = {"UPLOADED", "ANALYZING", "AWAITING_LABELS", "FAILED"}
_FINALIZE_OK = {"LABELS_SUBMITTED", "FINALIZING", "COMPLETED", "FAILED"}


def _start_job(db: Session, project, job_type: str, in_progress_state: str):
    job = models.Job(
        project_id=project.id,
        type=job_type,
        state="QUEUED",
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    project.state = in_progress_state
    project.error = None
    db.add(project)
    db.commit()
    return job


def _fail_trigger(db: Session, project, job, exc: Exception) -> None:
    """Trigger/dispatch failed before any compute started: mark both rows FAILED."""
    job.state = "FAILED"
    job.error = str(exc)
    job.finished_at = datetime.utcnow()
    project.state = "FAILED"
    project.error = str(exc)
    db.add_all([job, project])
    db.commit()


def _mark_dispatched(db: Session, job, run_id: str) -> None:
    """Record the run id. In Airflow mode the in-flight run is tracked by the
    DAG's own callback job, so flip this trigger job to RUNNING (nothing else
    will) instead of leaving it stuck at QUEUED. In local mode the background
    task updates this same job, so leave its lifecycle to the task."""
    job.celery_task_id = run_id
    if airflow_enabled():
        job.state = "RUNNING"
    db.add(job)
    db.commit()


@router.post("/projects/{project_id}/runs/analyze")
def trigger_analyze(project=Depends(get_project), db: Session = Depends(get_db)):
    if project.state not in _ANALYZE_OK:
        raise HTTPException(409, f"Cannot analyze from state {project.state}")
    if not project.orthos:
        raise HTTPException(400, "Upload at least one orthomosaic first")

    job = _start_job(db, project, "analyze", "ANALYZING")
    try:
        run_id = dispatch_analyze(project.id, job.id)
    except RuntimeError as exc:
        # Trigger failed before any compute started — surface FAILED so the user
        # can retry instead of seeing a project stuck in ANALYZING.
        _fail_trigger(db, project, job, exc)
        raise HTTPException(502, f"Failed to trigger analyze: {exc}") from exc

    _mark_dispatched(db, job, run_id)
    return {
        "state": project.state,
        "job_id": job.id,
        "run_id": run_id,
        "mode": "airflow" if airflow_enabled() else "local",
    }


@router.post("/projects/{project_id}/runs/finalize")
def trigger_finalize(project=Depends(get_project), db: Session = Depends(get_db)):
    if project.state not in _FINALIZE_OK:
        raise HTTPException(409, f"Cannot finalize from state {project.state}")
    n_labels = db.query(models.ClusterLabel).filter_by(project_id=project.id).count()
    if n_labels == 0:
        raise HTTPException(400, "Submit labels before finalizing")

    job = _start_job(db, project, "finalize", "FINALIZING")
    try:
        run_id = dispatch_finalize(project.id, job.id)
    except RuntimeError as exc:
        _fail_trigger(db, project, job, exc)
        raise HTTPException(502, f"Failed to trigger finalize: {exc}") from exc

    _mark_dispatched(db, job, run_id)
    return {
        "state": project.state,
        "job_id": job.id,
        "run_id": run_id,
        "mode": "airflow" if airflow_enabled() else "local",
    }
