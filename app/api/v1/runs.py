"""Asynchronous run triggers (frontend-facing).

These are the endpoints the frontend calls. They return immediately:
atomically move the project into the in-progress state, dispatch the work
(Airflow DAG run, or a local background thread), and return the run id. The
frontend then polls GET /projects/{id} until AWAITING_LABELS / COMPLETED /
FAILED.

Concurrency (v4 section 9.4): the in-progress states are intentionally NOT valid
launch states, and the transition is atomic - a duplicate trigger while a run
is active is rejected with 409 CONFLICT_BUSY rather than racing the destructive
reset_dirs.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_project
from app.core.models_registry import resolve_backbone, resolve_model_path
from app.core.storage import ensure_project_dirs
from app.db import models
from app.db.session import get_db
from app.schemas.project import AnalyzeTrigger
from app.services.airflow_client import airflow_enabled
from app.services.project_service import USED_RUN_STATES, archive_current_run
from app.services.run_dispatch import dispatch_analyze, dispatch_finalize
from app.services.state import transition_if

router = APIRouter()

# A NEW run may be launched only from these states (in-progress excluded on
# purpose). Launching from a used-run state archives that run and opens a fresh
# work/run_<n+1> folder (one-call re-analyze).
_ANALYZE_FROM = {"UPLOADED", "AWAITING_LABELS", "LABELS_SUBMITTED", "COMPLETED", "FAILED"}
_FINALIZE_FROM = {"LABELS_SUBMITTED", "COMPLETED", "FAILED"}
_BUSY = {"ANALYZING", "FINALIZING"}


def _gate(db: Session, project, allowed: set[str], in_progress_state: str, action: str):
    """Atomically enter the in-progress state, or raise the right 409."""
    if not transition_if(db, project, allowed, in_progress_state):
        db.refresh(project)
        if project.state in _BUSY:
            raise HTTPException(409, {
                "code": "CONFLICT_BUSY",
                "message": f"A run is already in progress (state {project.state})",
                "project_id": project.id,
            })
        raise HTTPException(409, {
            "code": "INVALID_STATE",
            "message": f"Cannot {action} from state {project.state}",
            "project_id": project.id,
        })


def _new_job(db: Session, project, job_type: str):
    job = models.Job(
        project_id=project.id, type=job_type, state="QUEUED",
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _fail_trigger(db: Session, project, job, exc: Exception) -> None:
    job.state = "FAILED"
    job.error = str(exc)
    job.finished_at = datetime.utcnow()
    project.state = "FAILED"
    project.error = str(exc)
    db.add_all([job, project])
    db.commit()


def _mark_dispatched(db: Session, job, run_id: str) -> None:
    job.celery_task_id = run_id
    if airflow_enabled():
        job.state = "RUNNING"
    db.add(job)
    db.commit()


def _validate_trigger_body(project, body) -> None:
    """Fail fast with 400 before any state transition happens."""
    if not body:
        return
    try:
        if body.model_key is not None:
            resolve_model_path(body.model_key)
        if body.params and "model_name" in body.params:
            resolve_backbone(body.params.get("model_name"))
    except ValueError as e:
        raise HTTPException(400, {"code": "BAD_REQUEST", "message": str(e),
                                  "project_id": project.id})
    if body.params:
        from app.api.v1.projects import _validate_param_overrides
        _validate_param_overrides(body.params)


def _apply_run_config(db: Session, project, body, pre_state: str) -> None:
    """Apply analyze-time configuration (run name + model + param overrides).

    Called after the atomic gate has moved the project into ANALYZING, so no
    concurrent run can interleave. If the previous state had already used the
    current run, that run is archived first and current_run is bumped so this
    run computes into a fresh folder (run versioning, v5)."""
    if pre_state in USED_RUN_STATES:
        archive_current_run(db, project, archived_state=pre_state)

    if body:
        if body.params:
            new_params = dict(project.params or {})
            new_params.update(body.params)
            if "model_name" in new_params:
                new_params["model_name"] = resolve_backbone(new_params.get("model_name"))
            project.params = new_params
        if body.model_key is not None:
            project.model_key = body.model_key
        if body.source_epsg is not None:
            project.source_epsg = body.source_epsg
        if body.run_name is not None:
            project.run_name = body.run_name.strip() or None

    db.add(project)
    db.commit()
    db.refresh(project)
    ensure_project_dirs(project.id, project.current_run)


@router.post("/projects/{project_id}/runs/analyze")
def trigger_analyze(
    body: AnalyzeTrigger | None = None,
    project=Depends(get_project),
    db: Session = Depends(get_db),
):
    """Fire (or re-fire) the analysis. Optional JSON body names the run and sets
    the detector / feature extractor / pipeline params in the same call, so the
    frontend's Analyze tab is a single button."""
    if not project.orthos:
        raise HTTPException(400, {
            "code": "BAD_REQUEST",
            "message": "Upload at least one orthomosaic first",
            "project_id": project.id,
        })
    _validate_trigger_body(project, body)
    pre_state = project.state
    _gate(db, project, _ANALYZE_FROM, "ANALYZING", "analyze")
    _apply_run_config(db, project, body, pre_state)
    job = _new_job(db, project, "analyze")
    try:
        run_id = dispatch_analyze(project.id, job.id)
    except RuntimeError as exc:
        _fail_trigger(db, project, job, exc)
        raise HTTPException(502, {
            "code": "DISPATCH_FAILED",
            "message": f"Failed to trigger analyze: {exc}",
            "project_id": project.id,
        }) from exc
    _mark_dispatched(db, job, run_id)
    return {
        "state": project.state, "job_id": job.id, "run_id": run_id,
        "run": project.current_run, "run_name": project.run_name,
        "mode": "airflow" if airflow_enabled() else "local",
    }


@router.post("/projects/{project_id}/runs/finalize")
def trigger_finalize(project=Depends(get_project), db: Session = Depends(get_db)):
    n_labels = db.query(models.ClusterLabel).filter_by(project_id=project.id).count()
    if n_labels == 0:
        raise HTTPException(400, {
            "code": "BAD_REQUEST",
            "message": "Submit labels before finalizing",
            "project_id": project.id,
        })
    _gate(db, project, _FINALIZE_FROM, "FINALIZING", "finalize")
    job = _new_job(db, project, "finalize")
    try:
        run_id = dispatch_finalize(project.id, job.id)
    except RuntimeError as exc:
        _fail_trigger(db, project, job, exc)
        raise HTTPException(502, {
            "code": "DISPATCH_FAILED",
            "message": f"Failed to trigger finalize: {exc}",
            "project_id": project.id,
        }) from exc
    _mark_dispatched(db, job, run_id)
    return {
        "state": project.state, "job_id": job.id, "run_id": run_id,
        "mode": "airflow" if airflow_enabled() else "local",
    }
