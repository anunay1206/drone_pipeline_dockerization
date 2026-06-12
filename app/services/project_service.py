from app.schemas.project import OrthoOut, ProjectOut

# States whose run has already produced (or attempted) results - re-configuring
# from here archives the run and opens a fresh work/run_<n+1> folder.
USED_RUN_STATES = {"AWAITING_LABELS", "LABELS_SUBMITTED", "COMPLETED", "FAILED"}


def archive_current_run(db, project, archived_state: str | None = None) -> None:
    """Append the current run's summary to project.runs and bump current_run so
    the next analyze computes into a fresh folder. Clears the per-run review
    fields and the cluster labels (they belong to the archived run's
    clustering). Does NOT commit - the caller owns the transaction."""
    from app.db import models

    history = list(project.runs or [])
    history.append(
        {
            "run": project.current_run or 1,
            "run_name": getattr(project, "run_name", None),
            "params": dict(project.params or {}),
            "model_key": project.model_key,
            "state": archived_state or project.state,
            "recommended_k": project.recommended_k,
            "available_k": project.available_k,
            "ortho": project.orthos[0].filename if project.orthos else None,
        }
    )
    project.runs = history
    project.current_run = (project.current_run or 1) + 1
    project.run_name = None
    project.recommended_k = None
    project.available_k = None
    db.query(models.ClusterLabel).filter_by(project_id=project.id).delete()


def _last_error(project) -> dict | None:
    """Structured failure info when the project is FAILED (v4 section 8.1).

    Combines the project's error text with the most recent job's stage so the
    frontend gets a machine code + stage instead of only free-text."""
    if project.state != "FAILED":
        return None
    stage = None
    try:
        jobs = [j for j in (project.jobs or []) if j.started_at]
        if jobs:
            stage = max(jobs, key=lambda j: j.started_at).current_stage
    except Exception:
        stage = None
    return {"code": "COMPUTE_FAILED", "stage": stage, "message": project.error}


def serialize_project(project) -> ProjectOut:
    """ORM Project -> API response model."""
    return ProjectOut(
        project_id=project.id,
        name=project.name,
        model_key=project.model_key,
        state=project.state,
        source_epsg=project.source_epsg,
        params=project.params or {},
        recommended_k=project.recommended_k,
        available_k=project.available_k,
        current_run=project.current_run or 1,
        run_name=getattr(project, "run_name", None),
        runs=project.runs or [],
        orthos=[OrthoOut.model_validate(o) for o in project.orthos],
        error=project.error,
        last_error=_last_error(project),
        created_at=project.created_at,
        updated_at=project.updated_at,
    )
