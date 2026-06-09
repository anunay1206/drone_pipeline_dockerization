"""Phase B — Species assignment + validation + export (synchronous).

Blocks until the pipeline finishes and returns the final results payload
directly. An internal Job row is created for audit.
"""
import traceback
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_project
from app.api.v1.results import build_results_payload
from app.db import models
from app.db.session import get_db
from app.workers.tasks import job_b_finalize

router = APIRouter()

# Includes FINALIZING so the Airflow DAG callback can run after the
# /runs/finalize trigger has already moved the project into the in-progress state.
_FINALIZE_OK = {"LABELS_SUBMITTED", "FINALIZING", "COMPLETED", "FAILED"}


@router.post("/projects/{project_id}/finalize")
def start_finalize(project=Depends(get_project), db: Session = Depends(get_db)):
    if project.state not in _FINALIZE_OK:
        raise HTTPException(409, f"Cannot finalize from state {project.state}")

    n_labels = (
        db.query(models.ClusterLabel).filter_by(project_id=project.id).count()
    )
    if n_labels == 0:
        raise HTTPException(400, "Submit labels before finalizing")

    previous_state = project.state
    job = models.Job(
        project_id=project.id,
        type="finalize",
        state="RUNNING",
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    project.state = "FINALIZING"
    project.error = None
    db.add(project)
    db.commit()

    try:
        # synchronous in-process execution; no broker required
        job_b_finalize.apply(args=[project.id, job.id]).get(propagate=True)
    except Exception as exc:
        db.refresh(project)
        db.refresh(job)
        detail = (
            f"Finalize failed at stage '{job.current_stage or 'unknown'}': "
            f"{project.error or str(exc)}"
        )
        if project.state == "FINALIZING":
            project.state = previous_state
            db.add(project)
            db.commit()
        raise HTTPException(status_code=500, detail=detail) from exc

    db.refresh(project)
    return build_results_payload(project)
