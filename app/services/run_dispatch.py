"""Dispatch the heavy compute for a project run.

Two modes, chosen automatically:

* **Airflow** (``airflow_base_url`` set) — trigger the configured DAG. The DAG
  task calls back into this service's ``POST /analyze`` / ``POST /finalize``
  compute endpoints, which do the actual work. Returns the Airflow
  ``dag_run_id``.

* **Local** (no Airflow configured) — run the Celery task body in a daemon
  thread in-process (``task.apply(...)`` executes synchronously). The HTTP
  trigger returns immediately and the frontend polls project state. This makes
  the full pipeline runnable locally without an Airflow stack or a broker.

Either way the endpoint returns fast; progress is tracked via the project state
machine in the DB (CREATED -> ... -> AWAITING_LABELS / COMPLETED / FAILED).
"""
import threading

from app.core.settings import settings
from app.services.airflow_client import airflow_enabled, trigger_dag


def _conf(project_id: str, job_id: str) -> dict:
    return {"project_id": project_id, "job_id": job_id}


def _run_local(task_name: str, project_id: str, job_id: str) -> str:
    """Execute a Celery task body in a background daemon thread."""

    def _target():
        # Imported lazily inside the thread so importing this module never pulls
        # in the heavy worker/ML stack.
        from app.workers.tasks import job_a_analyze, job_b_finalize

        task = {"job_a_analyze": job_a_analyze, "job_b_finalize": job_b_finalize}[task_name]
        try:
            task.apply(args=[project_id, job_id])
        except Exception:
            # The task's own _fail() has already recorded FAILED state + the
            # traceback in the DB; nothing more to do on this thread.
            pass

    threading.Thread(target=_target, name=f"{task_name}:{job_id}", daemon=True).start()
    return f"local:{job_id}"


def dispatch_analyze(project_id: str, job_id: str) -> str:
    if airflow_enabled():
        return trigger_dag(settings.analyze_dag_id, _conf(project_id, job_id))
    return _run_local("job_a_analyze", project_id, job_id)


def dispatch_finalize(project_id: str, job_id: str) -> str:
    if airflow_enabled():
        return trigger_dag(settings.finalize_dag_id, _conf(project_id, job_id))
    return _run_local("job_b_finalize", project_id, job_id)
