"""Celery application.

Two logical queues:
  * ``gpu`` — detection (Step 0) + DINOv2 feature extraction; concurrency capped
    to the number of GPUs. Must run with the *prefork* pool (one task per
    process) because the detection patch uses ``os.chdir`` for CWD isolation.
  * ``cpu`` — cropping, clustering, t-SNE, assign, validate, KMZ export.

Run workers, e.g.:
    celery -A app.workers.celery_app:celery_app worker -Q gpu -c 1 --pool=prefork
    celery -A app.workers.celery_app:celery_app worker -Q cpu -c 4
"""
from celery import Celery

from app.core.settings import settings

celery_app = Celery(
    "treecrown",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_always_eager=settings.celery_eager,
    task_eager_propagates=settings.celery_eager,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # long tasks: don't hoard the queue
    task_routes={
        "app.workers.tasks.job_a_analyze": {"queue": "gpu"},
        "app.workers.tasks.job_b_finalize": {"queue": "cpu"},
    },
)

# Register tasks (import after the app object exists to avoid a circular import).
import app.workers.tasks  # noqa: E402,F401
