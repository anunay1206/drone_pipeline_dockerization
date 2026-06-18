"""Application settings.

All values can be overridden via environment variables with the ``TCP_`` prefix
(e.g. ``TCP_DATABASE_URL``) or a local ``.env`` file. See ``.env.example``.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TCP_", extra="ignore"
    )

    # ── storage / paths ────────────────────────────────────────────────
    storage_root: str = "./storage"      # per-project artifacts live here
    models_dir: str = "."                # directory containing the .pth weight files

    # ── database / task queue ──────────────────────────────────────────
    # SQLite by default for easy local dev; point at Postgres in production.
    database_url: str = "sqlite:///./treecrown.db"
    redis_url: str = "redis://localhost:6379/0"

    # ── Airflow orchestration (optional) ───────────────────────────────
    # When airflow_base_url is set, the /runs/* trigger endpoints kick off the
    # corresponding Airflow DAG (which calls back into /analyze and /finalize).
    # When it is blank, the trigger endpoints fall back to running the compute
    # in a background thread in-process — so the full pipeline works locally
    # without an Airflow stack. REST auth is OPTIONAL: leave the credentials
    # blank to call the Airflow API unauthenticated.
    airflow_base_url: str = ""               # e.g. http://host.docker.internal:8080
    airflow_username: str | None = None      # Airflow REST basic-auth user (optional)
    airflow_password: str | None = None      # Airflow REST basic-auth password (optional)
    airflow_auth_token: str | None = None    # OR a bearer token for the Airflow REST API
    analyze_dag_id: str = "drone_analyze"
    finalize_dag_id: str = "drone_finalize"

    # ── model registry ─────────────────────────────────────────────────
    default_model_key: str = "urban_cambridge"
    # External model catalog (detectors + backbones). Mount this as a volume in
    # Docker so the catalog can change without rebuilding the image. If the file
    # is absent, models_registry falls back to built-in defaults.
    models_manifest: str = "./models.yaml"

    # ── uploads ────────────────────────────────────────────────────────
    max_upload_mb: int = 8192

    # ── auth (optional) ────────────────────────────────────────────────
    # If set, clients must send the header ``X-API-Key: <api_key>``.
    api_key: str | None = None

    # ── STAC ───────────────────────────────────────────────────────────
    # Public base URL used to make STAC Item asset/link hrefs absolute
    # (e.g. https://api.example.com). Leave blank to emit relative hrefs.
    public_base_url: str = ""

    # ── dev convenience ────────────────────────────────────────────────
    # Run Celery tasks inline in-process (no broker/worker needed). Note the
    # heavy ML deps must still be importable for an inline run to succeed.
    celery_eager: bool = False

    # ── retention / cleanup ────────────────────────────────────────────
    # A Celery Beat job (see app/workers/cleanup.py) deletes projects whose
    # last activity is older than retention_days — DB row + storage folder.
    retention_days: int = 7
    cleanup_enabled: bool = True
    cleanup_hour: int = 3          # daily run time (UTC), 0-23
    cleanup_minute: int = 0        # 0-59


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
