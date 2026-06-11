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

    # ── model registry ─────────────────────────────────────────────────
    default_model_key: str = "urban_cambridge"

    # ── uploads ────────────────────────────────────────────────────────
    max_upload_mb: int = 8192

    # ── auth (optional) ────────────────────────────────────────────────
    # If set, clients must send the header ``X-API-Key: <api_key>``.
    api_key: str | None = None

    # ── dev convenience ────────────────────────────────────────────────
    # Run Celery tasks inline in-process (no broker/worker needed). Note the
    # heavy ML deps must still be importable for an inline run to succeed.
    celery_eager: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
