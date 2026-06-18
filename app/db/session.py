from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.settings import settings

_connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

engine = create_engine(settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db():
    """FastAPI dependency: yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables. For production use Alembic migrations instead."""
    from app.db import models  # noqa: F401  (register mappers)
    from app.db.base import Base

    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_add_columns()


def _migrate_sqlite_add_columns() -> None:
    """Dev-convenience migration: add columns that ``create_all`` won't add to an
    existing SQLite file. Keeps older treecrown.db files working without a wipe.
    Use Alembic for real migrations / non-SQLite backends.
    """
    if not settings.database_url.startswith("sqlite"):
        return
    wanted = {
        "projects": {
            "current_run": "INTEGER DEFAULT 1",
            "runs": "JSON",
        },
    }
    with engine.begin() as conn:
        for table, cols in wanted.items():
            existing = {
                row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            }
            for name, ddl in cols.items():
                if name not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
