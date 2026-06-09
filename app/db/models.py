"""SQLAlchemy ORM models — the service's source of truth for state.

State machine (see API_DESIGN.md §4):
  CREATED -> UPLOADED -> ANALYZING -> AWAITING_LABELS
          -> LABELS_SUBMITTED -> FINALIZING -> COMPLETED
  (any heavy stage may go -> FAILED)
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, default="default", index=True)
    name: Mapped[str] = mapped_column(String, default="")
    model_key: Mapped[str] = mapped_column(String, default="urban_cambridge")
    state: Mapped[str] = mapped_column(String, default="CREATED", index=True)
    source_epsg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    recommended_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    available_k: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Run versioning: current_run points at the active work/run_<n> folder;
    # `runs` keeps a lightweight history of prior runs (params + outcome).
    current_run: Mapped[int] = mapped_column(Integer, default=1)
    runs: Mapped[list | None] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    orthos: Mapped[list["Ortho"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["Job"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    labels: Mapped[list["ClusterLabel"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Ortho(Base):
    __tablename__ = "orthos"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    stem: Mapped[str] = mapped_column(String)            # filename without extension
    filename: Mapped[str] = mapped_column(String)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    crs: Mapped[str | None] = mapped_column(String, nullable=True)
    bands: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="orthos")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    type: Mapped[str] = mapped_column(String)            # 'analyze' | 'finalize'
    state: Mapped[str] = mapped_column(String, default="QUEUED")  # QUEUED|RUNNING|SUCCEEDED|FAILED
    current_stage: Mapped[str | None] = mapped_column(String, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    log_path: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="jobs")


class ClusterLabel(Base):
    __tablename__ = "cluster_labels"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    chosen_k: Mapped[int] = mapped_column(Integer)
    cluster_id: Mapped[int] = mapped_column(Integer)
    species: Mapped[str] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="labels")
