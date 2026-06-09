from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.db import models
from app.db.session import get_db


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Optional API-key gate. If ``settings.api_key`` is unset, auth is open.

    Returns the user id (single-tenant 'default' for now).
    """
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return "default"


def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    user: str = Depends(require_api_key),
) -> models.Project:
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != user:
        raise HTTPException(status_code=403, detail="Forbidden")
    return project
