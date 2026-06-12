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
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHENTICATED", "message": "Invalid or missing API key"},
        )
    return "default"


def require_service_token(
    x_service_token: str | None = Header(default=None),
) -> str:
    """Compute-API gate (v4 §9.2). If ``settings.compute_token`` is unset the
    check is open (single-service/dev). Returns the system identity."""
    if settings.compute_token and x_service_token != settings.compute_token:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHENTICATED", "message": "Invalid or missing service token"},
        )
    return "system"


def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    user: str = Depends(require_api_key),
) -> models.Project:
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(
            status_code=404,
            detail={"code": "PROJECT_NOT_FOUND", "message": "Project not found",
                    "project_id": project_id},
        )
    if project.user_id != user:
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Forbidden", "project_id": project_id},
        )
    return project
