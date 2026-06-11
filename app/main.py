"""FastAPI application entry point.

Run:
    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()          # dev convenience; use Alembic migrations in production
    yield


app = FastAPI(
    title="Tree-Crown Species Pipeline API",
    version="0.1.0",
    description=(
        "Two-job, stage-gated async service wrapping the tree-crown detection / "
        "clustering / species-mapping pipeline. "
        "See API_SPECIFICATION.docx and API_DESIGN.md."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


app.include_router(api_router)
