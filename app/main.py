"""FastAPI application entry point.

Run:
    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# CORS — open during development. `allow_origins=["*"]` cannot be combined with
# `allow_credentials=True`; this API authenticates via the X-API-Key header (not
# cookies), so credentials are not needed. Tighten allow_origins to your real
# frontend origin(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/livez", tags=["meta"])
def livez():
    """Liveness probe - is the process up. Used by Docker/K8s healthchecks."""
    return {"status": "ok"}


app.include_router(api_router)
