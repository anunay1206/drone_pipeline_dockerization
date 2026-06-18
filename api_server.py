"""
FastAPI wrapper for the Tree Species Classification pipeline.

Exposes each pipeline step as an HTTP endpoint so it can be called from
Postman, curl, or any HTTP client.

Run with:
    pip install fastapi uvicorn python-multipart
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

Docs (interactive Swagger UI) at:
    http://localhost:8000/docs
"""

import os
import shutil
import threading
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import Config

# Lazy imports for heavy modules — only loaded when an endpoint is called.
# This keeps the API process light and avoids long startup time.


app = FastAPI(
    title="Tree Species Classification API",
    description=(
        "REST API wrapper around the drone-orthomosaic tree species "
        "classification pipeline. Each pipeline step is exposed as an "
        "endpoint that can be called from Postman."
    ),
    version="1.0.0",
)


# ----------------------------------------------------------------------
# In-memory job registry (for async pipeline runs)
# ----------------------------------------------------------------------
JOBS: Dict[str, Dict[str, Any]] = {}


def _new_job(step_name: str) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "step": step_name,
        "status": "pending",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "finished_at": None,
        "error": None,
        "logs": [],
    }
    return job_id


def _run_in_thread(job_id: str, fn, *args, **kwargs) -> None:
    def _runner():
        JOBS[job_id]["status"] = "running"
        try:
            result = fn(*args, **kwargs)
            JOBS[job_id]["result"] = result
            JOBS[job_id]["status"] = "completed"
        except Exception as exc:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"
            JOBS[job_id]["traceback"] = traceback.format_exc()
        finally:
            JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"

    t = threading.Thread(target=_runner, daemon=True)
    t.start()


# ----------------------------------------------------------------------
# Request/response models
# ----------------------------------------------------------------------
class ConfigPatch(BaseModel):
    ORTHO_PATH: Optional[str] = None
    WORKDIR: Optional[str] = None
    DETECTREE_MODEL: Optional[str] = None
    TILE_SIZE: Optional[int] = None
    BUFFER: Optional[int] = None
    IOU_THRESHOLD: Optional[float] = None
    CONF_THRESHOLD: Optional[float] = None
    STEP1_OUTPUT: Optional[str] = None
    MODEL_NAME: Optional[str] = None
    IMG_SIZE: Optional[int] = None
    BATCH_SIZE: Optional[int] = None
    PCA_COMPONENTS: Optional[int] = None
    K_LIST: Optional[List[int]] = None
    COPY_TO_CLUSTER_FOLDERS: Optional[bool] = None
    CHOSEN_K: Optional[int] = None
    STEP2_OUTPUT: Optional[str] = None
    GROUND_TRUTH_CSV: Optional[str] = None
    STEP3_VALIDATION_OUTPUT: Optional[str] = None
    STEP4_OUTPUT: Optional[str] = None
    SOURCE_EPSG: Optional[int] = None
    COLOR_PALETTE: Optional[List[str]] = None


class StepRunRequest(BaseModel):
    async_run: bool = Field(
        default=True,
        description="If true, run in background and return a job_id. "
                    "If false, block until the step completes.",
    )
    overrides: Optional[ConfigPatch] = Field(
        default=None,
        description="Optional config overrides for this run only.",
    )


class JobResponse(BaseModel):
    job_id: str
    status: str
    step: str


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _apply_overrides(config: Config, overrides: Optional[ConfigPatch]) -> Config:
    if overrides is None:
        return config
    for key, value in overrides.dict(exclude_none=True).items():
        setattr(config, key, value)
    return config


def _build_config(overrides: Optional[ConfigPatch] = None) -> Config:
    c = Config()
    return _apply_overrides(c, overrides)


# ----------------------------------------------------------------------
# Meta endpoints
# ----------------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health():
    """Liveness probe — returns OK if the API is up."""
    return {"status": "ok", "service": "tree-species-api", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/config", tags=["meta"])
def get_config():
    """Return the current default configuration values."""
    c = Config()
    return {
        k: getattr(c, k)
        for k in dir(c)
        if not k.startswith("_") and not callable(getattr(c, k))
    }


@app.patch("/config", tags=["meta"])
def patch_config(patch: ConfigPatch):
    """Update Config class attributes at runtime (in-memory)."""
    c = Config
    for key, value in patch.dict(exclude_none=True).items():
        setattr(c, key, value)
    return get_config()


# ----------------------------------------------------------------------
# File upload / download
# ----------------------------------------------------------------------
@app.post("/upload/orthomosaic", tags=["files"])
async def upload_orthomosaic(file: UploadFile = File(...)):
    """
    Upload a drone orthomosaic (.tif) to the working directory.
    The file is stored in <WORKDIR>/ortho/<filename>.
    """
    c = Config()
    target_dir = os.path.join(c.WORKDIR, "ortho")
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"saved_to": dest, "size_bytes": os.path.getsize(dest)}


@app.get("/download", tags=["files"])
def download(path: str):
    """Stream a file back to the caller. Use the absolute path."""
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(path, filename=os.path.basename(path))


@app.get("/outputs", tags=["files"])
def list_outputs():
    """List files in the WORKDIR tree."""
    c = Config()
    if not os.path.isdir(c.WORKDIR):
        return {"workdir": c.WORKDIR, "files": []}
    files = []
    for root, _, names in os.walk(c.WORKDIR):
        for n in names:
            full = os.path.join(root, n)
            files.append({"path": full, "size_bytes": os.path.getsize(full)})
    return {"workdir": c.WORKDIR, "files": files}


# ----------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------
@app.post("/pipeline/step0/detect", response_model=JobResponse, tags=["pipeline"])
def step0_detect(req: StepRunRequest):
    """Step 0 — Detect tree crowns with Detectree2."""
    from end_to_end_pipeline import step0_detection

    config = _build_config(req.overrides)
    job_id = _new_job("step0_detect")

    if req.async_run:
        _run_in_thread(job_id, step0_detection, config)
        return JobResponse(job_id=job_id, status="running", step="step0_detect")

    try:
        step0_detection(config)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
        return JobResponse(job_id=job_id, status="completed", step="step0_detect")
    except Exception as exc:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pipeline/step1/cluster", response_model=JobResponse, tags=["pipeline"])
def step1_cluster(req: StepRunRequest):
    """Step 1 — Crop crowns, extract DINOv2 features, multi-k clustering, t-SNE."""
    from end_to_end_pipeline import step1_clustering

    config = _build_config(req.overrides)
    job_id = _new_job("step1_cluster")

    if req.async_run:
        _run_in_thread(job_id, step1_clustering, config)
        return JobResponse(job_id=job_id, status="running", step="step1_cluster")

    step1_clustering(config)
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return JobResponse(job_id=job_id, status="completed", step="step1_cluster")


@app.post("/pipeline/step2/species", response_model=JobResponse, tags=["pipeline"])
def step2_species(req: StepRunRequest):
    """Step 2 — Assign species labels using the human-edited cluster map CSV."""
    from end_to_end_pipeline import step2_species as run_step2

    config = _build_config(req.overrides)
    job_id = _new_job("step2_species")

    if req.async_run:
        _run_in_thread(job_id, run_step2, config)
        return JobResponse(job_id=job_id, status="running", step="step2_species")

    run_step2(config)
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return JobResponse(job_id=job_id, status="completed", step="step2_species")


@app.post("/pipeline/step3/validate", response_model=JobResponse, tags=["pipeline"])
def step3_validate(req: StepRunRequest):
    """Step 3 — Validate predictions against ground truth CSV (optional)."""
    from end_to_end_pipeline import step3_validation

    config = _build_config(req.overrides)
    job_id = _new_job("step3_validate")

    if req.async_run:
        _run_in_thread(job_id, step3_validation, config)
        return JobResponse(job_id=job_id, status="running", step="step3_validate")

    step3_validation(config)
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return JobResponse(job_id=job_id, status="completed", step="step3_validate")


@app.post("/pipeline/step4/kmz", response_model=JobResponse, tags=["pipeline"])
def step4_kmz(req: StepRunRequest):
    """Step 4 — Export the species map to KMZ for Google Earth."""
    from end_to_end_pipeline import step4_kmz as run_step4

    config = _build_config(req.overrides)
    job_id = _new_job("step4_kmz")

    if req.async_run:
        _run_in_thread(job_id, run_step4, config)
        return JobResponse(job_id=job_id, status="running", step="step4_kmz")

    run_step4(config)
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return JobResponse(job_id=job_id, status="completed", step="step4_kmz")


@app.post("/pipeline/run-all", response_model=JobResponse, tags=["pipeline"])
def run_all(req: StepRunRequest):
    """
    Convenience endpoint — runs steps 0 → 1 → 2 → 3 → 4 sequentially.
    Note: step 1 normally requires a human-in-the-loop pause to label
    clusters. This endpoint assumes you have already filled in the
    cluster_species_map.csv before calling.
    """
    from end_to_end_pipeline import (
        step0_detection,
        step1_clustering,
        step2_species as run_step2,
        step3_validation,
        step4_kmz as run_step4,
    )

    config = _build_config(req.overrides)
    job_id = _new_job("run_all")

    def _all():
        step0_detection(config)
        step1_clustering(config)
        run_step2(config)
        step3_validation(config)
        run_step4(config)

    if req.async_run:
        _run_in_thread(job_id, _all)
        return JobResponse(job_id=job_id, status="running", step="run_all")

    _all()
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return JobResponse(job_id=job_id, status="completed", step="run_all")


# ----------------------------------------------------------------------
# Job tracking
# ----------------------------------------------------------------------
@app.get("/jobs", tags=["jobs"])
def list_jobs():
    return {"jobs": list(JOBS.values())}


@app.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JOBS[job_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
