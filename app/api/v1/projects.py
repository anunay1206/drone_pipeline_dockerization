"""Project lifecycle + uploads."""
import os
import re
import shutil
import tempfile
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_project, require_api_key
from app.core.models_registry import (
    DEFAULT_MODEL_KEY,
    list_backbones,
    list_models,
    resolve_backbone,
    resolve_model_path,
)
from app.core.settings import settings
from app.core.storage import delete_project_dir, ensure_project_dirs
from app.db import models
from app.db.session import get_db
from app.schemas.project import OrthoFromUrl, ProjectCreate, ProjectUpdate
from app.services.project_service import serialize_project

router = APIRouter()


@router.get("/detectors")
def get_detectors(user: str = Depends(require_api_key)):
    """List the registered Detectree2 detector weight files (+ availability/default)."""
    return list_models()


@router.get("/feature-extractors")
def get_feature_extractors(user: str = Depends(require_api_key)):
    """List the allowed DINOv2 feature-extractor models (valid model_name values)."""
    return list_backbones()


@router.post("/projects", status_code=201)
def create_project(
    body: ProjectCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_api_key),
):
    model_key = body.model_key or DEFAULT_MODEL_KEY
    try:
        resolve_model_path(model_key)
        # Validate the feature-extraction backbone against the allowlist so a bad
        # model_name fails here rather than deep in the worker (Step 1B).
        body.params.model_name = resolve_backbone(body.params.model_name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    project = models.Project(
        user_id=user,
        name=body.name,
        model_key=model_key,
        source_epsg=body.source_epsg,
        params=body.params.model_dump(),
        state="CREATED",
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    ensure_project_dirs(project.id, project.current_run)
    return serialize_project(project)


@router.get("/projects")
def list_projects(db: Session = Depends(get_db), user: str = Depends(require_api_key)):
    rows = (
        db.query(models.Project)
        .filter_by(user_id=user)
        .order_by(models.Project.created_at.desc())
        .all()
    )
    return [serialize_project(p) for p in rows]


@router.get("/projects/{project_id}")
def get_one(project=Depends(get_project)):
    return serialize_project(project)


# states whose run has already produced (or attempted) results — editing params
# from here opens a NEW run rather than overwriting the existing one.
_USED_RUN_STATES = {"AWAITING_LABELS", "LABELS_SUBMITTED", "COMPLETED", "FAILED"}


@router.patch("/projects/{project_id}")
def update_project(
    body: ProjectUpdate,
    project=Depends(get_project),
    db: Session = Depends(get_db),
):
    """Edit parameters and prepare a re-run on the SAME uploaded ortho.

    If the current run has already been used (analyzed/finalized/failed), its
    summary is archived to ``runs`` and ``current_run`` is bumped so the next
    analyze computes into a fresh ``work/run_<n+1>`` folder — previous runs are
    preserved on disk. If the current run hasn't been analyzed yet, params are
    just updated in place. Either way the project ends in ``UPLOADED``, ready for
    ``POST /runs/analyze``.
    """
    if project.state in ("ANALYZING", "FINALIZING"):
        raise HTTPException(409, "Cannot change parameters while a run is in progress")
    if not project.orthos:
        raise HTTPException(400, "Upload an orthomosaic before configuring a re-run")

    # Merge param overrides onto the existing params, then validate model choices.
    new_params = dict(project.params or {})
    if body.params:
        new_params.update(body.params)
    try:
        if body.model_key is not None:
            resolve_model_path(body.model_key)
        if "model_name" in new_params:
            new_params["model_name"] = resolve_backbone(new_params.get("model_name"))
    except ValueError as e:
        raise HTTPException(400, str(e))

    if project.state in _USED_RUN_STATES:
        history = list(project.runs or [])
        history.append(
            {
                "run": project.current_run or 1,
                "params": dict(project.params or {}),
                "model_key": project.model_key,
                "state": project.state,
                "recommended_k": project.recommended_k,
                "available_k": project.available_k,
            }
        )
        project.runs = history
        project.current_run = (project.current_run or 1) + 1
        project.recommended_k = None
        project.available_k = None
        # labels belong to the previous run's clustering — clear for the new run
        db.query(models.ClusterLabel).filter_by(project_id=project.id).delete()

    project.params = new_params
    if body.model_key is not None:
        project.model_key = body.model_key
    if body.source_epsg is not None:
        project.source_epsg = body.source_epsg
    project.state = "UPLOADED"
    project.error = None
    db.add(project)
    db.commit()
    db.refresh(project)
    ensure_project_dirs(project.id, project.current_run)
    return serialize_project(project)


@router.delete("/projects/{project_id}", status_code=204)
def delete_one(project=Depends(get_project), db: Session = Depends(get_db)):
    delete_project_dir(project.id)
    db.delete(project)
    db.commit()
    return None


@router.post("/projects/{project_id}/orthomosaic")
def upload_ortho(
    project=Depends(get_project),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload the project's orthomosaic.

    Each project holds **exactly one** orthomosaic. If one is already present
    when this is called, the previous file is deleted from disk and its DB row
    is replaced, so the call has set-semantics rather than append-semantics.
    """
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Orthomosaic must be a .tif/.tiff GeoTIFF")

    paths = ensure_project_dirs(project.id, project.current_run)
    _clear_existing_orthos(project, db, paths)

    stem = os.path.splitext(os.path.basename(file.filename))[0]
    dst = os.path.join(paths["input_ortho"], f"{stem}.tif")
    _stream_to_disk(file, dst)
    return _register_ortho(project, db, dst, stem)


@router.post("/projects/{project_id}/orthomosaic/from-url")
def upload_ortho_from_url(
    body: OrthoFromUrl,
    project=Depends(get_project),
    db: Session = Depends(get_db),
):
    """Register an orthomosaic by downloading it from a public Google Drive link.

    Synchronous: the request blocks while the server downloads the file, so set a
    long client read timeout for large orthos. Only Google Drive share links set to
    'anyone with the link' are supported. Same set-semantics as the file upload —
    any previously-registered ortho is replaced.
    """
    url = (body.url or "").strip()
    host = urlparse(url).netloc.lower()
    if not (host == "drive.google.com" or host.endswith(".google.com")):
        raise HTTPException(400, "Only Google Drive links are supported.")

    file_id = _extract_drive_id(url)
    if not file_id:
        raise HTTPException(400, "Could not parse a Google Drive file id from the URL.")

    try:
        import gdown
    except ImportError:
        raise HTTPException(
            503, "Server is missing the 'gdown' dependency required for URL uploads."
        )

    paths = ensure_project_dirs(project.id, project.current_run)
    tmp_dir = tempfile.mkdtemp(prefix="ortho_dl_", dir=paths["input_ortho"])
    try:
        try:
            out = gdown.download(id=file_id, output=tmp_dir + os.sep, quiet=True)
        except Exception as e:
            raise HTTPException(400, f"Google Drive download failed: {e}")
        if not out or not os.path.exists(out):
            raise HTTPException(
                400,
                "Download failed — the file may be private, deleted, or over its "
                "Google Drive download quota.",
            )

        max_bytes = settings.max_upload_mb * 1024 * 1024
        if os.path.getsize(out) > max_bytes:
            raise HTTPException(
                400, f"Downloaded file exceeds the {settings.max_upload_mb} MB limit."
            )
        if not out.lower().endswith((".tif", ".tiff")):
            raise HTTPException(400, "The Drive file is not a .tif/.tiff GeoTIFF.")

        stem = os.path.splitext(os.path.basename(out))[0]
        dst = os.path.join(paths["input_ortho"], f"{stem}.tif")
        _clear_existing_orthos(project, db, paths)
        shutil.move(out, dst)
        return _register_ortho(project, db, dst, stem)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/projects/{project_id}/ground-truth")
def upload_ground_truth(project=Depends(get_project), file: UploadFile = File(...)):
    """Upload a .zip whose top-level folders are species names containing crown
    .tif files (the structure step3_validate expects)."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Ground truth must be a .zip of <species>/*.tif folders")

    import zipfile

    paths = ensure_project_dirs(project.id, project.current_run)
    tmp = os.path.join(paths["input_gt"], "_upload.zip")
    _stream_to_disk(file, tmp)
    try:
        with zipfile.ZipFile(tmp) as z:
            z.extractall(paths["input_gt"])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    species = [
        d for d in os.listdir(paths["input_gt"])
        if os.path.isdir(os.path.join(paths["input_gt"], d))
    ]
    return {"state": project.state, "species_folders": sorted(species)}


# ── helpers ────────────────────────────────────────────────────────────
_DRIVE_ID_PATTERNS = [
    re.compile(r"/file/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]{10,})"),
    re.compile(r"/d/([A-Za-z0-9_-]{10,})"),
]


def _extract_drive_id(url: str) -> str | None:
    """Pull the file id out of common Google Drive URL shapes."""
    for rx in _DRIVE_ID_PATTERNS:
        m = rx.search(url)
        if m:
            return m.group(1)
    return None


def _clear_existing_orthos(project, db: Session, paths: dict) -> None:
    """Delete any previously-registered ortho (file + DB row) — set-semantics."""
    for existing in db.query(models.Ortho).filter_by(project_id=project.id).all():
        prev_path = os.path.join(paths["input_ortho"], existing.filename)
        if os.path.exists(prev_path):
            try:
                os.remove(prev_path)
            except OSError:
                pass
        db.delete(existing)


def _register_ortho(project, db: Session, dst: str, stem: str):
    """Record raster metadata, create the Ortho row, advance state, and serialize."""
    meta = _raster_meta(dst)
    if meta.get("crs_epsg") and not project.source_epsg:
        project.source_epsg = meta["crs_epsg"]

    o = models.Ortho(project_id=project.id, stem=stem)
    o.filename = f"{stem}.tif"
    o.width = meta.get("width")
    o.height = meta.get("height")
    o.crs = meta.get("crs")
    o.bands = meta.get("bands")
    o.size_bytes = os.path.getsize(dst)
    db.add(o)

    if project.state in ("CREATED", "UPLOADED"):
        project.state = "UPLOADED"
    db.add(project)
    db.commit()
    db.refresh(project)
    return serialize_project(project)


def _stream_to_disk(file: UploadFile, dst: str, chunk: int = 1024 * 1024) -> None:
    with open(dst, "wb") as out:
        while True:
            data = file.file.read(chunk)
            if not data:
                break
            out.write(data)
    file.file.close()


def _raster_meta(path: str) -> dict:
    """Best-effort raster metadata; empty dict if rasterio is unavailable."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            try:
                epsg = src.crs.to_epsg() if src.crs else None
            except Exception:
                epsg = None
            return {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs) if src.crs else None,
                "crs_epsg": epsg,
                "bands": src.count,
            }
    except Exception:
        return {}
