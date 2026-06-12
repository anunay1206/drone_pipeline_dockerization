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
from app.services.project_service import (
    USED_RUN_STATES,
    archive_current_run,
    serialize_project,
)

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
        raise HTTPException(400, {"code": "BAD_REQUEST", "message": str(e)})

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


# Dataset lock (v5 req-8): once the current run has been analyzed the input
# orthomosaic is immutable. Opening a NEW run (PATCH) returns the project to
# UPLOADED, which unlocks uploads again until that run is analyzed.
_ORTHO_UNLOCKED_STATES = {"CREATED", "UPLOADED"}


def _assert_ortho_unlocked(project) -> None:
    if project.state in ("ANALYZING", "FINALIZING"):
        raise HTTPException(409, {
            "code": "CONFLICT_BUSY",
            "message": "Cannot change the orthomosaic while a run is in progress",
            "project_id": project.id,
        })
    if project.state not in _ORTHO_UNLOCKED_STATES:
        raise HTTPException(423, {
            "code": "ORTHO_LOCKED",
            "message": (
                "The input dataset is locked because this run has already been "
                "analyzed. Open a new run (PATCH /projects/{id} or re-analyze) "
                "to replace the orthomosaic."
            ),
            "project_id": project.id,
        })


@router.patch("/projects/{project_id}")
def update_project(
    body: ProjectUpdate,
    project=Depends(get_project),
    db: Session = Depends(get_db),
):
    """Edit parameters and prepare a re-run on the SAME uploaded ortho.

    If the current run has already been used (analyzed/finalized/failed), its
    summary is archived to ``runs`` and ``current_run`` is bumped so the next
    analyze computes into a fresh ``work/run_<n+1>`` folder - previous runs are
    preserved on disk. If the current run hasn't been analyzed yet, params are
    just updated in place. Either way the project ends in ``UPLOADED``, ready for
    ``POST /runs/analyze``.
    """
    if project.state in ("ANALYZING", "FINALIZING"):
        raise HTTPException(409, {
            "code": "CONFLICT_BUSY",
            "message": "Cannot change parameters while a run is in progress",
            "project_id": project.id,
        })
    if not project.orthos:
        raise HTTPException(400, {
            "code": "BAD_REQUEST",
            "message": "Upload an orthomosaic before configuring a re-run",
            "project_id": project.id,
        })

    # Type-check provided param overrides up front so a bad value fails here with
    # 400 instead of deep in the worker with 500 (v4 section 8.3).
    if body.params:
        _validate_param_overrides(body.params)

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
        raise HTTPException(400, {"code": "BAD_REQUEST", "message": str(e),
                                  "project_id": project.id})

    if project.state in USED_RUN_STATES:
        archive_current_run(db, project)

    project.params = new_params
    if body.model_key is not None:
        project.model_key = body.model_key
    if body.source_epsg is not None:
        project.source_epsg = body.source_epsg
    if body.run_name is not None:
        project.run_name = body.run_name.strip() or None
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

    Locked (423 ORTHO_LOCKED) once the current run has been analyzed - the
    dataset can only change on a freshly-opened run.
    """
    _assert_ortho_unlocked(project)
    if not (file.filename or "").lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, {
            "code": "BAD_REQUEST",
            "message": "Orthomosaic must be a .tif/.tiff GeoTIFF",
            "project_id": project.id,
        })

    paths = ensure_project_dirs(project.id, project.current_run)
    _clear_existing_orthos(project, db, paths)

    stem = os.path.splitext(os.path.basename(file.filename))[0]
    dst = os.path.join(paths["input_ortho"], f"{stem}.tif")
    _stream_to_disk(file, dst, max_bytes=settings.max_upload_mb * 1024 * 1024)
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
    'anyone with the link' are supported. Same set-semantics as the file upload -
    any previously-registered ortho is replaced. Locked (423 ORTHO_LOCKED) once
    the current run has been analyzed.
    """
    _assert_ortho_unlocked(project)
    url = (body.url or "").strip()
    host = urlparse(url).netloc.lower()
    if not (host == "drive.google.com" or host.endswith(".google.com")):
        raise HTTPException(400, {"code": "BAD_REQUEST",
            "message": "Only Google Drive links are supported.", "project_id": project.id})

    file_id = _extract_drive_id(url)
    if not file_id:
        raise HTTPException(400, {"code": "BAD_REQUEST",
            "message": "Could not parse a Google Drive file id from the URL.",
            "project_id": project.id})

    try:
        import gdown
    except ImportError:
        raise HTTPException(503, {"code": "DEPENDENCY_MISSING",
            "message": "Server is missing the 'gdown' dependency required for URL uploads."})

    paths = ensure_project_dirs(project.id, project.current_run)
    tmp_dir = tempfile.mkdtemp(prefix="ortho_dl_", dir=paths["input_ortho"])
    try:
        try:
            out = gdown.download(id=file_id, output=tmp_dir + os.sep, quiet=True)
        except Exception as e:
            raise HTTPException(400, {"code": "BAD_REQUEST",
                "message": f"Google Drive download failed: {e}", "project_id": project.id})
        if not out or not os.path.exists(out):
            raise HTTPException(400, {"code": "BAD_REQUEST",
                "message": ("Download failed - the file may be private, deleted, or over "
                            "its Google Drive download quota."), "project_id": project.id})

        max_bytes = settings.max_upload_mb * 1024 * 1024
        if os.path.getsize(out) > max_bytes:
            raise HTTPException(413, {"code": "UPLOAD_TOO_LARGE",
                "message": f"Downloaded file exceeds the {settings.max_upload_mb} MB limit.",
                "project_id": project.id})
        if not out.lower().endswith((".tif", ".tiff")):
            raise HTTPException(400, {"code": "BAD_REQUEST",
                "message": "The Drive file is not a .tif/.tiff GeoTIFF.", "project_id": project.id})

        stem = os.path.splitext(os.path.basename(out))[0]
        dst = os.path.join(paths["input_ortho"], f"{stem}.tif")
        _clear_existing_orthos(project, db, paths)
        shutil.move(out, dst)
        return _register_ortho(project, db, dst, stem)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- ground-truth zip safety limits (defend against bombs / hostile archives) --
_GT_MAX_MEMBERS = 10000                          # absurd file counts -> reject
_GT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024     # 2 GiB uncompressed budget
_GT_MAX_FILE_BYTES = 250 * 1024 * 1024           # 250 MiB per extracted file
_GT_MAX_RATIO = 200                              # uncompressed/compressed ratio guard


@router.post("/projects/{project_id}/ground-truth")
def upload_ground_truth(project=Depends(get_project), file: UploadFile = File(...)):
    """Upload a .zip whose top-level folders are species names containing crown
    .tif files (the structure step3_validate expects).

    Hardened against zip bombs and hostile archives: the compressed upload is
    size-capped, the archive is inspected *before* anything is written, and only
    regular ``*.tif`` members are extracted - each into ``<species>/<file>.tif``
    with a sanitized path and copied through per-file and total-size budgets, so
    a lying header or a decompression bomb cannot fill the disk. Existing ground
    truth is replaced (set-semantics).
    """
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, {"code": "BAD_ARCHIVE",
            "message": "Ground truth must be a .zip of <species>/*.tif folders",
            "project_id": project.id})

    import zipfile

    paths = ensure_project_dirs(project.id, project.current_run)
    # Stage the upload OUTSIDE input_gt so input_gt can be wiped for set-semantics.
    tmp = os.path.join(paths["root"], "_gt_upload.zip")
    _stream_to_disk(file, tmp, max_bytes=settings.max_upload_mb * 1024 * 1024)

    try:
        try:
            zf = zipfile.ZipFile(tmp)
        except zipfile.BadZipFile:
            raise HTTPException(400, {"code": "BAD_ARCHIVE",
                "message": "File is not a valid .zip archive", "project_id": project.id})
        with zf as z:
            extracted = _safe_extract_gt_tifs(z, paths["input_gt"])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    if extracted == 0:
        raise HTTPException(400, {"code": "BAD_ARCHIVE",
            "message": "Archive contained no usable .tif ground-truth images",
            "project_id": project.id})

    species = sorted(
        d for d in os.listdir(paths["input_gt"])
        if os.path.isdir(os.path.join(paths["input_gt"], d))
    )
    return {
        "state": project.state,
        "species_folders": species,
        "files_extracted": extracted,
    }


# -- helpers --------------------------------------------------------------
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
    """Delete any previously-registered ortho (file + DB row) - set-semantics."""
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


def _validate_param_overrides(overrides: dict) -> None:
    """Validate provided pipeline-param overrides against their declared types so
    a bad value fails fast with 400 (v4 section 8.3). Unknown keys are left untouched."""
    from pydantic import TypeAdapter
    from app.schemas.project import PipelineParams
    fields = PipelineParams.model_fields
    for key, val in (overrides or {}).items():
        if key in fields:
            try:
                TypeAdapter(fields[key].annotation).validate_python(val)
            except Exception as e:
                raise HTTPException(400, {"code": "BAD_REQUEST",
                    "message": f"Invalid value for param '{key}': {e}"})


def _stream_to_disk(
    file: UploadFile, dst: str, chunk: int = 1024 * 1024, max_bytes: int | None = None
) -> None:
    """Stream an upload to disk. If ``max_bytes`` is given, abort (and delete the
    partial file) as soon as the body exceeds it - so an oversized upload can
    never be fully written."""
    written = 0
    with open(dst, "wb") as out:
        while True:
            data = file.file.read(chunk)
            if not data:
                break
            written += len(data)
            if max_bytes is not None and written > max_bytes:
                out.close()
                try:
                    os.remove(dst)
                except OSError:
                    pass
                file.file.close()
                raise HTTPException(413, {"code": "UPLOAD_TOO_LARGE",
                    "message": f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit."})
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


def _safe_member_path(name: str) -> str | None:
    """Return a sanitized ``<species>/<file>`` path for a zip member, or None if
    unsafe. Strips drive letters and leading separators, rejects ``..`` and
    absolute paths, flattens deep nesting to ``<species>/<file>``, and limits
    each component to a safe charset."""
    name = name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    fname = parts[-1]
    species = parts[-2] if len(parts) >= 2 else "unlabelled"

    def _clean(seg: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", seg).strip("._") or "x"

    safe_name = _clean(fname)
    if not safe_name.lower().endswith(".tif"):
        return None
    return os.path.join(_clean(species), safe_name)


def _safe_extract_gt_tifs(z, dest: str) -> int:
    """Validate a ground-truth zip and extract only safe ``*.tif`` members.

    Defends against zip bombs (member-count, total-size and per-member
    compression-ratio caps) and hostile paths (traversal, absolute paths,
    symlinks/devices). Returns the number of files written.
    """
    infos = z.infolist()
    if len(infos) > _GT_MAX_MEMBERS:
        raise HTTPException(400, {"code": "BAD_ARCHIVE", "message": f"Archive has too many entries (> {_GT_MAX_MEMBERS})"})

    # Pre-flight on the headers: catch obvious bombs before writing a single byte.
    declared_total = 0
    for info in infos:
        if info.is_dir():
            continue
        declared_total += info.file_size
        if info.file_size > _GT_MAX_FILE_BYTES:
            raise HTTPException(413, {"code": "UPLOAD_TOO_LARGE", "message": f"Archive contains an oversized member: {info.filename}"})
        if info.compress_size > 0 and (info.file_size / info.compress_size) > _GT_MAX_RATIO:
            raise HTTPException(400, {"code": "BAD_ARCHIVE", "message": "Archive looks like a decompression bomb (suspicious ratio)"})
    if declared_total > _GT_MAX_TOTAL_BYTES:
        raise HTTPException(413, {"code": "UPLOAD_TOO_LARGE", "message": "Archive exceeds the uncompressed size budget"})

    # Set-semantics: drop any previous ground truth, then recreate the folder.
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)

    written_total = 0
    count = 0
    for info in infos:
        if info.is_dir():
            continue
        # Skip anything that isn't a regular file (symlink/device/fifo). A mode of
        # 0 (common for Windows-made zips) is treated as a regular file.
        mode = (info.external_attr >> 16) & 0o170000
        if mode and mode != 0o100000:
            continue
        if not info.filename.lower().endswith(".tif"):
            continue
        rel = _safe_member_path(info.filename)
        if rel is None:
            continue
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
        # Stream-copy enforcing REAL byte counts (never trust the header).
        file_bytes = 0
        with z.open(info) as src, open(target, "wb") as out:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                file_bytes += len(buf)
                written_total += len(buf)
                if file_bytes > _GT_MAX_FILE_BYTES or written_total > _GT_MAX_TOTAL_BYTES:
                    out.close()
                    try:
                        os.remove(target)
                    except OSError:
                        pass
                    raise HTTPException(413, {"code": "UPLOAD_TOO_LARGE", "message": "Archive exceeds size limits during extraction"})
                out.write(buf)
        count += 1
    return count
