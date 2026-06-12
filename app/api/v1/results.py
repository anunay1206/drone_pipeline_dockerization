"""Final results summary + downloads (current run) + per-run history/comparison."""
import csv
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import get_project
from app.core.storage import project_paths

router = APIRouter()


def _run(project) -> int:
    return getattr(project, "current_run", 1) or 1


def _require_completed(project) -> None:
    """425 NOT_READY until the run has produced results (v4 section 8.3)."""
    if project.state != "COMPLETED":
        raise HTTPException(425, {"code": "NOT_READY",
            "message": f"Results not ready (state {project.state})",
            "project_id": project.id})


def build_results_payload(project) -> dict:
    """Build the final summary + download links for the current run.

    Extracted so the synchronous POST /finalize can return it directly.
    """
    p = project_paths(project.id, _run(project))
    master = os.path.join(p["step2_output"], "crown_master.csv")
    polyspecies = os.path.join(p["step2_output"], "polygon_species.csv")
    kmz = os.path.join(p["step4_output"], "species_map.kmz")
    cm = os.path.join(p["step3_output"], "confusion_matrix.png")
    stac = os.path.join(p["step4_output"], "stac_item.json")

    distribution: dict[str, int] = {}
    if os.path.exists(master):
        with open(master) as f:
            for row in csv.DictReader(f):
                sp = row.get("species", "unlabelled") or "unlabelled"
                distribution[sp] = distribution.get(sp, 0) + 1

    base = f"/api/v1/projects/{project.id}/results"
    return {
        "state": project.state,
        "run": _run(project),
        "species_distribution": distribution,
        "validation": _read_validation(p),
        "downloads": {
            "kmz": f"{base}/kmz" if os.path.exists(kmz) else None,
            "crown_master_csv": f"{base}/crown-master.csv" if os.path.exists(master) else None,
            "polygon_species_csv": f"{base}/polygon-species.csv" if os.path.exists(polyspecies) else None,
            "confusion_matrix_png": f"{base}/confusion-matrix.png" if os.path.exists(cm) else None,
            "stac_item_json": f"{base}/stac-item.json" if os.path.exists(stac) else None,
        },
    }


@router.get("/projects/{project_id}/results")
def results(project=Depends(get_project)):
    _require_completed(project)
    return build_results_payload(project)


@router.get("/projects/{project_id}/results/kmz")
def download_kmz(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step4_output"], "species_map.kmz")
    _require_completed(project)
    if not os.path.exists(f):
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "KMZ not found",
            "project_id": project.id})
    return FileResponse(
        f, media_type="application/vnd.google-earth.kmz", filename="species_map.kmz"
    )


@router.get("/projects/{project_id}/results/crown-master.csv")
def download_master(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step2_output"], "crown_master.csv")
    _require_completed(project)
    if not os.path.exists(f):
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "crown_master.csv not found",
            "project_id": project.id})
    return FileResponse(f, media_type="text/csv", filename="crown_master.csv")


@router.get("/projects/{project_id}/results/polygon-species.csv")
def download_polyspecies(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step2_output"], "polygon_species.csv")
    _require_completed(project)
    if not os.path.exists(f):
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "polygon_species.csv not found",
            "project_id": project.id})
    return FileResponse(f, media_type="text/csv", filename="polygon_species.csv")


@router.get("/projects/{project_id}/results/confusion-matrix.png")
def download_cm(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step3_output"], "confusion_matrix.png")
    _require_completed(project)
    if not os.path.exists(f):
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "confusion_matrix.png not found",
            "project_id": project.id})
    return FileResponse(f, media_type="image/png")


@router.get("/projects/{project_id}/results/stac-item.json")
def download_stac_item(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step4_output"], "stac_item.json")
    _require_completed(project)
    if not os.path.exists(f):
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "stac_item.json not found",
            "project_id": project.id})
    return FileResponse(f, media_type="application/json", filename="stac_item.json")


# -- run history: list + per-run results for comparison (v5) ----------------
_ASSETS = {
    "kmz": ("step4_output", "species_map.kmz",
            "application/vnd.google-earth.kmz", "species_map.kmz"),
    "crown-master.csv": ("step2_output", "crown_master.csv", "text/csv", "crown_master.csv"),
    "polygon-species.csv": ("step2_output", "polygon_species.csv", "text/csv", "polygon_species.csv"),
    "confusion-matrix.png": ("step3_output", "confusion_matrix.png", "image/png", None),
    "stac-item.json": ("step4_output", "stac_item.json", "application/json", "stac_item.json"),
}


def _asset_path(project_id: str, run: int, asset: str):
    spec = _ASSETS.get(asset)
    if spec is None:
        raise HTTPException(404, {"code": "NOT_FOUND", "message": f"Unknown asset '{asset}'"})
    dir_key, fname, media, download_name = spec
    p = project_paths(project_id, run)
    return os.path.join(p[dir_key], fname), media, download_name


def _run_results_payload(project, run: int) -> dict:
    """Results summary for a specific (possibly archived) run, with run-scoped
    download URLs. 404s if that run never produced final outputs."""
    p = project_paths(project.id, run)
    master = os.path.join(p["step2_output"], "crown_master.csv")
    kmz = os.path.join(p["step4_output"], "species_map.kmz")
    if not (os.path.exists(master) or os.path.exists(kmz)):
        raise HTTPException(404, {"code": "NOT_FOUND",
            "message": f"Run {run} has no final results (it may not have been finalized)",
            "project_id": project.id})

    distribution: dict[str, int] = {}
    if os.path.exists(master):
        with open(master) as f:
            for row in csv.DictReader(f):
                sp = row.get("species", "unlabelled") or "unlabelled"
                distribution[sp] = distribution.get(sp, 0) + 1

    base = f"/api/v1/projects/{project.id}/runs/{run}/results"
    downloads = {}
    for asset in _ASSETS:
        path, _, _ = _asset_path(project.id, run, asset)
        key = asset.replace("-", "_").replace(".", "_")
        downloads[key] = f"{base}/{asset}" if os.path.exists(path) else None

    return {
        "run": run,
        "species_distribution": distribution,
        "validation": _read_validation(p),
        "downloads": downloads,
    }


def _run_meta(project) -> list[dict]:
    """All runs (archived + current), oldest first, with results availability."""
    entries = []
    for h in (project.runs or []):
        entries.append(dict(h))
    entries.append({
        "run": project.current_run or 1,
        "run_name": getattr(project, "run_name", None),
        "params": dict(project.params or {}),
        "model_key": project.model_key,
        "state": project.state,
        "recommended_k": project.recommended_k,
        "available_k": project.available_k,
        "ortho": project.orthos[0].filename if project.orthos else None,
    })
    for e in entries:
        run = e.get("run") or 1
        p = project_paths(project.id, run)
        e["is_current"] = run == (project.current_run or 1)
        e["has_results"] = (
            os.path.exists(os.path.join(p["step2_output"], "crown_master.csv"))
            or os.path.exists(os.path.join(p["step4_output"], "species_map.kmz"))
        )
        e["results_url"] = (
            f"/api/v1/projects/{project.id}/runs/{run}/results" if e["has_results"] else None
        )
    return entries


@router.get("/projects/{project_id}/runs")
def list_runs(project=Depends(get_project)):
    """Run history for this project - used by the frontend's comparison view."""
    return {"current_run": project.current_run or 1, "runs": _run_meta(project)}


@router.get("/projects/{project_id}/runs/{run}/results")
def run_results(run: int, project=Depends(get_project)):
    if run < 1 or run > (project.current_run or 1):
        raise HTTPException(404, {"code": "NOT_FOUND",
            "message": f"Run {run} does not exist (runs 1..{project.current_run or 1})",
            "project_id": project.id})
    return _run_results_payload(project, run)


@router.get("/projects/{project_id}/runs/{run}/results/{asset}")
def run_asset(run: int, asset: str, project=Depends(get_project)):
    if run < 1 or run > (project.current_run or 1):
        raise HTTPException(404, {"code": "NOT_FOUND",
            "message": f"Run {run} does not exist", "project_id": project.id})
    path, media, download_name = _asset_path(project.id, run, asset)
    if not os.path.exists(path):
        raise HTTPException(404, {"code": "NOT_FOUND",
            "message": f"{asset} not found for run {run}", "project_id": project.id})
    kwargs = {"media_type": media}
    if download_name:
        kwargs["filename"] = download_name
    return FileResponse(path, **kwargs)


def _read_validation(p: dict):
    """Derive simple metrics from step3's validation_detail.csv (acc + counts)."""
    detail = os.path.join(p["step3_output"], "validation_detail.csv")
    if not os.path.exists(detail):
        return None
    try:
        with open(detail) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        total = len(rows)
        correct = sum(
            1 for r in rows if r.get("true_species") == r.get("pred_species")
        )
        return {
            "matched_samples": total,
            "accuracy": round(correct / total, 4) if total else None,
        }
    except Exception:
        return None
