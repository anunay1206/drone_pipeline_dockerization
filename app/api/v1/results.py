"""Final results summary + downloads."""
import csv
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import get_project
from app.core.storage import project_paths

router = APIRouter()


def _run(project) -> int:
    return getattr(project, "current_run", 1) or 1


def build_results_payload(project) -> dict:
    """Build the final summary + download links.

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
    return build_results_payload(project)


@router.get("/projects/{project_id}/results/kmz")
def download_kmz(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step4_output"], "species_map.kmz")
    if not os.path.exists(f):
        raise HTTPException(404, "KMZ not found")
    return FileResponse(
        f, media_type="application/vnd.google-earth.kmz", filename="species_map.kmz"
    )


@router.get("/projects/{project_id}/results/crown-master.csv")
def download_master(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step2_output"], "crown_master.csv")
    if not os.path.exists(f):
        raise HTTPException(404, "crown_master.csv not found")
    return FileResponse(f, media_type="text/csv", filename="crown_master.csv")


@router.get("/projects/{project_id}/results/polygon-species.csv")
def download_polyspecies(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step2_output"], "polygon_species.csv")
    if not os.path.exists(f):
        raise HTTPException(404, "polygon_species.csv not found")
    return FileResponse(f, media_type="text/csv", filename="polygon_species.csv")


@router.get("/projects/{project_id}/results/confusion-matrix.png")
def download_cm(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step3_output"], "confusion_matrix.png")
    if not os.path.exists(f):
        raise HTTPException(404, "confusion_matrix.png not found")
    return FileResponse(f, media_type="image/png")


@router.get("/projects/{project_id}/results/stac-item.json")
def download_stac_item(project=Depends(get_project)):
    f = os.path.join(project_paths(project.id, _run(project))["step4_output"], "stac_item.json")
    if not os.path.exists(f):
        raise HTTPException(404, "stac_item.json not found")
    return FileResponse(f, media_type="application/json", filename="stac_item.json")


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
