"""STAC Item emission for a completed pipeline run.

After ``job_b_finalize`` produces the KMZ + CSV outputs, we emit a SpatioTemporal
Asset Catalog (STAC) Item describing the run — a machine-readable, standards-based
manifest of the run's footprint, parameters and downloadable assets. It mirrors
``tree_crown_stac_item.example.yaml`` but is rendered as JSON (the canonical STAC
serialization) and filled with the run's real values.

The item is written to the run's ``step4_output/stac_item.json`` (next to the
KMZ) so it is preserved per run alongside the rest of that run's artifacts.

Asset/link hrefs point at the live result-download API endpoints (the same ones
``results.py`` exposes), so the manifest is usable as-is today. They are relative
by default; set ``TCP_PUBLIC_BASE_URL`` to emit absolute hrefs for a real STAC
catalog (e.g. ``https://api.example.com``).
"""
import csv
import json
import os
from datetime import datetime, timezone

from app.core.models_registry import default_backbone
from app.core.settings import settings
from app.core.storage import project_paths

# Human-readable descriptions for the columns of crown_master.csv. Any column not
# listed here still gets emitted, just with a generic description.
_COLUMN_DOCS = {
    "image_name": "Crown image filename",
    "polygon_id": "Crown polygon id",
    "site": "Orthomosaic stem",
    "cluster": "KMeans cluster id",
    "species": "Assigned species label",
    "true_species": "Ground-truth species label",
    "pred_species": "Predicted species label",
}

# STAC table-extension column types inferred from a sample value.
_INT_COLUMNS = {"polygon_id", "cluster"}


def _run(project) -> int:
    return getattr(project, "current_run", 1) or 1


def _href(rel_path: str) -> str:
    """Build an asset href. Relative by default; absolute if PUBLIC_BASE_URL set."""
    base = (getattr(settings, "public_base_url", "") or "").rstrip("/")
    return f"{base}{rel_path}" if base else rel_path


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in (text or "")]
    s = "".join(keep).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _footprint_wgs84(ortho_dir: str):
    """Return (geometry, bbox) in WGS84 from the union of ortho footprints.

    Reads each GeoTIFF's bounds and reprojects to EPSG:4326. Returns
    ``(None, None)`` if rasterio is unavailable or no readable ortho is found, so
    STAC emission never blocks finalize.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except Exception:
        return None, None

    if not os.path.isdir(ortho_dir):
        return None, None

    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    found = False
    for f in os.listdir(ortho_dir):
        if not f.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(ortho_dir, f)
        try:
            with rasterio.open(path) as src:
                if src.crs is None:
                    continue
                l, b, r, t = transform_bounds(
                    src.crs, "EPSG:4326", *src.bounds, densify_pts=21
                )
        except Exception:
            continue
        minx, miny = min(minx, l), min(miny, b)
        maxx, maxy = max(maxx, r), max(maxy, t)
        found = True

    if not found:
        return None, None

    bbox = [round(minx, 6), round(miny, 6), round(maxx, 6), round(maxy, 6)]
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[0], bbox[3]],
            [bbox[2], bbox[3]],
            [bbox[2], bbox[1]],
            [bbox[0], bbox[1]],
        ]],
    }
    return geometry, bbox


def _table_columns(master_csv: str) -> list[dict]:
    """Build STAC table-extension columns from crown_master.csv's header."""
    if not os.path.exists(master_csv):
        return []
    try:
        with open(master_csv, newline="") as f:
            header = next(csv.reader(f), [])
    except Exception:
        return []
    cols = []
    for name in header:
        cols.append({
            "name": name,
            "type": "int64" if name in _INT_COLUMNS else "string",
            "description": _COLUMN_DOCS.get(name, f"Column {name}"),
        })
    return cols


def build_stac_item(project, chosen_k: int | None = None) -> dict:
    """Construct the STAC Item dict for the project's current run."""
    run = _run(project)
    paths = project_paths(project.id, run)
    params = dict(getattr(project, "params", None) or {})

    master_csv = os.path.join(paths["step2_output"], "crown_master.csv")
    poly_csv = os.path.join(paths["step2_output"], "polygon_species.csv")
    kmz = os.path.join(paths["step4_output"], "species_map.kmz")
    cm_png = os.path.join(paths["step3_output"], "confusion_matrix.png")

    geometry, bbox = _footprint_wgs84(paths["input_ortho"])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    item_id = f"{_slug(project.name) or 'tree_crown'}_{project.id[:8]}_run{run}"

    properties = {
        "title": "Tree-Crown Species Map",
        "description": (
            "Per-crown tree-species map produced by the Tree-Crown pipeline: "
            "Detectree2 crown detection, DINOv2 feature extraction, KMeans "
            "clustering, and human-assigned species labels. Polygons reprojected "
            "to WGS84."
        ),
        "datetime": now,
        "keywords": ["forestry", "tree-crown", "species", "drone", "orthomosaic"],
        # run parameters
        "project_id": project.id,
        "run": run,
        "detector_model": project.model_key,
        "feature_extractor": params.get("model_name") or default_backbone(),
        "source_epsg": getattr(project, "source_epsg", None) or 32643,
        "chosen_k": chosen_k,
        # table extension
        "table:columns": _table_columns(master_csv),
    }

    # Asset/link hrefs mirror the live result-download endpoints in results.py.
    results_base = f"/api/v1/projects/{project.id}/results"
    assets: dict[str, dict] = {}
    if os.path.exists(kmz):
        assets["kmz"] = {
            "href": _href(f"{results_base}/kmz"),
            "type": "application/vnd.google-earth.kmz",
            "title": "Species map for Google Earth",
            "roles": ["data"],
        }
    if os.path.exists(master_csv):
        assets["crown_master"] = {
            "href": _href(f"{results_base}/crown-master.csv"),
            "type": "text/csv",
            "title": "Per-crown master table",
            "roles": ["data"],
        }
    if os.path.exists(poly_csv):
        assets["polygon_species"] = {
            "href": _href(f"{results_base}/polygon-species.csv"),
            "type": "text/csv",
            "title": "Polygon-to-species table",
            "roles": ["data"],
        }
    if os.path.exists(cm_png):
        assets["confusion_matrix"] = {
            "href": _href(f"{results_base}/confusion-matrix.png"),
            "type": "image/png",
            "title": "Validation confusion matrix",
            "roles": ["overview"],
        }
    # Detection overlay (served by the clustering router) as the thumbnail.
    det = paths["detectree"]
    if os.path.isdir(det) and any(
        os.path.exists(os.path.join(det, s, "overlay.png")) for s in os.listdir(det)
    ):
        assets["thumbnail"] = {
            "href": _href(f"/api/v1/projects/{project.id}/detection/overlay.png"),
            "type": "image/png",
            "title": "Detection overlay",
            "roles": ["thumbnail"],
        }

    item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/table/v1.2.0/schema.json"
        ],
        "id": item_id,
        "collection": "tree_crown_runs",
        "geometry": geometry,
        "properties": properties,
        "assets": assets,
        "links": [
            {
                "rel": "self",
                "href": _href(f"{results_base}/stac-item.json"),
                "type": "application/json",
            }
        ],
    }
    if bbox is not None:
        item["bbox"] = bbox
    return item


def stac_item_path(project) -> str:
    """Filesystem path of the run's STAC item."""
    return os.path.join(
        project_paths(project.id, _run(project))["step4_output"], "stac_item.json"
    )


def write_stac_item(project, chosen_k: int | None = None) -> str:
    """Build and persist the STAC Item to the run's step4 output. Returns path."""
    item = build_stac_item(project, chosen_k=chosen_k)
    out = stac_item_path(project)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(item, f, indent=2)
    return out
