"""Submit the cluster -> species mapping by uploading the filled CSV.

This closes the human-in-the-loop gate between DAG 1 and DAG 2.
"""
import csv
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_project
from app.db import models
from app.db.session import get_db
from app.services.pipeline_adapter import normalize_species, write_species_map_csv

router = APIRouter()

_LABEL_STATES = {"AWAITING_LABELS", "LABELS_SUBMITTED", "COMPLETED"}
_REQUIRED_COLS = {"cluster", "species"}


@router.post("/projects/{project_id}/labels")
def submit_labels(
    project=Depends(get_project),
    chosen_k: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload the filled ``k{chosen_k}_cluster_species_map.csv``.

    The CSV must have at least the columns ``cluster`` and ``species``;
    ``cluster_folder`` and ``notes`` are accepted but optional. One row per
    cluster (0 to chosen_k − 1). Empty species values are stored as
    ``unlabelled``; whitespace, case, and hyphens in species names are
    normalised server-side so 'Non Acacia', 'non-acacia', and 'non_acacia'
    all become ``non_acacia``.
    """
    if project.state not in _LABEL_STATES:
        raise HTTPException(409, f"Cannot submit labels from state {project.state}")
    if project.available_k and chosen_k not in project.available_k:
        raise HTTPException(400, f"chosen_k must be one of {project.available_k}")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Labels file must be a .csv")

    # Parse the upload. utf-8-sig strips a BOM if Excel added one.
    try:
        raw = file.file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "Labels file must be UTF-8 encoded text")
    finally:
        file.file.close()

    reader = csv.DictReader(io.StringIO(raw))
    cols = set(reader.fieldnames or [])
    missing = _REQUIRED_COLS - cols
    if missing:
        raise HTTPException(
            400, f"Labels CSV is missing required column(s): {sorted(missing)}"
        )

    mapping: dict[int, dict] = {}
    for row in reader:
        try:
            cid = int(str(row.get("cluster", "")).strip())
        except (TypeError, ValueError):
            continue
        if cid < 0 or cid >= chosen_k:
            continue  # silently ignore rows outside the chosen_k range
        mapping[cid] = {
            "species": normalize_species(row.get("species", "")),
            "notes": (row.get("notes") or "").strip(),
        }

    if not mapping:
        raise HTTPException(400, "Labels CSV had no usable rows for chosen_k")

    # Replace any previous mapping for this project.
    db.query(models.ClusterLabel).filter_by(project_id=project.id).delete()
    for cid, m in mapping.items():
        db.add(
            models.ClusterLabel(
                project_id=project.id,
                chosen_k=chosen_k,
                cluster_id=cid,
                species=m["species"],
                notes=m["notes"],
            )
        )

    # persist chosen_k into params (new dict so SQLAlchemy flags it dirty)
    params = dict(project.params or {})
    params["chosen_k"] = chosen_k
    project.params = params
    db.add(project)
    db.commit()

    # write the canonical CSV the pipeline's step2 reads
    write_species_map_csv(project, chosen_k, mapping)

    project.state = "LABELS_SUBMITTED"
    db.add(project)
    db.commit()

    counts: dict[str, int] = {}
    for v in mapping.values():
        counts[v["species"]] = counts.get(v["species"], 0) + 1
    return {
        "state": project.state,
        "chosen_k": chosen_k,
        "species_counts_preview": counts,
    }
