"""Per-project artifact storage layout.

Maps the pipeline's hard-coded ``Config`` directories onto a single isolated
tree per project so concurrent jobs never collide. Local filesystem for now;
swap the body of these helpers for an S3/MinIO backend later without touching
the rest of the app.
"""
import os
import shutil

from app.core.settings import settings


def project_root(project_id: str) -> str:
    return os.path.join(settings.storage_root, "projects", project_id)


def project_paths(project_id: str) -> dict:
    """All directories used for a project (see API_DESIGN.md §5)."""
    root = project_root(project_id)
    work = os.path.join(root, "work")
    return {
        "root": root,
        "input_ortho": os.path.join(root, "input", "ortho"),
        "input_gt": os.path.join(root, "input", "ground_truth"),
        "work": work,                                       # == Config.WORKDIR
        "detectree": os.path.join(work, "detectree"),
        "ortho": os.path.join(work, "ortho"),               # == Config.ORTHO_FOLDER
        "polygons": os.path.join(work, "polygons"),         # == Config.POLY_FOLDER
        "step1_output": os.path.join(work, "step1_output"),
        "step2_output": os.path.join(work, "step2_output"),
        "step3_output": os.path.join(work, "step3_output"),
        "step4_output": os.path.join(work, "step4_output"),
        "logs": os.path.join(work, "logs"),
    }


def ensure_project_dirs(project_id: str) -> dict:
    paths = project_paths(project_id)
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def reset_dirs(project_id: str, keys: list[str]) -> dict:
    """Remove and recreate the named work directories for a clean re-run.

    Used at the start of a job so stale cached artifacts from a previous run
    (e.g. ``dinov2_features.npy``, ``tsne_coordinates.csv``, old clusters) can
    never leak into a fresh computation.
    """
    paths = project_paths(project_id)
    for key in keys:
        d = paths[key]
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    return paths


def delete_project_dir(project_id: str) -> None:
    root = project_root(project_id)
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
