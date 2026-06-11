"""Registry of available Detectree2 weight files.

The user supplies the ``.pth`` files in ``settings.models_dir``. A project picks
one via ``model_key``; if omitted the configured default is used.
"""
import os

from app.core.settings import settings

# model_key -> filename (resolved against settings.models_dir)
MODEL_FILES: dict[str, str] = {
    "urban_cambridge": "urban_trees_Cambridge_20230630.pth",
    "paracou": "220723_withParacouUAV.pth",
    "randresize": "230103_randresize_full.pth",
}

DEFAULT_MODEL_KEY = settings.default_model_key


def list_models() -> list[dict]:
    """Return the registry with availability + default flags for the UI."""
    out = []
    for key, fname in MODEL_FILES.items():
        path = os.path.join(settings.models_dir, fname)
        out.append(
            {
                "key": key,
                "filename": fname,
                "available": os.path.exists(path),
                "default": key == DEFAULT_MODEL_KEY,
            }
        )
    return out


def resolve_model_path(model_key: str | None) -> tuple[str, str]:
    """Return (resolved_key, absolute_path) for a model_key (or the default)."""
    key = model_key or DEFAULT_MODEL_KEY
    if key not in MODEL_FILES:
        raise ValueError(
            f"Unknown model_key '{key}'. Valid keys: {sorted(MODEL_FILES)}"
        )
    path = os.path.abspath(os.path.join(settings.models_dir, MODEL_FILES[key]))
    return key, path
