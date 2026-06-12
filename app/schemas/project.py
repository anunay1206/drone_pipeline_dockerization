from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.models_registry import default_backbone


class PipelineParams(BaseModel):
    """User-tunable knobs; mirror the pipeline's Config fields."""

    # detection (Step 0)
    tile_size: int = 10
    buffer: int = 10
    iou_threshold: float = 0.9
    conf_threshold: float = 0.85
    # features + clustering (Step 1)
    k_list: list[int] = Field(default_factory=lambda: [2, 4, 6, 8, 10])
    pca_components: int | None = 50
    batch_size: int = 16
    img_size: int = 224
    model_name: str = Field(default_factory=default_backbone)


class ProjectCreate(BaseModel):
    """Project creation takes only a display name (v5): the UUID is generated
    server-side and returned as the reference for every other endpoint. Model +
    parameter configuration moved to the analyze trigger. model_key /
    source_epsg / params are still accepted for backwards compatibility."""

    name: str = ""
    model_key: str | None = None          # None -> server default (urban_cambridge)
    source_epsg: int | None = None        # None -> auto-detected from the GeoTIFF
    params: PipelineParams = Field(default_factory=PipelineParams)


class AnalyzeTrigger(BaseModel):
    """Optional body for POST /runs/analyze: name this run and configure the
    detector / feature extractor / pipeline params in the same call. params is
    merged onto the project's existing params (not replaced wholesale)."""

    run_name: str | None = None
    model_key: str | None = None
    source_epsg: int | None = None
    params: dict | None = None


class ProjectUpdate(BaseModel):
    """Partial update for a re-run: change params (and optionally model/EPSG) and
    open the next run on the same uploaded ortho. All fields optional; params is
    merged onto the existing params, not replaced wholesale."""

    model_key: str | None = None
    source_epsg: int | None = None
    params: dict | None = None
    run_name: str | None = None


class OrthoFromUrl(BaseModel):
    """Request body for registering an ortho from a public Google Drive link."""

    url: str


class OrthoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stem: str
    filename: str
    width: int | None = None
    height: int | None = None
    crs: str | None = None
    bands: int | None = None


class ProjectOut(BaseModel):
    project_id: str
    name: str
    model_key: str
    state: str
    source_epsg: int | None = None
    params: dict
    recommended_k: int | None = None
    available_k: list[int] | None = None
    current_run: int = 1
    run_name: str | None = None
    runs: list = []
    orthos: list[OrthoOut] = []
    error: str | None = None
    # Structured failure info when state == FAILED (v4 section 8.1): {code, stage, message}.
    last_error: dict | None = None
    created_at: datetime
    updated_at: datetime
