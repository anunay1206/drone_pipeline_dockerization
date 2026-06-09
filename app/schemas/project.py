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
    name: str = ""
    model_key: str | None = None          # None -> server default (urban_cambridge)
    source_epsg: int | None = None        # None -> auto-detected from the GeoTIFF
    params: PipelineParams = Field(default_factory=PipelineParams)


class ProjectUpdate(BaseModel):
    """Partial update for a re-run: change params (and optionally model/EPSG) and
    open the next run on the same uploaded ortho. All fields optional; ``params``
    is merged onto the existing params, not replaced wholesale."""

    model_key: str | None = None
    source_epsg: int | None = None
    params: dict | None = None


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
    runs: list = []
    orthos: list[OrthoOut] = []
    error: str | None = None
    created_at: datetime
    updated_at: datetime
