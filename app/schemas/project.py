from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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
    model_name: str = "vit_base_patch14_dinov2.lvd142m"


class PipelineParamsUpdate(BaseModel):
    """Partial update for user-tunable pipeline knobs."""

    tile_size: int | None = None
    buffer: int | None = None
    iou_threshold: float | None = None
    conf_threshold: float | None = None
    k_list: list[int] | None = None
    pca_components: int | None = None
    batch_size: int | None = None
    img_size: int | None = None
    model_name: str | None = None


class ProjectCreate(BaseModel):
    name: str = ""
    model_key: str | None = None          # None -> server default (urban_cambridge)
    source_epsg: int | None = None        # None -> auto-detected from the GeoTIFF
    params: PipelineParams = Field(default_factory=PipelineParams)


class ProjectUpdate(BaseModel):
    name: str | None = None
    model_key: str | None = None
    source_epsg: int | None = None
    params: PipelineParamsUpdate | None = None


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
    orthos: list[OrthoOut] = []
    error: str | None = None
    created_at: datetime
    updated_at: datetime
