# ════════════════════════════════════════════════════════════════════════
# Tree-Crown Species Pipeline — single shared image
# Used by three services (api / worker / beat) with different start commands.
# Defaults to CPU torch for portability; see DOCKER.md for the GPU build.
# ════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache

# System libs:
#   build-essential + git  -> compile detectron2 from source
#   libgl1 libglib2.0-0     -> OpenCV / matplotlib runtime needed by detectron2/detectree2
#   curl                    -> container HEALTHCHECK against /livez
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pkg_resources is imported by detectron2.model_zoo; it was removed from
# setuptools 81+. Pin below 81 so that import keeps working.
RUN pip install "setuptools<81" wheel

# 1) Torch FIRST (CPU wheels) so the detectron2 source build links against it.
#    Override TORCH_INDEX for a CUDA build (see DOCKER.md).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url ${TORCH_INDEX}

# 2) API / service-layer deps.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 3) Pipeline deps (geospatial + ML support). Listed explicitly so we control
#    ordering; torch/detectron2/detectree2 from requirements.txt are handled
#    separately above/below.
RUN pip install \
        timm \
        rasterio geopandas shapely fiona \
        numpy pandas scikit-learn \
        matplotlib seaborn \
        tqdm simplekml

# 4) detectron2 (pinned to the commit this project is known to build with) and
#    detectree2.
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@e0ec4e189d438848521aee7926f9900e114229f5"
RUN pip install detectree2

# 5) Enforce the pipeline's Pillow pin LAST (earlier installs may bump it).
RUN pip install "Pillow==9.5.0"

# 6) Application code (weights, orthos, venv, storage excluded via .dockerignore).
COPY . .

EXPOSE 8000

# Default command runs the API; worker/beat override this in docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
