# Running the Tree-Crown Pipeline with Docker

Four services, one shared image:

| Service | Role |
|---------|------|
| `redis`  | Celery broker + result backend + cleanup lock |
| `api`    | FastAPI — uploads, project lifecycle, review/results JSON |
| `worker` | Celery worker — **analyze + cluster (DAG 1)** and **finalize / KMZ (DAG 2)** |
| `beat`   | Celery Beat — nightly retention cleanup |

Storage and model weights are **mounted**, not baked into the image.

## 1. One-time setup

Put the detector weights and the model catalog under `./models/`:

```
models/
  models.yaml
  urban_trees_Cambridge_20230630.pth
  220723_withParacouUAV.pth
  230103_randresize_full.pth
```

(`models.yaml` references each `.pth` by bare filename, resolved against `/models` inside the container.)

Create your env file:

```bash
cp .env.example .env
```

## 2. Build & run

```bash
docker compose build          # first build compiles detectron2 from source — slow (~10-20 min)
docker compose up -d          # start redis + api + worker + beat
docker compose ps             # check health
curl http://localhost:8000/livez
```

The API is on `http://localhost:8000`. Project data persists in the `app-data` volume
(`/data/storage/<project_id>/...` and the SQLite DB at `/data/treecrown.db`).

## 3. How the workflow runs

- `POST /api/v1/projects/{id}/analyze` and `/finalize` enqueue Celery tasks; the **worker**
  container executes them (detection -> clustering -> t-SNE, then assign -> validate -> KMZ).
- The worker drains both queues (`gpu,cpu`) with the prefork pool (detection needs per-process
  CWD isolation). Scale it with `docker compose up -d --scale worker=2`.
- `beat` fires `cleanup_expired_projects` nightly (default 03:00 UTC), deleting projects idle
  longer than `TCP_RETENTION_DAYS`.

## 4. GPU build (optional)

The default image installs CPU torch so it runs anywhere. For GPU:

1. Base the image on a CUDA image (e.g. `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` with
   Python 3.10) and build torch with the matching CUDA index:
   ```bash
   docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118
   ```
2. Install the NVIDIA Container Toolkit on the host and uncomment the `deploy.resources.devices`
   block under `worker:` in `docker-compose.yml`.

## 5. Notes

- **detectron2** is pinned to commit `e0ec4e1` (the version this project builds with).
- **setuptools<81** and **Pillow==9.5.0** are pinned in the image (required by detectron2's
  model zoo and the pipeline respectively).
- The first DINOv2 run downloads weights from Hugging Face into `/data/hf-cache` (on the volume),
  so it is only downloaded once.
- Static file serving of `/data/storage/<id>/output/...` (so the frontend can link to results)
  is a small follow-up: mount the storage dir into the API with FastAPI `StaticFiles`.
