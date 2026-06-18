# Tree-Crown Species Pipeline — API

A stage-gated, asynchronous FastAPI service wrapping the existing detection /
clustering / species-mapping pipeline. Full design rationale is in
[`API_DESIGN.md`](./API_DESIGN.md).

The pipeline has a **mandatory human-in-the-loop break**: it runs detection +
clustering, **pauses** so you can review the two cluster-analysis visuals
(t-SNE + k-selection) and assign a species to each cluster, then resumes to
produce the Google-Earth KMZ. The API models this as two background jobs around
a `AWAITING_LABELS` gate.

## Layout

```
app/
  main.py                 FastAPI app (uvicorn app.main:app)
  core/                   settings, storage layout, model registry
  db/                     SQLAlchemy models + session
  schemas/                Pydantic request/response models
  services/               pipeline_adapter (build per-project Config), serializers
  workers/                Celery app + job_a_analyze / job_b_finalize tasks
  api/v1/                 routers: projects, analyze, clustering, labels, finalize, results
predict.py                pipeline Step 0 (Detectree2)        ← patched, imported by workers
tree_crown_pipeline.py    pipeline Steps 1–4                  ← patched, imported by workers
```

The pipeline modules stay at the repo root and are imported by the Celery
workers (lazily), so the FastAPI process itself doesn't need the heavy ML stack.

## Install

```bash
# API layer
pip install -r requirements-api.txt
# pipeline layer (on the worker host: GPU + heavy deps)
pip install -r requirements.txt        # see README.md for detectron2/detectree2 steps
```

## Configure

```bash
cp .env.example .env
# ensure the three .pth files are in TCP_MODELS_DIR (repo root by default):
#   urban_trees_Cambridge_20230630.pth   (key: urban_cambridge, default)
#   220723_withParacouUAV.pth            (key: paracou)
#   230103_randresize_full.pth           (key: randresize)
```

## Run (3 processes)

```bash
# 1) API
uvicorn app.main:app --reload

# 2) GPU worker — detection + features (prefork, one task per process)
celery -A app.workers.celery_app:celery_app worker -Q gpu -c 1 --pool=prefork -l info

# 3) CPU worker — clustering, t-SNE, assign, validate, export
celery -A app.workers.celery_app:celery_app worker -Q cpu -c 4 -l info
```

Requires a running Redis (`TCP_REDIS_URL`). Interactive docs at `/docs`.

> Dev without Redis/workers: set `TCP_CELERY_EAGER=true` to run tasks inline in
> the API process. The ML stack must still be importable for a real run.

## End-to-end flow

```bash
BASE=http://localhost:8000/api/v1

# 1. create a project (model_key optional -> default urban_cambridge)
PID=$(curl -s -X POST $BASE/projects -H 'Content-Type: application/json' \
      -d '{"name":"Sanjay Van S3","model_key":"urban_cambridge"}' | jq -r .project_id)

# 2. upload one or more orthomosaics (call repeatedly for multi-ortho)
curl -X POST $BASE/projects/$PID/orthomosaic -F file=@ortho/s3_tree.tif

# 3. (optional) ground truth for validation: zip of <species>/*.tif folders
# curl -X POST $BASE/projects/$PID/ground-truth -F file=@labels.zip

# 4. start Phase A (detect + cluster)
JOB=$(curl -s -X POST $BASE/projects/$PID/analyze | jq -r .job_id)

# 5. poll until SUCCEEDED  (state -> AWAITING_LABELS)
curl -s $BASE/jobs/$JOB | jq '{state, current_stage, progress}'

# 6. review the two DAGs + clusters
curl -s $BASE/projects/$PID/clustering | jq
#    open  .../clustering/k-selection.png   and   .../clustering/6/tsne.png
#    inspect cluster thumbnails:  .../clustering/6/clusters

# 7. submit cluster -> species labels
curl -X POST $BASE/projects/$PID/labels -H 'Content-Type: application/json' -d '{
  "chosen_k": 6,
  "mapping": [
    {"cluster":0,"species":"acacia"},
    {"cluster":1,"species":"non_acacia"}
  ]
}'

# 8. finalize (assign + validate + export)
JOB2=$(curl -s -X POST $BASE/projects/$PID/finalize | jq -r .job_id)
curl -s $BASE/jobs/$JOB2 | jq '{state, current_stage}'

# 9. results + download KMZ
curl -s $BASE/projects/$PID/results | jq
curl -s -OJ $BASE/projects/$PID/results/kmz
```

## Notes / production

* SQLite + `init_db()` is for dev. Use Postgres + Alembic in production.
* Detection workers must use the **prefork** pool — the Step 0 patch uses
  `os.chdir` for per-job CWD isolation, which is process-global.
* Models are warm-loaded once per worker process (cached by `model_key` /
  DINOv2 config).
* Storage is local filesystem behind `app/core/storage.py`; swap for S3/MinIO
  without touching the routers.
