# Calling the Tree-Crown Species Pipeline API from Postman

Your project already exposes a real FastAPI service at `app/main.py`
(stage-gated, async, Celery-backed). This guide gives you a ready-to-import
Postman collection that mirrors every endpoint, with sample request bodies
and test scripts that auto-fill the `project_id` / `job_id` / `chosen_k`
environment variables as you go.

## Files

| File | Purpose |
|------|---------|
| `tree_classification.postman_collection.json` | The collection — 8 folders, every `/api/v1/*` endpoint plus `/health` and `/api/v1/models`. |
| `tree_classification.postman_environment.json` | Environment — `base_url`, optional `api_key`, and auto-filled flow variables. |
| `POSTMAN_API_GUIDE.md` | This file. |
| `api_server.py` | **Obsolete** — an earlier shim I created before I discovered `app/main.py`. You can delete it. |

## Import in Postman

1. **File → Import** (or the Import button) → drag in both JSON files.
2. Top-right environment selector → choose **Tree Classification - Local**.
3. If you set `TCP_API_KEY` in `.env`, open the environment and paste the
   key into `api_key`. If not, leave it blank.

The collection sets collection-level auth (`X-API-Key: {{api_key}}`) so every
request inherits it. When the server has no key configured it just ignores
the header.

## Run the server (WSL)

```bash
source venv/bin/activate

# 1) API process
# Use --host 0.0.0.0 so Windows Postman/browser can reach the WSL server.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 2) GPU worker — Phase A (detection + features), one task per process
celery -A app.workers.celery_app:celery_app worker -Q gpu -c 1 --pool=prefork -l info

# 3) CPU worker — Phase B (clustering, assign, validate, export)
celery -A app.workers.celery_app:celery_app worker -Q cpu -c 4 -l info
```

Workers need Redis (`TCP_REDIS_URL` in `.env`). For a quick test without
Redis or Celery, set `TCP_CELERY_EAGER=true` in `.env` — tasks then run
inline inside the API process, but the ML stack must still be importable
in that process.

### Reaching the WSL server from Windows Postman

The log you posted shows uvicorn was started without `--host`, so it binds to
`127.0.0.1` inside WSL. From Windows-side Postman, restart uvicorn with
`--host 0.0.0.0`:

```bash
uvicorn app.main:app --port 8000 --host 0.0.0.0
```

Then either keep `base_url = http://localhost:8000` (mirrored WSL networking)
or switch it to `http://<WSL-IP>:8000` — find the IP with `hostname -I` inside
WSL.

## Happy-path flow

The collection folders are ordered to match the pipeline phases. Run the
requests top-to-bottom in this order:

1. **Meta → Health check** — confirm 200 OK.
2. **Meta → List models** — confirm `urban_cambridge`, `paracou`, `randresize`
   weight files are present.
3. **Projects → Create project** — body already filled. Response 201 saves
   `project_id` into the env via a test script.
4. **Uploads → Upload orthomosaic** — Body → form-data → key `file`,
   type `File`, pick `examples/S3C.tif` (or any GeoTIFF). Repeat for
   multi-ortho projects.
5. *(Optional)* **Uploads → Upload ground-truth** — zip of
   `<species>/*.tif` folders for validation metrics.
6. **Analyze (Phase A) → Start analyze** — saves the returned `job_id`.
7. **Analyze (Phase A) → Poll job status** — re-send until `state` becomes
   `SUCCEEDED`. The project state moves to `AWAITING_LABELS`.
8. **Clustering Review → Clustering overview** — read `recommended_k` and
   the metric table; decide on `chosen_k`.
9. **Clustering Review → k-selection plot (PNG)** + **t-SNE for chosen_k** +
   **Clusters with sample crowns** — visual inspection. In Postman click
   the response → "Save Response → Save to a file" to keep the PNGs.
10. **Labels → Submit cluster → species labels** — edit the sample body so
    the `mapping` has one entry per cluster (0..k-1) and the species names
    are correct (e.g. `"acacia"`, `"non_acacia"`). The test script saves
    `chosen_k` into the env.
11. **Finalize (Phase B) → Start finalize** — saves a new `job_id`
    (overwrites the Phase A one).
12. **Analyze (Phase A) → Poll job status** — re-use the same request to
    poll Phase B until `SUCCEEDED`.
13. **Results → Results summary** — species distribution + validation +
    download URLs.
14. **Results → Download KMZ** — Send → "Save Response → Save to a file"
    → open the `.kmz` in Google Earth.

## Auth notes

`X-API-Key` is **optional**:

- If `TCP_API_KEY` is unset in `.env`, the server's `require_api_key`
  dependency lets every request through and returns the single-tenant user
  id `default`.
- If `TCP_API_KEY` is set, every request must send the matching value in
  the `X-API-Key` header. The collection already does this — just fill the
  `api_key` env variable.

## State machine cheat sheet

```
CREATED ─upload─► UPLOADED ─POST analyze─► ANALYZING ─worker done─► AWAITING_LABELS
                                                                          │
                                                                  POST labels
                                                                          ▼
                                                                  LABELS_SUBMITTED
                                                                          │
                                                                  POST finalize
                                                                          ▼
                                                                      FINALIZING ─worker done─► COMPLETED
```

The analyze endpoint accepts `UPLOADED`, `AWAITING_LABELS`, `FAILED`.
The finalize endpoint accepts `LABELS_SUBMITTED`, `COMPLETED`, `FAILED`.
A 409 response means the project is in the wrong state for that call.

## Troubleshooting

- **404 on `/config`** — that endpoint is from the obsolete `api_server.py`,
  not from `app/main.py`. The new collection no longer calls it.
- **401 on every request** — `TCP_API_KEY` is set but the `api_key` env var
  is wrong/blank.
- **400 "Upload at least one orthomosaic first"** on analyze — call the
  upload endpoint first; check `GET /api/v1/projects/{project_id}` shows
  the ortho in the `orthos` array.
- **409 "Cannot analyze from state ..."** — the project isn't in an
  allowed state; see the cheat sheet above.
- **Job stuck in QUEUED** — no Celery worker is running, or it's listening
  on a different queue (`gpu` vs `cpu`), or Redis isn't running. For a
  quick local test set `TCP_CELERY_EAGER=true` in `.env`.
- **404 on KMZ download** — finalize hasn't produced `species_map.kmz` yet;
  poll the finalize job until `SUCCEEDED` first.
