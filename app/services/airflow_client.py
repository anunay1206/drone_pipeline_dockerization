"""Minimal Airflow REST client used by the /runs/* trigger endpoints.

Only one operation is needed: trigger a DAG run. Uses the stdlib ``urllib`` so
the API process gains no new dependency. Auth is optional — if no credentials
are configured the request is sent unauthenticated.

Airflow REST (stable API v1):
    POST {base}/api/v1/dags/{dag_id}/dagRuns
    body: {"conf": {...}}
    -> {"dag_run_id": "...", "state": "queued", ...}
"""
import base64
import json
import urllib.error
import urllib.request

from app.core.settings import settings


def airflow_enabled() -> bool:
    """True when an Airflow base URL is configured (otherwise dispatch is local)."""
    return bool((settings.airflow_base_url or "").strip())


def _auth_header() -> dict:
    """Optional auth header. Bearer token wins over basic-auth; both optional."""
    if settings.airflow_auth_token:
        return {"Authorization": f"Bearer {settings.airflow_auth_token}"}
    if settings.airflow_username:
        raw = f"{settings.airflow_username}:{settings.airflow_password or ''}".encode()
        return {"Authorization": "Basic " + base64.b64encode(raw).decode()}
    return {}


def trigger_dag(dag_id: str, conf: dict, timeout: int = 30) -> str:
    """Trigger a DAG run and return its ``dag_run_id``.

    Raises ``RuntimeError`` with a readable message on any HTTP/transport error
    so the caller can surface a 502 to the client.
    """
    base = settings.airflow_base_url.rstrip("/")
    url = f"{base}/api/v1/dags/{dag_id}/dagRuns"
    body = json.dumps({"conf": conf or {}}).encode()

    headers = {"Content-Type": "application/json", **_auth_header()}
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode() or "{}")
        return data.get("dag_run_id", "")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"Airflow returned {e.code} for {dag_id}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Airflow at {base}: {e.reason}") from e
