"""Structured error envelope shared by both backends.

See API_SPECIFICATION_v4 section 8. Every 4xx/5xx response is normalised to:

    {"error": {"code": "...", "message": "...",
               "project_id": ..., "stage": ..., "details": ...}}

Call sites may either raise ``ApiError`` (precise code) or the usual
``HTTPException(status, "msg")`` (code derived from the status) - or
``HTTPException(status, {"code": ..., "message": ...})`` to set a precise code
without importing ApiError. A catch-all handler converts anything else into a
500 INTERNAL envelope so stack traces never reach clients.
"""
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Default machine code per HTTP status (used when a call site doesn't supply one).
_STATUS_CODE = {
    400: "BAD_REQUEST",
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "INVALID_STATE",
    413: "UPLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
    423: "ORTHO_LOCKED",
    425: "NOT_READY",
    429: "RATE_LIMITED",
    500: "INTERNAL",
    502: "DISPATCH_FAILED",
    503: "DEPENDENCY_MISSING",
    504: "COMPUTE_TIMEOUT",
}


class ApiError(Exception):
    """Raise for a precise error envelope with a stable machine code."""

    def __init__(self, status_code, code, message, *,
                 project_id=None, stage=None, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.project_id = project_id
        self.stage = stage
        self.details = details
        super().__init__(message)


def _envelope(code, message, project_id=None, stage=None, details=None):
    return {
        "error": {
            "code": code,
            "message": message,
            "project_id": project_id,
            "stage": stage,
            "details": details,
        }
    }


async def _api_error_handler(request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.message, exc.project_id, exc.stage, exc.details),
    )


async def _http_exception_handler(request, exc: StarletteHTTPException):
    detail = exc.detail
    headers = getattr(exc, "headers", None)
    if isinstance(detail, dict):
        code = detail.get("code") or _STATUS_CODE.get(exc.status_code, "ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(
                code,
                detail.get("message", ""),
                detail.get("project_id"),
                detail.get("stage"),
                detail.get("details"),
            ),
            headers=headers,
        )
    code = _STATUS_CODE.get(exc.status_code, "ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(code, str(detail)),
        headers=headers,
    )


async def _validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_envelope(
            "VALIDATION_ERROR",
            "Request validation failed",
            details=jsonable_encoder(exc.errors()),
        ),
    )


async def _unhandled_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=_envelope("INTERNAL", "Internal server error"),
    )


def install_error_handlers(app) -> None:
    """Register the envelope handlers on a FastAPI app (call once at startup)."""
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
