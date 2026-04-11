from __future__ import annotations

from collections.abc import Awaitable, Callable
import logging
import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routers.ask import router as ask_router
from api.routers.health import router as health_router
from api.routers.ingest import router as ingest_router

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Public Health RAG Assistant",
    description="Domain-focused Retrieval-Augmented Generation API for UK health protection reports.",
    version="1.0.0",
)
app.include_router(health_router)
app.include_router(ask_router)
app.include_router(ingest_router)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    malformed_json = any(err.get("type") == "json_invalid" for err in errors)
    message = "Malformed JSON request body. Use proper JSON serialization and escaped control characters."
    if not malformed_json:
        message = "Request validation failed."
    return JSONResponse(
        status_code=422,
        content={
            "message": message,
            "detail": errors,
        },
    )


@app.middleware("http")
async def add_process_time_header(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    started = time.perf_counter()
    response: Response
    try:
        response = await call_next(request)
    except HTTPException as exc:
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
    except Exception:
        logger.exception("Unhandled request error")
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )

    response.headers["X-Process-Time"] = str(_elapsed_ms(started))
    return response


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
