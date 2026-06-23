"""RiftRace FastAPI application entrypoint.

Run from backend/ with: uvicorn api.main:app --reload --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .logging_config import configure_logging, logger
from .routers import explain, flagged, graph, investigate, predict, stats
from .services import graph_service, llm_service, model_service, rag_service

configure_logging()

# llm_service is deliberately excluded here -- it depends on an optional
# external API key (GROQ_API_KEY) and must degrade gracefully at request
# time, not block the whole app from starting when it's unset.
REQUIRED_PATHS = [*model_service.REQUIRED_FILES, *graph_service.REQUIRED_FILES, *rag_service.REQUIRED_FILES]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail-fast startup: refuse to accept traffic if required files are
    missing, or if the model/graph fail to actually load. A missing model
    file used to look "healthy" at startup and only surface as a bare 500
    on the first real /predict request -- this catches that before deploy
    traffic ever reaches the app.
    """
    logger.info("Validating required files and loading model/graph...")
    missing = [str(p) for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        message = "Missing required files, refusing to start:\n  " + "\n  ".join(missing)
        logger.error(message)
        raise RuntimeError(message)

    try:
        model_service.warm_up()
        graph_service.warm_up()
        rag_service.warm_up()
    except Exception as exc:
        logger.exception("Failed to load required artifacts at startup")
        raise RuntimeError(f"RiftRace API failed to initialize: {exc}") from exc

    logger.info("RiftRace API ready.")
    yield
    logger.info("Shutting down RiftRace API.")


app = FastAPI(
    title="RiftRace API",
    description="Graph intelligence API for detecting illicit Bitcoin transactions.",
    version="0.1.0",
    lifespan=lifespan,
)

class RequestLoggingMiddleware:
    """Logs method, path, status code, and latency for every request.

    Deliberately a plain ASGI middleware, not @app.middleware("http")
    (Starlette's BaseHTTPMiddleware). BaseHTTPMiddleware's call_next()
    re-raises exceptions even after a registered exception handler has
    already built and sent a response for them -- which would silently
    skip this logging on every error path, since the line after call_next()
    never runs. Wrapping `send` directly avoids that entirely.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f'{scope["method"]} {scope["path"]} -> {status_code} ({duration_ms:.1f}ms)')


# Allow the React dev server to call this API directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Create React App default
        "http://localhost:5173",  # Vite default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added after CORSMiddleware so it ends up as the outermost layer (each
# add_middleware call wraps around the previous ones).
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches anything not already handled (e.g. HTTPException, pydantic's
    RequestValidationError keep their own normal responses) and turns it
    into structured JSON instead of FastAPI's bare default 500, with the
    real stack trace going to the server log, not the client.
    """
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred while processing this request.",
        },
    )


app.include_router(predict.router, tags=["prediction"])
app.include_router(graph.router, tags=["graph"])
app.include_router(stats.router, tags=["stats"])
app.include_router(flagged.router, tags=["flagged"])
app.include_router(explain.router, tags=["explain"])
app.include_router(investigate.router, tags=["investigate"])


@app.get("/")
def root() -> dict:
    """Bare liveness ping: process is up. Use /health for a real readiness
    check that confirms the model can actually serve predictions.
    """
    return {"status": "ok", "service": "RiftRace API"}


@app.get("/health")
def health_check() -> dict:
    try:
        model_status = model_service.health_check()
        graph_status = graph_service.health_check()
        rag_status = rag_service.health_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {exc}") from exc

    return {"status": "ok", **model_status, **graph_status, **rag_status, "llm_configured": llm_service.is_configured()}
