"""Local HTTP API. One resident model, serialized GPU access, no external calls."""

import hmac
import os
from math import isfinite
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi import Request as HttpRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from .compat import (
    SystemOneRequest,
    UnknownModel,
    from_native,
    list_models,
    resolve_model,
    to_native,
)
from .responses import Response
from .schema import Request

API_KEY_ENV = "RIZZO_API_KEY"
PLAYGROUND = Path(__file__).with_name("playground.html")
SNAKE = Path(__file__).with_name("snake.html")
LOGO = Path(__file__).with_name("logo.png")


def jsonable(value):
    """The same structure, with whatever `json.dumps` would refuse replaced by its text.

    Two things reach here: non-JSON floats (NaN, Infinity, which `json.loads` accepts on the
    way in) echoed back as the offending input, and the exception object a validator raised.
    """
    if isinstance(value, float):
        return value if isfinite(value) else f"<{value}>"
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def create_app(engine, api_key=None):
    app = FastAPI(
        title="Rizzo Flow",
        version="0.2.0",
        description="Typed decisions with a local Spark-X2.5 model; no text generation.",
    )
    api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)

    @app.exception_handler(RequestValidationError)
    def invalid_request(request: HttpRequest, error: RequestValidationError):
        # `json.loads` accepts NaN and Infinity, JSON does not. Validation rejects them, but the
        # 422 body echoes the offending input, and serializing that would fail inside the
        # response and turn a client error into a 500. Report them instead of echoing them.
        return JSONResponse(status_code=422, content={"detail": jsonable(error.errors())})

    def authorize(authorization: str | None = Header(default=None)):
        # Bearer auth mirrors the hosted API; it is enforced only when a key is configured.
        if api_key and not hmac.compare_digest(
            (authorization or "").encode(), f"Bearer {api_key}".encode()
        ):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ready", "model": engine.backend.metadata}

    @app.post("/v1/decisions", response_model=Response, dependencies=[Depends(authorize)])
    def decisions(request: Request):
        try:
            return engine.decide(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    def systemone(request: SystemOneRequest):
        try:
            served = resolve_model(request.model, engine.backend.metadata)
            native, options = to_native(request)
            return from_native(request, engine.decide(native), options, served)
        except UnknownModel as error:
            # The hosted API answers an unserved model name with a 400 and a typed detail.
            raise HTTPException(
                status_code=400,
                detail={"error_type": "api_usage_error", "message": str(error)},
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        return list_models(engine.backend.metadata)

    @app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
    def playground():
        return PLAYGROUND.read_text(encoding="utf-8")

    @app.get("/snake", response_class=HTMLResponse, include_in_schema=False)
    def snake():
        return SNAKE.read_text(encoding="utf-8")

    @app.get("/playground/logo.png", include_in_schema=False)
    def logo():
        return FileResponse(LOGO, media_type="image/png")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/playground")

    return app
