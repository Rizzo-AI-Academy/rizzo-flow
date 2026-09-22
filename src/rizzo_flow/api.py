"""Local HTTP API. One resident model, serialized GPU access, no external calls."""

import hmac
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .compat import SystemOneRequest, from_native, list_models, resolve_model, to_native
from .engine import Engine
from .responses import Response
from .schema import Request

API_KEY_ENV = "RIZZO_API_KEY"
PLAYGROUND = Path(__file__).with_name("playground.html")
SNAKE = Path(__file__).with_name("snake.html")
LOGO = Path(__file__).with_name("logo.png")


def create_app(engine: Engine, api_key: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Rizzo Flow",
        version="0.2.0",
        description="Typed decisions with a local Spark-X2.5 model; no text generation.",
    )
    api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)

    def authorize(authorization: str | None = Header(default=None)):
        # Bearer auth mirrors the hosted API; it is enforced only when a key is configured.
        if api_key and not hmac.compare_digest(
            (authorization or "").encode(), f"Bearer {api_key}".encode()
        ):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ready", "model": engine.backend.metadata.as_dict()}

    @app.post("/v1/decisions", response_model=Response)
    def decisions(request: Request):
        try:
            return engine.decide(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    def systemone(request: SystemOneRequest):
        try:
            metadata = engine.backend.metadata
            resolve_model(request.model, metadata)  # rejects a model this server cannot serve
            native, options = to_native(request)
            return from_native(request, engine.decide(native), options, metadata)
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
