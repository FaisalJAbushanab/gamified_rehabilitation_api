"""Naming Gaming App API (GANVA v2)."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import APP_VERSION, config_hash, get_settings
from .routers import admin, auth, participants, review, sessions, stimuli
from .security import audio_key, jwt_secret
from .storage import media_root


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail fast on missing production secrets.
    jwt_secret()
    audio_key()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Naming Gaming App API", version=APP_VERSION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (auth, participants, stimuli, sessions, review, admin):
        app.include_router(r.router)
    app.mount("/media", StaticFiles(directory=media_root()), name="media")

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": APP_VERSION, "config_hash": config_hash()}

    return app


app = create_app()
