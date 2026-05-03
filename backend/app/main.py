import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .db import db, ensure_indexes, ping_mongo
from .routers import router
from .security import get_password_hash

# SPA on disk (three levels up from backend/app/main.py → cgmp_app/frontend).
_HERE = Path(__file__).resolve()
_INDEX_HTML = _HERE.parent.parent.parent / "frontend" / "index.html"

# StaticFiles requires these paths at import time; keep in sync with lifespan.
os.makedirs(settings.uploads_dir, exist_ok=True)
os.makedirs(settings.processed_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)
os.makedirs(settings.registry_dir, exist_ok=True)
os.makedirs(os.path.join(settings.registry_dir, "users"), exist_ok=True)
os.makedirs(os.path.join(settings.registry_dir, "batches"), exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.uploads_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)
    os.makedirs(settings.reports_dir, exist_ok=True)
    os.makedirs(settings.registry_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.registry_dir, "users"), exist_ok=True)
    os.makedirs(os.path.join(settings.registry_dir, "batches"), exist_ok=True)
    await ensure_indexes()
    disable_seed = os.getenv("DISABLE_SEED_ADMIN", "").strip().lower() in ("1", "true", "yes")
    if not disable_seed:
        admin = await db.users.find_one({"username": "admin"})
        if not admin:
            await db.users.insert_one(
                {
                    "_id": "seed-admin",
                    "username": "admin",
                    "password_hash": get_password_hash("admin123"),
                    "role": "admin",
                    "is_active": True,
                    "must_change_password": False,
                }
            )
        reg_u = os.path.join(settings.registry_dir, "users", "seed-admin.json")
        if not os.path.isfile(reg_u):
            try:
                with open(reg_u, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "_id": "seed-admin",
                            "username": "admin",
                            "role": "admin",
                            "is_active": True,
                            "must_change_password": False,
                            "registry_note": "Seed administrator (JSON mirror of MongoDB user record).",
                        },
                        f,
                        indent=2,
                    )
            except OSError:
                pass
    _log = logging.getLogger("uvicorn.error")
    if _INDEX_HTML.is_file():
        _log.info("SPA index: %s", _INDEX_HTML)
    else:
        _log.warning("SPA index missing (GET / will 503): %s", _INDEX_HTML)
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

# JWT is sent in Authorization header; cookies not used — credentials=False avoids invalid *+credentials CORS pairing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _spa_index_response(request: Request) -> Response:
    """HTML shell for GET / and HEAD / (some stacks skip explicit routes)."""
    if not _INDEX_HTML.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Frontend not found at {_INDEX_HTML}",
        )
    if request.method == "HEAD":
        st = _INDEX_HTML.stat()
        return Response(
            content=b"",
            media_type="text/html; charset=utf-8",
            headers={"Content-Length": str(st.st_size)},
        )
    return FileResponse(_INDEX_HTML, media_type="text/html; charset=utf-8")


# Register early so / is never shadowed by later mounts.
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def spa_index(request: Request) -> Response:
    """Serve login shell — same origin as /api so no ?api_base needed."""
    return _spa_index_response(request)


@app.api_route("/index.html", methods=["GET", "HEAD"], include_in_schema=False)
async def spa_index_alias(request: Request) -> Response:
    return _spa_index_response(request)


app.include_router(router, prefix="/api")
app.mount("/processed_images", StaticFiles(directory=settings.processed_dir), name="processed_images")
app.mount("/reports", StaticFiles(directory=settings.reports_dir), name="reports")


@app.get("/health")
async def health() -> dict[str, str]:
    mongo_ok = await ping_mongo()
    status = "ok" if mongo_ok else "degraded"
    return {"status": status, "mongodb": "up" if mongo_ok else "down"}


@app.exception_handler(StarletteHTTPException)
async def _spa_fallback_on_404(request: Request, exc: StarletteHTTPException) -> Response:
    """If routing still misses GET /, serve index instead of JSON {\"detail\":\"Not Found\"}."""
    if exc.status_code == 404 and request.scope.get("type") == "http":
        if request.method in ("GET", "HEAD"):
            path = request.url.path or "/"
            if path in ("/", "") or path.rstrip("/") == "":
                if _INDEX_HTML.is_file():
                    return _spa_index_response(request)
    return await http_exception_handler(request, exc)
