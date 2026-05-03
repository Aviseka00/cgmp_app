"""HTTP smoke tests against a running MongoDB (local or CI). Run: pytest tests/ -v"""

from __future__ import annotations

import io
import os
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def _minimal_rgb_png(width: int = 16, height: int = 16) -> bytes:
    """Valid PNG (RGB 8-bit) for upload/analyze tests; hand-built chunks often have bad CRCs."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    body = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    return body


@pytest.fixture(scope="session")
def client() -> TestClient:
    """One TestClient for the session: Motor must not outlive a closed event loop."""
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") in ("ok", "degraded")


def test_root_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_login_admin(client: TestClient) -> None:
    r = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("role") == "admin"
    assert data.get("access_token")


def test_dashboard_requires_auth(client: TestClient) -> None:
    r = client.get("/api/dashboard/summary")
    assert r.status_code == 401


def test_dashboard_with_token(client: TestClient) -> None:
    r = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    tok = r.json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    r2 = client.get("/api/dashboard/summary", headers=h)
    assert r2.status_code == 200, r2.text
    assert "total_batches" in r2.json() or "admin" in r2.json()


def test_notifications_list(client: TestClient) -> None:
    r = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    tok = r.json()["access_token"]
    r2 = client.get("/api/notifications", headers={"Authorization": f"Bearer {tok}"})
    assert r2.status_code == 200
    assert isinstance(r2.json(), list)


def test_reviewers_list(client: TestClient) -> None:
    r = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    tok = r.json()["access_token"]
    r2 = client.get("/api/users/reviewers", headers={"Authorization": f"Bearer {tok}"})
    assert r2.status_code == 200
    assert isinstance(r2.json(), list)


def test_create_batch_and_upload_flow(client: TestClient, tmp_path) -> None:
    r = client.post("/api/auth/login", data={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    tok = r.json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}

    code = f"TST-{os.urandom(3).hex().upper()}"
    fd = {
        "batch_code": code,
        "batch_name": "pytest batch",
        "description": "",
        "sample_id": "",
        "sample_description": "",
    }
    r2 = client.post("/api/batches", headers=h, data=fd)
    assert r2.status_code == 200, r2.text
    bid = r2.json()["batch_id"]

    png = _minimal_rgb_png()
    files = [("files", ("x.png", io.BytesIO(png), "image/png"))]
    r3 = client.post(f"/api/batches/{bid}/upload", headers=h, files=files)
    assert r3.status_code == 200, r3.text

    if not os.path.isfile(settings.best_model_path):
        pytest.skip(f"Model missing at {settings.best_model_path!r} — skip analyze")

    r4 = client.post(f"/api/batches/{bid}/analyze", headers=h)
    assert r4.status_code == 200, r4.text
    out = r4.json()
    assert out.get("images") == 1
    assert "outputs" in out
