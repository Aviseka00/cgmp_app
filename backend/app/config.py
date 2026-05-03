import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory (contains uploads/, app/, etc.)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
# cgmp_app/ (repository folder)
_REPO_ROOT = _BACKEND_ROOT.parent


class Settings(BaseSettings):
    app_name: str = "cGMP Cell Analysis Platform"
    secret_key: str = "change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "cgmp_cell_app"
    uploads_dir: str = str(_BACKEND_ROOT / "uploads")
    processed_dir: str = str(_BACKEND_ROOT / "processed_images")
    reports_dir: str = str(_BACKEND_ROOT / "reports")
    # JSON snapshots under registry/users and registry/batches (complements MongoDB).
    registry_dir: str = str(_BACKEND_ROOT / "registry")
    # Weights: use env in production. Default fits a checked-out clone with optional weights/ in the repo.
    best_model_path: str = str(_REPO_ROOT / "weights" / "production_model.pth")
    # Vendored inference package at repo root (see /cell_detection). Docker sets CELL_DETECTION_DIR=/opt/cell_detection.
    cell_detection_dir: str = str(_REPO_ROOT / "cell_detection")
    model_tile: int = 512
    model_stride: int = 256
    model_threshold: float = 0.5
    # Post-threshold cleanup: morphological opening removes isolated 1–2px hits (kernel 0 = off; use odd ≥3).
    speckle_morph_open_kernel: int = 3
    # Drop components below this area (px²), after opening.
    min_detection_area_pixels: int = 500
    # Also drop if min(bounding box width, height) is below this (stops “chunky” small dots); 0 = disable.
    min_detection_short_side_pixels: int = 8
    max_upload_mb: int = 100
    allowed_upload_extensions: str = ".jpg,.jpeg,.png,.webp,.tif,.tiff,.bmp"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    def upload_extension_allowed(self, filename: str | None) -> bool:
        ext = (os.path.splitext(filename or "")[1] or "").lower()
        allowed = {x.strip().lower() for x in self.allowed_upload_extensions.split(",") if x.strip()}
        return ext in allowed


settings = Settings()
