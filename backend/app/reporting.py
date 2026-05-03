import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from .config import settings
from .db import db


def _safe_dir_segment(code: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (code or "").strip()).strip("_")
    return s or "batch"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def save_batch_report(batch_code: str, report_type: str, payload: dict[str, Any]) -> dict[str, str]:
    os.makedirs(settings.reports_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{batch_code}_{report_type}_{ts}.json"
    path = os.path.join(settings.reports_dir, filename)
    wrapped = {
        "report_type": report_type,
        "generated_at": _utc(),
        "batch_code": batch_code,
        "payload": payload,
    }
    serialized = json.dumps(wrapped, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialized)

    file_hash = _sha256_hex(serialized.encode("utf-8"))
    sidecar_path = path + ".sha256"
    with open(sidecar_path, "w", encoding="utf-8") as cf:
        cf.write(file_hash + "\n")

    prev = await db.report_checksums.find_one({}, sort=[("created_at", -1)])
    prev_hash = prev["chain_hash"] if prev else ""
    chain_material = {
        "batch_code": batch_code,
        "report_type": report_type,
        "report_path": path,
        "file_hash": file_hash,
        "prev_hash": prev_hash,
        "ts": _utc(),
    }
    chain_hash = _sha256_hex(_canonical_json_bytes(chain_material))
    record = {
        "_id": f"{batch_code}-{ts}-{report_type}",
        "batch_code": batch_code,
        "report_type": report_type,
        "report_path": path,
        "sidecar_path": sidecar_path,
        "file_hash": file_hash,
        "prev_hash": prev_hash,
        "chain_hash": chain_hash,
        "created_at": _utc(),
    }
    await db.report_checksums.insert_one(record)
    try:
        os.chmod(path, 0o444)
        os.chmod(sidecar_path, 0o444)
    except OSError:
        # Best effort on Windows/filesystems that do not fully support chmod semantics.
        pass
    return {
        "report_path": path,
        "checksum": file_hash,
        "chain_hash": chain_hash,
        "sidecar_path": sidecar_path,
        "report_url": _report_file_to_url(path),
    }


def _report_file_to_url(abs_path: str) -> str:
    norm_path = os.path.normpath(abs_path)
    base = os.path.normpath(settings.reports_dir)
    try:
        rel = os.path.relpath(norm_path, base)
    except ValueError:
        return ""
    if rel.startswith(".."):
        return ""
    return "/reports/" + rel.replace("\\", "/")


async def save_analysis_run_report(
    batch_code: str,
    run_number: int,
    archive_id: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    """
    Persist a numbered segmentation analysis report under
    reports/batches/<batch_code>/analysis/run_XXXX_<archive>.json
    and record checksum metadata (same chain as formal batch reports).
    """
    seg = _safe_dir_segment(batch_code)
    subdir = os.path.join(settings.reports_dir, "batches", seg, "analysis")
    os.makedirs(subdir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"run_{run_number:04d}_{archive_id[:8]}_{ts}.json"
    path = os.path.join(subdir, fname)
    wrapped = {
        "report_category": "analysis_run",
        "report_type": "analysis_run",
        "run_number": run_number,
        "archive_id": archive_id,
        "generated_at": _utc(),
        "batch_code": batch_code,
        "payload": payload,
    }
    serialized = json.dumps(wrapped, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialized)

    file_hash = _sha256_hex(serialized.encode("utf-8"))
    sidecar_path = path + ".sha256"
    with open(sidecar_path, "w", encoding="utf-8") as cf:
        cf.write(file_hash + "\n")

    prev = await db.report_checksums.find_one({}, sort=[("created_at", -1)])
    prev_hash = prev["chain_hash"] if prev else ""
    chain_material = {
        "batch_code": batch_code,
        "report_type": "analysis_run",
        "report_path": path,
        "file_hash": file_hash,
        "prev_hash": prev_hash,
        "ts": _utc(),
    }
    chain_hash = _sha256_hex(_canonical_json_bytes(chain_material))
    record = {
        "_id": f"{batch_code}-analysis-{archive_id}-{ts}",
        "batch_code": batch_code,
        "report_type": "analysis_run",
        "report_path": path,
        "sidecar_path": sidecar_path,
        "file_hash": file_hash,
        "prev_hash": prev_hash,
        "chain_hash": chain_hash,
        "created_at": _utc(),
        "run_number": run_number,
        "archive_id": archive_id,
    }
    await db.report_checksums.insert_one(record)
    try:
        os.chmod(path, 0o444)
        os.chmod(sidecar_path, 0o444)
    except OSError:
        pass
    return {
        "report_path": path,
        "checksum": file_hash,
        "chain_hash": chain_hash,
        "sidecar_path": sidecar_path,
        "report_url": _report_file_to_url(path),
    }


async def verify_report_checksum(report_path: str) -> dict[str, Any]:
    rec = await db.report_checksums.find_one({"report_path": report_path})
    if not rec:
        return {"exists": False, "is_valid": False, "reason": "No checksum record found"}
    if not os.path.isfile(report_path):
        return {"exists": True, "is_valid": False, "reason": "Report file missing"}
    with open(report_path, "rb") as f:
        current_hash = _sha256_hex(f.read())
    expected = rec["file_hash"]
    is_valid = current_hash == expected
    return {
        "exists": True,
        "is_valid": is_valid,
        "expected_hash": expected,
        "current_hash": current_hash,
        "chain_hash": rec["chain_hash"],
        "report_path": report_path,
    }
