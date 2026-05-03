import json
import os
import re
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pymongo.errors import DuplicateKeyError

from .audit import log_activity
from .config import settings
from .db import db
from .deps import get_current_user, require_roles
from .inference import analyze_image
from .mongo_utils import as_str_id, ids_equal, match_created_by, serialize_document, user_id_query
from .pdf_export import build_analysis_archive_pdf
from .reporting import _safe_dir_segment, save_analysis_run_report, save_batch_report, verify_report_checksum
from .security import create_access_token, get_password_hash, verify_password


router = APIRouter()


def _registry_users_dir() -> str:
    return os.path.join(settings.registry_dir, "users")


def _registry_batches_dir() -> str:
    return os.path.join(settings.registry_dir, "batches")


def _write_user_registry_file(doc: dict[str, Any]) -> None:
    try:
        os.makedirs(_registry_users_dir(), exist_ok=True)
        sid = as_str_id(doc.get("_id"))
        snap = {k: v for k, v in serialize_document(doc).items() if k != "password_hash"}
        path = os.path.join(_registry_users_dir(), f"{sid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)
    except Exception:
        pass


def _remove_user_registry_files(user_id: str, stored_id: Any = None) -> None:
    keys = {user_id, as_str_id(user_id), as_str_id(stored_id)}
    for key in {k for k in keys if k}:
        path = os.path.join(_registry_users_dir(), f"{key}.json")
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _write_batch_registry_file(doc: dict[str, Any]) -> None:
    os.makedirs(_registry_batches_dir(), exist_ok=True)
    sid = as_str_id(doc.get("_id"))
    path = os.path.join(_registry_batches_dir(), f"{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialize_document(doc), f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _oid() -> str:
    return str(uuid.uuid4())


async def _next_image_sequence(batch_id: str, batch_code: str) -> int:
    """Next 1-based image index so repeat uploads to the same batch do not collide on (batch_id, filename)."""
    pattern = re.compile(r"^" + re.escape(batch_code) + r"_(\d+)(?:\.[^.]+)?$", re.IGNORECASE)
    max_n = 0
    async for doc in db.images.find({"batch_id": batch_id}, {"filename": 1}):
        fn = doc.get("filename") or ""
        m = pattern.match(fn)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


async def _save_upload_with_limit(upload: UploadFile, dest_path: str, max_bytes: int) -> None:
    chunk_size = 1024 * 1024
    written = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum upload size ({settings.max_upload_mb} MB)",
                    )
                out.write(chunk)
    except HTTPException:
        if os.path.isfile(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise


def _batch_image_mutations_blocked(batch: dict[str, Any]) -> str | None:
    """Return error message if uploads/analysis are not allowed for this batch status."""
    s = batch.get("status") or ""
    if s in ("passed", "rejected"):
        return "This batch is closed (accepted or rejected). Open Result folders to view read-only records."
    if s == "submitted":
        return "This batch is awaiting reviewer decision; images cannot be changed until a decision is recorded."
    return None


def _processed_path_to_url(path: str) -> str | None:
    if not path:
        return None
    norm_path = os.path.normpath(path)
    base = os.path.normpath(settings.processed_dir)
    try:
        rel = os.path.relpath(norm_path, base)
    except ValueError:
        return None
    if rel.startswith(".."):
        return None
    return "/processed_images/" + rel.replace("\\", "/")


def _batch_assigned_to_reviewer(batch: dict[str, Any], reviewer_id: Any) -> bool:
    ar = batch.get("assigned_reviewer_id")
    if ar is None or ar == "":
        return True
    return ids_equal(ar, reviewer_id)


def _reviewer_may_read_batch(batch: dict[str, Any], user: dict[str, Any]) -> bool:
    uname = user.get("username") or ""
    st = batch.get("status") or ""
    if st == "submitted":
        return _batch_assigned_to_reviewer(batch, user.get("_id"))
    if batch.get("reviewed_by") == uname:
        return True
    return False


def _may_read_batch(batch: dict[str, Any], user: dict[str, Any]) -> bool:
    if user.get("role") == "admin":
        return True
    if user.get("role") == "user":
        return ids_equal(batch.get("created_by"), user.get("_id"))
    if user.get("role") == "reviewer":
        return _reviewer_may_read_batch(batch, user)
    return False


@router.post("/auth/login")
async def login(username: str = Form(...), password: str = Form(...)) -> dict[str, Any]:
    username = (username or "").strip()
    user = await db.users.find_one({"username": username})
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account disabled")
    token = create_access_token(subject=user["username"], role=user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"],
        "must_change_password": bool(user.get("must_change_password", False)),
    }


@router.post("/auth/change-password")
async def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    if not verify_password(old_password, current_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    await db.users.update_one(
        {"_id": current_user["_id"]},
        {
            "$set": {
                "password_hash": get_password_hash(new_password),
                "must_change_password": False,
                "password_changed_at": _now(),
            }
        },
    )
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="change_password",
        entity_type="user",
        entity_id=str(current_user["_id"]),
        details={},
    )
    return {"message": "Password updated"}


@router.post("/admin/create-user")
async def create_user(
    username: str = Form(...),
    role: str = Form(...),
    generate_temporary_password: str = Form("false"),
    password: str = Form(""),
    current_user: dict[str, Any] = Depends(require_roles("admin")),
) -> dict[str, Any]:
    username = (username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if role not in {"user", "reviewer", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    exists = await db.users.find_one({"username": username})
    if exists:
        raise HTTPException(status_code=409, detail="Username exists")

    gen_tmp = generate_temporary_password.lower() in ("true", "1", "on", "yes")
    if gen_tmp:
        plain_password = secrets.token_urlsafe(12)
    else:
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters or use temporary password")
        plain_password = password

    must_change = role in {"user", "reviewer"} or (role == "admin" and gen_tmp)
    uid = _oid()
    created_at = _now()
    new_doc = {
        "_id": uid,
        "username": username,
        "password_hash": get_password_hash(plain_password),
        "role": role,
        "is_active": True,
        "must_change_password": must_change,
        "created_at": created_at,
        "created_by_admin": current_user["username"],
    }
    try:
        await db.users.insert_one(new_doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Username already exists") from None
    try:
        _write_user_registry_file({k: v for k, v in new_doc.items() if k != "password_hash"})
    except OSError:
        pass
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="create_user",
        entity_type="user",
        entity_id=uid,
        details={
            "new_username": username,
            "new_role": role,
            "temporary_password": gen_tmp,
        },
    )
    out: dict[str, Any] = {
        "message": "User created",
        "user_id": uid,
        "must_change_password": must_change,
        "user": {
            "_id": uid,
            "username": username,
            "role": role,
            "is_active": True,
            "must_change_password": must_change,
            "created_at": created_at,
            "created_by_admin": current_user["username"],
        },
    }
    if gen_tmp:
        out["temporary_password"] = plain_password
        out["notice"] = "Share this password once; user must change it after first login."
    return out


@router.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: dict[str, Any] = Depends(require_roles("admin")),
) -> dict[str, str]:
    if ids_equal(user_id, current_user["_id"]):
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    target = await db.users.find_one(user_id_query(user_id))
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.get("role") == "admin":
        admin_count = await db.users.count_documents({"role": "admin", "is_active": True})
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last administrator")
    del_result = await db.users.delete_one({"_id": target["_id"]})
    if del_result.deleted_count != 1:
        raise HTTPException(
            status_code=500,
            detail="Database did not delete the user record (check MongoDB permissions or connectivity)",
        )
    _remove_user_registry_files(user_id, target["_id"])
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="delete_user",
        entity_type="user",
        entity_id=as_str_id(target["_id"]),
        details={"deleted_username": target.get("username")},
    )
    return {"message": "User deleted"}


@router.get("/admin/users")
async def list_users(_: dict[str, Any] = Depends(require_roles("admin"))) -> list[dict[str, Any]]:
    users = await db.users.find({}, {"password_hash": 0}).to_list(length=1000)
    return [serialize_document(u) for u in users]


@router.post("/admin/users/{user_id}/reset-password")
async def admin_reset_user_password(
    user_id: str,
    generate_temporary_password: str = Form("true"),
    new_password: str = Form(""),
    current_user: dict[str, Any] = Depends(require_roles("admin")),
) -> dict[str, Any]:
    if ids_equal(user_id, current_user["_id"]):
        raise HTTPException(status_code=400, detail="Use the change-password screen to update your own password")
    target = await db.users.find_one(user_id_query(user_id))
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    gen_tmp = generate_temporary_password.lower() in ("true", "1", "on", "yes")
    if gen_tmp:
        plain = secrets.token_urlsafe(12)
    else:
        if len(new_password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters or use temporary password")
        plain = new_password
    must_change = target.get("role") in {"user", "reviewer"} or (target.get("role") == "admin" and gen_tmp)
    await db.users.update_one(
        {"_id": target["_id"]},
        {
            "$set": {
                "password_hash": get_password_hash(plain),
                "must_change_password": must_change,
                "password_reset_by": current_user["username"],
                "password_reset_at": _now(),
            }
        },
    )
    try:
        u2 = await db.users.find_one({"_id": target["_id"]}, {"password_hash": 0})
        if u2:
            _write_user_registry_file(u2)
    except OSError:
        pass
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="admin_reset_password",
        entity_type="user",
        entity_id=as_str_id(target["_id"]),
        details={"target_username": target.get("username"), "temporary": gen_tmp},
    )
    out: dict[str, Any] = {"message": "Password updated", "must_change_password": must_change}
    if gen_tmp:
        out["temporary_password"] = plain
        out["notice"] = "Share once; user must change password after login."
    return out


@router.patch("/admin/users/{user_id}/role")
async def admin_set_user_role(
    user_id: str,
    role: str = Form(...),
    current_user: dict[str, Any] = Depends(require_roles("admin")),
) -> dict[str, Any]:
    if role not in {"user", "reviewer", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    if ids_equal(user_id, current_user["_id"]):
        raise HTTPException(status_code=400, detail="Ask another administrator to change your role")
    target = await db.users.find_one(user_id_query(user_id))
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    prev_role = target.get("role")
    if prev_role == "admin" and role != "admin":
        admin_count = await db.users.count_documents({"role": "admin", "is_active": True})
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot remove the last administrator")
    await db.users.update_one({"_id": target["_id"]}, {"$set": {"role": role}})
    try:
        u2 = await db.users.find_one({"_id": target["_id"]}, {"password_hash": 0})
        if u2:
            _write_user_registry_file(u2)
    except OSError:
        pass
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="admin_set_user_role",
        entity_type="user",
        entity_id=as_str_id(target["_id"]),
        details={
            "target_username": target.get("username"),
            "previous_role": prev_role,
            "new_role": role,
        },
    )
    return {"message": "Role updated", "role": role}


@router.patch("/admin/batches/{batch_id}")
async def admin_update_batch_metadata(
    batch_id: str,
    batch_name: str = Form(""),
    sample_id: str = Form(""),
    description: str = Form(""),
    sample_description: str = Form(""),
    current_user: dict[str, Any] = Depends(require_roles("admin")),
) -> dict[str, str]:
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    name_clean = batch_name.strip()
    if not name_clean:
        raise HTTPException(status_code=400, detail="Batch name is required")
    await db.batches.update_one(
        {"_id": batch_id},
        {
            "$set": {
                "batch_name": name_clean,
                "sample_id": sample_id.strip(),
                "description": description.strip(),
                "sample_description": sample_description.strip(),
                "updated_at": _now(),
            }
        },
    )
    try:
        b2 = await db.batches.find_one({"_id": batch_id})
        if b2:
            _write_batch_registry_file(b2)
    except OSError:
        pass
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="admin_update_batch",
        entity_type="batch",
        entity_id=batch_id,
        batch_id=batch_id,
        details={"batch_code": batch.get("batch_code", "")},
    )
    return {"message": "Batch updated"}


@router.delete("/admin/batches/{batch_id}")
async def admin_delete_batch(
    batch_id: str,
    current_user: dict[str, Any] = Depends(require_roles("admin")),
) -> dict[str, str]:
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    code = batch.get("batch_code") or ""
    await db.images.delete_many({"batch_id": batch_id})
    await db.notifications.delete_many({"batch_id": batch_id})
    await db.analysis_archives.delete_many({"batch_id": batch_id})
    await db.audit_logs.delete_many({"batch_id": batch_id})
    await db.batches.delete_one({"_id": batch_id})
    try:
        br = os.path.join(_registry_batches_dir(), f"{as_str_id(batch.get('_id'))}.json")
        if os.path.isfile(br):
            os.remove(br)
    except OSError:
        pass
    if code:
        up_dir = os.path.join(settings.uploads_dir, code)
        pr_dir = os.path.join(settings.processed_dir, code)
        if os.path.isdir(up_dir):
            shutil.rmtree(up_dir, ignore_errors=True)
        if os.path.isdir(pr_dir):
            shutil.rmtree(pr_dir, ignore_errors=True)
        seg = _safe_dir_segment(code)
        rep_batch = os.path.join(settings.reports_dir, "batches", seg)
        if os.path.isdir(rep_batch):
            shutil.rmtree(rep_batch, ignore_errors=True)
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="admin_delete_batch",
        entity_type="batch",
        entity_id=batch_id,
        batch_id=batch_id,
        details={"batch_code": code},
    )
    return {"message": "Batch and related records removed"}


@router.post("/batches")
async def create_batch(
    batch_code: str = Form(...),
    batch_name: str = Form(...),
    description: str = Form(""),
    sample_id: str = Form(""),
    sample_description: str = Form(""),
    current_user: dict[str, Any] = Depends(require_roles("user", "admin")),
) -> dict[str, Any]:
    exists = await db.batches.find_one({"batch_code": batch_code})
    if exists:
        raise HTTPException(status_code=409, detail="Batch code exists")
    bid = _oid()
    now = _now()
    doc = {
        "_id": bid,
        "batch_code": batch_code,
        "batch_name": batch_name,
        "description": description,
        "sample_id": sample_id.strip(),
        "sample_description": sample_description.strip(),
        "created_by": current_user["_id"],
        "created_by_username": current_user["username"],
        "status": "created",
        "created_at": now,
        "updated_at": now,
        "submitted_at": None,
        "assigned_reviewer_id": None,
        "assigned_reviewer_username": None,
        "review_decision": None,
        "review_comments": None,
        "report_path": None,
        "rejection_report_path": None,
    }
    await db.batches.insert_one(doc)
    try:
        _write_batch_registry_file(doc)
    except OSError:
        pass
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="create_batch",
        entity_type="batch",
        entity_id=bid,
        batch_id=bid,
        details={"batch_code": batch_code, "batch_name": batch_name},
    )
    return {"batch_id": bid, "status": "created"}


@router.post("/batches/{batch_id}/upload")
async def upload_images(
    batch_id: str,
    files: list[UploadFile] = File(...),
    current_user: dict[str, Any] = Depends(require_roles("user", "admin")),
) -> dict[str, Any]:
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if current_user["role"] != "admin" and not ids_equal(batch.get("created_by"), current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not your batch")
    if msg := _batch_image_mutations_blocked(batch):
        raise HTTPException(status_code=400, detail=msg)

    base = os.path.join(settings.uploads_dir, batch["batch_code"])
    os.makedirs(base, exist_ok=True)
    start_idx = await _next_image_sequence(batch_id, batch["batch_code"])
    saved = 0
    for offset, f in enumerate(files):
        if not settings.upload_extension_allowed(f.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Disallowed file type: {f.filename!r}. Allowed: {settings.allowed_upload_extensions}",
            )
        idx = start_idx + offset
        ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
        proper_name = f"{batch['batch_code']}_{idx:04d}{ext}"
        path = os.path.join(base, proper_name)
        await _save_upload_with_limit(f, path, settings.max_upload_bytes)
        img_doc = {
            "_id": _oid(),
            "batch_id": batch_id,
            "filename": proper_name,
            "input_path": path,
            "analysis": None,
            "created_at": _now(),
        }
        await db.images.insert_one(img_doc)
        saved += 1
    await db.batches.update_one(
        {"_id": batch_id},
        {"$set": {"status": "uploaded", "updated_at": _now()}},
    )
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="upload_images",
        entity_type="batch",
        entity_id=batch_id,
        batch_id=batch_id,
        details={"count": saved},
    )
    return {"message": "Uploaded", "count": saved}


@router.post("/batches/{batch_id}/analyze")
async def analyze_batch(
    batch_id: str,
    current_user: dict[str, Any] = Depends(require_roles("user", "admin")),
) -> dict[str, Any]:
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if current_user["role"] != "admin" and not ids_equal(batch.get("created_by"), current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not your batch")
    if msg := _batch_image_mutations_blocked(batch):
        raise HTTPException(status_code=400, detail=msg)

    images = await db.images.find({"batch_id": batch_id}).to_list(length=10000)
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")

    out_dir = os.path.join(settings.processed_dir, batch["batch_code"])
    os.makedirs(out_dir, exist_ok=True)
    total_cells = 0
    analyzed = 0
    analyzed_outputs: list[dict[str, Any]] = []
    for image in images:
        stem = os.path.splitext(image["filename"])[0]
        mask_path = os.path.join(out_dir, f"{stem}_mask.png")
        heatmap_path = os.path.join(out_dir, f"{stem}_heatmap.png")
        res = analyze_image(image["input_path"], mask_path, heatmap_path)
        res["mask_url"] = _processed_path_to_url(mask_path)
        res["heatmap_url"] = _processed_path_to_url(heatmap_path)
        await db.images.update_one(
            {"_id": image["_id"]},
            {"$set": {"analysis": res, "analyzed_at": _now()}},
        )
        total_cells += int(res["cell_count"])
        analyzed += 1
        analyzed_outputs.append(
            {
                "image_id": as_str_id(image["_id"]),
                "filename": image["filename"],
                "cell_count": int(res["cell_count"]),
                "mask_url": res.get("mask_url"),
                "heatmap_url": res.get("heatmap_url"),
            }
        )

    await db.batches.update_one(
        {"_id": batch_id},
        {"$set": {"status": "analyzed", "updated_at": _now(), "total_cells": total_cells}},
    )
    analyzed_at = _now()
    archive_id = _oid()
    run_number = await db.analysis_archives.count_documents({"batch_id": batch_id}) + 1
    report_payload: dict[str, Any] = {
        "archive_id": archive_id,
        "run_number": run_number,
        "batch_id": batch_id,
        "batch_code": batch["batch_code"],
        "batch_name": batch.get("batch_name", ""),
        "analyzed_at": analyzed_at,
        "total_cells": total_cells,
        "image_count": analyzed,
        "outputs": analyzed_outputs,
        "created_by": as_str_id(batch.get("created_by")),
        "created_by_username": batch.get("created_by_username", ""),
        "analyzed_by": as_str_id(current_user["_id"]),
        "analyzed_by_username": current_user["username"],
    }
    report_meta = await save_analysis_run_report(batch["batch_code"], run_number, archive_id, report_payload)
    await db.analysis_archives.insert_one(
        {
            "_id": archive_id,
            "batch_id": batch_id,
            "batch_code": batch["batch_code"],
            "batch_name": batch["batch_name"],
            "analyzed_at": analyzed_at,
            "total_cells": total_cells,
            "image_count": analyzed,
            "outputs": analyzed_outputs,
            "created_by": batch["created_by"],
            "created_by_username": batch.get("created_by_username", ""),
            "analyzed_by": current_user["_id"],
            "analyzed_by_username": current_user["username"],
            "analysis_run_number": run_number,
            "analysis_report_path": report_meta["report_path"],
            "analysis_report_url": report_meta.get("report_url", ""),
        }
    )
    try:
        bdoc = await db.batches.find_one({"_id": batch_id})
        if bdoc:
            _write_batch_registry_file(bdoc)
    except OSError:
        pass

    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="analyze_batch",
        entity_type="batch",
        entity_id=batch_id,
        batch_id=batch_id,
        details={
            "images": analyzed,
            "total_cells": total_cells,
            "archive_id": archive_id,
            "analysis_run_number": run_number,
            "analysis_report_url": report_meta.get("report_url", ""),
        },
    )
    return {
        "message": "Analysis complete",
        "images": analyzed,
        "total_cells": total_cells,
        "outputs": analyzed_outputs,
        "analyzed_at": analyzed_at,
        "archive_id": archive_id,
        "analysis_run_number": run_number,
        "analysis_report_url": report_meta.get("report_url", ""),
    }


@router.post("/batches/{batch_id}/submit")
async def submit_for_review(
    batch_id: str,
    assigned_reviewer_id: str = Form(""),
    current_user: dict[str, Any] = Depends(require_roles("user", "admin")),
) -> dict[str, Any]:
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if current_user["role"] != "admin" and not ids_equal(batch.get("created_by"), current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not your batch")
    if batch.get("status") != "analyzed":
        raise HTTPException(status_code=400, detail="Analyze batch first")
    rid = (assigned_reviewer_id or "").strip()
    if current_user["role"] == "user" and not rid:
        raise HTTPException(status_code=400, detail="Select a reviewer to submit to")
    rev_doc = await db.users.find_one(user_id_query(rid))
    if rid:
        if not rev_doc or rev_doc.get("role") != "reviewer":
            raise HTTPException(status_code=400, detail="Invalid reviewer account")
        if not rev_doc.get("is_active", True):
            raise HTTPException(status_code=400, detail="Selected reviewer is inactive")
    now = _now()
    set_doc: dict[str, Any] = {
        "status": "submitted",
        "submitted_at": now,
        "updated_at": now,
    }
    if rid:
        set_doc["assigned_reviewer_id"] = rev_doc["_id"]
        set_doc["assigned_reviewer_username"] = rev_doc.get("username", "")
    else:
        set_doc["assigned_reviewer_id"] = None
        set_doc["assigned_reviewer_username"] = None
    await db.batches.update_one({"_id": batch_id}, {"$set": set_doc})
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action="submit_for_review",
        entity_type="batch",
        entity_id=batch_id,
        batch_id=batch_id,
        details={
            "assigned_reviewer_id": rid,
            "assigned_reviewer_username": set_doc.get("assigned_reviewer_username"),
        },
    )
    return {"message": "Submitted for review"}


@router.get("/batches")
async def list_batches(current_user: dict[str, Any] = Depends(get_current_user)) -> list[dict[str, Any]]:
    if current_user["role"] == "reviewer":
        raw = await db.batches.find({}).sort("created_at", -1).to_list(length=1000)
        raw = [b for b in raw if _reviewer_may_read_batch(b, current_user)]
        return [serialize_document(b) for b in raw]
    query: dict[str, Any] = {}
    if current_user["role"] == "user":
        query = match_created_by(current_user["_id"])
    raw = await db.batches.find(query).sort("created_at", -1).to_list(length=1000)
    return [serialize_document(b) for b in raw]


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str, current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if not _may_read_batch(batch, current_user):
        raise HTTPException(status_code=403, detail="Not allowed to view this batch")
    return serialize_document(batch)


@router.get("/users/reviewers")
async def list_active_reviewers(
    current_user: dict[str, Any] = Depends(require_roles("user", "admin")),
) -> list[dict[str, Any]]:
    rows = await db.users.find({"role": "reviewer", "is_active": True}).sort("username", 1).to_list(500)
    return [{"_id": as_str_id(r["_id"]), "username": r.get("username", "")} for r in rows]


@router.get("/batches/{batch_id}/images")
async def list_batch_images(
    batch_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[dict[str, Any]]:
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if current_user["role"] == "user" and not ids_equal(batch.get("created_by"), current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not your batch")
    if current_user["role"] == "reviewer" and not _may_read_batch(batch, current_user):
        raise HTTPException(status_code=403, detail="Not allowed to view this batch")
    images = await db.images.find({"batch_id": batch_id}).to_list(length=10000)
    for item in images:
        analysis = item.get("analysis") or {}
        if analysis:
            if not analysis.get("mask_url") and analysis.get("mask_path"):
                analysis["mask_url"] = _processed_path_to_url(analysis["mask_path"])
            if not analysis.get("heatmap_url") and analysis.get("heatmap_path"):
                analysis["heatmap_url"] = _processed_path_to_url(analysis["heatmap_path"])
            item["analysis"] = analysis
    return images


@router.get("/review/queue")
async def review_queue(current_user: dict[str, Any] = Depends(require_roles("reviewer", "admin"))) -> list[dict[str, Any]]:
    raw = await db.batches.find({"status": "submitted"}).sort("submitted_at", 1).to_list(length=1000)
    if current_user["role"] == "reviewer":
        raw = [b for b in raw if _batch_assigned_to_reviewer(b, current_user["_id"])]
    if not raw:
        return []
    raw_ids = [b["_id"] for b in raw]
    count_rows = await db.images.aggregate(
        [
            {"$match": {"batch_id": {"$in": raw_ids}}},
            {"$group": {"_id": "$batch_id", "n": {"$sum": 1}}},
        ]
    ).to_list(length=10000)
    count_map: dict[str, int] = {}
    for row in count_rows:
        count_map[as_str_id(row.get("_id"))] = int(row.get("n") or 0)
    out: list[dict[str, Any]] = []
    for b in raw:
        d = serialize_document(b)
        bid = as_str_id(d.get("_id"))
        d["image_count"] = int(count_map.get(bid, 0))
        d["submitter_username"] = (b.get("created_by_username") or "").strip() or "—"
        d["submitter_user_id"] = as_str_id(b.get("created_by")) if b.get("created_by") is not None else ""
        out.append(d)
    return out


@router.get("/review/stats")
async def review_stats_endpoint(
    current_user: dict[str, Any] = Depends(require_roles("reviewer", "admin")),
) -> dict[str, Any]:
    if current_user["role"] == "admin":
        return {"role": "admin"}
    uid = current_user["_id"]
    submitted = await db.batches.find({"status": "submitted"}).to_list(length=5000)
    pending = sum(1 for b in submitted if _batch_assigned_to_reviewer(b, uid))
    uname = current_user["username"]
    accepted = await db.batches.count_documents({"status": "passed", "reviewed_by": uname})
    rejected = await db.batches.count_documents({"status": "rejected", "reviewed_by": uname})
    reanalysis = await db.batches.count_documents({"status": "reanalysis_required", "reviewed_by": uname})
    return {
        "role": "reviewer",
        "pending": pending,
        "accepted": accepted,
        "rejected": rejected,
        "reanalysis": reanalysis,
    }


@router.post("/review/{batch_id}/decision")
async def review_decision(
    batch_id: str,
    decision: str = Form(...),
    comments: str = Form(""),
    signature_username: str = Form(...),
    signature_password: str = Form(...),
    signature_meaning: str = Form(...),
    signature_reason: str = Form(...),
    current_user: dict[str, Any] = Depends(require_roles("reviewer", "admin")),
) -> dict[str, Any]:
    if decision not in {"pass", "reanalyze", "reject"}:
        raise HTTPException(status_code=400, detail="Invalid decision")
    batch = await db.batches.find_one({"_id": batch_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if batch.get("status") != "submitted":
        raise HTTPException(status_code=400, detail="Batch is not in submitted state")
    if len(comments.strip()) < 10:
        raise HTTPException(status_code=400, detail="Decision comment must be at least 10 characters")
    if current_user["role"] == "reviewer" and not _batch_assigned_to_reviewer(batch, current_user["_id"]):
        raise HTTPException(status_code=403, detail="This batch is assigned to another reviewer")
    if signature_username != current_user["username"]:
        raise HTTPException(status_code=400, detail="Signature username must match logged in reviewer")
    if not verify_password(signature_password, current_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid electronic signature password")
    if len(signature_meaning.strip()) < 5 or len(signature_reason.strip()) < 5:
        raise HTTPException(status_code=400, detail="Signature meaning/reason must be descriptive")

    status_map = {"pass": "passed", "reanalyze": "reanalysis_required", "reject": "rejected"}
    new_status = status_map[decision]

    images = await db.images.find({"batch_id": batch_id}).to_list(length=10000)
    summary = {
        "reviewed_by": current_user["username"],
        "reviewed_at": _now(),
        "decision": decision,
        "comments": comments,
        "electronic_signature": {
            "signed_by": signature_username,
            "signed_at": _now(),
            "meaning": signature_meaning.strip(),
            "reason": signature_reason.strip(),
        },
        "batch": batch,
        "image_count": len(images),
        "total_cells": int(sum((img.get("analysis") or {}).get("cell_count", 0) for img in images)),
    }
    report_type = "pass_report" if decision == "pass" else "reanalysis_report" if decision == "reanalyze" else "rejection_report"
    report_meta = await save_batch_report(batch["batch_code"], report_type, summary)
    report_path = report_meta["report_path"]

    update_doc: dict[str, Any] = {
        "status": new_status,
        "review_decision": decision,
        "review_comments": comments,
        "reviewed_by": current_user["username"],
        "reviewed_at": _now(),
        "electronic_signature": summary["electronic_signature"],
        "report_checksum": report_meta["checksum"],
        "report_chain_hash": report_meta["chain_hash"],
        "updated_at": _now(),
    }
    if decision == "pass":
        update_doc["report_path"] = report_path
    elif decision == "reject":
        update_doc["rejection_report_path"] = report_path
    else:
        update_doc["report_path"] = report_path

    await db.batches.update_one({"_id": batch_id}, {"$set": update_doc})
    await db.notifications.insert_one(
        {
            "_id": _oid(),
            "user_id": batch["created_by"],
            "batch_id": batch_id,
            "message": f"Batch {batch['batch_code']} review decision: {decision}",
            "created_at": _now(),
            "is_read": False,
        }
    )
    await log_activity(
        user_id=current_user["_id"],
        username=current_user["username"],
        role=current_user["role"],
        action=f"review_{decision}",
        entity_type="batch",
        entity_id=batch_id,
        batch_id=batch_id,
        details={
            "comments": comments,
            "report_path": report_path,
            "signature_meaning": signature_meaning.strip(),
            "signature_reason": signature_reason.strip(),
            "report_checksum": report_meta["checksum"],
        },
    )
    return {
        "message": "Decision saved",
        "status": new_status,
        "report_path": report_path,
        "report_checksum": report_meta["checksum"],
        "report_chain_hash": report_meta["chain_hash"],
    }


@router.get("/notifications")
async def get_notifications(current_user: dict[str, Any] = Depends(get_current_user)) -> list[dict[str, Any]]:
    return await db.notifications.find({"user_id": current_user["_id"]}).sort("created_at", -1).to_list(length=1000)


async def _image_count_for_batches(batch_filter: dict[str, Any]) -> int:
    ids = await db.batches.distinct("_id", batch_filter)
    if not ids:
        return 0
    return await db.images.count_documents({"batch_id": {"$in": ids}})


@router.get("/dashboard/summary")
async def dashboard_summary(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if current_user["role"] == "admin":
        match: dict[str, Any] = {}
    elif current_user["role"] == "reviewer":
        match = {"status": {"$in": ["submitted", "passed", "reanalysis_required", "rejected"]}}
    else:
        match = match_created_by(current_user["_id"])

    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]
    status_rows = await db.batches.aggregate(pipeline).to_list(length=100)
    statuses = {row["_id"]: row["count"] for row in status_rows}
    total_batches = int(sum(statuses.values()))
    if current_user["role"] == "admin":
        total_images = await db.images.count_documents({})
    else:
        total_images = await _image_count_for_batches(match)

    out: dict[str, Any] = {
        "role": current_user["role"],
        "total_batches": total_batches,
        "total_images": total_images,
        "status_counts": statuses,
    }

    if current_user["role"] == "admin":
        role_rows = await db.users.aggregate([{"$group": {"_id": "$role", "count": {"$sum": 1}}}]).to_list(20)
        users_by_role = {str(r["_id"]): int(r["count"]) for r in role_rows}
        img_status_rows = await db.images.aggregate(
            [
                {"$lookup": {"from": "batches", "localField": "batch_id", "foreignField": "_id", "as": "b"}},
                {"$match": {"b": {"$ne": []}}},
                {"$unwind": "$b"},
                {"$group": {"_id": "$b.status", "count": {"$sum": 1}}},
            ]
        ).to_list(50)
        images_by_batch_status = {str(r["_id"]): int(r["count"]) for r in img_status_rows}
        recent = await db.batches.find({}).sort("updated_at", -1).limit(30).to_list(30)
        recent_batches = []
        for b in recent:
            recent_batches.append(
                {
                    "_id": as_str_id(b["_id"]),
                    "batch_code": b.get("batch_code", ""),
                    "batch_name": b.get("batch_name", ""),
                    "status": b.get("status", ""),
                    "total_cells": b.get("total_cells"),
                    "created_by_username": b.get("created_by_username", ""),
                    "updated_at": b.get("updated_at"),
                }
            )
        out["admin"] = {
            "users_by_role": users_by_role,
            "total_users": int(sum(users_by_role.values())),
            "images_by_batch_status": images_by_batch_status,
            "recent_batches": recent_batches,
        }

    return out


@router.get("/audit/{batch_id}")
async def batch_audit_logs(
    batch_id: str,
    _: dict[str, Any] = Depends(require_roles("reviewer", "admin")),
) -> list[dict[str, Any]]:
    rows = await db.audit_logs.find({"batch_id": batch_id}).sort("timestamp", 1).to_list(length=5000)
    return [serialize_document(r) for r in rows]


@router.get("/reports/verify")
async def verify_report(
    report_path: str,
    _: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await verify_report_checksum(report_path)


@router.get("/archive/analysis-history")
async def analysis_history(
    current_user: dict[str, Any] = Depends(get_current_user),
    limit: int = 100,
    batch_id: str | None = None,
) -> list[dict[str, Any]]:
    q: dict[str, Any] = {}
    if current_user["role"] == "user":
        q = match_created_by(current_user["_id"])
    if batch_id:
        bid_q = {"batch_id": batch_id}
        q = {"$and": [q, bid_q]} if q else bid_q
    cursor = db.analysis_archives.find(q).sort("analyzed_at", -1).limit(min(limit, 500))
    rows = await cursor.to_list(length=500)
    return [serialize_document(r) for r in rows]


@router.get("/archive/analysis-report/{archive_id}/pdf")
async def analysis_archive_pdf(
    archive_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    """PDF summary for an analysis run (operators see own batches; reviewers/admins see all)."""
    doc = await db.analysis_archives.find_one({"_id": archive_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Analysis archive not found")
    if current_user["role"] == "user":
        if not ids_equal(doc.get("created_by"), current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not your batch")
    pdf_bytes = build_analysis_archive_pdf(serialize_document(doc))
    bc = _safe_dir_segment(str(doc.get("batch_code") or "batch"))
    rn = doc.get("analysis_run_number")
    try:
        rn_part = f"_run{int(rn):04d}" if rn is not None else ""
    except (TypeError, ValueError):
        rn_part = ""
    short_id = (archive_id or "")[:8] if archive_id else "report"
    fname = f"analysis_{bc}{rn_part}_{short_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{fname}"',
            "Cache-Control": "private, no-store",
        },
    )
