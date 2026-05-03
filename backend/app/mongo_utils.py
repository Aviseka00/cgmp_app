"""Helpers for BSON/ObjectId vs string ids and JSON-safe API responses."""

from __future__ import annotations

from typing import Any

from bson import ObjectId


def as_str_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, ObjectId):
        return str(value)
    return str(value)


def serialize_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert ObjectId values to str for JSON responses."""
    out: dict[str, Any] = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, dict):
            out[k] = serialize_document(v)
        elif isinstance(v, list):
            out[k] = [
                serialize_document(i) if isinstance(i, dict) else (str(i) if isinstance(i, ObjectId) else i)
                for i in v
            ]
        else:
            out[k] = v
    return out


def user_id_query(user_id: str) -> dict[str, Any]:
    """
    Match a user document by _id whether stored as UUID string, legacy ObjectId, etc.
    """
    uid = (user_id or "").strip()
    if not uid:
        return {"_id": "__invalid__"}
    ors: list[dict[str, Any]] = [{"_id": uid}]
    if len(uid) == 24:
        try:
            oid = ObjectId(uid)
            ors.append({"_id": oid})
        except Exception:
            pass
    return {"$or": ors} if len(ors) > 1 else ors[0]


def ids_equal(a: Any, b: Any) -> bool:
    return as_str_id(a) == as_str_id(b)


def match_created_by(current_user_id: Any) -> dict[str, Any]:
    """Query fragment for documents where created_by matches the current user id."""
    s = as_str_id(current_user_id)
    parts: list[dict[str, Any]] = [{"created_by": current_user_id}, {"created_by": s}]
    if len(s) == 24:
        try:
            parts.append({"created_by": ObjectId(s)})
        except Exception:
            pass
    # Deduplicate identical conditions
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for p in parts:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return {"$or": unique} if len(unique) > 1 else unique[0]
