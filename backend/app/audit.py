import hashlib
import json
import logging
from typing import Any

from bson import ObjectId

from .db import db
from .deps import utcnow_iso

_log = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """BSON/ObjectId and nested structures safe for json.dumps."""
    if obj is None:
        return None
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(_json_safe(data), sort_keys=True, separators=(",", ":"), default=str)


async def log_activity(
    *,
    user_id: str | ObjectId,
    username: str,
    role: str,
    action: str,
    entity_type: str,
    entity_id: str | ObjectId,
    batch_id: str | ObjectId | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    try:
        prev = await db.audit_logs.find_one(
            {"batch_id": batch_id} if batch_id else {"entity_type": {"$exists": True}},
            sort=[("timestamp", -1)],
        )
        prev_hash = prev["hash"] if prev else ""
        payload = {
            "timestamp": utcnow_iso(),
            "user_id": str(user_id) if user_id is not None else "",
            "username": username,
            "role": role,
            "action": action,
            "entity_type": entity_type,
            "entity_id": str(entity_id) if entity_id is not None else "",
            "batch_id": str(batch_id) if batch_id is not None else None,
            "details": details or {},
            "prev_hash": prev_hash,
        }
        curr_hash = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
        payload["hash"] = curr_hash
        await db.audit_logs.insert_one(payload)
    except Exception as exc:
        # Never fail primary operations (user/batch writes) because audit storage broke.
        _log.exception("audit_logs failed (non-fatal): %s", exc)
