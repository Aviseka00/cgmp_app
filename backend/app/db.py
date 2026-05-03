from motor.motor_asyncio import AsyncIOMotorClient

from .config import settings


client = AsyncIOMotorClient(settings.mongodb_uri)
db = client[settings.mongodb_db]


async def ping_mongo() -> bool:
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def ensure_indexes() -> None:
    await db.users.create_index("username", unique=True)
    await db.batches.create_index("batch_code", unique=True)
    await db.images.create_index([("batch_id", 1), ("filename", 1)], unique=True)
    await db.audit_logs.create_index([("batch_id", 1), ("timestamp", 1)])
    await db.notifications.create_index([("user_id", 1), ("created_at", -1)])
    await db.report_checksums.create_index("report_path", unique=True)
    await db.report_checksums.create_index([("batch_code", 1), ("created_at", -1)])
    await db.analysis_archives.create_index([("analyzed_at", -1)])
    await db.analysis_archives.create_index([("created_by", 1), ("analyzed_at", -1)])
