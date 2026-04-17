import os
from typing import Optional, Dict, Any
from datetime import datetime

_MONGO_URI = os.getenv("MONGODB_URI")
_MONGO_DB = os.getenv("MONGODB_DB", "vehicle_detection")
_MONGO_COLLECTION = os.getenv("MONGODB_COLLECTION", "detections")

_client = None


def get_client():
    """Returns a cached MongoDB client if MONGODB_URI is set."""
    global _client
    if not _MONGO_URI:
        return None
    if _client is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except Exception:
            return None
        _client = AsyncIOMotorClient(_MONGO_URI)
    return _client


async def save_detection(payload: Dict[str, Any]) -> None:
    """Persist detection results to MongoDB if configured."""
    client = get_client()
    if client is None:
        return

    db = client[_MONGO_DB]
    collection = db[_MONGO_COLLECTION]
    payload["created_at"] = datetime.utcnow()
    await collection.insert_one(payload)
