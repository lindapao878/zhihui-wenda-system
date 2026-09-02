"""Readiness probe: check downstream middleware connectivity."""
from __future__ import annotations

import os
from typing import Dict, List

from knowledge.utils.logger_util import logger


def check_milvus() -> bool:
    """Ping Milvus by listing collections."""
    try:
        from knowledge.utils.milvus_util import get_milvus_client

        client = get_milvus_client()
        if client is None:
            return False
        client.list_collections()
        return True
    except Exception as exc:
        logger.warning("就绪检查 Milvus 失败: {}", exc)
        return False


def check_mongodb() -> bool:
    """Ping MongoDB via server ping command."""
    try:
        from knowledge.utils.mongo_history_util import get_mongo_client

        client = get_mongo_client()
        if client is None:
            return False
        client.admin.command("ping")  # serverSelectionTimeoutMS is on MongoClient
        return True
    except Exception as exc:
        logger.warning("就绪检查 MongoDB 失败: {}", exc)
        return False


def check_minio() -> bool:
    """Check MinIO by verifying the knowledge bucket exists."""
    try:
        from knowledge.utils.minio_util import get_minio_client

        client = get_minio_client()
        if client is None:
            return False
        bucket_name = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")
        return client.bucket_exists(bucket_name)
    except Exception as exc:
        logger.warning("就绪检查 MinIO 失败: {}", exc)
        return False


def readiness_check() -> Dict[str, object]:
    """Run all middleware checks and return a structured result."""
    checks = {
        "milvus": check_milvus(),
        "mongodb": check_mongodb(),
        "minio": check_minio(),
    }
    failed: List[str] = [name for name, ok in checks.items() if not ok]
    return {
        "ready": len(failed) == 0,
        "checks": checks,
        "failed": failed,
    }
