"""MinIO client helper."""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from knowledge.utils.logger_util import logger

load_dotenv()

_client = None


def get_minio_client():
    global _client
    if _client is not None:
        return _client

    try:
        from minio import Minio

        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        secure = os.getenv("MINIO_SECURE", "False").lower() in {"1", "true", "yes", "on"}

        _client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

        bucket_name = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")
        try:
            if not _client.bucket_exists(bucket_name):
                _client.make_bucket(bucket_name)
                logger.info("创建 MinIO 存储桶: {}", bucket_name)
            _set_public_read_policy(_client, bucket_name)
        except Exception as exc:
            logger.warning("MinIO 存储桶检查失败: {}", exc)
    except Exception as exc:
        logger.error("创建 MinIO 客户端失败: {}", exc)
        return None

    return _client


def _set_public_read_policy(client, bucket_name):
    """让桶内对象可通过 URL 匿名只读访问。"""
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }
    client.set_bucket_policy(bucket_name, json.dumps(policy))
    logger.info("MinIO 存储桶已配置公开只读策略: {}", bucket_name)
