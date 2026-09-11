"""Test-package isolation.

unittest discovery imports this package before test modules. The patches below
keep unit tests away from local MongoDB, Milvus, MinIO, and local model weights.
They are intentionally started for the whole test run; individual tests can
still override any of them with their own ``unittest.mock.patch``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

_ACTIVE_PATCHES = []


def _start_patch(target: str, **kwargs):
    active_patch = patch(target, **kwargs)
    active_patch.start()
    _ACTIVE_PATCHES.append(active_patch)


# Task state is exercised with the in-memory fallback. This removes the real
# MongoDB ping that previously dominated the unit-test runtime.
_start_patch("knowledge.utils.task_store._get_collection", return_value=None)

# Dataset-version tests use the process-local fallback; Mongo mode is mocked per test.
_start_patch("knowledge.utils.dataset_version_util._get_metadata_collection", return_value=None)
_start_patch("knowledge.utils.query_cache.get_dataset_version", return_value=1)

# Chat history nodes should not depend on a running MongoDB instance.
_start_patch("knowledge.utils.mongo_history_util.get_recent_messages", return_value=[])
_start_patch(
    "knowledge.utils.mongo_history_util.save_chat_message",
    return_value="test-history-id",
)
_start_patch("knowledge.utils.mongo_history_util.update_message_item_names")

# /ready endpoint tests should verify routing and authentication, not local
# middleware availability.
_start_patch("knowledge.utils.health_util.check_milvus", return_value=True)
_start_patch("knowledge.utils.health_util.check_mongodb", return_value=True)
_start_patch("knowledge.utils.health_util.check_minio", return_value=True)

# Optional heavy clients. Tests that need their behavior mock the concrete
# service methods themselves; no test should load a real model.
_start_patch("knowledge.utils.bge_m3_embedding_util.get_beg_m3_embedding_model", return_value=None)
_start_patch("knowledge.utils.bge_rerank_util.get_reranker_model", return_value=None)
_start_patch("knowledge.utils.milvus_util.get_milvus_client", return_value=MagicMock())
_start_patch("knowledge.utils.minio_util.get_minio_client", return_value=MagicMock())
