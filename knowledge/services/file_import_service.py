"""File import service."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

from knowledge.core.paths import get_temp_data_dir
from knowledge.processor.import_process.main_graph import kb_import_graph_app
from knowledge.processor.import_process.state import create_default_state
from knowledge.processor.import_process.config import get_config
from knowledge.utils.task_util import TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, TASK_STATUS_PROCESSING, update_task_status
from knowledge.utils.task_util import set_task_result
from knowledge.utils.milvus_string_util import escape_milvus_string
from knowledge.utils.dataset_version_util import get_dataset_version, increment_dataset_version
from knowledge.utils.document_registry_util import (
    delete_by_title,
    find_active_document,
    is_registry_available,
    mark_active,
    mark_failed,
    supersede_title,
)
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.logger_util import logger



class ImportFileService:
    @staticmethod
    def _compute_content_hash(file: UploadFile) -> str:
        stream = getattr(file, "file", None)
        if stream is None:
            return ""
        digest = hashlib.sha256()
        original_position = stream.tell()
        stream.seek(0)
        try:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        except Exception as exc:
            logger.warning("计算上传文件 SHA-256 失败: {}", exc)
            return ""
        finally:
            try:
                stream.seek(original_position)
            except Exception:
                pass
        return digest.hexdigest()

    def check_duplicate_file(self, file: UploadFile) -> bool:
        """上传前预检：优先按内容 SHA-256，注册表不可用时回退 file_title。"""
        content_hash = self._compute_content_hash(file)
        if not content_hash:
            return False
        if is_registry_available():
            return bool(find_active_document(content_hash))

        try:
            from knowledge.processor.import_process.config import get_config
            from knowledge.utils.milvus_util import get_milvus_client

            collection = get_config().chunks_collection
            client = get_milvus_client()
            if client is None or not client.has_collection(collection_name=collection):
                return False
            file_title = Path(file.filename or "").stem
            if not file_title:
                return False
            rows = client.query(
                collection_name=collection,
                filter=f'file_title == "{escape_milvus_string(file_title)}"',
                output_fields=["file_title"],
                limit=1,
            )
            return bool(rows)
        except Exception as exc:
            logger.warning("去重预检失败: {}", exc)
            return False

    def process_upload_file(self, file: UploadFile) -> Tuple[str, str, str, str]:
        task_id = str(uuid.uuid4())
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        file_dir = os.path.join(get_temp_data_dir(), task_id)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        original_name = Path(file.filename or "upload.pdf").name
        import_file_path = os.path.join(file_dir, original_name)

        content_hash = self._compute_content_hash(file)
        stream = getattr(file, "file", None)
        if stream is None:
            raise RuntimeError("上传文件流不可用")
        stream.seek(0)
        with open(import_file_path, "wb") as buffer:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        logger.info("上传文件 {} -> {}", original_name, import_file_path)
        return task_id, file_dir, import_file_path, content_hash

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str, content_hash: str = "") -> None:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        file_title = Path(import_file_path).stem
        state = create_default_state(
            task_id=task_id,
            file_dir=file_dir,
            import_file_path=import_file_path,
            content_hash=content_hash,
        )
        try:
            final_state = kb_import_graph_app.invoke(state)
            logger.info("导入任务完成: {}, 切片数={}", task_id, len(final_state.get("chunks", [])))
            if content_hash:
                mark_active(content_hash, file_title, task_id)
                supersede_title(file_title, content_hash)
            dataset_version = increment_dataset_version()
            logger.info("导入成功，dataset_version={}，查询缓存已按版本自然失效: {}", dataset_version, task_id)
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as exc:
            logger.exception("导入任务失败: {}", task_id)
            if content_hash:
                mark_failed(content_hash, file_title, task_id)
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(exc))

    def delete_document(self, file_title: str) -> dict:
        """按 file_title 删除三张 Milvus 集合和 kb_documents 注册记录。"""
        config = get_config()
        client = get_milvus_client()
        if client is None:
            raise RuntimeError("Milvus 客户端不可用")

        safe_title = escape_milvus_string(file_title)
        filter_expr = f'file_title == "{safe_title}"'
        collections = [
            ("kb_chunks", config.chunks_collection),
            ("kb_item_names", config.item_name_collection),
            ("kb_entity_names", config.entity_name_collection),
        ]
        deleted = {}
        for label, collection_name in collections:
            try:
                if not client.has_collection(collection_name=collection_name):
                    deleted[label] = 0
                    continue
                result = client.delete(collection_name=collection_name, filter=filter_expr)
                if hasattr(result, "delete_count"):
                    count = result.delete_count
                elif isinstance(result, dict):
                    count = result.get("delete_count", 0)
                else:
                    count = len(result)
                deleted[label] = int(count)
            except Exception as exc:
                logger.warning("删除集合 {} 失败: {}", collection_name, exc)
                deleted[label] = 0
        registry_deleted = delete_by_title(file_title)
        total_deleted = sum(deleted.values())
        dataset_version = increment_dataset_version() if (total_deleted or registry_deleted) else get_dataset_version()
        return {
            "file_title": file_title,
            "deleted": deleted,
            "registry_deleted": registry_deleted,
            "dataset_version": dataset_version,
        }
