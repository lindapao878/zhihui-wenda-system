"""File import service."""
from __future__ import annotations

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
from knowledge.utils.query_cache import query_cache
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.logger_util import logger



class ImportFileService:
    def check_duplicate_file(self, file: UploadFile) -> bool:
        """上传前预检：Milvus kb_chunks 是否已有同 file_title。"""
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

    def process_upload_file(self, file: UploadFile) -> Tuple[str, str, str]:
        task_id = str(uuid.uuid4())
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        file_dir = os.path.join(get_temp_data_dir(), task_id)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        original_name = Path(file.filename or "upload.pdf").name
        import_file_path = os.path.join(file_dir, original_name)

        with open(import_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        logger.info("上传文件 {} -> {}", original_name, import_file_path)
        return task_id, file_dir, import_file_path

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str) -> None:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        state = create_default_state(
            task_id=task_id,
            file_dir=file_dir,
            import_file_path=import_file_path,
        )
        try:
            final_state = kb_import_graph_app.invoke(state)
            logger.info("导入任务完成: {}, 切片数={}", task_id, len(final_state.get("chunks", [])))
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as exc:
            query_cache.clear()
            logger.info("导入完成，已清空查询缓存: {}", task_id)
            logger.exception("导入任务失败: {}", task_id)
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(exc))

    def delete_document(self, file_title: str) -> dict:
        """按 file_title 删除三张 Milvus 集合中的文档记录。"""
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
        return {"file_title": file_title, "deleted": deleted}
