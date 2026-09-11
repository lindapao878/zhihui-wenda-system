"""Dataset-version cache invalidation and structured cache tests."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils import dataset_version_util
from knowledge.utils.query_cache import QueryCache
from knowledge.utils.task_util import get_task_result


class TestDatasetVersion(unittest.TestCase):
    def test_cache_key_changes_with_dataset_version(self):
        cache = QueryCache(ttl_seconds=300)
        with patch("knowledge.utils.query_cache.get_dataset_version", return_value=1):
            cache.set("三体简介", {"answer": "旧答案"})
            self.assertIsNotNone(cache.get("三体简介"))

        with patch("knowledge.utils.query_cache.get_dataset_version", return_value=2):
            self.assertIsNone(cache.get("三体简介"))

        with patch("knowledge.utils.query_cache.get_dataset_version", return_value=1):
            self.assertEqual(cache.get("三体简介")["answer"], "旧答案")

    def test_mongo_dataset_version_read_and_increment(self):
        collection = MagicMock()
        collection.find_one.return_value = {"version": 3}
        collection.find_one_and_update.return_value = {"version": 4}

        with patch(
            "knowledge.utils.dataset_version_util._get_metadata_collection",
            return_value=collection,
        ):
            self.assertEqual(dataset_version_util.get_dataset_version(), 3)
            self.assertEqual(dataset_version_util.increment_dataset_version(), 4)

        collection.find_one.assert_called_once_with(
            {"_id": dataset_version_util.VERSION_DOCUMENT_ID}
        )
        self.assertEqual(
            collection.find_one_and_update.call_args.kwargs["upsert"], True
        )


class TestStructuredCache(unittest.TestCase):
    def test_answer_node_builds_structured_cache_payload(self):
        node = AnswerOutputNode()
        task_id = "structured-cache-task"
        state = {
            "task_id": task_id,
            "session_id": "",
            "original_query": "如何配置？",
            "rewritten_query": "如何配置系统？",
            "answer": "系统配置包括 Milvus、MongoDB 和 MinIO。",
            "item_names": ["配置模块"],
            "related_entities": ["Milvus"],
            "reranked_docs": [
                {
                    "chunk_id": 1,
                    "file_title": "配置指南.md",
                    "url": "http://minio.local/config.jpg",
                    "content": "![配置图](http://minio.local/config.jpg)",
                    "score": 0.92,
                }
            ],
            "is_stream": False,
        }

        with patch(
            "knowledge.processor.query_process.nodes.answer_output_node.query_cache"
        ) as mock_cache:
            node.process(state)

        self.assertEqual(
            get_task_result(task_id, "source_refs"),
            [{
                "chunk_id": 1,
                "file_title": "配置指南.md",
                "url": "http://minio.local/config.jpg",
                "score": 0.92,
            }],
        )
        self.assertEqual(
            get_task_result(task_id, "image_urls"),
            ["http://minio.local/config.jpg"],
        )

        first_call = mock_cache.set.call_args_list[0]
        self.assertEqual(first_call.args[0], "如何配置系统？")
        self.assertEqual(first_call.args[1]["image_urls"], ["http://minio.local/config.jpg"])
        self.assertEqual(first_call.args[1]["source_refs"][0]["chunk_id"], 1)
        self.assertEqual(first_call.args[1]["item_names"], ["配置模块"])
        self.assertEqual(first_call.args[1]["related_entities"], ["Milvus"])

    def test_cache_hit_restores_all_structured_fields(self):
        node = ItemNameConfirmNode()
        node._item_name_extractor = MagicMock()
        node._item_name_extractor.extract_item_name.return_value = {
            "item_names": [],
            "rewritten_query": "重写后的问题",
        }
        node._item_name_aligner = MagicMock()
        node._item_name_aligner.match_align_filter.return_value = ([], [])

        with patch(
            "knowledge.processor.query_process.nodes.item_name_confirm_node.query_cache"
        ) as mock_cache:
            mock_cache.get.return_value = {
                "answer": "缓存答案",
                "image_urls": ["http://example.com/a.jpg"],
                "source_refs": [{"chunk_id": 9}],
                "item_names": ["商品A"],
                "related_entities": ["实体B"],
            }
            result = node.process({
                "original_query": "原始问题",
                "session_id": "",
                "rewritten_query": "",
                "item_names": [],
            })

        self.assertEqual(result["answer"], "缓存答案")
        self.assertEqual(result["image_urls"], ["http://example.com/a.jpg"])
        self.assertEqual(result["source_refs"], [{"chunk_id": 9}])
        self.assertEqual(result["item_names"], ["商品A"])
        self.assertEqual(result["related_entities"], ["实体B"])


class TestImportVersionIncrement(unittest.TestCase):
    def test_successful_import_increments_dataset_version(self):
        service = ImportFileService()
        with patch(
            "knowledge.services.file_import_service.kb_import_graph_app.invoke",
            return_value={"chunks": []},
        ), patch(
            "knowledge.services.file_import_service.increment_dataset_version",
            return_value=7,
        ) as mock_increment:
            service.run_import_graph("import-version-task", "dir", "file.md")

        mock_increment.assert_called_once()

    def test_failed_import_does_not_increment_dataset_version(self):
        service = ImportFileService()
        with patch(
            "knowledge.services.file_import_service.kb_import_graph_app.invoke",
            side_effect=RuntimeError("boom"),
        ), patch(
            "knowledge.services.file_import_service.increment_dataset_version",
            return_value=7,
        ) as mock_increment:
            service.run_import_graph("import-failed-task", "dir", "file.md")

        mock_increment.assert_not_called()

    def test_delete_document_increments_version_when_rows_deleted(self):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.delete.return_value = {"delete_count": 1}
        with patch(
            "knowledge.services.file_import_service.get_milvus_client",
            return_value=mock_client,
        ), patch(
            "knowledge.services.file_import_service.increment_dataset_version",
            return_value=8,
        ) as mock_increment:
            result = ImportFileService().delete_document("guide.md")

        mock_increment.assert_called_once()
        self.assertEqual(result["dataset_version"], 8)


if __name__ == "__main__":
    unittest.main()
