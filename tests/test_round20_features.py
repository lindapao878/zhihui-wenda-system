"""轮次20 单元测试：查询缓存、导入去重预检、batch_import 并发参数。

全部使用 mock 数据，不依赖外部服务。
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import batch_import
from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils.query_cache import QueryCache


class TestQueryCache(unittest.TestCase):
    def test_set_get_hit(self):
        cache = QueryCache(ttl_seconds=300)
        cache.set("三体简介", "三体是刘慈欣创作的长篇科幻小说。")
        self.assertEqual(cache.get("三体简介"), "三体是刘慈欣创作的长篇科幻小说。")

    def test_ttl_expiry(self):
        cache = QueryCache(ttl_seconds=-1)
        cache.set("问题", "答案")
        self.assertIsNone(cache.get("问题"))

    def test_empty_query_ignored(self):
        cache = QueryCache(ttl_seconds=300)
        cache.set("", "答案")
        self.assertIsNone(cache.get(""))


class TestAnswerCache(unittest.TestCase):
    def setUp(self):
        self.node = AnswerOutputNode()

    @patch("knowledge.processor.query_process.nodes.answer_output_node.query_cache")
    def test_cache_answer_writes_normal_answer(self, mock_cache):
        state = {"rewritten_query": "三体简介", "answer": "三体是一部科幻小说。"}
        self.node._cache_answer(state)
        mock_cache.set.assert_called_once_with("三体简介", "三体是一部科幻小说。")

    @patch("knowledge.processor.query_process.nodes.answer_output_node.query_cache")
    def test_cache_answer_skips_clarification(self, mock_cache):
        state = {"rewritten_query": "三体", "answer": "我不确定您指的是哪款产品？"}
        self.node._cache_answer(state)
        mock_cache.set.assert_not_called()


class TestQueryCacheHit(unittest.TestCase):
    @patch("knowledge.processor.query_process.nodes.item_name_confirm_node.query_cache")
    def test_cache_hit_fills_answer(self, mock_cache):
        node = ItemNameConfirmNode()
        node._item_name_extractor = MagicMock()
        node._item_name_extractor.extract_item_name.return_value = {
            "item_names": [],
            "rewritten_query": "重写后的问题",
        }
        node._item_name_aligner = MagicMock()
        node._item_name_aligner.match_align_filter.return_value = ([], [])
        mock_cache.get.return_value = "缓存的回答内容"

        state = {
            "original_query": "原始问题",
            "session_id": "s1",
            "answer": "",
            "rewritten_query": "",
            "item_names": [],
        }
        out = node.process(state)
        self.assertEqual(out["answer"], "缓存的回答内容")
        mock_cache.get.assert_called_once_with("重写后的问题")


class TestImportDedupPrecheck(unittest.TestCase):
    @patch("knowledge.processor.import_process.config.get_config")
    @patch("knowledge.utils.milvus_util.get_milvus_client")
    def test_check_duplicate_true(self, mock_get_client, mock_get_config):

        mock_get_config.return_value.chunks_collection = "kb_chunks"
        fake_client = MagicMock()
        fake_client.has_collection.return_value = True
        fake_client.query.return_value = [{"file_title": "测试文档"}]
        mock_get_client.return_value = fake_client

        fake_file = MagicMock()
        fake_file.filename = "测试文档.md"
        service = ImportFileService()
        self.assertTrue(service.check_duplicate_file(fake_file))
        fake_client.query.assert_called_once()

    @patch("knowledge.processor.import_process.config.get_config")
    @patch("knowledge.utils.milvus_util.get_milvus_client")
    def test_check_duplicate_false(self, mock_get_client, mock_get_config):
        mock_get_config.return_value.chunks_collection = "kb_chunks"
        fake_client = MagicMock()
        fake_client.has_collection.return_value = True
        fake_client.query.return_value = []
        mock_get_client.return_value = fake_client

        fake_file = MagicMock()
        fake_file.filename = "新文档.md"
        service = ImportFileService()
        self.assertFalse(service.check_duplicate_file(fake_file))

    @patch("knowledge.processor.import_process.config.get_config")
    @patch("knowledge.utils.milvus_util.get_milvus_client")
    def test_check_duplicate_collection_missing(self, mock_get_client, mock_get_config):
        mock_get_config.return_value.chunks_collection = "kb_chunks"
        fake_client = MagicMock()
        fake_client.has_collection.return_value = False
        mock_get_client.return_value = fake_client

        fake_file = MagicMock()
        fake_file.filename = "新文档.md"
        service = ImportFileService()
        self.assertFalse(service.check_duplicate_file(fake_file))
        fake_client.query.assert_not_called()


class TestUpload409(unittest.TestCase):
    def test_upload_returns_409_for_duplicate(self):
        from fastapi.testclient import TestClient

        from knowledge.api.import_router import app, get_import_file_service

        fake_service = ImportFileService()
        fake_service.check_duplicate_file = MagicMock(return_value=True)
        app.dependency_overrides[get_import_file_service] = lambda: fake_service
        try:
            resp = TestClient(app).post(
                "/upload",
                files={"file": ("已导入.md", b"test", "text/markdown")},
            )
            self.assertEqual(resp.status_code, 409)
            self.assertIn("跳过重复导入", resp.json()["detail"])
        finally:
            app.dependency_overrides.clear()


class TestBatchImportWorkers(unittest.TestCase):
    @patch("sys.argv", ["batch_import.py", "--dry-run"])
    def test_workers_default_2(self):
        args = batch_import.parse_args()
        self.assertEqual(args.workers, 2)

    @patch("sys.argv", ["batch_import.py", "--workers", "4", "--dry-run"])
    def test_workers_parse(self):
        args = batch_import.parse_args()
        self.assertEqual(args.workers, 4)


if __name__ == "__main__":
    unittest.main()
