"""轮次23 单元测试：图片提取从检索文档中直接正则提取。

图片提取实现说明：
1. 检查文档 url 字段（图片后缀）
2. 扫描 content/text 正文中的 Markdown 图片语法 ![alt](url)
与 LLM 答案中的【图片】标记提取合并去重，保证 final 事件 image_urls 非空。
全部使用构造的 mock 数据直调节点，不依赖外部服务。
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.utils.sse_util import SSEEvent, create_sse_queue, get_sse_queue
from knowledge.utils.task_util import get_done_task_list, get_task_result


class TestExtractImagesFromDocs(unittest.TestCase):
    def setUp(self):
        self.node = AnswerOutputNode()

    def test_extract_from_doc_content(self):
        docs = [{
            "content": "参考下图 ![面板图](http://minio.local/p1.jpg) 如下",
            "title": "操作手册",
            "chunk_id": "c1",
            "url": "",
            "source": "local",
        }]
        urls = self.node._extract_images_from_docs(docs)
        self.assertIn("http://minio.local/p1.jpg", urls)

    def test_extract_from_doc_url_field(self):
        docs = [
            {"content": "", "url": "http://search.local/photo.png"},
            {"content": "页面链接 http://search.local/page.html", "url": "http://search.local/page.html"},
        ]
        urls = self.node._extract_images_from_docs(docs)
        self.assertEqual(urls, ["http://search.local/photo.png"])

    def test_extract_dedup(self):
        docs = [
            {"content": "![a](http://minio.local/x.jpg) 与 ![b](http://minio.local/x.jpg)", "url": "http://minio.local/x.jpg"},
        ]
        urls = self.node._extract_images_from_docs(docs)
        self.assertEqual(urls, ["http://minio.local/x.jpg"])

    def test_merge_image_urls_order_and_dedup(self):
        merged = self.node._merge_image_urls(
            ["http://minio.local/a.jpg", "http://minio.local/b.jpg"],
            ["http://minio.local/b.jpg", "http://minio.local/c.jpg"],
        )
        self.assertEqual(merged, ["http://minio.local/a.jpg", "http://minio.local/b.jpg", "http://minio.local/c.jpg"])


class TestProcessImagesFromDocs(unittest.TestCase):
    @patch("knowledge.processor.query_process.nodes.answer_output_node.query_cache")
    def test_final_event_has_images_without_llm_marker(self, mock_cache):
        task_id = "img-doc-test"
        create_sse_queue(task_id)

        node = AnswerOutputNode()
        state = {
            "task_id": task_id,
            "session_id": "",
            "is_stream": True,
            "original_query": "如何查看面板图片？",
            "rewritten_query": "面板图片位置",
            "answer": "面板图片位于机器正面。",  # 无【图片】标记
            "item_names": [],
            "history": [],
            "reranked_docs": [{
                "content": "机器正面示意图 ![面板图](http://minio.local/panel.jpg)",
                "title": "操作手册",
                "chunk_id": "c1",
                "url": "",
                "source": "local",
            }],
            "related_entities": [],
        }
        out = node(state)

        self.assertIn("answer_output_node", get_done_task_list(task_id))
        image_urls = get_task_result(task_id, "image_urls")
        self.assertIn("http://minio.local/panel.jpg", image_urls)
        self.assertEqual(out["answer"], "面板图片位于机器正面。")

        events = []
        stream_queue = get_sse_queue(task_id)
        if stream_queue is not None:
            while not stream_queue.empty():
                events.append(stream_queue.get())
        final = next((e for e in events if e.get("event") == SSEEvent.FINAL), None)
        self.assertIsNotNone(final)
        self.assertIn("http://minio.local/panel.jpg", final["data"]["image_urls"])


if __name__ == "__main__":
    unittest.main()
