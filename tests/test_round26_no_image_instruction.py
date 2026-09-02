"""轮次26 单元测试：answer.prompt 不再要求 LLM 输出【图片】区块。

图片完全由 _extract_images_from_docs 从 reranked_docs 的 Markdown 图片语法提取：
1. prompt 中已删除图片区块指令，仅保留"基于参考内容回答、不编造"
2. 查询返回的 answer 为纯文字（不含 http URL），image_urls 来自文档图片
全部使用构造的 mock 数据直调节点，不依赖外部服务。
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.utils.sse_util import SSEEvent, create_sse_queue, get_sse_queue
from knowledge.utils.task_util import get_task_result

PROMPT_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "prompts" / "query" / "answer.prompt"


class TestPromptHasNoImageInstruction(unittest.TestCase):
    def test_prompt_no_image_block(self):
        text = PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("不要编造不存在的事实", text)
        self.assertNotIn("图片区块", text)
        self.assertNotIn("【图片】", text)
        self.assertNotIn("每行一个URL", text)
        self.assertNotIn("<图片URL1>", text)
        self.assertNotIn("<图片URL2>", text)


class TestAnswerPlainTextImagesFromDocs(unittest.TestCase):
    @patch("knowledge.processor.query_process.nodes.answer_output_node.query_cache")
    def test_plain_answer_and_doc_images(self, mock_cache):
        task_id = "r26-no-img"
        create_sse_queue(task_id)

        node = AnswerOutputNode()
        state = {
            "task_id": task_id,
            "session_id": "",
            "is_stream": True,
            "original_query": "面板图片在哪里？",
            "rewritten_query": "面板图片位置",
            "answer": "面板图片位于机器正面。",  # 纯文字，无【图片】区块、无 URL
            "item_names": [],
            "history": [],
            "related_entities": [],
            "reranked_docs": [{
                "content": "机器正面示意图 ![面板图](http://minio.local/panel.jpg)",
                "title": "操作手册",
                "chunk_id": "c1",
                "url": "",
                "source": "local",
            }],
        }

        out = node.process(state)

        answer = out["answer"]
        self.assertNotIn("http", answer)
        self.assertNotIn("【图片】", answer)

        image_urls = get_task_result(task_id, "image_urls")
        self.assertEqual(image_urls, ["http://minio.local/panel.jpg"])

        events = []
        stream_queue = get_sse_queue(task_id)
        if stream_queue is not None:
            while not stream_queue.empty():
                events.append(stream_queue.get())
        final = next((e for e in events if e.get("event") == SSEEvent.FINAL), None)
        self.assertIsNotNone(final)
        self.assertEqual(final["data"]["image_urls"], ["http://minio.local/panel.jpg"])


if __name__ == "__main__":
    unittest.main()
