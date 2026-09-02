"""轮次25 单元测试：图片 URL 黏连拆分。

LLM 有时把两个图片 URL 放在同一行且没有换行（...jpghttp://...jpg），
后端 _extract_image_urls 改用 re.findall 提取所有 http(s):// URL，
并在相邻 URL 黏连处按 http 边界二次拆分；前端 extractUrlsLoose 同理。
这里用构造的 mock 答案文本直接调用节点方法，不依赖外部服务。
"""
from __future__ import annotations

import unittest

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode


class TestExtractStickyImageUrls(unittest.TestCase):
    def setUp(self):
        self.node = AnswerOutputNode()

    def test_two_sticky_urls_on_one_line(self):
        answer = "【图片】\nhttp://minio.local/a.jpghttp://minio.local/b.jpg"
        urls = self.node._extract_image_urls(answer)
        self.assertEqual(urls, ["http://minio.local/a.jpg", "http://minio.local/b.jpg"])

    def test_markdown_and_sticky_url(self):
        answer = "【图片】\n![](http://minio.local/a.jpg)http://minio.local/b.jpg"
        urls = self.node._extract_image_urls(answer)
        self.assertEqual(
            urls, ["http://minio.local/a.jpg", "http://minio.local/b.jpg"]
        )

    def test_sticky_urls_before_sentence_tail(self):
        answer = "参考【图片】http://minio.local/x.jpghttp://minio.local/y.jpg 结束"
        urls = self.node._extract_image_urls(answer)
        self.assertEqual(urls, ["http://minio.local/x.jpg", "http://minio.local/y.jpg"])

    def test_merge_with_doc_images_dedup(self):
        answer = "【图片】\nhttp://minio.local/a.jpghttp://minio.local/b.jpg"
        from_llm = self.node._extract_image_urls(answer)
        from_docs = ["http://minio.local/b.jpg", "http://minio.local/c.png"]
        merged = self.node._merge_image_urls(from_llm, from_docs)
        self.assertEqual(
            merged,
            [
                "http://minio.local/a.jpg",
                "http://minio.local/b.jpg",
                "http://minio.local/c.png",
            ],
        )


if __name__ == "__main__":
    unittest.main()
