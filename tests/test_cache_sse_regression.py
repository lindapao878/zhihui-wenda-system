"""Cache-hit SSE regression test: answer_output_node must report done + final.

Verifies the fix for the old bug where cache-hit streaming left the progress
bar waiting because _cache_answer referenced undefined task_id/is_stream.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from knowledge.processor.query_process.main_graph import query_app
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameAligner, ItemNameExtractor
from knowledge.utils.query_cache import query_cache
from knowledge.utils.sse_util import create_sse_queue, get_sse_queue
from knowledge.utils.task_util import get_done_task_list, get_task_result


class TestCacheHitSseRegression(unittest.TestCase):
    def test_cache_hit_stream_reports_done_and_final(self):
        task_id = "cache-hit-sse-test"
        create_sse_queue(task_id)
        query_cache.clear()
        query_cache.set("重写后的测量电压问题", "这是缓存的测量电压答案")

        with patch.object(ItemNameExtractor, "extract_item_name", return_value={
            "item_names": [],
            "rewritten_query": "重写后的测量电压问题",
        }), patch.object(ItemNameAligner, "match_align_filter", return_value=([], [])), patch(
            "knowledge.processor.query_process.nodes.item_name_confirm_node.get_recent_messages",
            return_value=[],
        ), patch(
            "knowledge.processor.query_process.nodes.item_name_confirm_node.update_message_item_names",
        ):
            query_app.invoke({
                "original_query": "如何使用万用表测量电压？",
                "rewritten_query": "",
                "session_id": "",  # empty session avoids real Mongo history writes
                "task_id": task_id,
                "is_stream": True,
                "item_names": [],
                "history": [],
            })

        done_list = get_done_task_list(task_id)
        self.assertIn("answer_output_node", done_list)
        self.assertEqual(get_task_result(task_id, "answer"), "这是缓存的测量电压答案")

        events = []
        stream_queue = get_sse_queue(task_id)
        if stream_queue is not None:
            while not stream_queue.empty():
                events.append(stream_queue.get())
        self.assertTrue(any(event.get("event") == "final" for event in events))


if __name__ == "__main__":
    unittest.main()
