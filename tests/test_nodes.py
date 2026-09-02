"""Node unit tests: DocumentSplitNode, RrfNode, RerankNode.

All tests use constructed mock data, no external service dependencies.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from pathlib import Path

from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.query_process.nodes.rerank_node import RerankNode
from knowledge.processor.query_process.nodes.rrf_node import RrfNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.config import QueryConfig
from knowledge.processor.import_process.config import ImportConfig
from knowledge.processor.import_process.exceptions import PdfConversionError
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils.task_util import TASK_STATUS_FAILED, get_task_result, get_task_status


def _make_import_state(**overrides) -> ImportGraphState:
    defaults: ImportGraphState = {
        "task_id": "test-task",
        "file_dir": "",
        "import_file_path": "",
        "file_title": "test_doc",
        "md_content": "",
        "chunks": [],
        "is_md_read_enabled": False,
        "is_docx_read_enabled": False,
        "is_pdf_read_enabled": False,
        "item_name": "",
        "md_path": "",
        "pdf_path": "",
        "docx_path": "",
    }
    defaults.update(overrides)
    return defaults


def _make_query_state(**overrides) -> QueryGraphState:
    defaults: QueryGraphState = {
        "session_id": "test-session",
        "task_id": "test-task",
        "message_id": "",
        "original_query": "test question",
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "rrf_chunks": [],
        "web_search_docs": [],
        "reranked_docs": [],
        "prompt": "",
        "answer": "",
        "item_names": [],
        "related_entities": [],
        "rewritten_query": "",
        "history": [],
        "kg_triples": [],
        "is_stream": False,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# DocumentSplitNode
# ---------------------------------------------------------------------------

class TestDocumentSplitNode(unittest.TestCase):

    def setUp(self):
        self.node = DocumentSplitNode()

    def test_split_long_section(self):
        """Long markdown content > max_content_length (2000) should be split."""
        # Build a paragraph of ~2500 chars (well above default max=2000)
        long_para = "长段落" * 500  # 3 chars * 500 = 1500, need > 2000
        while len(long_para) < 2500:
            long_para += " 扩充内容填充文字。"
        md = f"# 长文档标题\n\n{long_para}\n\n## 第二部分\n\n正常段落内容。\n"
        state = _make_import_state(md_content=md, file_title="长文档")
        result = self.node.process(state)
        chunks = result.get("chunks", [])
        # With content > 2000 chars, splitting should produce > 1 chunk
        self.assertGreater(len(chunks), 1, f"Expected >1 chunks, got {len(chunks)}")
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("title", chunk)
            self.assertGreater(len(chunk["content"]), 0)

    def test_merge_short_sections(self):
        """Two short sections under same parent_title should be merged into one."""
        md = (
            "# 主标题\n\n"
            "一些简短的开头文字。\n\n"
            "## 小节 A\n\n"
            "简短内容 A，不足以独立成段。\n\n"
            "## 小节 B\n\n"
            "简短内容 B，同样很短。\n"
        )
        state = _make_import_state(md_content=md, file_title="短文档")
        result = self.node.process(state)
        chunks = result.get("chunks", [])
        # The two ## sections should be merged (both < 500 chars, same parent_title)
        self.assertGreater(len(chunks), 0)
        # Short sections merge: original 3 sections (1 heading + 2 subsections)
        # become fewer chunks after merging short ones together
        all_content = " ".join(c.get("content", "") for c in chunks)
        self.assertIn("简短内容 A", all_content)
        self.assertIn("简短内容 B", all_content)
        self.assertLess(len(chunks), 3,
                        f"Merged should reduce chunk count below 3, got {len(chunks)}")

    def test_empty_content(self):
        """Empty md_content should produce empty chunks list."""
        state = _make_import_state(md_content="", file_title="空文档")
        result = self.node.process(state)
        chunks = result.get("chunks", [])
        self.assertEqual(len(chunks), 0, f"Expected 0 chunks, got {len(chunks)}")


# ---------------------------------------------------------------------------
# RrfNode
# ---------------------------------------------------------------------------

class TestRrfNode(unittest.TestCase):

    def setUp(self):
        self.node = RrfNode()

    def _make_chunk(self, chunk_id: str, content: str = "", title: str = ""):
        return {"entity": {"chunk_id": chunk_id, "content": content, "title": title}}

    def test_fusion_dedup_and_sort(self):
        """Dual-path RRF: overlapping chunk_ids are deduplicated by higher rank."""
        vec_chunks = [
            self._make_chunk("c1", "first doc"),
            self._make_chunk("c2", "second doc"),
            self._make_chunk("c3", "third doc"),
        ]
        hyde_chunks = [
            self._make_chunk("c2", "second doc"),  # overlap with vec
            self._make_chunk("c4", "fourth doc"),
        ]
        state = _make_query_state(
            embedding_chunks=vec_chunks,
            hyde_embedding_chunks=hyde_chunks,
        )
        result = self.node.process(state)
        rrf = result.get("rrf_chunks", [])

        # 4 unique chunks, all should appear (default rrf_max_results=10 ≥ 4)
        self.assertEqual(len(rrf), 4, f"Expected 4 unique chunks, got {len(rrf)}")
        chunk_ids = [d["chunk_id"] for d in rrf]
        self.assertCountEqual(chunk_ids, ["c1", "c2", "c3", "c4"])

        # c1 should rank first (top-1 in vector, no hyde competition)
        # c2 appears in BOTH paths (rank 2 in vec, rank 1 in hyde) → highest RRF score
        self.assertEqual(rrf[0]["chunk_id"], "c2",
                         f"c2 should win (dual-path), got {rrf[0]['chunk_id']}")

    def test_empty_input_returns_empty(self):
        """Both input lists empty → rrf_chunks = []."""
        state = _make_query_state(
            embedding_chunks=[],
            hyde_embedding_chunks=[],
        )
        result = self.node.process(state)
        self.assertEqual(result.get("rrf_chunks", []), [])


# ---------------------------------------------------------------------------
# RerankNode
# ---------------------------------------------------------------------------

class TestRerankNode(unittest.TestCase):

    def setUp(self):
        self.node = RerankNode()

    def _make_rrf_doc(self, chunk_id: str, content: str = "", title: str = ""):
        return {"chunk_id": chunk_id, "content": content or f"doc {chunk_id}", "title": title}

    def test_cliff_cutoff(self):
        """When a large score gap exists, cliff cutoff truncates at the gap."""
        docs = [
            self._make_rrf_doc("c1", "top result"),
            self._make_rrf_doc("c2", "good result"),
            self._make_rrf_doc("c3", "decent result"),
            self._make_rrf_doc("c4", "weak result"),   # gap here
            self._make_rrf_doc("c5", "bad result"),
        ]
        fake_scores = [0.95, 0.88, 0.82, 0.30, 0.12]
        # gap between c3(0.82) and c4(0.30): abs_gap=0.52 > 0.5 → cutoff at pos 3

        def fake_rerank(_self, _query, merged):
            return [{**doc, "score": s} for doc, s in zip(merged, fake_scores)]

        with patch.object(RerankNode, "_rerank_merged_docs", fake_rerank):
            state = _make_query_state(
                original_query="test query",
                rrf_chunks=docs,
            )
            result = self.node.process(state)
            reranked = result.get("reranked_docs", [])
            self.assertEqual(len(reranked), 3,
                             f"Expected cutoff at 3, got {len(reranked)}")
            self.assertAlmostEqual(reranked[0]["score"], 0.95, places=2)

    def test_empty_input_returns_empty(self):
        """No rrf_chunks and no web_search_docs → empty reranked_docs."""
        state = _make_query_state(rrf_chunks=[], web_search_docs=[])
        result = self.node.process(state)
        self.assertEqual(result.get("reranked_docs", []), [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Min score filter (shared via BaseNode._apply_min_score_filter)
# ---------------------------------------------------------------------------

class TestMinScoreFilter(unittest.TestCase):

    def _make_hit(self, distance: float, chunk_id: str = "c1"):
        return {"id": chunk_id, "distance": distance, "entity": {"chunk_id": chunk_id}}

    def test_filters_below_threshold(self):
        """Threshold > 0 drops low-score hits."""
        config = QueryConfig(milvus_min_cosine_score=0.5)
        node = RrfNode()
        node.config = config  # inject config directly
        hits = [
            self._make_hit(0.95, "c1"),
            self._make_hit(0.82, "c2"),
            self._make_hit(0.48, "c3"),  # below 0.5
            self._make_hit(0.30, "c4"),  # below 0.5
            self._make_hit(0.60, "c5"),
        ]
        filtered = node._apply_min_score_filter(hits)
        self.assertEqual(len(filtered), 3)
        kept_ids = [h["id"] for h in filtered]
        self.assertIn("c1", kept_ids)
        self.assertIn("c2", kept_ids)
        self.assertIn("c5", kept_ids)
        self.assertNotIn("c3", kept_ids)
        self.assertNotIn("c4", kept_ids)

    def test_zero_threshold_passes_all(self):
        """Threshold 0.0 skips filtering entirely."""
        config = QueryConfig(milvus_min_cosine_score=0.0)
        node = RrfNode()
        node.config = config
        hits = [self._make_hit(0.95), self._make_hit(0.10)]
        filtered = node._apply_min_score_filter(hits)
        self.assertEqual(len(filtered), 2)

    def test_empty_hits(self):
        """Empty list returns empty list."""
        config = QueryConfig(milvus_min_cosine_score=0.5)
        node = RrfNode()
        node.config = config
        self.assertEqual(node._apply_min_score_filter([]), [])


# ---------------------------------------------------------------------------
# MinerU timeout and import error persistence
# ---------------------------------------------------------------------------

class TestPdfToMdTimeout(unittest.TestCase):

    class _FakeProc:
        pid = 999999
        stdout = None

        def poll(self):
            return None  # never exits

    def test_timeout_kills_tree_and_raises_pdf_conversion_error(self):
        config = ImportConfig(mineru_timeout_seconds=1)
        node = PdfToMdNode(config=config)
        import_file_path = Path("dummy.pdf")
        fake_proc = self._FakeProc()
        with (
            patch("knowledge.processor.import_process.nodes.pdf_to_md_node.subprocess.Popen", return_value=fake_proc),
            patch.object(PdfToMdNode, "_kill_process_tree") as mock_kill,
        ):
            with self.assertRaises(PdfConversionError) as ctx:
                node._execute_mineru(import_file_path, Path("out"))
        self.assertIn("超时", str(ctx.exception))
        mock_kill.assert_called_once()


class TestImportErrorPersistence(unittest.TestCase):

    def test_run_import_graph_saves_error_on_failure(self):
        service = ImportFileService()
        with patch(
            "knowledge.services.file_import_service.kb_import_graph_app.invoke",
            side_effect=RuntimeError("boom"),
        ):
            service.run_import_graph("task-err", "dir", "file.md")
        self.assertEqual(get_task_status("task-err"), TASK_STATUS_FAILED)
        self.assertEqual(get_task_result("task-err", "error"), "boom")

class TestDeleteDocument(unittest.TestCase):
    def test_delete_document_uses_delete_count(self):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.delete.return_value = {"delete_count": 3, "cost": 0}
        config = ImportConfig(
            chunks_collection="kb_chunks",
            item_name_collection="kb_item_names",
            entity_name_collection="kb_entity_names",
        )
        with patch("knowledge.services.file_import_service.get_milvus_client", return_value=mock_client), patch(
            "knowledge.services.file_import_service.get_config", return_value=config
        ):
            result = ImportFileService().delete_document('a"b')
        self.assertEqual(result["deleted"], {
            "kb_chunks": 3, "kb_item_names": 3, "kb_entity_names": 3,
        })
if __name__ == "__main__":
    unittest.main()
