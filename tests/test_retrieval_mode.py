from __future__ import annotations

import unittest

from knowledge.processor.query_process.config import normalize_retrieval_mode
from knowledge.processor.query_process.main_graph import create_query_graph, route_after_vector


class TestRetrievalModeValidation(unittest.TestCase):
    def test_supported_modes_normalize(self):
        self.assertEqual(normalize_retrieval_mode("basic"), "basic")
        self.assertEqual(normalize_retrieval_mode("HYDE"), "hyde")
        self.assertEqual(normalize_retrieval_mode("full"), "full")
        self.assertEqual(normalize_retrieval_mode("auto"), "auto")

    def test_unsupported_mode_raises(self):
        with self.assertRaises(ValueError):
            normalize_retrieval_mode("unknown")


class TestRetrievalModeGraph(unittest.TestCase):
    def _node_names(self, retrieval_mode: str):
        graph = create_query_graph(retrieval_mode=retrieval_mode)
        return set(graph.get_graph().nodes)

    def test_basic_graph_omits_heavy_nodes(self):
        names = self._node_names("basic")
        self.assertIn("search_embedding", names)
        self.assertNotIn("search_embedding_hyde", names)
        self.assertNotIn("web_search_mcp", names)
        self.assertNotIn("knowledge_graph_query", names)

    def test_hyde_graph_skips_web_and_entity_recall(self):
        names = self._node_names("hyde")
        self.assertIn("search_embedding", names)
        self.assertIn("search_embedding_hyde", names)
        self.assertNotIn("web_search_mcp", names)
        self.assertNotIn("knowledge_graph_query", names)

    def test_full_graph_keeps_all_nodes(self):
        names = self._node_names("full")
        self.assertIn("search_embedding", names)
        self.assertIn("search_embedding_hyde", names)
        self.assertIn("web_search_mcp", names)
        self.assertIn("knowledge_graph_query", names)

    def test_auto_graph_keeps_extension_path(self):
        names = self._node_names("auto")
        self.assertIn("multi_search_extended", names)
        self.assertIn("search_embedding_hyde", names)
        self.assertIn("web_search_mcp", names)
        self.assertIn("knowledge_graph_query", names)


class TestAutoRoute(unittest.TestCase):
    def test_empty_vector_result_chooses_extension(self):
        self.assertEqual(route_after_vector({"embedding_chunks": []}), "extended")
        self.assertEqual(route_after_vector({}), "extended")

    def test_nonempty_vector_result_chooses_direct(self):
        self.assertEqual(
            route_after_vector({"embedding_chunks": [{"chunk_id": 1}]}),
            "direct",
        )


if __name__ == "__main__":
    unittest.main()
