"""Minimal observability tests: stage durations and model readiness."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from knowledge.processor.import_process.base import BaseNode as ImportBaseNode
from knowledge.processor.query_process.base import BaseNode as QueryBaseNode
from knowledge.services.query_service import QueryService
from knowledge.utils import task_store


class _QueryTimingNode(QueryBaseNode):
    name = "vector_search_node"

    def process(self, state):
        return {**state, "ok": True}


class _FailingQueryNode(QueryBaseNode):
    name = "answer_output_node"

    def process(self, state):
        raise RuntimeError("boom")


class _ImportTimingNode(ImportBaseNode):
    name = "pdf_to_md_node"

    def process(self, state):
        return {**state, "ok": True}


class TestTaskDurationStore(unittest.TestCase):
    def setUp(self):
        task_store._memory.clear()

    def test_record_and_get_duration_in_memory_mode(self):
        task_store.record_duration("obs-memory", "vector_search_node", 12.5)
        self.assertEqual(
            task_store.get_durations("obs-memory"),
            {"vector_search_node": 12.5},
        )

    def test_record_and_get_duration_in_mongo_mode(self):
        collection = MagicMock()
        collection.find_one.return_value = {"durations_ms": {"rerank_node": 7.25}}
        with patch(
            "knowledge.utils.task_store._get_collection",
            return_value=collection,
        ):
            task_store.record_duration("obs-mongo", "rerank_node", 7.25)
            durations = task_store.get_durations("obs-mongo")

        update_filter, update_doc = collection.update_one.call_args.args
        self.assertEqual(update_filter, {"_id": "obs-mongo"})
        self.assertIn("durations_ms.rerank_node", update_doc["$set"])
        self.assertEqual(durations, {"rerank_node": 7.25})

    def test_invalid_duration_is_ignored(self):
        task_store.record_duration("obs-invalid", "bad.stage", 1)
        task_store.record_duration("obs-invalid", "vector_search_node", float("nan"))
        self.assertEqual(task_store.get_durations("obs-invalid"), {})


class TestNodeDurations(unittest.TestCase):
    def setUp(self):
        task_store._memory.clear()

    def test_query_node_duration_is_recorded(self):
        result = _QueryTimingNode()({"task_id": "obs-node", "is_stream": False})
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(
            task_store.get_durations("obs-node").get("vector_search_node", -1), 0
        )

    def test_query_node_failure_still_records_duration(self):
        with self.assertRaises(RuntimeError):
            _FailingQueryNode()({"task_id": "obs-failed-node", "is_stream": False})
        self.assertIn(
            "answer_output_node",
            task_store.get_durations("obs-failed-node"),
        )

    def test_import_node_duration_is_recorded(self):
        result = _ImportTimingNode()({"task_id": "obs-import-node"})
        self.assertTrue(result["ok"])
        self.assertIn(
            "pdf_to_md_node",
            task_store.get_durations("obs-import-node"),
        )


class TestQueryServiceDuration(unittest.TestCase):
    def setUp(self):
        task_store._memory.clear()

    def test_total_query_duration_is_returned(self):
        service = QueryService()
        task_id = "obs-query"
        service.submit_query(task_id, is_stream=False)
        with patch("knowledge.services.query_service.query_app") as query_app:
            query_app.invoke.return_value = {}
            service.run_query_graph(task_id, "session", "question", False)

        info = service.get_task_info(task_id)
        self.assertGreaterEqual(info["durations_ms"].get("total", -1), 0)

    def test_total_query_duration_is_recorded_on_failure(self):
        service = QueryService()
        task_id = "obs-query-failed"
        service.submit_query(task_id, is_stream=False)
        with patch("knowledge.services.query_service.query_app") as query_app:
            query_app.invoke.side_effect = RuntimeError("graph failed")
            service.run_query_graph(task_id, "session", "question", False)

        info = service.get_task_info(task_id)
        self.assertGreaterEqual(info["durations_ms"].get("total", -1), 0)


class TestReadyModelStatus(unittest.TestCase):
    def test_model_statuses_do_not_load_models(self):
        from knowledge.utils.health_util import get_model_statuses

        with patch(
            "knowledge.utils.bge_m3_embedding_util.is_bge_m3_model_loaded",
            return_value=True,
        ), patch(
            "knowledge.utils.bge_rerank_util.is_reranker_model_loaded",
            return_value=False,
        ):
            self.assertEqual(
                get_model_statuses(),
                {"bge_m3_embedding": "loaded", "bge_reranker": "not_loaded"},
            )

    def test_ready_endpoint_reports_model_status_without_gating_ready(self):
        from knowledge.api.query_router import app

        client = TestClient(app)
        with patch(
            "knowledge.utils.health_util.check_milvus", return_value=False
        ), patch(
            "knowledge.utils.health_util.check_mongodb", return_value=True
        ), patch(
            "knowledge.utils.health_util.check_minio", return_value=True
        ), patch(
            "knowledge.utils.health_util.get_model_statuses",
            return_value={
                "bge_m3_embedding": "loaded",
                "bge_reranker": "not_loaded",
            },
        ):
            response = client.get("/ready")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["models"]["bge_m3_embedding"], "loaded")
        self.assertEqual(payload["models"]["bge_reranker"], "not_loaded")


if __name__ == "__main__":
    unittest.main()
