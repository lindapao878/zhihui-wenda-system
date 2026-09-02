"""Round 34 enterprise hardening tests: API key auth, health probe, task store."""
from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


class TestApiKeyAuth(unittest.TestCase):
    """Verify X-API-Key authentication on the import router."""

    def setUp(self):
        from knowledge.api.import_router import app
        self.client = TestClient(app)

    @patch("knowledge.core.security.get_app_api_key", return_value="")
    def test_no_auth_when_key_empty(self, _):
        """Empty APP_API_KEY skips auth — gets 400 (bad file) not 401."""
        resp = self.client.post(
            "/upload",
            files={"file": ("bad.txt", b"x", "text/plain")},
        )
        self.assertEqual(resp.status_code, 400)

    @patch("knowledge.core.security.get_app_api_key", return_value="secret123")
    def test_401_without_header(self, _):
        resp = self.client.post(
            "/upload",
            files={"file": ("test.md", b"x", "text/markdown")},
        )
        self.assertEqual(resp.status_code, 401)

    @patch("knowledge.core.security.get_app_api_key", return_value="secret123")
    def test_401_with_wrong_key(self, _):
        resp = self.client.post(
            "/upload",
            files={"file": ("test.md", b"x", "text/markdown")},
            headers={"X-API-Key": "wrong"},
        )
        self.assertEqual(resp.status_code, 401)

    @patch("knowledge.core.security.get_app_api_key", return_value="secret123")
    def test_health_exempt_from_auth(self, _):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    @patch("knowledge.core.security.get_app_api_key", return_value="secret123")
    def test_ready_exempt_from_auth(self, _):
        resp = self.client.get("/ready")
        # 200 or 503 depending on middleware, but never 401
        self.assertNotEqual(resp.status_code, 401)


class TestCORSConfig(unittest.TestCase):
    def test_parse_origins(self):
        from knowledge.core.security import get_allowed_origins
        with patch.dict("os.environ", {"ALLOWED_ORIGINS": "http://a.com, http://b.com "}):
            origins = get_allowed_origins()
            self.assertEqual(origins, ["http://a.com", "http://b.com"])


class TestHealthProbe(unittest.TestCase):
    def test_all_pass(self):
        from knowledge.utils.health_util import readiness_check
        with patch("knowledge.utils.health_util.check_milvus", return_value=True), \
             patch("knowledge.utils.health_util.check_mongodb", return_value=True), \
             patch("knowledge.utils.health_util.check_minio", return_value=True):
            result = readiness_check()
            self.assertTrue(result["ready"])
            self.assertEqual(result["failed"], [])

    def test_one_fails(self):
        from knowledge.utils.health_util import readiness_check
        with patch("knowledge.utils.health_util.check_milvus", return_value=False), \
             patch("knowledge.utils.health_util.check_mongodb", return_value=True), \
             patch("knowledge.utils.health_util.check_minio", return_value=True):
            result = readiness_check()
            self.assertFalse(result["ready"])
            self.assertIn("milvus", result["failed"])


class TestTaskStoreMemoryFallback(unittest.TestCase):
    """Exercise task_store in-memory mode (MongoDB unreachable)."""

    @patch("knowledge.utils.task_store._get_collection", return_value=None)
    def test_add_running_and_done(self, _):
        from knowledge.utils import task_store
        task_store._memory.clear()
        task_store.add_running("t1", "node_a")
        self.assertIn("node_a", task_store.get_running_list("t1"))
        task_store.add_done("t1", "node_a")
        self.assertNotIn("node_a", task_store.get_running_list("t1"))
        self.assertIn("node_a", task_store.get_done_list("t1"))

    @patch("knowledge.utils.task_store._get_collection", return_value=None)
    def test_update_and_get_status(self, _):
        from knowledge.utils import task_store
        task_store._memory.clear()
        task_store.update_status("t2", "processing")
        self.assertEqual(task_store.get_status("t2"), "processing")

    @patch("knowledge.utils.task_store._get_collection", return_value=None)
    def test_set_and_get_result(self, _):
        from knowledge.utils import task_store
        task_store._memory.clear()
        task_store.set_result("t3", "answer", "hello")
        self.assertEqual(task_store.get_result("t3", "answer"), "hello")
        self.assertIsNone(task_store.get_result("t3", "missing"))

    @patch("knowledge.utils.task_store._get_collection", return_value=None)
    def test_clear_task(self, _):
        from knowledge.utils import task_store
        task_store._memory.clear()
        task_store.add_running("t4", "node_a")
        task_store.clear("t4")
        self.assertEqual(task_store.get_running_list("t4"), [])

    @patch("knowledge.utils.task_store._get_collection", return_value=None)
    def test_init_on_startup_memory_mode(self, _):
        """init_on_startup in memory mode should return 0, not raise."""
        from knowledge.utils.task_util import init_on_startup
        init_on_startup()


if __name__ == "__main__":
    unittest.main()
