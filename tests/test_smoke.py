import unittest

from fastapi.testclient import TestClient

from knowledge.api.import_router import app
from knowledge.prompts.loader import load_prompt
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT, KG_EXTRACT_SYSTEM_PROMPT
from knowledge.prompts.upload.import_prompt import ITEM_NAME_SYSTEM_PROMPT


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_import_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_upload_rejects_unsupported_type(self):
        resp = self.client.post(
            "/upload",
            files={"file": ("bad.txt", b"not a doc", "text/plain")},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("不支持", resp.json()["detail"])

    def test_prompt_files_are_loaded(self):
        self.assertTrue(ITEM_NAME_SYSTEM_PROMPT)
        self.assertTrue(ANSWER_PROMPT)
        self.assertTrue(KG_EXTRACT_SYSTEM_PROMPT)
        self.assertTrue(load_prompt("query", "hyde.prompt"))


if __name__ == "__main__":
    unittest.main()
