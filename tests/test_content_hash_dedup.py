"""Content-hash dedup tests for kb_documents registry and import flow."""
from __future__ import annotations

import hashlib
import io
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from knowledge.api.import_router import app, get_import_file_service
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils import document_registry_util


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TestContentHashPrecheck(unittest.TestCase):
    def _file(self, filename: str, content: bytes):
        fake = MagicMock()
        fake.filename = filename
        fake.file = io.BytesIO(content)
        return fake

    @patch("knowledge.services.file_import_service.find_active_document", return_value={"_id": "h"})
    @patch("knowledge.services.file_import_service.is_registry_available", return_value=True)
    def test_same_content_different_filename_is_duplicate(self, mock_available, mock_find):
        service = ImportFileService()
        file = self._file("a.md", b"same content")
        self.assertTrue(service.check_duplicate_file(file))
        mock_find.assert_called_once_with(_sha256(b"same content"))

    @patch("knowledge.services.file_import_service.find_active_document", return_value=None)
    @patch("knowledge.services.file_import_service.is_registry_available", return_value=True)
    def test_same_filename_different_content_is_not_duplicate(self, mock_available, mock_find):
        service = ImportFileService()
        file = self._file("a.md", b"new content")
        self.assertFalse(service.check_duplicate_file(file))
        mock_find.assert_called_once_with(_sha256(b"new content"))

    @patch("knowledge.services.file_import_service.get_milvus_client")
    @patch("knowledge.services.file_import_service.is_registry_available", return_value=False)
    def test_registry_unavailable_falls_back_to_file_title(self, mock_available, mock_client):
        fake_client = MagicMock()
        fake_client.has_collection.return_value = True
        fake_client.query.return_value = [{"file_title": "a"}]
        mock_client.return_value = fake_client

        service = ImportFileService()
        file = self._file("a.md", b"old content")
        self.assertTrue(service.check_duplicate_file(file))


class TestContentHashStatusTransitions(unittest.TestCase):
    def test_successful_import_marks_active_and_supersedes(self):
        service = ImportFileService()
        with patch(
            "knowledge.services.file_import_service.kb_import_graph_app.invoke",
            return_value={"chunks": [{"chunk_id": 1}]},
        ), patch(
            "knowledge.services.file_import_service.increment_dataset_version",
            return_value=9,
        ), patch(
            "knowledge.services.file_import_service.mark_active",
        ) as mock_active, patch(
            "knowledge.services.file_import_service.supersede_title",
        ) as mock_supersede:
            service.run_import_graph("task", "dir", "file.md", "hash123")

        mock_active.assert_called_once_with("hash123", "file", "task")
        mock_supersede.assert_called_once_with("file", "hash123")

    def test_failed_import_marks_failed_without_version_increment(self):
        service = ImportFileService()
        with patch(
            "knowledge.services.file_import_service.kb_import_graph_app.invoke",
            side_effect=RuntimeError("boom"),
        ), patch(
            "knowledge.services.file_import_service.increment_dataset_version",
        ) as mock_increment, patch(
            "knowledge.services.file_import_service.mark_failed",
        ) as mock_failed:
            service.run_import_graph("task", "dir", "file.md", "hash123")

        mock_failed.assert_called_once_with("hash123", "file", "task")
        mock_increment.assert_not_called()

    def test_delete_removes_registry_and_increments_version(self):
        fake_client = MagicMock()
        fake_client.has_collection.return_value = True
        fake_client.delete.return_value = {"delete_count": 1}
        with patch(
            "knowledge.services.file_import_service.get_milvus_client", return_value=fake_client
        ), patch(
            "knowledge.services.file_import_service.delete_by_title", return_value=2
        ), patch(
            "knowledge.services.file_import_service.increment_dataset_version", return_value=5
        ) as mock_increment:
            result = ImportFileService().delete_document("guide.md")

        self.assertEqual(result["registry_deleted"], 2)
        mock_increment.assert_called_once()


class TestUploadRouterRegistersHash(unittest.TestCase):
    def test_upload_registers_importing_and_passes_hash(self):
        fake_service = ImportFileService()
        fake_service.check_duplicate_file = MagicMock(return_value=False)
        fake_service.process_upload_file = MagicMock(
            return_value=("task-id", "task-dir", "task-dir/guide.md", "hash123")
        )
        fake_service.run_import_graph = MagicMock()
        app.dependency_overrides[get_import_file_service] = lambda: fake_service
        try:
            with patch("knowledge.api.import_router.register_importing") as mock_register:
                response = TestClient(app).post(
                    "/upload",
                    files={"file": ("guide.md", b"content", "text/markdown")},
                )
            self.assertEqual(response.status_code, 200)
            mock_register.assert_called_once_with("hash123", "guide", "task-id")
            fake_service.run_import_graph.assert_called_once_with(
                "task-id", "task-dir", "task-dir/guide.md", "hash123"
            )
        finally:
            app.dependency_overrides.clear()


class TestDocumentRegistryUtil(unittest.TestCase):
    def test_register_importing_uses_content_hash_as_id(self):
        collection = MagicMock()
        with patch(
            "knowledge.utils.document_registry_util._get_collection", return_value=collection
        ):
            document_registry_util.register_importing("hash123", "guide.md", "task-id")

        document = collection.replace_one.call_args.args[1]
        self.assertEqual(document["_id"], "hash123")
        self.assertEqual(document["content_hash"], "hash123")
        self.assertEqual(document["status"], "importing")

    def test_delete_by_title_uses_file_title_filter(self):
        collection = MagicMock()
        with patch(
            "knowledge.utils.document_registry_util._get_collection", return_value=collection
        ):
            document_registry_util.delete_by_title("guide.md")

        collection.delete_many.assert_called_once_with({"file_title": "guide.md"})


if __name__ == "__main__":
    unittest.main()
