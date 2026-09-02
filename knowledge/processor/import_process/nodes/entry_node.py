"""Entry node that routes PDF and Markdown files."""
from __future__ import annotations

from pathlib import Path

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_dir = state.get("file_dir")
        import_file_path = state.get("import_file_path")

        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或者文件不存在", self.name)

        path = Path(import_file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".docx":
            state["is_docx_read_enabled"] = True
            state["docx_path"] = import_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError(f"文件类型{suffix}不支持", self.name)

        state["file_title"] = path.stem
        return state


