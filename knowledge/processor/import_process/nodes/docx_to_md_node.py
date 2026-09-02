"""DOCX to Markdown conversion node."""
from __future__ import annotations

from pathlib import Path

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import FileProcessingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class DocxToMdNode(BaseNode):
    name = "docx_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        docx_path = state.get("docx_path", "")
        file_dir = state.get("file_dir", "")

        if not docx_path:
            raise ValidationError("docx文件不存在", self.name)

        docx_path_obj = Path(docx_path)
        if not docx_path_obj.exists():
            raise FileProcessingError("docx文件路径无效", self.name)

        file_dir_obj = Path(file_dir) if file_dir else docx_path_obj.parent
        file_dir_obj.mkdir(parents=True, exist_ok=True)
        md_path = file_dir_obj / f"{docx_path_obj.stem}.md"

        md_content = self._docx_to_markdown(docx_path_obj)
        md_path.write_text(md_content, encoding="utf-8")

        state["md_path"] = str(md_path)
        state["md_content"] = md_content
        return state

    def _docx_to_markdown(self, path: Path) -> str:
        try:
            from docx import Document
        except Exception as exc:
            raise FileProcessingError(f"python-docx 不可用: {exc}", self.name)

        document = Document(str(path))
        lines = [f"# {path.stem}", ""]

        # python-docx 只暴露顶层段落和表格，按文档顺序逐项处理。
        from docx.document import Document as _Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        body = document.element.body
        for child in body.iterchildren():
            if child.tag == qn("w:p"):
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()
                if text:
                    lines.append(text)
                    lines.append("")
            elif child.tag == qn("w:tbl"):
                table = Table(child, document)
                self._append_table(lines, table)

        return "\n".join(lines).strip()

    def _append_table(self, lines, table):
        for row_index, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
            if row_index == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        lines.append("")
