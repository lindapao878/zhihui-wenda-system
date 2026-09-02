"""Markdown document splitter node."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer
from knowledge.utils.logger_util import logger


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)
        sections = self._split_by_headings(md_content, file_title)
        final_chunks = self.split_and_merge(sections, max_content_length, min_content_length)
        chunks = self._assemble_chunk(final_chunks)
        state["chunks"] = chunks
        self._log_summary(md_content, chunks, max_content_length)
        self._backup_chunks(state, chunks)
        return state

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        config = get_config()
        md_content = state.get("md_content", "")
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        file_title = state.get("file_title", "")
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValueError("切片长度参数校验失败")
        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[dict]:
        in_fence = False
        body_lines: List[str] = []
        sections: List[dict] = []
        current_level = 0
        current_title = ""
        hierarchy = [""] * 7
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        content_lines = md_content.split("\n")

        def flush():
            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title,
                })
            body_lines.clear()

        for content_line in content_lines:
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence
            match = heading_re.match(content_line) if not in_fence else None
            if match:
                flush()
                level = len(match.group(1))
                current_level = level
                current_title = content_line
                hierarchy[level] = current_title
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
            else:
                body_lines.append(content_line)
        flush()
        return sections

    def split_and_merge(self, sections, max_content_length, min_content_length):
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))
        return self.merge_short_section(current_sections, min_content_length)

    def split_long_section(self, section, max_content_length):
        title = section.get("title", "")
        body = section.get("body", "")
        file_title = section.get("file_title", "")
        parent_title = section.get("parent_title", "")

        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)

        if len(title) > 50:
            title = title[:50]

        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)
        if total_length <= max_content_length:
            section["title"] = title
            section["body"] = body
            return [section]

        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=False,
        )
        texts = text_splitter.split_text(body)
        if len(texts) <= 1:
            return [section]

        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                "title": title + "-" + str(index + 1),
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": str(index + 1),
            })
        return sub_section

    def merge_short_section(self, current_sections, min_content_length):
        if not current_sections:
            return []

        current_section = current_sections[0]
        final_sections = []
        for next_section in current_sections[1:]:
            same_parent = current_section["parent_title"] == next_section["parent_title"]
            if same_parent and len(current_section.get("body", "")) < min_content_length:
                current_section["body"] = (
                    current_section.get("body", "").rstrip()
                    + "\n\n"
                    + next_section.get("body", "").lstrip()
                )
                current_section["title"] = current_section["parent_title"]
                current_section["part"] = 0
            else:
                final_sections.append(current_section)
                current_section = next_section
        final_sections.append(current_section)

        part_counter: Dict[str, int] = {}
        result = []
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section.get("parent_title")
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter[parent_title]
                final_section["part"] = new_part
                final_section["title"] = final_section["title"] + f"- {new_part}"
            result.append(final_section)
        return result

    def _assemble_chunk(self, final_chunks):
        chunks = []
        for chunk in final_chunks:
            title = chunk.get("title")
            file_title = chunk.get("file_title")
            parent_title = chunk.get("parent_title")
            body = chunk.get("body")
            content = f"{title}\n\n{body}"
            assembled = {
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title,
                "content": content,
            }
            if "part" in chunk:
                assembled["part"] = chunk.get("part")
            chunks.append(assembled)
        return chunks

    def _log_summary(self, raw_content, chunks, max_length):
        logger.info("原文档行数: {}", raw_content.count("\n") + 1)
        logger.info("最终切分章节数: {}", len(chunks))
        logger.info("最大切片长度: {}", max_length)
        for i, sec in enumerate(chunks[:5]):
            logger.info("  {}. {}...", i + 1, sec.get("title", "")[:30])

    def _backup_chunks(self, state, sections):
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("备份失败: {}", exc)
