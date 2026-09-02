"""Markdown image processing node."""
from __future__ import annotations

import base64
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError, ImageProcessingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_util import get_minio_client
from knowledge.utils.logger_util import logger


class MarkDownImageNode(BaseNode):
    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)

        if not image_dir.exists():
            logger.warning("文件{}暂无图片要处理", md_path_obj.name)
            state["md_content"] = md_content
            return state

        target_images_context = self._scan_images_and_context(image_dir, md_content, config)
        images_summaries = self._extract_img_summary(md_path_obj.stem, target_images_context, config)
        new_md_content = self._upload_img_and_update_new_md(
            md_path_obj.stem, md_content, images_summaries, target_images_context, config
        )
        self._backup_new_md_file(md_path_obj, new_md_content)
        state["md_content"] = new_md_content
        return state

    def _get_img_md_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        md_path = state.get("md_path", "")
        if not md_path:
            raise ValidationError("md文件不存在", self.name)

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径无效", self.name)

        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()

        return md_content, md_path_obj, md_path_obj.parent / "images"

    def _scan_images_and_context(self, image_dir, md_content, config):
        target_images_context = []
        for img_name in os.listdir(image_dir):
            file_ext = os.path.splitext(img_name)[1]
            if file_ext not in config.image_extensions:
                continue
            img_path = str(image_dir / img_name)
            img_context = self._find_img_context_with_limit(
                md_content, img_name, config.img_content_length
            )
            if not img_context:
                continue
            target_images_context.append((img_name, img_path, img_context[0]))
        return target_images_context

    def _find_img_context_with_limit(self, md_content, img_name, max_chars=200):
        re_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        md_lines = md_content.split("\n")
        imgs_context = []

        for line_idx, line in enumerate(md_lines):
            if not re_pattern.search(line):
                continue

            head_title = ""
            head_index = -1
            for i in range(line_idx - 1, -1, -1):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    head_title = md_lines[i]
                    head_index = i
                    break

            pre_content = md_lines[head_index + 1 : line_idx]
            img_pre_context = self._extract_img_context_with_limit(pre_content, max_chars, direction="front")

            section_index = len(md_lines)
            for i in range(line_idx + 1, section_index):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break

            post_content = md_lines[line_idx + 1 : section_index]
            img_post_context = self._extract_img_context_with_limit(post_content, max_chars, direction="end")
            imgs_context.append((head_title, img_pre_context, img_post_context))

        return imgs_context

    def _extract_img_context_with_limit(self, extract_content, max_chars, direction):
        current_paragraph = []
        final_paragraph = []
        for line in extract_content:
            clen_strip = line.strip()
            if not clen_strip:
                if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))
                    current_paragraph = []
            else:
                if re.match(r"^!\[.*?\]\(.*?\)$", clen_strip):
                    if current_paragraph:
                        final_paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)

        if current_paragraph:
            final_paragraph.append("\n".join(current_paragraph))

        if direction == "front":
            final_paragraph.reverse()

        total = 0
        selected = []
        for para in final_paragraph:
            para_len = len(para)
            if total + para_len > max_chars and selected:
                break
            selected.append(para)
            total += para_len

        if direction == "front":
            selected.reverse()
        return "\n\n".join(selected)

    def _extract_img_summary(self, document_title, target_images_context, config):
        summaries = {}
        request_timestamps: Deque[float] = deque()

        try:
            client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_api_base)
        except Exception:
            logger.error("VLM客户端创建失败")
            return summaries

        for img_name, img_path, images_context in target_images_context:
            self._enforce_rate_limit(request_timestamps, config.requests_per_minute, 60)
            summaries[img_name] = self._get_img_summary(
                config, client, document_title, img_path, images_context
            )
        return summaries

    def _enforce_rate_limit(self, request_timestamps, max_requests, window_seconds=60):
        current_time = time.time()
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                logger.info("达到速率限制，暂停 {:.2f} 秒...", sleep_duration)
                time.sleep(sleep_duration)
            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()
        request_timestamps.append(current_time)

    def _get_img_summary(self, config, client, document_title, img_path, images_context):
        section_title, pre_context, post_context = images_context
        context_parts = [p for p in [section_title, pre_context, post_context] if p]
        final_context = "\n".join(context_parts) if context_parts else "暂无可用上下文"

        try:
            with open(img_path, "rb") as f:
                local_img_content = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return "暂无图片"

        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
背景信息：
    1. 所属文档标题："{document_title}"
    2. 图片上下文：{final_context}
请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
生成一个精准的中文标题（不要包含"图片"二字）。"""},
                            {"type": "image_url", "image_url": {"url": f"data:{self._image_media_type(img_path)};base64,{local_img_content}"}},
                        ],
                    }
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "图片描述"

    @staticmethod
    def _image_media_type(img_path: str) -> str:
        ext = Path(img_path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

    def _upload_img_and_update_new_md(self, document_name, md_content, images_summaries, target_images_context, config):
        remote_urls = {}
        minio_client = get_minio_client()
        if minio_client is None:
            logger.warning("无法将本地的图片上传到minio")

        for img_name, img_path, _ in target_images_context:
            object_name = f"{document_name}/{img_name}"
            if minio_client is None:
                remote_urls[img_name] = f"http://minio_mock/{document_name}/{img_name}"
                continue
            try:
                minio_client.fput_object(config.minio_bucket, object_name, img_path)
                remote_url = config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name
                remote_urls[img_name] = remote_url
            except Exception:
                logger.warning("{}上传到minio失败", img_name)
                remote_urls[img_name] = f"http://minio_mock/{document_name}/{img_name}"

        new_md_content = md_content
        for img_name, images_summary in images_summaries.items():
            remote_url = remote_urls.get(img_name)
            if not remote_url:
                continue
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)", re.IGNORECASE
            )
            new_md_content = replace_pattern.sub(
                f"![{images_summary}]({remote_url})", new_md_content
            )
        return new_md_content

    def _backup_new_md_file(self, md_path_obj, new_md_content):
        new_file_path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            return str(new_file_path)
        except IOError as exc:
            raise ImageProcessingError(f"文件写入失败: {exc}", node_name=self.name)
