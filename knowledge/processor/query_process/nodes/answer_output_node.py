"""Answer generation and history persistence node."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import SSEEvent, push_sse_event
from knowledge.utils.task_util import set_task_result
from knowledge.utils.query_cache import query_cache
from knowledge.utils.logger_util import logger

# 图片提取常量：url 字段后缀判定 + Markdown 图片语法正则
_IMAGE_URL_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
_MD_IMG_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        task_id = state.get("task_id")
        is_stream = state.get("is_stream")

        if state.get("answer"):
            self._push_existing_answer(state)
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(state, prompt)

        self._cache_answer(state)

        image_urls = self._extract_image_urls(state.get("answer", ""))
        doc_image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])
        image_urls = self._merge_image_urls(image_urls, doc_image_urls)
        if task_id:
            set_task_result(task_id, "image_urls", image_urls)

        if is_stream:
            push_sse_event(
                task_id,
                SSEEvent.FINAL,
                {"answer": state.get("answer", ""), "image_urls": image_urls},
            )

        self._write_history(state)
        return state

    def _cache_answer(self, state: QueryGraphState) -> None:
        """把普通问答结果写入内存缓存；澄清问句不缓存。"""
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        answer = state.get("answer", "")
        if not rewritten_query or not answer:
            return
        if "？" in answer or "不确定" in answer:
            return
        query_cache.set(rewritten_query, answer)
        original_query = state.get("original_query", "")
        if original_query:
            query_cache.set(original_query, answer)

    def _push_existing_answer(self, state):
        set_task_result(state["task_id"], "answer", state["answer"])

    def _build_prompt(self, state):
        char_budget = self.config.max_context_chars
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names") or []
        related_entities = state.get("related_entities") or []
        logger.info("related_entities={}, count={}", related_entities[:5] if len(related_entities) > 5 else related_entities, len(related_entities))

        context_str, char_budget = self._format_reranked_docs(state.get("reranked_docs") or [], char_budget)
        history_str, char_budget = self._format_chat_history(state.get("history") or [], char_budget)
        graph_str, _char_budget = self._format_kg_triples(state.get("kg_triples") or [], char_budget)

        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str or "暂无历史对话",
            item_names=", ".join(item_names),
            related_entities=", ".join(related_entities) or "无",
            graph_relation_description=graph_str or "无图谱关系",
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs, char_budget):
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                value = str(doc.get(field, "")).strip()
                if value:
                    meta_tags.append(template.format(value))
            score = doc.get("score")
            if score is not None:
                meta_tags.append(f"[score={float(score):.4f}]")
            doc_entry = " ".join(meta_tags) + "\n" + content
            if used_chars + len(doc_entry) > char_budget:
                break
            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history, char_budget):
        formatted_lines = []
        used_chars = 0
        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue
            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1
            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars
            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    @staticmethod
    def _format_kg_triples(kg_triples, char_budget):
        formatted_lines = []
        used_chars = 0
        for triple in kg_triples:
            text = str(triple).strip()
            if not text:
                continue
            if used_chars + len(text) > char_budget:
                break
            formatted_lines.append(text)
            used_chars += len(text) + 1
        return "\n".join(formatted_lines), char_budget - used_chars

    def _generate_answer(self, state, prompt):
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError("LLM 客户端初始化失败")

        task_id = state["task_id"]
        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
            set_task_result(task_id, "answer", state["answer"])
        else:
            state["answer"] = self._invoke_generate(prompt)
            set_task_result(task_id, "answer", state["answer"])

    def _invoke_generate(self, prompt):
        llm_client = get_llm_client()
        if llm_client is None:
            return "抱歉，生成回答时出现错误。"
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as exc:
            logger.error("生成回答出错: {}", exc)
            return "抱歉，生成回答时出现错误。"

    def _stream_generate(self, llm_client, prompt, task_id):
        accumulated_answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id, "delta", {"delta": delta_text})
        except Exception as exc:
            logger.error("流式生成出错: {}", exc)
        return accumulated_answer

    @staticmethod
    def _extract_image_urls(answer: str) -> List[str]:
        """从 LLM 答案文本中提取【图片】标记后的图片 URL（作为补充来源）。

        改用 re.findall 提取所有 http(s):// URL，并在相邻 URL 黏连处按 http 边界拆分，
        避免多个 URL 挤在同一行时被误判为一个无效的超长 URL。"""
        if not answer:
            return []
        marker = re.search(r'【\s*图片\s*】|\[\s*图片\s*\]', answer)
        if not marker:
            return []
        tail = answer[marker.end():]
        urls = []
        for match in re.findall(r'https?://[^\s，。]+', tail):
            for piece in re.split(r'(?=https?://)', match):
                piece = piece.strip()
                # 去掉黏连处或句子结尾混入的尾部标点
                piece = re.sub(r'[)\]}>>，,。;；】”、]+$', '', piece)
                if piece.startswith('http://') or piece.startswith('https://'):
                    urls.append(piece)
        return urls

    @staticmethod
    def _extract_images_from_docs(docs) -> List[str]:
        """从检索文档中直接提取图片 URL，避免依赖 LLM 输出图片标记。

        策略一：检查文档 url 字段，后缀为常见图片格式则收集；
        策略二：用正则扫描 content/text 正文中的 Markdown 图片语法 ![alt](url)。
        """
        if not docs:
            return []
        urls: List[str] = []
        seen = set()
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            url = (doc.get("url") or "").strip()
            if url and url.lower().endswith(_IMAGE_URL_SUFFIXES) and url not in seen:
                seen.add(url)
                urls.append(url)
            text = doc.get("content") or doc.get("text") or ""
            for img_url in _MD_IMG_PATTERN.findall(str(text)):
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    seen.add(img_url)
                    urls.append(img_url)
        return urls

    @staticmethod
    def _merge_image_urls(*url_lists) -> List[str]:
        """合并多个来源的图片 URL，按传入顺序去重保序。"""
        merged: List[str] = []
        seen = set()
        for urls in url_lists:
            for url in urls or []:
                url = str(url).strip()
                if url and url not in seen:
                    seen.add(url)
                    merged.append(url)
        return merged

    def _write_history(self, state):
        session_id = state.get("session_id", "")
        if not session_id:
            return
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_names = state.get("item_names") or []
        try:
            save_chat_message(session_id=session_id, role="user", text=state["original_query"], rewritten_query=rewritten_query, item_names=item_names)
            if state.get("answer"):
                save_chat_message(session_id=session_id, role="assistant", text=state["answer"], rewritten_query=rewritten_query, item_names=item_names)
        except Exception as exc:
            logger.warning("写入历史记录失败: {}", exc)
