"""Product-name confirmation and query rewriting node."""
from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model
from knowledge.utils.llm_client_util import get_llm_client, invoke_llm_with_json_fallback
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, get_milvus_client
from knowledge.utils.mongo_history_util import get_recent_messages, update_message_item_names
from knowledge.utils.query_cache import query_cache
from knowledge.utils.logger_util import logger



class ItemNameAligner:
    def __init__(self):
        self.config = get_config()
    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        search_result = self._match_vector(item_names)
        confirmed, options = self._item_name_score_align(search_result)
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)
        return confirmed, options

    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        milvus_client = get_milvus_client()
        embedding_model = get_beg_m3_embedding_model()
        if milvus_client is None or embedding_model is None:
            return []

        hybrid_embedding_result = generate_hybrid_embeddings(embedding_model, item_names)
        search_results = []

        for index, extract_item_name in enumerate(item_names):
            requests = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result["dense"][index],
                sparse_vector=hybrid_embedding_result["sparse"][index],
            )
            result = execute_hybrid_search_query(
                milvus_client,
                collection_name=self.config.item_name_collection,
                search_requests=requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"],
                limit=5,
            )
            matches = []
            for hit in (result[0] if result else []):
                entity = hit.get("entity", {})
                matches.append({"item_name": entity.get("item_name"), "score": hit.get("distance")})
            search_results.append({"extracted_name": extract_item_name, "matches": matches})
        return search_results

    def _item_name_score_align(self, search_results):
        confirmed = []
        options = []

        for result in search_results:
            extracted_name = result.get("extracted_name")
            matches = sorted(result.get("matches"), key=lambda x: x["score"], reverse=True)
            high = [m for m in matches if m.get("score") >= self.config.item_name_high_confidence]

            if high:
                exact = next((h for h in high if str(h["item_name"]) == extracted_name), None)
                if exact:
                    picked = exact["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    for h in high[:3]:
                        picked = h.get("item_name")
                        if picked not in options and picked not in confirmed:
                            options.append(picked)
            else:
                for m in matches[:3]:
                    if m["score"] >= self.config.item_name_mid_confidence and m["item_name"] not in options and m["item_name"] not in confirmed:
                        options.append(m["item_name"])

        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed, search_results):
        item_name_score = {}
        for search_result in search_results:
            for match in search_result.get("matches"):
                name = match.get("item_name")
                if name in confirmed:
                    item_name_score[name] = max(item_name_score.get(name, 0), match.get("score", 0))
        if not item_name_score:
            return confirmed
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items() if max_score - score <= 0.15]


class ItemNameExtractor:
    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        result = {"item_names": [], "rewritten_query": original_query}

        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text if history_text else "暂无上下文", query=original_query
        )
        try:
            llm_response = invoke_llm_with_json_fallback([
                SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
                HumanMessage(content=human_prompt),
            ])
            if llm_response is None:
                return result
            parsed = self._clean_parse(llm_response.content.strip())
            result["rewritten_query"] = parsed.get("rewritten_query") or original_query
            result["item_names"] = parsed.get("item_names")
        except Exception as exc:
            logger.error("清洗以及解析LLM的输出失败: {}", exc)
        return result

    def _clean_parse(self, llm_response):
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(content)
            item_names = parsed.get("item_names")
            if not isinstance(item_names, list):
                item_names = []
            item_names = [name for name in item_names if str(name).strip()]
            rewritten_query = parsed.get("rewritten_query")
            if not isinstance(rewritten_query, str):
                rewritten_query = ""
            return {"item_names": item_names, "rewritten_query": rewritten_query.strip()}
        except JSONDecodeError as exc:
            raise ValueError(f"JSON反序列LLM的输出失败：{exc}")


class ItemNameConfirmNode(BaseNode):
    name = "item_name_confirm_node"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = state.get("original_query", "")
        session_id = state.get("session_id", "")

        chat_history = get_recent_messages(session_id, limit=10)
        history_text = ""
        for msg in chat_history:
            history_text += f"{msg.get('role')}: {msg.get('text', '')}\n"

        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)
        item_names = clean_llm_result.get("item_names")
        rewritten_query = clean_llm_result.get("rewritten_query")

        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        self._decide(state, item_names, confirmed, options, rewritten_query)
        logger.info("商品名确认: extracted={} confirmed={} options={}", item_names, confirmed, options)

        self._apply_query_cache(state)

        if confirmed:
            ids_to_update = [str(msg["_id"]) for msg in chat_history if not msg.get("item_names")]
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as exc:
                    logger.warning("回填历史 item_names 失败: {}", exc)

        state["history"] = chat_history
        return state

    def _decide(self, state, item_names, confirmed, options, rewritten_query):
        if confirmed:
            state["rewritten_query"] = rewritten_query
            state["item_names"] = confirmed
        elif options:
            state["answer"] = f"我不确定您指的是哪款产品。您是在询问以下产品吗：{'、'.join(options)}？"
        else:
            state["rewritten_query"] = rewritten_query or state.get("original_query", "")
            state["item_names"] = []

    def _apply_query_cache(self, state: QueryGraphState) -> None:
        if state.get("answer"):
            return
        rewritten_query = state.get("rewritten_query", "")
        cached_answer = query_cache.get(rewritten_query)
        if cached_answer:
            state["answer"] = cached_answer
            logger.info(
                "QUERY_CACHE_HIT rewritten_query={} answer_len={}", rewritten_query, len(cached_answer)
            )
            return
        original_query = state.get("original_query", "")
        if original_query and original_query != rewritten_query:
            cached_answer = query_cache.get(original_query)
            if cached_answer:
                state["answer"] = cached_answer
                logger.info(
                    "QUERY_CACHE_HIT original_query={} answer_len={}", original_query, len(cached_answer)
                )
