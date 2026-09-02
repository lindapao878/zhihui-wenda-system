"""Knowledge-graph relation extraction node."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import KG_EXTRACT_SYSTEM_PROMPT, KG_EXTRACT_USER_PROMPT_TEMPLATE
from knowledge.utils.llm_client_util import get_llm_client, invoke_llm_with_json_fallback
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, get_milvus_client
from knowledge.utils.logger_util import logger



class KnowledgeGraphQueryNode(BaseNode):
    name = "knowledge_graph_query_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        question = state.get("rewritten_query") or state.get("original_query", "")
        docs = state.get("reranked_docs") or state.get("rrf_chunks") or []
        context = self._build_context(docs)
        state["kg_triples"] = self._extract_triples(question, context)
        state["related_entities"] = self._retrieve_entities(question, state.get("item_names") or [])
        return state

    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, doc in enumerate(docs[:8], 1):
            if not isinstance(doc, dict):
                continue
            content = (doc.get("content") or "").strip()
            if not content:
                continue
            title = (doc.get("title") or "").strip()
            head = f"{title}: " if title else ""
            lines.append(f"[{idx}] {head}{content}")
        return "\n".join(lines)[:6000]

    def _extract_triples(self, question: str, context: str) -> List[str]:
        if not question or not context:
            return []

        prompt = KG_EXTRACT_USER_PROMPT_TEMPLATE.format(question=question, context=context)
        try:
            response = invoke_llm_with_json_fallback([
                SystemMessage(content=KG_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            if response is None:
                return []
            parsed = self._parse_json(getattr(response, "content", ""))
            triples = parsed.get("triples") or []
        except Exception as exc:
            logger.warning("图谱关系抽取失败: {}", exc)
            return []

        results = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            head = str(triple.get("head", "")).strip()
            relation = str(triple.get("relation", "")).strip()
            tail = str(triple.get("tail", "")).strip()
            if head and relation and tail:
                results.append(f"{head} -> {relation} -> {tail}")
        return results[:5]

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    def _retrieve_entities(self, question: str, item_names: List[str]) -> List[str]:
        """从 kb_entity_names 检索与问题相关的实体名，补充知识图谱上下文。"""
        if not question:
            return []
        if not item_names:
            logger.info("实体检索跳过: 未确认商品名")
            return []
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        if embedding_model is None or milvus_client is None:
            logger.info("实体检索跳过: BGE 或 Milvus 不可用")
            return []
        try:
            logger.info("开始实体检索, query={}", question[:60])
            embedding_result = generate_hybrid_embeddings(embedding_model, [question])
            if not embedding_result:
                return []
            requests = create_hybrid_search_requests(
                dense_vector=embedding_result["dense"][0],
                sparse_vector=embedding_result["sparse"][0],
                limit=10,
            )
            result = execute_hybrid_search_query(
                milvus_client,
                collection_name=self.config.entity_name_collection,
                search_requests=requests,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                output_fields=["entity_name"],
                limit=10,
            )
            if not result or not result[0]:
                logger.info("实体检索无结果")
                return []
            entities: List[str] = []
            seen: set = set()
            for hit in result[0]:
                entity = hit.get("entity", {})
                name = (entity.get("entity_name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    entities.append(name)
            logger.info("实体检索完成, 命中 {} 个实体: {}", len(entities), entities[:10])
            return entities
        except Exception as exc:
            logger.warning("实体检索失败: {}", exc)
            return []
