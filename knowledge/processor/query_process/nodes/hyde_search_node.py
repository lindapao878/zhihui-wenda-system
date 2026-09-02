"""HyDE search node."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_string_util import escape_milvus_string
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, get_milvus_client


class HyDeSearchNode(BaseNode):
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        validated_query, validate_item_names = self._validate_query_inputs(state)
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        if not embedding_model or not milvus_client:
            return {}

        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_hybrid_embeddings(embedding_model, [embedding_document])
        if not embedding_result:
            return {}

        expr = self._item_name_filter_expr(validate_item_names)
        requests = create_hybrid_search_requests(
            dense_vector=embedding_result["dense"][0],
            sparse_vector=embedding_result["sparse"][0],
            expr=expr or None,
            limit=self.config.hyde_search_limit,
        )
        result = execute_hybrid_search_query(
            milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=requests,
            ranker_weights=(0.8, 0.2), norm_score=True,
            output_fields=["chunk_id", "content", "title", "parent_title", "file_title", "item_name"],
            limit=self.config.hyde_search_limit,
        )
        if not result or not result[0]:
            return {}
        filtered = self._apply_min_score_filter(result[0])
        if not filtered:
            return {}
        return {"hyde_embedding_chunks": filtered}

    def _validate_query_inputs(self, state):
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(self.name, "rewritten_query", str)
        if not isinstance(item_names, list):
            raise StateFieldError(self.name, "item_names", list)
        return rewritten_query, item_names

    def _generate_hy_document(self, validated_query, validate_item_names):
        llm_client = get_llm_client()
        if llm_client is None:
            return ""

        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(rewritten_query=validated_query)
        system_prompt = "你是知识库问答助手，擅长根据用户问题撰写准确、简洁的参考回答。"
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return getattr(llm_response, "content", "").strip()
        except Exception:
            return ""

    def _item_name_filter_expr(self, validate_item_names):
        if not validate_item_names:
            return ''
        quoted = ", ".join(f'"{escape_milvus_string(value)}"' for value in validate_item_names)
        return f" item_name in [{quoted}]"
