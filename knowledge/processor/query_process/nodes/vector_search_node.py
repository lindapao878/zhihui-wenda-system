"""Vector search node."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model
from knowledge.utils.milvus_string_util import escape_milvus_string
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, get_milvus_client


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        validated_query, validate_item_names = self._validate_query_inputs(state)
        embedding_model = get_beg_m3_embedding_model()
        milvus_client = get_milvus_client()
        if embedding_model is None or milvus_client is None:
            return {}

        embedding_result = generate_hybrid_embeddings(embedding_model, [validated_query])
        if not embedding_result:
            return {}

        item_name_filter_expr = self._item_name_filter(validate_item_names)
        requests = create_hybrid_search_requests(
            dense_vector=embedding_result["dense"][0],
            sparse_vector=embedding_result["sparse"][0],
            expr=item_name_filter_expr or None,
            limit=self.config.embedding_search_limit,
        )
        result = execute_hybrid_search_query(
            milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=requests,
            ranker_weights=(0.8, 0.2), norm_score=True,
            output_fields=["chunk_id", "content", "title", "parent_title", "file_title", "item_name"],
            limit=self.config.embedding_search_limit,
        )
        if not result or not result[0]:
            return {}
        filtered = self._apply_min_score_filter(result[0])
        if not filtered:
            return {}
        return {"embedding_chunks": filtered}

    def _validate_query_inputs(self, state):
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(self.name, "rewritten_query", str)
        if not isinstance(item_names, list):
            raise StateFieldError(self.name, "item_names", list)
        return rewritten_query, item_names

    def _item_name_filter(self, validate_item_names: List[str]) -> str:
        if not validate_item_names:
            return ''
        quoted = ", ".join(f'"{escape_milvus_string(value)}"' for value in validate_item_names)
        return f" item_name in [{quoted}]"




