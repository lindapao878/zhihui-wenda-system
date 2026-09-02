"""RRF fusion node."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:
        vector_search_chunks = state.get("embedding_chunks") or []
        hyde_search_chunks = state.get("hyde_embedding_chunks") or []

        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result": (self._normalize_input(hyde_search_chunks), 1.0),
        }
        rrf_merge_results = self._rrf_merge(list(search_source.values()), self._rrf_k, self._top_k)
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        state["rrf_chunks"] = rrf_chunks
        return state

    def _normalize_input(self, rrf_input):
        result = []
        for doc in rrf_input or []:
            if not isinstance(doc, dict):
                continue
            entity = doc.get("entity")
            if entity:
                result.append(entity)
        return result

    def _rrf_merge(self, rrf_inputs, rrf_k, top_k):
        chunk_scores = {}
        chunk_data = {}

        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (rrf_k + i)
                chunk_data.setdefault(chunk_id, doc)

        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_results[:top_k] if top_k else sorted_results
