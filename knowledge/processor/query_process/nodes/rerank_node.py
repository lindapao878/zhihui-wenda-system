"""Rerank node with cliff cutoff."""
from __future__ import annotations

from typing import Any, Dict, List

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bge_rerank_util import get_reranker_model
from knowledge.utils.logger_util import logger


class RerankNode(BaseNode):
    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        user_query = state.get("rewritten_query", "") or state.get("original_query", "")
        merged_docs = self._merge_multi_source_docs(state)
        local_count = sum(1 for doc in merged_docs if doc.get("source") == "local")
        logger.info("rerank 输入: 本地={} web={} 总数={}", local_count, len(merged_docs) - local_count, len(merged_docs))
        reranked_docs = self._rerank_merged_docs(user_query, merged_docs)
        logger.info("rerank 排序完成: 总数={}", len(reranked_docs))
        state["reranked_docs"] = self._cliff_cutoff(reranked_docs)
        logger.info("rerank cutoff 保留: {}", len(state["reranked_docs"]))
        return state

    def _cliff_cutoff(self, ranked_docs):
        if not ranked_docs:
            return []

        upper_bound = min(self.config.rerank_max_top_k, len(ranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)
        cutoff_pos = upper_bound

        for i in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[i].get("score")
            next_score = ranked_docs[i + 1].get("score")
            if current_score is None or next_score is None:
                continue
            abs_gap = current_score - next_score
            rel_gap = abs_gap / (abs(current_score) + 1e-6)
            if abs_gap >= self.config.rerank_gap_abs or rel_gap >= self.config.rerank_gap_ratio:
                cutoff_pos = i + 1
                break
        return ranked_docs[:cutoff_pos]

    def _merge_multi_source_docs(self, state):
        final_docs = []
        for rrf_doc in state.get("rrf_chunks") or []:
            if not isinstance(rrf_doc, dict):
                continue
            content = rrf_doc.get("content", "").strip()
            if not content:
                continue
            final_docs.append({
                "content": content,
                "title": rrf_doc.get("title", ""),
                "chunk_id": rrf_doc.get("chunk_id"),
                "url": "",
                "source": "local",
            })

        for web_doc in state.get("web_search_docs") or []:
            if not isinstance(web_doc, dict):
                continue
            content = (web_doc.get("content") or web_doc.get("snippet") or "").strip()
            if not content:
                continue
            final_docs.append({
                "content": content,
                "title": web_doc.get("title", ""),
                "chunk_id": None,
                "url": web_doc.get("url", ""),
                "source": "web",
            })
        return final_docs

    def _rerank_merged_docs(self, user_query, merged_multi_docs):
        if not merged_multi_docs:
            return []

        rerank_model = get_reranker_model()
        if rerank_model is None:
            return [{**doc, "score": None} for doc in merged_multi_docs]

        pairs = [(user_query, doc.get("content")) for doc in merged_multi_docs]
        try:
            scores = rerank_model.compute_score(sentence_pairs=pairs)
            score_docs = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, scores)]
            return sorted(score_docs, key=lambda x: x["score"] or 0, reverse=True)
        except Exception as exc:
            logger.error("Rerank 重排序失败: {}", exc)
            return [{**doc, "score": None} for doc in merged_multi_docs]
