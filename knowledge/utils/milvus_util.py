"""Milvus client and hybrid search helpers."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from knowledge.utils.logger_util import logger

load_dotenv()

_client = None


def get_milvus_client():
    global _client
    if _client is not None:
        return _client

    try:
        from pymilvus import MilvusClient

        uri = os.getenv("MILVUS_URL", "http://localhost:19530")
        _client = MilvusClient(uri=uri)
        return _client
    except Exception as exc:
        logger.error("创建 Milvus 客户端失败: {}", exc)
        return None


def create_hybrid_search_requests(
    dense_vector: List[float],
    sparse_vector: Dict[int, float],
    expr: Optional[str] = None,
    limit: int = 5,
):
    try:
        from pymilvus import AnnSearchRequest
    except Exception as exc:
        logger.error("导入 pymilvus 失败: {}", exc)
        return []

    dense_request = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        expr=expr,
        limit=limit,
    )
    sparse_request = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector", param={"metric_type": "IP"},
        expr=expr,
        limit=limit,
    )
    return [dense_request, sparse_request]


def execute_hybrid_search_query(
    milvus_client,
    collection_name: str,
    search_requests,
    ranker_weights: Tuple[float, float] = (0.5, 0.5),
    norm_score: bool = False,
    output_fields: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Optional[List]:
    if output_fields is None:
        output_fields = ["item_name"]

    try:
        from pymilvus import WeightedRanker

        ranker = WeightedRanker(
            ranker_weights[0],
            ranker_weights[1],
            norm_score=norm_score,
        )
        search_kwargs = {
            "collection_name": collection_name,
            "reqs": search_requests,
            "ranker": ranker,
            "output_fields": output_fields,
        }
        if limit is not None:
            search_kwargs["limit"] = limit
        results = milvus_client.hybrid_search(**search_kwargs)
        logger.info(
            "混合检索完成: collection={}, 命中={}",
            collection_name,
            len(results[0]) if results else 0,
        )
        return results
    except Exception as exc:
        logger.error("混合检索执行失败: {}", exc)
        return None


def batch_hybrid_search(
    embedding_model,
    queries: List[str],
    collection_name: str,
    limit: int = 5,
    output_fields: Optional[List[str]] = None,
    ranker_weights: Tuple[float, float] = (0.8, 0.2),
    norm_score: bool = True,
) -> List[Optional[List]]:
    from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings

    if not queries or embedding_model is None:
        return []

    client = get_milvus_client()
    if client is None:
        return []

    hybrid_embeddings = generate_hybrid_embeddings(embedding_model, queries)
    if not hybrid_embeddings:
        return []

    results = []
    for index, query in enumerate(queries):
        requests = create_hybrid_search_requests(
            dense_vector=hybrid_embeddings["dense"][index],
            sparse_vector=hybrid_embeddings["sparse"][index],
            limit=limit,
        )
        result = execute_hybrid_search_query(
            client,
            collection_name=collection_name,
            search_requests=requests,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            output_fields=output_fields,
            limit=limit,
        )
        results.append(result[0] if result else None)
    return results

