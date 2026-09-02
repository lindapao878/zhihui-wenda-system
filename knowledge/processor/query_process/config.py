"""Knowledge query flow configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryConfig:
    max_context_chars: int = field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000")))

    rerank_max_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10")))
    rerank_min_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3")))
    rerank_gap_ratio: float = field(default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25")))
    rerank_gap_abs: float = field(default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5")))

    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    rrf_max_results: int = field(default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10")))

    embedding_search_limit: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "10")))
    hyde_search_limit: int = field(default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5")))

    milvus_min_cosine_score: float = field(
        default_factory=lambda: float(os.getenv("MILVUS_MIN_COSINE_SCORE", "0.0"))
    )

    item_name_high_confidence: float = field(default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.7")))
    item_name_mid_confidence: float = field(default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.6")))
    item_name_max_options: int = field(default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "5")))
    item_name_dense_weight: float = field(default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5")))
    item_name_sparse_weight: float = field(default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5")))

    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    default_model: str = field(default_factory=lambda: os.getenv("MODEL", ""))
    item_model: str = field(default_factory=lambda: os.getenv("ITEM_MODEL", ""))

    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "kb_chunks"))
    item_name_collection: str = field(default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "kb_item_names"))
    entity_name_collection: str = field(default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "kb_entity_names"))

    mcp_dashscope_base_url: str = field(default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", ""))
    mcp_dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_API_KEY", "")
    )
    query_cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_TTL_SECONDS", "300"))
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
