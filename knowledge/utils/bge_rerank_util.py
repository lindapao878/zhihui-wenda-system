"""BGE reranker model singleton."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from knowledge.utils.logger_util import logger

load_dotenv()

_reranker_model = None


class BgeReranker:
    def __init__(self, model):
        self._model = model

    def compute_score(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        try:
            scores = self._model.compute_score(sentence_pairs)
        except TypeError:
            scores = self._model.compute_score(*zip(*sentence_pairs))
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]


def get_reranker_model() -> Optional[BgeReranker]:
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    try:
        from FlagEmbedding import FlagReranker

        model_path = os.getenv(
            "BGE_RERANKER_LARGE", "BAAI/bge-reranker-large"
        )
        use_fp16 = os.getenv("BGE_RERANKER_FP16", "False").lower() in {
            "1", "true", "yes", "on"
        }
        device = os.getenv("BGE_RERANKER_DEVICE", "cpu")

        try:
            model = FlagReranker(model_path, use_fp16=use_fp16, device=device)
        except TypeError:
            model = FlagReranker(model_path, use_fp16=use_fp16)

        _reranker_model = BgeReranker(model)
    except Exception as exc:
        logger.error("加载 BGE-Reranker 模型失败: {}", exc)
        return None

    return _reranker_model
