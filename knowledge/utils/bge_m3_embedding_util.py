"""BGE-M3 dense/sparse embedding helpers."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from knowledge.utils.logger_util import logger

load_dotenv()

_bge_m3_model = None


class _BgeM3EmbeddingWrapper:
    def __init__(self, model):
        self._model = model

    def _to_csr(self, lexical_weights):
        from scipy import sparse

        indptr = [0]
        indices = []
        data = []
        tokenizer = getattr(self._model, 'tokenizer', None)

        n_docs = len(lexical_weights)

        for weights in lexical_weights:
            for token, weight in weights.items():
                if isinstance(token, int):
                    token_id = token
                elif isinstance(token, str) and token.isdigit():
                    token_id = int(token)
                elif isinstance(token, str) and tokenizer is not None:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                else:
                    token_id = -1
                if token_id in (None, -1):
                    continue
                indices.append(int(token_id))
                data.append(float(weight))
            indptr.append(len(indices))

        if not data:
            logger.warning("BGE-M3 稀疏向量为空，文档数: {}", n_docs)
            return sparse.csr_matrix((n_docs, 1), dtype='float32')
        return sparse.csr_matrix((data, indices, indptr), shape=(n_docs, max(indices) + 1), dtype='float32')

    def encode_documents(self, documents):
        result = self._model.encode(documents, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return {'dense': result.get('dense_vecs'), 'sparse': self._to_csr(result.get('lexical_weights') or [])}

    def __call__(self, documents):
        return self.encode_documents(documents)


def get_beg_m3_embedding_model():
    global _bge_m3_model
    if _bge_m3_model is not None:
        return _bge_m3_model

    try:
        from FlagEmbedding import BGEM3FlagModel

        model_name = os.getenv("BGE_M3_PATH", "") or "BAAI/bge-m3"
        device = os.getenv("BGE_DEVICE", "cpu")
        use_fp16 = os.getenv("BGE_FP16", "False").lower() in {"1", "true", "yes", "on"}

        model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

        _bge_m3_model = _BgeM3EmbeddingWrapper(model)
    except Exception as exc:
        logger.error("加载 BGE-M3 模型失败: {}", exc)
        return None

    return _bge_m3_model


def is_bge_m3_model_loaded() -> bool:
    """Return current model singleton state without loading the model."""
    return _bge_m3_model is not None


def normalize_sparse_vector(sparse_dict: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(value * value for value in sparse_dict.values()))
    if norm == 0:
        return dict(sparse_dict)
    return {key: value / norm for key, value in sparse_dict.items()}


def _extract_sparse_vectors(raw_embeddings, text_count: int) -> List[Dict[int, float]]:
    sparse_matrix = raw_embeddings["sparse"]
    sparse_vectors = []

    for i in range(text_count):
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i + 1]
        sparse_dict = dict(zip(sparse_matrix.indices[row_start:row_end].tolist(), sparse_matrix.data[row_start:row_end].tolist()))
        sparse_vectors.append(normalize_sparse_vector(sparse_dict))

    return sparse_vectors


def generate_hybrid_embeddings(embedding_model, embedding_documents: List[str]) -> Dict[str, Any]:
    try:
        raw_embeddings = embedding_model(embedding_documents)
        dense_vectors = [emb.tolist() for emb in raw_embeddings["dense"]]
        sparse_vectors = _extract_sparse_vectors(raw_embeddings, len(embedding_documents))
        return {"dense": dense_vectors, "sparse": sparse_vectors}
    except Exception as exc:
        logger.error("生成混合向量失败: {}", exc)
        return {}
