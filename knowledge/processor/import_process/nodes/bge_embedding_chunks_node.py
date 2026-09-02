"""BGE-M3 chunk embedding node."""
from __future__ import annotations

from typing import Any, Dict, List

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import EmbeddingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model
from knowledge.utils.logger_util import logger


class BgeEmbeddingChunksNode(BaseNode):
    name = "beg_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        validated_chunks, config = self._validate_get_inputs(state)
        embedding_batch_chunk_size = getattr(config, "embedding_batch_size", 16)
        total_length = len(validated_chunks)
        final_chunks = []

        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = validated_chunks[i : i + embedding_batch_chunk_size]
            final_chunks.extend(self._process_batch_chunks(batch, i, total_length))

        state["chunks"] = final_chunks
        return state

    def _process_batch_chunks(self, batch: List[Dict[str, Any]], start_index: int, total_length: int):
        embedding_contents = []
        for chunk in batch:
            content = chunk.get("content")
            item_name = chunk.get("item_name")
            embedding_contents.append(f"{item_name}\n{content}")

        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            raise EmbeddingError('BGE-M3 模型不可用', self.name)

        try:
            embedding_result = embedding_model.encode_documents(documents=embedding_contents)
        except Exception as exc:
            logger.error('嵌入向量嵌入失败: {}', exc)
            raise EmbeddingError(f'嵌入向量生成失败: {exc}', self.name)

        if not embedding_result:
            return batch

        for index, chunk in enumerate(batch):
            dense_vector = embedding_result["dense"][index].tolist()
            csr_array = embedding_result["sparse"]
            start = csr_array.indptr[index]
            end = csr_array.indptr[index + 1]
            token_id = csr_array.indices[start:end].tolist()
            weight = csr_array.data[start:end].tolist()
            chunk["dense_vector"] = dense_vector
            chunk["sparse_vector"] = dict(zip(token_id, weight))

        return batch

    def _validate_get_inputs(self, state):
        config = get_config()
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks为空或者无效", self.name)
        return chunks, config


