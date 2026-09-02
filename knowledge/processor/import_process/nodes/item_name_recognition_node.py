"""Product name recognition node."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import EmbeddingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.upload.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model
from knowledge.utils.milvus_string_util import escape_milvus_string
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.logger_util import logger


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title, chunks, config = self._validate_inputs(state)
        item_name_context = self._prepare_item_name_context(chunks, config)
        item_name = self._recognition_item_name_by_llm(file_title, item_name_context)
        dense_vector, sparse_vector = self._embedding_item_name(item_name)
        self._save_to_milvus(file_title, item_name, dense_vector, sparse_vector, config)
        self._fill_item_name(item_name, state, chunks)
        return state

    def _validate_inputs(self, state):
        config = get_config()
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        if not file_title:
            raise ValidationError("文件标题为空", self.name)
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunk为空或者无效", self.name)
        return file_title, chunks, config

    def _prepare_item_name_context(self, chunks, config):
        result = []
        total = 0
        for index, chunk in enumerate(chunks[: config.item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue
            content = chunk.get("content")
            spices = f"【切片】- {index + 1} - {content}"
            total += len(spices)
            result.append(spices)
            if total > config.item_name_chunk_size:
                break
        return "\n\n".join(result)[: config.item_name_chunk_size]

    def _recognition_item_name_by_llm(self, file_title, item_name_context):
        llm_client = get_llm_client()
        if llm_client is None:
            return file_title

        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            item_name = getattr(llm_response, "content", "").strip()
            if not item_name or item_name.upper() == "UNKNOWN":
                return file_title
            return item_name
        except Exception:
            return file_title

    def _embedding_item_name(self, item_name: str) -> Tuple[list, dict]:
        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            raise EmbeddingError(f"嵌入商品名:{item_name}失败,模型不可用", self.name)

        try:
            embedding_result = embedding_model.encode_documents([item_name])
            dense = embedding_result["dense"][0].tolist()
            start_index = embedding_result["sparse"].indptr[0]
            end_index = embedding_result["sparse"].indptr[1]
            weights = embedding_result["sparse"].data[start_index:end_index].tolist()
            token_ids = embedding_result["sparse"].indices[start_index:end_index].tolist()
            sparse = dict(zip(token_ids, weights))
            return dense, sparse
        except Exception as exc:
            raise EmbeddingError(f"嵌入商品名:{item_name}失败,原因是：{exc}", self.name)

    def _save_to_milvus(self, file_title, item_name, dense_vector, sparse_vector, config):
        if not dense_vector or not sparse_vector:
            return
        milvus_client = get_milvus_client()
        if milvus_client is None:
            return

        collection_name = config.item_name_collection
        try:
            if not milvus_client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(milvus_client, collection_name)
            safe_title = escape_milvus_string(file_title)
            milvus_client.delete(
                collection_name=collection_name,
                filter=f'file_title == "{safe_title}"',
            )
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
            milvus_client.insert(collection_name=collection_name, data=[data])
        except Exception as exc:
            logger.error("Milvus 数据库保存操作彻底失败: {}", exc)

    def _create_item_name_collection(self, client, collection_name):
        schema = client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_name="sparse_inverted_index", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name
