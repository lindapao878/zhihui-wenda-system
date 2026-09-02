"""Milvus persistence node for embedded chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_string_util import escape_milvus_string
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.logger_util import logger



@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusSchemaBuilder:
    @staticmethod
    def build(client: MilvusClient, dim: int):
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        for scalar_field in _SCALAR_FIELDS:
            kwargs = {"field_name": scalar_field.field_name, "datatype": scalar_field.datatype}
            if scalar_field.max_length is not None:
                kwargs["max_length"] = scalar_field.max_length
            schema.add_field(**kwargs)
        return schema


class _MilvusIndexBuilder:
    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        index = client.prepare_index_params(collection_name=collection_name)
        index.add_index(field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index(field_name="sparse_vector", index_name="sparse_vector_index", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        return index


class _MilvusInserter:
    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]):
        file_titles = list({str(chunk.get("file_title", "")) for chunk in chunks if chunk.get("file_title")})
        if file_titles:
            for title in file_titles:
                safe_title = escape_milvus_string(title)
                try:
                    self._client.delete(
                        collection_name=self._collection_name,
                        filter=f'file_title == "{safe_title}"',
                    )
                except Exception as exc:
                    logger.warning("删除旧切片失败 file_title={}: {}", title, exc)

        inserted_result = self._client.insert(collection_name=self._collection_name, data=chunks)
        ids = inserted_result.get("ids", [])
        for chunk, chunk_id in zip(chunks, ids):
            chunk["chunk_id"] = chunk_id

        try:
            self._client.flush(collection_name=self._collection_name)
        except Exception as exc:
            logger.warning("Milvus flush 失败: {}", exc)

        return chunks


class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        validated_chunks, dim, config = self._validate_get_inputs(state)
        milvus_client = get_milvus_client()
        if milvus_client is None:
            return state

        collection = getattr(config, "chunks_collection")
        self._ensure_has_collection(milvus_client, collection, dim)
        final_chunks = _MilvusInserter(milvus_client, collection).insert(validated_chunks)
        state["chunks"] = final_chunks
        return state

    def _validate_get_inputs(self, state):
        config = get_config()
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        invalid_count = 0
        validated_chunks = []
        for chunk in chunks:
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                invalid_count += 1
                logger.error("待入库的切块chunk的混合向量不存在")

        if invalid_count:
            raise ValidationError(f"存在 {invalid_count} 个切片缺少混合向量，终止入库", self.name)
        if not validated_chunks:
            raise ValidationError("入库的chunk都无效", self.name)

        dim = len(validated_chunks[0].get("dense_vector"))
        return validated_chunks, dim, config

    def _ensure_has_collection(self, milvus_client, collection_name, dim, delete_flag=False):
        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            milvus_client.drop_collection(collection_name=collection_name)
        if milvus_client.has_collection(collection_name=collection_name):
            return

        schema = _MilvusSchemaBuilder.build(milvus_client, dim)
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index)
