"""Entity name recognition and Milvus persistence node."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import EmbeddingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model
from knowledge.utils.llm_client_util import invoke_llm_with_json_fallback
from knowledge.utils.milvus_string_util import escape_milvus_string
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.logger_util import logger

_ENTITY_EXTRACT_SYSTEM = "你是专业的实体识别助手，擅长从文档内容中提取关键实体名称（人名、地名、机构、产品、概念等）。"
_ENTITY_EXTRACT_USER = """请从以下文档内容中提取所有重要的实体名称。
输出 JSON 格式：{{"entities": ["实体1", "实体2", ...]}}。每个实体名称不超过20个字。

文档内容：
{context}"""


class EntityRecognitionNode(BaseNode):
    name = "entity_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title, chunks, config = self._validate_inputs(state)
        context = self._build_context(chunks)
        entities = self._extract_entities(file_title, context)
        if not entities:
            logger.info("未提取到实体，跳过实体入库")
            state["_entity_names"] = []
            return state

        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            logger.warning("BGE-M3 模型不可用，跳过实体入库")
            state["_entity_names"] = []
            return state

        dense_vectors, sparse_vectors = self._embed_entities(entities, embedding_model)
        self._save_to_milvus(file_title, entities, dense_vectors, sparse_vectors, config)
        state["_entity_names"] = entities
        return state

    def _validate_inputs(self, state):
        config = get_config()
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        if not file_title:
            raise ValidationError("文件标题为空", self.name)
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks 为空或无效", self.name)
        return file_title, chunks, config

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        total = 0
        max_chars = 3000
        for idx, chunk in enumerate(chunks):
            content = (chunk.get("content") or "").strip() if isinstance(chunk, dict) else ""
            if not content:
                continue
            piece = f"【片段{idx + 1}】{content}"
            total += len(piece)
            if total > max_chars and parts:
                break
            parts.append(piece)
        return "\n\n".join(parts)[:max_chars]

    def _extract_entities(self, file_title: str, context: str) -> List[str]:
        if not context:
            return []
        user_prompt = _ENTITY_EXTRACT_USER.format(context=context)
        try:
            response = invoke_llm_with_json_fallback([
                SystemMessage(content=_ENTITY_EXTRACT_SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            if response is None:
                return []
            parsed = self._parse_json(getattr(response, "content", ""))
            raw = parsed.get("entities", [])
        except Exception as exc:
            logger.warning("LLM 实体提取失败: {}", exc)
            return []

        entities: List[str] = []
        for e in raw:
            name = str(e).strip()[:20]
            if name and name.lower() not in ("none", "unknown", ""):
                if name not in entities:
                    entities.append(name)
        return entities[:20]

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    def _embed_entities(self, entities: List[str], model) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        dense_list: List[List[float]] = []
        sparse_list: List[Dict[int, float]] = []

        for entity in entities:
            try:
                result = model.encode_documents([entity])
                dense_vec = result["dense"][0].tolist()
                csr = result["sparse"]
                start = csr.indptr[0]
                end = csr.indptr[1]
                token_ids = csr.indices[start:end].tolist()
                weights = csr.data[start:end].tolist()
                sparse_vec = dict(zip(token_ids, weights))
            except Exception as exc:
                raise EmbeddingError(f"实体嵌入失败 '{entity}': {exc}", self.name)
            dense_list.append(dense_vec)
            sparse_list.append(sparse_vec)
        return dense_list, sparse_list

    def _save_to_milvus(self, file_title, entities, dense_vectors, sparse_vectors, config):
        milvus_client = get_milvus_client()
        if milvus_client is None:
            logger.warning("Milvus 客户端不可用，跳过实体入库")
            return

        collection_name = config.entity_name_collection
        self._ensure_collection(milvus_client, collection_name, len(dense_vectors[0]))

        # 删除旧实体
        safe_title = escape_milvus_string(file_title)
        try:
            milvus_client.delete(
                collection_name=collection_name,
                filter=f'file_title == "{safe_title}"',
            )
        except Exception as exc:
            logger.warning("删除旧实体失败: {}", exc)

        # 批量插入
        rows = []
        for entity, dense, sparse in zip(entities, dense_vectors, sparse_vectors):
            rows.append({
                "entity_name": entity,
                "file_title": file_title,
                "dense_vector": dense,
                "sparse_vector": sparse,
            })

        try:
            milvus_client.insert(collection_name=collection_name, data=rows)
            milvus_client.flush(collection_name=collection_name)
            logger.info("已入库 {} 个实体到 {}", len(rows), collection_name)
        except Exception as exc:
            logger.error("实体入库失败: {}", exc)

    def _ensure_collection(self, client, collection_name: str, dim: int):
        if client.has_collection(collection_name=collection_name):
            return

        schema = client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="entity_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_name="dense_vector_index",
            index_type="AUTOINDEX", metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector", index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema, index_params=index_params,
        )
        client.load_collection(collection_name=collection_name)
        logger.info("已创建 kb_entity_names 集合 (dim={})", dim)
