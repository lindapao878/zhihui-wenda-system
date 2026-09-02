# Codex 工作规则
 
## 0. 第一原则
少说、多做、做对。

## 1. 先理解，再行动
- 先读 `AGENTS.md`、`PROJECT_STATUS.md` 和 `.env`。
- 不清楚时先搜索现有实现，不急于重写。
- 需求模糊但可合理推断时，直接做，不反复追问。

## 2. 小步推进
- 每次只做一个逻辑完整的改动。
- 不顺手重构无关代码。
- 保持 diff 小、可回滚。

## 3. 优先复用
- 优先使用项目已有工具、函数和模式。
- 已有第三方库时不重复造轮子。
- 参考代码只迁移必要部分。

## 4. 控制 token 消耗
- 不递归扫描大型目录。
- 长任务使用后台运行和日志文件。
- 不输出 Docker、模型下载、依赖安装的大段进度。
- 修改后只做必要验证。
- 不反复执行同一批命令。
- 不要在回答中重复粘贴刚修改的代码内容。

## 5. 输出规则
- 使用中文。
- 直接完成工作。
- 完成后只用一两句话说明：改了什么文件、验证结果。
- 不要输出"下一步操作建议"。
- 不要长篇总结。
- 如果有必须让用户决策的问题，简短列出。

## 6. 提高缓存命中
- 保持规则文本稳定，不频繁改动提示词。
- 将固定规则放在最前面，具体任务放在最后面。
- 任务描述简洁，避免大段背景重复。
- 复用已有 `AGENTS.md` 内容，不在对话中重复其全文。
- 新线程只补充增量信息，不重述项目现状。

## 7. 工程判断
- 优先选择无聊但可靠的方案。
- 测试范围与风险匹配。
- 遇到问题先定位最小原因再修复。
- 不假装知道；不确定时明确说。

## 8. 项目上下文
- 主代码在 `knowledge/`，项目状态详见 `PROJECT_STATUS.md`。
- 目标：逐步完善 RAG 知识库项目，同时降低 Codex token 消耗。

---

# Project Context

智慧问答系统 is a RAG knowledge-base project under `knowledge/`.

## 技术栈
- LLM: SiliconFlow API，模型 `zai-org/GLM-5.2`（.env 可配，不支持 response_format=json_object，有 fallback）
- Embeddings: 本地 BGE-M3（CPU 模式，1024 维，dense+sparse 混合向量）
- Reranker: 本地 BGE-Reranker-Large（CPU 模式）
- VLM: `Qwen/Qwen3-VL-32B-Instruct`（SiliconFlow，导入时图片摘要用）
- MCP: 百炼 WebSearch，Streamable HTTP 协议（/mcp 端点，非 SSE）
- 日志: loguru（控制台+文件双输出，每日轮转，按天保留，位置精准）
- 转义: `escape_milvus_string`（统一处理 `\ " \r \n \t`，所有 Milvus filter 构造处必须调用）
- Middleware: Milvus(v2.4.5)、MongoDB(7.0)、MinIO via Docker Compose

## Milvus 集合
| 集合名 | 用途 | 主键 |
|--------|------|------|
| kb_chunks | 文档切片 | chunk_id INT64 auto_id |
| kb_item_names | 商品名 | pk INT64 auto_id |
| kb_entity_names | 实体名 | pk INT64 auto_id |

## 服务端口
- Import API: 8000
- Query API: 8001
- Milvus: 19530
- MongoDB: 27017
- MinIO: 9000

## API 接口
### 导入服务（端口 8000）
- POST /upload — 上传文档（PDF/MD/DOCX）
- GET /status/{task_id} — 查询导入任务状态
- DELETE /document/{file_title} — 删除指定文档（清三张集合）
- GET /health — 健康检查
### 查询服务（端口 8001）
- POST /query — 提交查询（流式/非流式）
- GET /stream/{task_id} — SSE 流式接收答案
- GET /status — 服务状态
- GET /history/{session_id} — 获取对话历史
- DELETE /history/{session_id} — 清空对话历史

## Useful Commands
- Start middleware:
  `docker compose -p zhihui-wenda-system up -d`
- Start APIs (Windows):
  `.venv\Scripts\python.exe -m uvicorn knowledge.api.import_router:app --port 8000`
  `.venv\Scripts\python.exe -m uvicorn knowledge.api.query_router:app --port 8001`
- Syntax check:
  `.venv\Scripts\python.exe -m py_compile <file>`
- Run tests:
  `.venv\Scripts\python.exe -m unittest discover -s tests -v`
- Batch import:
  `.venv\Scripts\python.exe batch_import.py --dry-run`
  `.venv\Scripts\python.exe batch_import.py --workers 2 --retry 2 --retry-delay 10`

## 已知瓶颈
- BGE-M3 和 Reranker 跑在 CPU 上，查询链路总耗时约 60-80s（无缓存时）
- MinerU 解析 PDF 在 CPU 上极慢（1 页约 15min），且偶发卡死；有 10min 超时防护
- 无 GPU 环境；查询缓存可大幅降低重复查询延迟（82s->4s）
- 检索排序和答案质量参数在 .env 可调（调优后: MILVUS_MIN_COSINE_SCORE=0.6, ITEM_NAME_MID_CONFIDENCE=0.7, RERANK_MAX_TOP_K=5, EMBEDDING_SEARCH_LIMIT=5）

## Avoid
- Do not recursively scan large local data folders unless explicitly needed.
- Do not print huge Docker, MinerU, or model download logs to the thread.
- Use background jobs with log files for long batch imports.
- Do not use SSE protocol for MCP (DashScope /mcp endpoint requires Streamable HTTP).
- Do not assume JSON mode works with SiliconFlow (use invoke_llm_with_json_fallback).
- Do not construct Milvus filter expressions without `escape_milvus_string` (from knowledge.utils.milvus_string_util).
- Do not add `images/` or `*_智慧问答系统*.md` to git (local-only assets, already in .gitignore).
- Do not use `logging.getLogger` -- use `from knowledge.utils.logger_util import logger` instead.
