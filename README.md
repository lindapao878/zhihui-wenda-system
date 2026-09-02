# 智慧问答系统

企业级 RAG 知识库问答系统：支持 PDF / Markdown / DOCX 导入，提供商品名确认、多路检索（向量 + HyDE + MCP 网络搜索）、RRF 融合、Rerank 重排序、知识图谱补充与流式问答。

## 当前阶段确认

本项目已完成：

- 导入链路：PDF / Markdown / DOCX 全流程（MinerU、python-docx、VLM 图片摘要、分块、商品名/实体识别、BGE-M3 向量化、Milvus 入库）
- 查询链路：商品名确认、并行检索、RRF 融合、Rerank 截断、知识图谱补充、流式 SSE 答案
- 工程硬化：接口鉴权（X-API-Key）、CORS 白名单、MongoDB 任务持久化、健康探针、全局异常处理、CPU/GPU 双配置文件、Docker 容器化
- 质量底稿：53 个单元测试全绿，批量导入支持断点续传 + MD5 去重 + 失败重试

尚未落地（属于后续阶段，不影响当前使用）：

- Prometheus 指标与 Grafana 监控
- GitHub Actions CI/CD（lint + test 流水线）
- 多实例共享缓存（当前查询缓存为进程内存）
- 优雅关闭与运行中任务中断恢复（重启时 processing 任务标记为 failed）
- MinerU 独立 GPU 镜像（当前镜像不含 MinerU，PDF 导入走本机裸跑或后续单独镜像）

结论：**功能已基本完善并具备企业级基础硬化**，距完整企业级运维体系还差监控与发布流水线。

## 功能特性

- **文档导入**：自动按 `.pdf` / `.md` / `.docx` 路由解析，PDF 走 MinerU 子进程，DOCX 按段落+表格顺序转 Markdown，图片由 VLM 生成中文摘要并上传 MinIO
- **智能分块**：按标题层级分块 + RecursiveCharacterTextSplitter 长块拆分 + 短块合并，HTML 表格线性化，chunks.json 备份
- **结构化识别**：LLM 提取商品名与实体三元组，写入独立 Milvus 集合并回填切片字段
- **混合检索**：BGE-M3 dense + sparse 混合向量，WeightedRanker 融合；HyDE 假设文档检索；MCP 网络搜索（Streamable HTTP /mcp）
- **结果融合**：RRF（k=60）融合本地两路，BGE-Reranker 重排序 + cliff cutoff 动态截断，知识图谱补充相关实体
- **流式问答**：后端 SSE（ready → progress → delta → final），前端打字机效果、Markdown 渲染、图片直链展示
- **批量导入**：断点续传、内容 MD5 去重、并发控制、失败自动重试
- **工程治理**：统一 loguru 日志、Milvus 字符串统一转义、接口鉴权加固、任务状态持久化

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | SiliconFlow API（默认 zai-org/GLM-5.2） |
| VLM | Qwen/Qwen3-VL-32B-Instruct（SiliconFlow，图片摘要） |
| Embedding | 本地 BGE-M3（1024 维，dense + sparse 混合向量） |
| Reranker | 本地 BGE-Reranker-Large |
| 向量数据库 | Milvus 2.4.5（standalone） |
| 文档数据库 | MongoDB 7.0 |
| 对象存储 | MinIO |
| Web 框架 | FastAPI + Uvicorn |
| 工作流编排 | LangGraph（导入/查询双图） |
| MCP 网络搜索 | 百炼 WebSearch（Streamable HTTP，非 SSE） |
| 日志 | loguru（控制台 + 文件双输出） |

## 目录结构

```
智慧问答系统/
├── knowledge/                  # 主程序包
│   ├── api/                    # FastAPI 路由
│   │   ├── import_router.py    # 导入服务（8000）：upload/status/delete/health/ready/前端
│   │   └── query_router.py     # 查询服务（8001）：query/status/stream/history/health/ready/前端
│   ├── core/                   # 基础设施
│   │   ├── deps.py             # FastAPI 依赖注入（服务单例）
│   │   ├── paths.py            # 项目路径（前端页、临时目录）
│   │   └── security.py         # X-API-Key 鉴权 + CORS 白名单
│   ├── front/                  # 静态前端页面
│   │   ├── import.html         # 导入页（拖拽上传/进度面板/可折叠日志）
│   │   └── chat.html           # 聊天页（暗色模式/Markdown 渲染/示例问题/复制按钮）
│   ├── processor/
│   │   ├── import_process/     # 导入 LangGraph 链路
│   │   │   ├── main_graph.py   # entry → 条件路由 → 解析 → 图片 → 分块 → 商品名 → 实体 → 向量化 → 入库
│   │   │   ├── config.py       # 导入侧参数（分块、图片限速、Milvus/MinIO/维度）
│   │   │   ├── state.py        # 导入图状态定义
│   │   │   ├── base.py         # 节点基类
│   │   │   ├── exceptions.py   # 导入异常
│   │   │   └── nodes/
│   │   │       ├── entry_node.py                  # 入口：按扩展名路由
│   │   │       ├── pdf_to_md_node.py              # MinerU 子进程解析 PDF
│   │   │       ├── docx_to_md_node.py             # python-docx 转 Markdown
│   │   │       ├── md_img_node.py                 # 图片 VLM 摘要 + MinIO 上传
│   │   │       ├── document_split_node.py         # 分块
│   │   │       ├── item_name_recognition_node.py  # 商品名识别 + kb_item_names 入库
│   │   │       ├── entity_recognition_node.py     # 实体/三元组识别
│   │   │       ├── bge_embedding_chunks_node.py   # BGE-M3 批量混合向量化
│   │   │       └── import_milvus_node.py          # 建索引 + 按 file_title 去重 + 写入
│   │   └── query_process/     # 查询 LangGraph 链路
│   │       ├── main_graph.py   # 商品名确认 → 并行检索 → RRF → Rerank → KG → 答案
│   │       ├── config.py       # 查询侧参数（rerank/RRF/置信度/检索 limit）
│   │       ├── state.py / base.py / exceptions.py
│   │       └── nodes/
│   │           ├── item_name_confirm_node.py  # 商品名提取/对齐/澄清 + 查询改写 + 缓存
│   │           ├── vector_search_node.py      # BGE-M3 混合向量检索
│   │           ├── hyde_search_node.py        # HyDE 假设文档检索
│   │           ├── mcp_search_node.py         # 百炼 WebSearch MCP
│   │           ├── rrf_node.py                # RRF 融合
│   │           ├── rerank_node.py             # BGE-Reranker + cliff cutoff
│   │           ├── knowledge_graph_node.py    # 三元组提取 + 实体检索补充
│   │           └── answer_output_node.py      # 答案生成 + SSE/历史写入 + 图片提取
│   ├── prompts/                  # 提示词文件（.prompt + loader）
│   │   ├── loader.py             # 按 category+name 加载
│   │   ├── query/                # answer / hyde / item_name_extract / kg_extract
│   │   └── upload/               # import_item_name（导入商品名识别）
│   ├── schema/                   # Pydantic 请求/响应模型
│   │   ├── task_schema.py        # TaskStatusResponse
│   │   ├── upload_schema.py      # UploadResponse
│   │   └── query_schema.py       # QueryRequest / QueryResponse / StreamSubmitResponse
│   ├── services/                 # 业务服务层
│   │   ├── file_import_service.py  # 上传落盘、去重预检、删除文档、导入图调度
│   │   ├── query_service.py        # 查询任务调度、历史读写
│   │   └── task_service.py         # 任务状态薄封装
│   └── utils/                     # 工具层
│       ├── milvus_util.py           # Milvus 客户端单例 + hybrid search
│       ├── milvus_string_util.py    # filter 字符串安全转义（红线项）
│       ├── bge_m3_embedding_util.py # BGE-M3 加载 + dense/sparse 向量
│       ├── bge_rerank_util.py       # FlagReranker 封装
│       ├── llm_client_util.py       # ChatOpenAI 单例 + JSON 回退
│       ├── minio_util.py            # MinIO 客户端 + 公开只读策略
│       ├── mongo_history_util.py    # MongoDB 客户端 + 历史 CRUD
│       ├── task_store.py            # Mongo kb_tasks 持久化（内存回退）
│       ├── task_util.py             # 任务状态管理（向前兼容）
│       ├── query_cache.py           # 查询结果进程内存缓存（TTL）
│       ├── sse_util.py              # SSE 队列生命周期
│       ├── health_util.py           # /ready 就绪探针
│       ├── markdown_util.py         # Markdown 表格转文本
│       └── logger_util.py           # 统一 loguru 日志
├── tests/                       # 单元测试（8 个文件 / 53 例，全绿）
├── batch_import.py              # 批量导入脚本（断点续传 + MD5 去重 + 重试）
├── Dockerfile                   # API 镜像（不含模型权重与 MinerU）
├── .dockerignore                # 镜像构建排除清单
├── docker-compose.yml           # Milvus/MongoDB/MinIO/etcd 中间件 + CPU/GPU API profile
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板（.env 不入库）
├── DEPLOY.md                    # 部署说明（裸跑 + 容器化）
├── PROJECT_STATUS.md            # 轮次进度与设计要点
├── AGENTS.md                    # Codex 工作规则
└── README.md                    # 本文件
```

## 快速开始

### 0. 前置条件

- Python 3.12+
- Docker / Docker Compose（中间件始终容器化）
- 本地模型权重（BGE-M3 / BGE-Reranker-Large），路径在 `.env` 指定

### 1. 启动中间件

```bash
docker compose -p zhihui-wenda-system up -d
```

将启动以下容器：Milvus(19530)、MongoDB(27017)、MinIO(9000/9001)、etcd(2379)、milvus-minio(19000/19001)。

### 2. 配置环境变量

将 `.env.example` 复制为 `.env`：

```powershell
Copy-Item .env.example .env
```

必填项：`OPENAI_API_KEY`（SiliconFlow）、`BGE_M3_PATH`、`BGE_RERANKER_LARGE`、`MINERU_MODEL_SOURCE`（如需 PDF）。生产环境还需设置 `APP_API_KEY`。

### 3. 安装依赖

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 4. 启动服务（裸进程）

```powershell
# 导入 API（8000）
.venv\Scripts\python.exe -m uvicorn knowledge.api.import_router:app --port 8000

# 查询 API（8001）
.venv\Scripts\python.exe -m uvicorn knowledge.api.query_router:app --port 8001
```

### 5. 访问前端

- 导入页面：http://localhost:8000/import.html（或 / 或 /import）
- 查询页面：http://localhost:8001/chat.html（或 /）

## 配置说明（.env）

### 模型与 LLM

| 变量 | 说明 | 默认示例 |
|------|------|----------|
| `OPENAI_API_KEY` | SiliconFlow API 密钥（必填） | 你的密钥 |
| `OPENAI_API_BASE` | OpenAI 兼容地址 | `https://api.siliconflow.cn/v1` |
| `MODEL` / `LLM_DEFAULT_MODEL` | 问答主模型 | `zai-org/GLM-5.2` |
| `ITEM_MODEL` | 商品名/实体模型 | `zai-org/GLM-5.2` |
| `VL_MODEL` | 图片摘要 VLM | `Qwen/Qwen3-VL-32B-Instruct` |
| `LLM_DEFAULT_TEMPERATURE` | 生成温度 | `0.1` |
| `BGE_M3_PATH` | BGE-M3 权重目录 | `./model_cache/bge-m3` |
| `BGE_DEVICE` / `BGE_FP16` | 嵌入设备与半精度 | `cpu` / `False` |
| `BGE_RERANKER_LARGE` | Reranker 权重目录 | `./model_cache/bge-reranker-large` |
| `BGE_RERANKER_DEVICE` / `BGE_RERANKER_FP16` | Rerank 设备与半精度 | `cpu` / `False` |

### MinerU / 临时目录

| 变量 | 说明 | 默认 |
|------|------|------|
| `MINERU_MODEL_SOURCE` | MinerU 模型来源 | `modelscope` |
| `MODELSCOPE_CACHE` | modelscope 缓存目录 | `./model_cache/modelscope` |
| `HF_HOME` | HuggingFace 缓存目录 | `./model_cache/huggingface` |
| `MINERU_TIMEOUT_SECONDS` | PDF 解析超时 | `600` |
| `TEMP_DATA_DIR` | 导入临时目录 | `./temp_data` |

### 存储中间件

| 变量 | 说明 | 默认 |
|------|------|------|
| `MILVUS_URL` | Milvus 地址 | `http://localhost:19530` |
| `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` / `ENTITY_NAME_COLLECTION` | 三张集合名 | `kb_chunks` / `kb_item_names` / `kb_entity_names` |
| `MILVUS_METRIC_TYPE` / `MILVUS_MIN_COSINE_SCORE` | 度量方式 / 最低相似度阈值 | `COSINE` / `0.6` |
| `MONGO_URL` / `MONGO_DB_NAME` | MongoDB 地址与库名 | `mongodb://localhost:27017` / `kb001` |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | MinIO 连接 | `localhost:9000` / `minioadmin` / `minioadmin` |
| `MINIO_BUCKET_NAME` / `MINIO_SECURE` | 桶名与协议 | `knowledge-base` / `False` |

### 查询质量调优

| 变量 | 说明 | 默认 |
|------|------|------|
| `MAX_CONTEXT_CHARS` | 答案上下文最大字符数 | `6000` |
| `RERANK_MAX_TOP_K` / `RERANK_MIN_TOP_K` | 重排序截断上下限 | `5` / `2` |
| `RERANK_GAP_RATIO` / `RERANK_GAP_ABS` | cliff cutoff 相对/绝对 gap | `0.25` / `0.5` |
| `RRF_K` / `RRF_MAX_RESULTS` | RRF 参数/结果数 | `60` / `5` |
| `EMBEDDING_SEARCH_LIMIT` / `HYDE_SEARCH_LIMIT` | 两路检索条数 | `5` / `3` |
| `ITEM_NAME_HIGH_CONFIDENCE` / `ITEM_NAME_MID_CONFIDENCE` | 商品名对齐阈值 | `0.7` / `0.7` |
| `ITEM_NAME_MAX_OPTIONS` | 对齐候选上限 | `5` |
| `ITEM_NAME_DENSE_WEIGHT` / `ITEM_NAME_SPARSE_WEIGHT` | 商品名检索权重 | `0.5` / `0.5` |
| `QUERY_CACHE_TTL_SECONDS` / `QUERY_CACHE_MAX_ITEMS` | 查询缓存 TTL/容量 | `300` / `200` |

### 日志 / MCP / 鉴权

| 变量 | 说明 | 默认 |
|------|------|------|
| `LOG_CONSOLE_ENABLE` / `LOG_CONSOLE_LEVEL` | 控制台日志开关/级别 | `True` / `INFO` |
| `LOG_FILE_ENABLE` / `LOG_FILE_LEVEL` / `LOG_FILE_RETENTION` | 文件日志开关/级别/保留天 | `True` / `INFO` / `7 days` |
| `MCP_DASHSCOPE_BASE_URL` | 百炼 WebSearch MCP 地址 | `https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp` |
| `MCP_DASHSCOPE_API_KEY` | MCP 密钥（网络搜索，可选） | 你的密钥 |
| `APP_API_KEY` | 接口鉴权密钥（生产必填；留空跳过） | 空 |
| `ALLOWED_ORIGINS` | CORS 白名单（逗号分隔） | `http://localhost:8000,http://localhost:8001` |
| `MODEL_HOST_PATH` | Docker 挂载的宿主模型路径 | `./model_cache/models` |

## API 接口

所有业务端点需要请求头 `X-API-Key`（值为 `APP_API_KEY`），未设置密钥时跳过鉴权。`/health`、`/ready`、前端页面及 SSE `/stream/{task_id}` 免鉴权。

### 导入服务（8000）

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/health` | 否 | 存活探针 |
| GET | `/ready` | 否 | 就绪探针（Milvus/MongoDB/MinIO） |
| POST | `/upload` | 是 | 上传文档（multipart 字段 `file`，仅 .pdf/.md/.docx） |
| GET | `/status/{task_id}` | 是 | 查询导入任务状态 |
| DELETE | `/document/{file_title}` | 是 | 删除文档（清三张集合） |
| GET | `/import.html` `/import` `/front/*` | 否 | 前端页面与静态资源 |

上传响应：

```json
{ "message": "文件上传成功", "task_id": "<uuid>" }
```

任务状态响应：

```json
{
  "status": "processing",
  "done_list": ["entry_node", "document_split_node"],
  "running_list": ["bge_embedding_node"],
  "error": null
}
```

删除响应：`{ "file_title": "...", "deleted": <删除条数> }`。

### 查询服务（8001）

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| GET | `/health` | 否 | 存活探针 |
| GET | `/ready` | 否 | 就绪探针 |
| POST | `/query` | 是 | 提交问答（`is_stream=true` 时后台执行并返回 task_id） |
| GET | `/status/{task_id}` | 是 | 查询任务状态 |
| GET | `/stream/{task_id}` | 否 | SSE 流式接收答案 |
| GET | `/history/{session_id}?limit=50` | 是 | 获取对话历史 |
| DELETE | `/history/{session_id}` | 是 | 清空对话历史 |

`POST /query` 请求体：

```json
{
  "query": "xxx 的用途是什么？",
  "session_id": "可选，会话 ID",
  "is_stream": true
}
```

流式提交响应：`{ "message": "Query submitted", "session_id": "...", "task_id": "..." }`。

非流式响应：

```json
{
  "message": "处理完成",
  "session_id": "...",
  "answer": "...",
  "done_list": [],
  "running_list": [],
  "error": null,
  "image_urls": []
}
```

SSE 事件：`ready`（就绪）、`progress`（节点开始/完成）、`delta`（答案增量）、`final`（结束并携带完整结果）、`error`（失败）。

### 调用示例

```bash
curl -X POST http://localhost:8000/upload -F "file=@test.md" -H "X-API-Key: $APP_API_KEY"
curl -X POST http://localhost:8001/query -H "Content-Type: application/json" -H "X-API-Key: $APP_API_KEY" -d '{"query":"xxx","is_stream":false}'
curl http://localhost:8001/stream/<task_id>
```

## 鉴权与 CORS

- 鉴权实现：`knowledge/core/security.py` 的 `verify_api_key` 依赖，读取 `APP_API_KEY`，校验 `X-API-Key`，异常返回 401。
- `APP_API_KEY` 留空：本地开发自动跳过，启动日志会告警；生产必须显式设置。
- CORS 白名单来自 `ALLOWED_ORIGINS`（逗号分隔），默认覆盖两个前端源，不再使用 `*` + credentials 的错误组合。

## 健康检查

| 端点 | 用途 | 行为 |
|------|------|------|
| `GET /health` | liveness | 进程存活即返回 `{"status":"ok"}`，不查依赖 |
| `GET /ready` | readiness | 依次 ping Milvus(`list_collections`) → MongoDB(`ping`) → MinIO(`bucket_exists`)，任一失败返回 503 与失败项 |

容器就绪探针应指向 `/ready`，存活探针指向 `/health`。

## 任务持久化

- 任务状态写入 MongoDB `kb_tasks` 集合：`{_id: task_id, status, running_nodes, done_nodes, result, updated_at}`。
- MongoDB 不可用时自动降级为进程内存模式，接口仍响应。
- 服务重启时，所有 `processing` 状态任务被标记为 `failed`（`result.interrupted=true`），避免状态悬挂。
- 已完成任务状态持久保留，可继续查询。
- 查询缓存（`query_cache.py`）仍为进程内存（TTL 默认 300s），多实例共享缓存属后续阶段。

## 批量导入

```powershell
# 先启动导入服务，再执行：
.venv\Scripts\python.exe batch_import.py --dry-run        # 只扫描统计
.venv\Scripts\python.exe batch_import.py --workers 2 --retry 2 --retry-delay 10
```

参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--roots` | 扫描目录（可多个） | `data/import` |
| `--url` | 导入 API 地址 | `http://localhost:8000` |
| `--progress` | 断点续传进度文件 | `import_progress.txt` |
| `--log` | 批量导入日志 | `batch_import.log` |
| `--max-wait` | 单任务最长等待（秒） | `1200` |
| `--dry-run` | 只扫描不导入 | `False` |
| `--workers` | 并发线程数 | `2` |
| `--retry` | 失败重试次数 | `2` |
| `--retry-delay` | 重试间隔（秒） | `10` |
| `--exclude` | 排除的子目录 | 无 |

进度文件每行记录 `路径 + TAB(MD5)`：

```
<绝对路径>\a.md<TAB>3f7c...
<绝对路径>\b.docx<TAB>ab12...
```

- 已记录且内容 MD5 未变的文件自动跳过；内容变化会重新导入；中断后重新执行可续传。
- API 侧另有按 `file_title` 的重复预检，重复上传返回 409。

## 测试

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前 8 个测试文件共 53 例，覆盖：健康/就绪探针、导入/查询节点、SSE 生命周期、查询缓存、图片 URL 提取与黏连、最小相似度过滤、Milvus 转义、文档删除、鉴权与任务持久化。测试使用 mock，不依赖真实中间件。

## 部署

两种方式，详见 [DEPLOY.md](DEPLOY.md)：

- **裸进程**：本机启动两个 Uvicorn 进程 + `docker compose up` 中间件。
- **全容器化**：同一镜像分 CPU / GPU 两个 profile：

```bash
# CPU（默认）
docker compose --profile cpu -p zhihui-wenda-system up -d --build

# GPU（需 nvidia-container-toolkit + CUDA torch）
docker compose --profile gpu -p zhihui-wenda-system up -d --build
```

镜像不含模型权重与 MinerU：BGE 权重经 `MODEL_HOST_PATH` 挂载到容器 `/models`（只读），模型缓存与 `temp_data/`、`logs/`、`volumes/` 挂载持久化。PDF 导入在本机裸跑或后续单独 MinerU 镜像。

## 日志与运维

- 日志统一走 `knowledge/utils/logger_util.py`（loguru）：控制台 + `logs/` 文件，每日轮转、按天保留（默认 7 天）。
- 导入日志文件：`import_server.log` / `batch_import.log`；查询日志：`query_api_8001.log`。
- MinIO 桶自动创建并设置公开只读策略，图片直链可被前端直接加载。
- 常用排障：Milvus/中间件未启动时 `/ready` 返回 503；查询长时间无响应先看 `query_api_8001.log`；PDF 导入慢主要来自 MinerU CPU 解析。

## 已知边界

- PDF 导入在 CPU 上极慢（1 页约 15 分钟），有 10 分钟超时防护；GPU 部署可显著改善查询延迟，但 MinerU 仍需单独方案。
- 查询缓存、任务执行均为单进程内状态：多实例部署时需引入共享缓存与任务恢复机制。
- 商品名澄清、HyDE、重排序等参数在 `.env` 可调，默认值已按实测调优。

## 许可证

本项目为开源项目，使用 [MIT License](LICENSE) 发布。