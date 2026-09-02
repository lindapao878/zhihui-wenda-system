# 智慧问答系统 PROJECT_STATUS

> 最后更新: 2026-09-02 (轮次38)
> 用途: 供协调者快速恢复上下文，避免重复扫描项目
> 日志同步规则: 每完成一项工作就更新本文件，记录做了什么、遇到的问题及解决办法

---

## 一、已完成功能

### 导入链路 (knowledge/processor/import_process/)

| 节点 | 文件 | 功能 | 状态 |
|------|------|------|------|
| EntryNode | entry_node.py | 按 .pdf/.docx/.md 路由 is_*_read_enabled + 设置 file_title | 完成 |
| PdfToMdNode | pdf_to_md_node.py | MinerU CLI 子进程，显式传 env(MINERU_MODEL_SOURCE/MODELSCOPE_CACHE/HF_HOME/MODELSCOPE_OFFLINE)，输出路径 file_dir/stem/hybrid_auto/stem.md | 完成，CPU 模式 1 页约 15min，偶发卡死 |
| DocxToMdNode | docx_to_md_node.py | python-docx 按文档顺序解析段落+表格，转 Markdown 表格 | 完成 |
| MarkDownImageNode | md_img_node.py | 扫描图片目录 → 提取标题/前后文上下文 → VLM 生成摘要(限速15req/min) → 上传 MinIO → 替换 MD 内图片链接为远程 URL | 完成，VL_MODEL 已配置并重新导入验证 |
| DocumentSplitNode | document_split_node.py | 按标题层级分块 + RecursiveCharacterTextSplitter 长块拆分 + 短块合并 + HTML 表格线性化 + chunks.json 备份 | 完成 |
| ItemNameRecognitionNode | item_name_recognition_node.py | LLM 提取商品名 + BGE-M3 向量化 + 写入 kb_item_names(INT64 主键) + 回填 chunks 的 item_name 字段 | 完成，主键已从 VARCHAR 修为 INT64 |
| BgeEmbeddingChunksNode | bge_embedding_chunks_node.py | BGE-M3 批量混合向量化(dense+sparse)，batch_size=8，embedding 内容=item_name+content | 完成 |
| ImportMilvusNode | import_milvus_node.py | 自动建集合/索引(dense COSINE AUTOINDEX + sparse SPARSE_INVERTED_INDEX IP) + 按 file_title 去重删除旧切片 + flush | 完成 |
| main_graph.py | - | LangGraph: entry → 条件路由 → pdf/docx → md_img → split → item_name → embedding → milvus → END | 完成 |

### 查询链路 (knowledge/processor/query_process/)

| 节点 | 文件 | 功能 | 状态 |
|------|------|------|------|
| ItemNameConfirmNode | item_name_confirm_node.py | LLM 提取商品名(JSON 回退) + 历史 10 条上下文 + 向量混合检索匹配 item_name 集合 + 高(0.7)/中(0.6)置信度对齐 + 澄清问句 + 查询改写 + 回填历史 item_names | 完成 |
| VectorSearchNode | vector_search_node.py | BGE-M3 混合向量检索 + item_name in[...] 过滤(WeightedRanker 0.8/0.2) | 完成 |
| HyDeSearchNode | hyde_search_node.py | LLM 生成假设性文档 → 拼接 query+hy_doc 向量检索(item_name 过滤) | 完成 |
| McpSearchNode | mcp_search_node.py | MCPServerStreamableHttp 连百炼 WebSearch MCP(/mcp) → 解析 pages → snippet/title/url | 已完成，Streamable HTTP 协议，百炼 WebSearch MCP |
| RrfNode | rrf_node.py | RRF 融合 vector+hyde 两路(k=60)，web 文档无 chunk_id 故跳过 RRF | 完成 |
| RerankNode | rerank_node.py | BGE-Reranker 重排序 + cliff cutoff(绝对 gap 0.5 / 相对 gap 0.25，保留 2-5 条) + 融合本地+web 文档 | 完成 |
| KnowledgeGraphQueryNode | knowledge_graph_node.py | LLM 三元组提取 + 从 kb_entity_names 混合检索实体补充上下文 | 完成，related_entities 已传递到 answer prompt |
| AnswerOutputNode | answer_output_node.py | LLM 答案生成(流式 SSE delta + 非流式) + 图片 URL 提取 + MongoDB 写入历史 | 完成 |
| main_graph.py | - | LangGraph: item_name_confirm → 条件路由(有澄清答案直接输出 / 否则进检索) → multi_search 并行 fan-out → vector+hyde+mcp → join → rrf → rerank → kg → answer → END | 完成 |

### 工具层 (knowledge/utils/)

| 文件 | 功能 | 状态 |
|------|------|------|
| milvus_util.py | Milvus 客户端单例 + hybrid search 请求构造 + WeightedRanker + batch_hybrid_search | 完成 |
| bge_m3_embedding_util.py | BGE-M3 模型加载(CSR 稀疏矩阵包装器) + dense/sparse 混合向量生成 + L2 归一化 | 完成 |
| bge_rerank_util.py | FlagReranker 加载 + compute_score 封装(兼容标量/列表返回) | 完成 |
| llm_client_util.py | ChatOpenAI 单例 + JSON 模式 + invoke_llm_with_json_fallback(GLM 不支持 json_object 时回退) | 完成 |
| minio_util.py | MinIO 客户端单例 + 自动建桶 + 公开只读策略 | 完成 |
| mongo_history_util.py | MongoDB 客户端单例 + 对话历史 CRUD + item_names 回填 | 完成 |
| sse_util.py | SSE 事件队列(Queue) + ready/progress/delta/final + 断连检测 + final 后自动清理 | 完成 |
| task_util.py / task_store.py | 任务状态对外接口(task_util) + MongoDB kb_tasks 持久化(task_store)；Mongo 不可用时内存降级；重启时 processing→failed | 完成 |
| markdown_util.py | HTML 表格转文本(剥离 thead/tbody，tr→换行，td/th→管道符) | 完成 |
| logger_util.py | 统一 loguru 日志：控制台+文件双输出、每日轮转、按天保留、UTF-8、异步安全、位置精准 | 完成 |
| milvus_string_util.py | Milvus filter 表达式字符串安全转义(\\ " \r\n\t) | 完成 |

### API 层 (knowledge/api/)

| 文件 | 端口 | 功能 | 状态 |
|------|------|------|------|
| import_router.py | 8000 | POST /upload(PDF/MD/DOCX)→后台 BackgroundTask；GET /status/{task_id}；DELETE /document/{file_title}；GET /health；静态页面 | 完成 |
| query_router.py | 8001 | POST /query(流式/非流式)；GET /status；GET /stream/{task_id}(SSE)；GET/DELETE /history/{session_id} | 完成 |

### 前端 (knowledge/front/)

| 文件 | 功能 | 状态 |
|------|------|------|
| chat.html | 聊天界面：头像/气泡/进度条/图片渲染/SSE 流式/历史加载/清空确认 | 完成（轮次35修复JS） |
| import.html | 文件上传界面 | 完成（轮次35修复跳转） |

### 配置层

| 文件 | 关键配置 | 状态 |
|------|------|------|
| import_process/config.py | 分块 2000/500、图片限速 15/min、Milvus/MinIO、embedding dim 1024 | 完成 |
| query_process/config.py | rerank 5/2、rrf k=60、检索 limit、item_name 置信度 0.7/0.6 | 完成 |
| .env | 模型=zai-org/GLM-5.2、SiliconFlow API、BGE-M3 本地路径、Milvus/Mongo/MinIO 连接 | 完成，已从 GLM-4.5-Air 修正回 GLM-5.2 |
| docker-compose.yml | 5 服务：Milvus(19530)/MongoDB(27017)/MinIO(9000)/etcd(2379)/milvus-minio(19000) | 完成 |

### 提示词 (.prompt 文件化)

| 路径 | 用途 | 状态 |
|------|------|------|
| prompts/query/answer.prompt | 答案生成(已移除图片指令，图片从 reranked_docs 提取) | 完成 |
| prompts/query/hyde.prompt | HyDE 假设文档生成 | 完成 |
| prompts/query/item_name_extract.prompt | 商品名提取(JSON) | 完成 |
| prompts/query/kg_extract_system.prompt | 知识图谱系统提示 | 完成 |
| prompts/query/kg_extract.prompt | 知识图谱用户提示 | 完成 |
| prompts/upload/import_item_name_system.prompt | 导入商品名系统提示 | 完成 |
| prompts/upload/import_item_name.prompt | 导入商品名用户提示 | 完成 |
| prompts/loader.py | 按 category+name 加载 .prompt | 完成 |

### Schema

| 文件 | 内容 | 状态 |
|------|------|------|
| task_schema.py | TaskStatusResponse | 完成 |
| upload_schema.py | UploadResponse | 完成 |
| query_schema.py | QueryRequest/QueryResponse/StreamSubmitResponse | 完成 |

### 测试

| 文件 | 用例数 | 覆盖 | 状态 |
|------|------|------|------|
| tests/ (8 个文件) | 53 | health/ready/鉴权/任务持久化/导入查询节点/SSE缓存/图片/URL黏连/最小相似度过滤/文档删除 | 完成 |

### 批量导入脚本

| 文件 | 说明 | 状态 |
|------|------|------|
| batch_import.py | 合并后单一脚本：断点续传 + 文件内容 MD5 去重 + ThreadPoolExecutor(--workers 默认2) + 失败重试(--retry/--retry-delay) + 进度文件加锁 | 完成 |
- 进度：import_progress.txt 同时记录路径和 hash，兼容旧格式
- 执行：不做实际批量导入，除非用户明确要求

---

## 二、Milvus 集合设计

| 集合名 | 用途 | 主键 | 向量字段 | 状态 |
|--------|------|------|---------|------|
| kb_chunks | 文档切片存储 | chunk_id INT64 auto_id | dense(FLOAT_VECTOR 1024 COSINE) + sparse(SPARSE_FLOAT_VECTOR IP) | 完成，row_count 已有测试数据 |
| kb_item_names | 商品名存储 | pk INT64 auto_id | dense + sparse | 完成，主键已修复，row_count=1(三体简介) |
| kb_entity_names | 实体存储 | pk INT64 auto_id | dense(COSINE) + sparse(IP) + entity_name + file_title | 完成，row_count=11 |

---

## 三、本轮修复历史

| 轮次 | 修复内容 | 涉及文件 |
|------|---------|---------|
| 1 | MD 端到端冒烟测试，导入查询链路跑通 | - |
| 2 | kb_item_names 主键 VARCHAR→INT64；SiliconFlow GLM 不支持 response_format=json_object，新增 JSON 回退 | item_name_recognition_node.py, llm_client_util.py, item_name_confirm_node.py, knowledge_graph_node.py |
| 3 | MinerU 环境变量显式传递(subprocess.Popen 加 env=) | pdf_to_md_node.py |
| - | .env 模型从 GLM-4.5-Air 修正回 GLM-5.2 | .env |
| - | SSE final 事件后不再挂起(队列自动清理) | sse_util.py |
| - | 流式查询每节点推送 progress 事件 | 各查询节点 |
| - | 商品名澄清和流式答案写入 task_result | query_service.py / answer_output_node.py |

### 最近轮次记录(做了什么、遇到的问题及解决办法)

| 轮次 | 做了什么 | 遇到的问题及解决办法 | 涉及文件 |
|------|---------|----------------------|---------|
| 20 | 批量导入并发控制(--workers 默认2、ThreadPoolExecutor、进度加锁)；查询结果缓存(rewritten_query hash、TTL)；导入前按 file_title 去重预检 | 缓存命中需明确日志；改为可配置 TTL 并在命中时输出 cache hit | batch_import.py / query_cache.py / import_router.py |
| 22 | 修复缓存命中时进度条不结束 | answer_output_node 已生成答案但 done_list 缺该节点；补 add_done_task 与 final 事件 | query_service.py / answer_output_node.py / sse_util.py |
| 23 | 图片提取对齐实现：从 reranked_docs 正则提取 ![...](url) | 原实现依赖 LLM 答案中的【图片】标记，容易漏图；改为 docs 提取 + 答案提取合并去重 | answer_output_node.py / chat.html |
| 24 | 配置 VL_MODEL 并重启导入服务，重新导入含图文档 | 此前 VL_MODEL 为空导致 VLM 摘要降级为“图片描述”；重新导入后摘要为有意义中文描述 | md_img_node.py / .env |
| 25 | 修复图片 URL 黏连 | LLM 把多个 URL 放同一行，按行 split 误判为一条超长 URL；后端 re.findall + http 边界拆分，前端 extractUrlsLoose 同步 | answer_output_node.py / chat.html |
| 26 | 移除 answer.prompt 的【图片】指令 | LLM 不主动输出 URL 时无图；图片完全由 _extract_images_from_docs 提供，答案保持纯文字 | prompts/query/answer.prompt / answer_output_node.py |
| 收尾 | MinIO 直链 403 修复 | 桶为私有且无公开读策略；minio_util 初始化时设置公开只读策略(对齐实现)，重启后直链 200 | knowledge/utils/minio_util.py |
| 27 | 补齐 Milvus 字符串转义 | 原删除旧切片/商品名过滤只转义 \" 或完全未转义，商品名含 \ \n \t 时 filter 解析失败；新增 escape_milvus_string 统一转义并应用到三处 | knowledge/utils/milvus_string_util.py / import_milvus_node.py / item_name_recognition_node.py / vector_search_node.py |
| 28 | 补齐统一日志系统(loguru) | 原各模块仅 logging.getLogger 或 basicConfig，无文件输出；迁移 25 个文件到统一 logger，多行调用 % 风格残留补转 {} 风格，新增 LOG_* 配置与 logs/ 文件输出 | knowledge/utils/logger_util.py / requirements.txt / .env / 25 个日志调用文件 |
| 29 | 检索与答案质量摸底+调参 | Q3 本地 5 条全被 0.75 阈值滤掉、Q4 Git 被商品名澄清误杀、实体检索对非商品查询输出无关实体；降 MILVUS_MIN_COSINE_SCORE 0.75→0.6、ITEM_NAME_MID_CONFIDENCE 0.6→0.7、实体检索仅商品确认后启用、补 rerank/MCP/商品名日志；4 条查询重测全通过。另：沙箱网络限制致 LLM/MCP Connection error，需外网授权重启 8001 | .env / rerank_node.py / mcp_search_node.py / knowledge_graph_node.py / item_name_confirm_node.py |
| 30 | 小规模验证 batch_import.py | MD/DOCX/PDF 各 1 文件成功入库，进度文件记录 路径+MD5，断点重跑只处理未完成文件，MD5 去重 3/3 全部跳过；发现极简手写 PDF 不被 MinerU 解析（改用 reportlab 生成标准 PDF 46s 成功）、DOCX 重新生成后二进制 MD5 变化触发重导被 API 409 预检拦截（hash 校准后正常跳过） | batch_import.py / import_progress.txt / temp_data/round30_batch |
| 31 | 补齐转义遗漏 + 修复 hybrid_search limit 传递 | hyde/entity/file_import 三处 filter 改用 escape_milvus_string；execute_hybrid_search_query 默认 limit 改为 None 不隐含截断，vector/hyde/kg/item_name 四调用点显式传 limit；8 个文件 py_compile 通过，39 个单元测试全绿 | knowledge/utils/milvus_util.py / hyde_search_node.py / entity_recognition_node.py / file_import_service.py / vector_search_node.py / knowledge_graph_node.py / item_name_confirm_node.py |
| 32 | 缓存失效 + 文档删除接口 + 批量导入重试 | 导入完成后清空查询缓存并输出日志；新增 DELETE /document/{file_title} 删除三张集合；batch_import 增加 --retry/--retry-delay 重试循环。发现 pymilvus delete 返回 OmitZeroDict，len() 误当删除条数，改为读 delete_count 后 DELETE 对不存在标题返回 0；4 个文件 py_compile 通过，40 个单元测试全绿 | file_import_service.py / import_router.py / batch_import.py / tests/test_nodes.py |
| 33 | 仓库整理与文档重写 | 清理非项目素材，重写 README/.gitignore/.env.example，补充部署文档与开源许可 | .gitignore / README.md / .env.example / AGENTS.md / PROJECT_STATUS.md / LICENSE |

| 34 | 企业级硬化：接口鉴权 + 任务持久化 + 健康探针 + 容器化 | 新增 security.py(X-API-Key 鉴权+CORS 收紧)、health_util.py(/ready 就绪探针)、task_store.py(MongoDB 持久化+内存回退)；task_util.py 委托 task_store 保持 API 不变；两个 router 加 Depends(verify_api_key)+/ready+全局异常处理+init_on_startup；Dockerfile+.dockerignore；docker-compose.yml 加 CPU/GPU profile API 服务+模型权重 volume 挂载；.env/.env.example 加 APP_API_KEY/ALLOWED_ORIGINS/MODEL_HOST_PATH；DEPLOY.md 重写；13 个新单测，总计 53 例全绿 | security.py / health_util.py / task_store.py / task_util.py / import_router.py / query_router.py / Dockerfile / .dockerignore / docker-compose.yml / DEPLOY.md / .env / .env.example / test_round34_hardening.py |
| 35 | 修复 chat.html JS 语法错误 + 导入页跳转聊天 | submitQuery 的 JSON.stringify 多了一个右括号，整个脚本解析失败导致聊天页按钮失效、连接状态停在“未连接”；mdToHtml 替换为精简版，node --check 通过；import.html 顶部“前往问答”改为指向 8001/chat.html，并清除 </html> 后残留代码；8000/8001 /health 均 200 | knowledge/front/chat.html / knowledge/front/import.html / PROJECT_STATUS.md |
| 36 | README 详细化 + Dockerfile 修正 | 重写 README：目录结构/技术栈/配置说明/API 文档/鉴权与 CORS/健康探针/任务持久化/批量导入/部署/日志运维/已知边界，修正测试数 40→53 与查询任务接口（/status/{task_id}）；移除 Dockerfile 中无效的 COPY prompts/（prompts 实际在 knowledge/prompts 下）；53 例测试全绿 | README.md / Dockerfile / PROJECT_STATUS.md |
| 37 | 开源化修正 | 移除 README“私有项目，未授权不得使用”，改为 MIT License 声明；新增 LICENSE(MIT)；README 许可证段链接到 LICENSE | README.md / LICENSE / PROJECT_STATUS.md |
| 38 | 项目更名为智慧问答系统 | 全局替换掌柜智库→智慧问答系统、zhanguanzhiku→zhihui-wenda-system；本地资料/课件/临时脚本明确不入仓库；已迁移至新公开仓库 zhihui-wenda-system；旧仓库 lindapao878/- 因令牌缺少删除权限，待手动删除 | AGENTS.md / README.md / DEPLOY.md / PROJECT_STATUS.md / docker-compose.yml / 前端页面与代码注释 / .gitignore / .dockerignore |

---

## 四、待完成事项(按 AGENTS.md 优先级)

1. ✅ **related_entities 已传递到答案 prompt** — 已完成(轮次8)，answer.prompt 含【相关实体】槽位
2. ✅ **kb_entity_names 集合实现** — 已完成(轮次7)
2. ✅ **batch_import 脚本合并** — 已完成(轮次9)，单一 batch_import.py + MD5 去重
3. ✅ **MCP 网络搜索端点配置** — 已完成，Streamable HTTP 协议，百炼 WebSearch MCP
4. ✅ **测试覆盖补充** — 已完成(轮次10-12)，15 个单元测试全绿
5. **MinerU CPU 慢** — 已加 10min 超时防护；CPU 单页约 15min，待 GPU 或换方案
6. ✅ **VL_MODEL 已配置生效** — Qwen/Qwen3-VL-32B-Instruct，含图文档已重新导入并验证图片展示
7. ✅ **检索排序和答案质量调优** — 已完成(轮次29)，MILVUS_MIN_COSINE_SCORE 0.75->0.6、ITEM_NAME_MID_CONFIDENCE 0.6->0.7、实体检索仅商品确认后启用
8. **最终批量导入** — 全部功能已完善，可触发

---

## 五、设计要点

- **条件路由**：item_name_confirm 有澄清问句时直接走 answer_output，不进检索
- **并行检索**：multi_search 节点(fan-out) → vector + hyde + mcp → join 节点(barrier) → rrf
- **RRF 只融合本地两路**(vector+hyde)，web 文档无 chunk_id 直入 rerank
- **Rerank cliff cutoff**：在 min_top_k~max_top_k 区间找最大 gap(绝对或相对)截断
- **SSE 生命周期**：ready → progress(每节点开始/完成) → delta(流式 token) → final → 队列清理
- **任务管理**：MongoDB kb_tasks 持久化(task_store.py)，Mongo 不可用时内存降级；重启时 processing 任务标记为 failed(interrupted)
- **去重**：导入时按 file_title 执行 Milvus delete 再 insert，无内容级 hash 去重
- **JSON 回退**：invoke_llm_with_json_fallback 先试 json_object，400 时回退普通模式
- **BGE-M3 包装器**：_BgeM3EmbeddingWrapper 将 lexical_weights 转为 CSR 稀疏矩阵
- **embedding 内容**：导入侧 = item_name + "\n" + content；查询侧 = rewritten_query(+hy_doc)`n- **缓存失效**：导入完成后 query_cache.clear()，避免新文档入库后仍命中旧缓存`n- **Milvus 转义**：所有 filter 构造处必须用 escape_milvus_string（轮次27/31 补齐 6 处）`n- **日志统一**：全部模块使用 loguru via knowledge.utils.logger_util，不再用 logging.getLogger
