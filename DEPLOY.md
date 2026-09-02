# 智慧问答系统部署说明

## 环境

- Python 3.12+
- Docker Compose 用于 Milvus、MongoDB、MinIO（中间件始终容器化）
- API 可裸跑或容器化
- 本地模型缓存：`BGE_M3_PATH`、`BGE_RERANKER_LARGE`

## .env 关键配置

```bash
# 项目 LLM 模型
MODEL=zai-org/GLM-5.2
ITEM_MODEL=zai-org/GLM-5.2

# MinerU 解析超时（秒），默认 600，即 10 分钟
MINERU_TIMEOUT_SECONDS=600

# 向量检索最低余弦相似度，低于该分数的结果会被过滤
MILVUS_MIN_COSINE_SCORE=0.6

# 接口鉴权（留空跳过，生产环境必须设置）
APP_API_KEY=
ALLOWED_ORIGINS=http://localhost:8000,http://localhost:8001

# Docker: 模型权重的宿主机路径（容器内挂载到 /models）
MODEL_HOST_PATH=F:/codex/Models
```

## 方式一：裸进程部署

```powershell
Copy-Item .env.example .env
# 按需填写 LLM、Milvus、MongoDB、MinIO 等配置

docker compose -p zhihui-wenda-system up -d

.venv\Scripts\python.exe -m uvicorn knowledge.api.import_router:app --port 8000
.venv\Scripts\python.exe -m uvicorn knowledge.api.query_router:app --port 8001
```

## 方式二：全容器化部署

```bash
# CPU 模式（默认）
docker compose --profile cpu -p zhihui-wenda-system up -d --build

# GPU 模式（需要 nvidia-container-toolkit + CUDA torch）
docker compose --profile gpu -p zhihui-wenda-system up -d --build
```

两种模式共用同一 Dockerfile。镜像不含模型权重和 MinerU——权重通过 volume 挂载（`MODEL_HOST_PATH`），PDF 导入需本机裸跑或单独 MinerU 镜像。

## 健康检查

| 端点 | 用途 | 检查内容 |
|------|------|---------|
| GET /health | liveness | 仅确认进程存活 |
| GET /ready | readiness | 依次 ping Milvus → MongoDB → MinIO，任一不可用返回 503 |

## 鉴权

- 所有业务端点需要 `X-API-Key` 请求头，值为 `APP_API_KEY`。
- `/health`、`/ready`、前端页面、SSE `/stream/{task_id}` 豁免。
- `APP_API_KEY` 为空时跳过鉴权（仅限本地开发；生产环境必须设置，启动日志会告警）。

## 验证

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 批量导入

先确保导入服务已启动，再执行：

```powershell
.venv\Scripts\python.exe batch_import.py
```

进度写入 `import_progress.txt`（每行记录路径和 MD5 hash），中断后重新执行即可断点续传，内容未变化的已导入文件会自动跳过。