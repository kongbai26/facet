# Facet

[English](./README.md) · [简体中文](./README.zh-CN.md)

> Knowledge, in focus.

Facet 是本地部署的知识库系统，由 RAG 驱动。后端 FastAPI + ChromaDB + SQLite，前端 React + Tailwind CSS。支持文档上传、混合检索、流式回答、来源卡片、持久化会话和会话级知识库范围。

本项目的默认存储是单机 SQLite + Chroma，适合单实例部署，不支持多个 worker 共享同一份 `data/`。

当前版本已经按公司内网使用场景收口为单实例、单 worker、tenant-aware 的部署形态：

- 文档、会话和向量集合都按 `tenant` 逻辑隔离。
- 现有前端继续走 RAG API。
- 内部系统可以直接走独立 Vector API，不需要经过前端。
- 启动时会自动补默认工作区、回填旧数据的 tenant/知识库元数据，并执行轻量恢复。
- 每个会话可检索全部知识库，或选择 1～N 个知识库；范围会同时约束向量召回、BM25、历史复用和重试。
- 对已建立父级证据索引的短文档，可选全文模式；服务端会按当前模型上下文预算拒绝截断式“全文”。

前端构建后由后端托管，**单端口运行**，无需额外启动前端服务。

## 快速开始

```bash
# 1. 从源码仓库安装运行依赖（当前为 source-first 发布方式）
pip install -e .
# 开发项目时改用：pip install -e ".[dev]"

# 2. 启动（前端已预构建，无需安装 Node.js）
facet dev
# 浏览器打开 http://localhost:8000，首次通过 Web 完成管理员初始化

# 正常/生产启动（不自动重载，仅监听本机）
facet serve

# 需要让局域网访问时，显式指定监听地址
facet serve --host 0.0.0.0 --port 8000
```

仓库已经包含预构建的 `web/dist`，普通用户不需要安装 Node.js。首次打开网页时，向导会填写对话模型、向量模型、可选重排模型和管理员密码，并将私密配置保存到本机忽略的 `config/.env`。只有修改前端源码时，才在项目根目录执行：

```bash
cd web && npm ci && npm run build && cd ..
```

安装完成后，`facet --help` 可查看全部选项；也可使用 `python -m app` 作为等价入口。

`.env` 常用项：

```bash
LLM_API_BASE=http://...         # LLM API 地址
EMBEDDING_API_BASE=http://...   # Embedding API 地址
LLM_API_KEY=                    # OpenAI 兼容服务若不需要，可留空
EMBEDDING_API_KEY=              # OpenAI 兼容服务若不需要，可留空
SESSION_SECRET=                 # 可留空，首次启动会自动生成并写回
# AUTH_BEARER_TOKEN=            # 可选的固定管理员令牌；Agent 更推荐使用有 scope/过期时间的 API key
```

本地源码开发时，留空的 `SESSION_SECRET` 会被写回 `config/.env`。默认配置仅监听 `127.0.0.1`，适合本机首次初始化；公开部署前请改为 production、启用 HTTPS Cookie，并设置随机的 `AUTH_BOOTSTRAP_ADMIN_TOKEN`。

首次启动流程：

1. 启动服务
2. 打开 Web，进入首次初始化向导，填写模型服务和管理员密码
3. 生产环境额外设置随机的 `AUTH_BOOTSTRAP_ADMIN_TOKEN`，并在向导中输入
4. 初始化完成后自动登录；生产环境应删除首次初始化令牌

说明：

- 管理员密码真值保存在 `data/metadata.db` 的 SQLite 凭据表中，不再以明文写入 `.env`
- 生产环境首次初始化必须设置 `AUTH_BOOTSTRAP_ADMIN_TOKEN`，并在 `POST /api/v1/auth/setup` 中携带 `Authorization: Bearer <token>`；这可防止公网首访者抢先创建管理员。
- 如果检测到历史 `AUTH_PASSWORD` / `AUTH_API_KEY` / `ADMIN_PASSWORD`，系统会在启动时自动迁移到 SQLite，并在系统状态里提示遗留明文配置
- `data/metadata.db` 需要持久化，否则管理员凭据、会话和文档元数据都会丢失
- 部署时必须持久化 `data/`，每次启动通过 `.env` 或环境变量提供同一组配置；首次启动后仍需访问 Web 完成初始化
- 暂时没有真实 LLM 时，可以把 `LLM_PROVIDER` 设为 `mock`，或者保持 `LLM_API_KEY` / `LLM_API_BASE` 为空，系统会自动使用模拟回答
- 切换 embedding 模型后，只要原文件还在，系统会在启动时自动把受影响文档加入后台重建队列，不需要重新上传
- 本地 Chroma 持久化模式保持单实例、单 worker，不要直接使用多个 Uvicorn worker 共享同一 `data/chroma`。

## 架构

```
┌─────────────────────────────────────────────────┐
│                   浏览器                         │
│        http://localhost:8000 (前端 + API)        │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│         FastAPI (单端口 / tenant-aware)          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Auth API │ │ Doc API  │ │ Chat API │ │ Vec API │ │
│  └──────────┘ └──────────┘ └────────┬─┘ └──┬─────┘ │
│                                     │           │
│  ┌──────────────────────────────────▼─────────┐ │
│  │              Pipeline                      │ │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐  │ │
│  │  │ Ingest   │ │ Retrieve │ │ Generate  │  │ │
│  │  │ 解析+切片 │ │ 向量+BM25 │ │ RAG/直答  │  │ │
│  │  └──────────┘ └──────────┘ └───────────┘  │ │
│  └────────────────────────────────────────────┘ │
│                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ ChromaDB │ │ SQLite   │ │ 文件系统          │ │
│  │ 向量存储  │ │ 元数据    │ │ 上传文件          │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────┘
```

### 核心流程

**文档摄入**：上传 → 解析（PDF/TXT/MD/DOCX/HTML）→ 切片（递归/语义）→ Embedding → ChromaDB

**聊天问答**：
```
用户提问 → 检索决策/历史复用
        → 会话范围过滤（全部知识库或指定的多个知识库）
        → 向量召回 + BM25 召回 + RRF 融合
        → 小范围多库时按知识库并行召回候选，再统一重排
        → 受控意图扩展/结构化排序
        → 可选 HTTP reranker 精排
        → 父子块去重 + 证据门控
        → 无足够证据：NO_EVIDENCE（auto/knowledge）或按模式直答
        → 有足够证据：RAG 生成 + 引用校验
```

### 部署边界

这套项目现在推荐的使用方式是：

1. 一个 FastAPI 服务实例。
2. 一个 SQLite 元数据数据库。
3. 一个本地向量库和 BM25 缓存目录。
4. 一个后台 ingest worker。
5. 一个默认工作区，必要时再按 tenant 做逻辑隔离。

如果是公司内部本地使用，不建议一开始就把它当成多实例集群来运维。
前端不需要单独启动开发服务器。服务运行时只使用仓库中预构建的 `web/dist`，普通用户无需安装 Node.js；`web/node_modules` 仅用于修改前端时的本地构建，可以在构建完成后删除。

生产环境使用 HTTPS 时保持 `auth.cookie_secure: true`。默认不开放跨域来源；如果前端与 API 不是同源部署，应将 `server.cors_origins` 显式设置为实际的 HTTPS 前端地址。管理员密码登录默认每个来源 5 分钟最多允许 10 次失败尝试。

### 交付前清理

`data/` 是运行时状态目录，包含 SQLite 元数据、Chroma 向量库、BM25 缓存、上传文件和启动恢复备份。交付新环境时可以清空它，让系统在首次启动时重新初始化。

### 启动恢复

服务启动时自动检测并修复异常状态：

| 检测项 | 处理 |
|--------|------|
| 旧数据缺少 tenant 元数据 | 回填默认工作区 tenant |
| 摄入中断（status=processing, 无 chunks） | 清理残留，标记 failed |
| 摄入完成但状态未更新（status=processing, 有 chunks + 向量） | 自动恢复为 ready |
| 删除中断（status=deleting） | 继续删除 |
| 文件缺失（status=ready, 文件不在） | 标记 failed |
| 向量丢失（status=ready, chunks>0, 向量=0） | 标记 failed |
| 模型不一致（embedding_model 变化） | 自动加入后台重建队列；无法自动重建时标记 failed |
| 重复文档（相同 content_hash） | 保留最优，删除重复 |

当启动恢复发现元数据异常时，会先备份 `data/metadata.db`。默认只保留最近 `7` 天且最多 `5` 个备份，防止 `data/` 持续膨胀；这两个值可通过 `config/config.yaml` 的 `app.metadata_backup_retention_days` 和 `app.metadata_backup_max_files` 调整。

上线前和维护窗口可执行：

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
python -m scripts.backup_data --output ./data/backups
```

`/health` 只表示进程存活；`/ready` 检查本地持久化目录是否可用。备份命令不包含 `config/.env`，执行时应暂停写入。

## 配置

所有配置在 `config/config.yaml`，敏感信息在 `config/.env`。

### chat — 聊天参数

```yaml
chat:
  history_limit: 8              # 带入上下文的历史消息轮数
  history_truncate: 4000        # 每条历史消息最大字符数
  title_max_length: 32          # 会话标题最大字符数
  max_concurrent_streams: 3     # 同一进程最多并行检索/生成的会话数
  rewrite_context_truncate: 1000  # 查询改写时历史上下文截断字符数
  answer_validation_max_retries: 0  # 0：确定性清理后保留可用回答，避免格式重写拖慢或失败
  stream_validate_before_emit: false  # 默认实时转发；会话可单独选择严格校验
```

### retrieval — 检索参数

```yaml
retrieval:
  top_k: 5                      # 返回结果数
  score_threshold: 0.2          # 向量检索分数阈值（低于此值的结果被过滤）
  relevance_threshold: 0.35     # 过滤低分候选；auto/knowledge 证据不足时返回 NO_EVIDENCE
  hybrid:
    enabled: true                # 混合检索开关
    vector_weight: 0.6           # 向量检索权重
    bm25_weight: 0.4             # BM25 检索权重
    fusion_method: "rrf"         # 融合方法（rrf | weighted）
    rrf_k: 60                    # RRF 常数
    bm25_top_k: 20               # BM25 候选数量
  query_rewrite:
    enabled: true                # LLM 查询改写开关，补足同义表达和章节标题措辞
    strategy: "expand"           # 改写策略（expand | decompose）
  support_grader:
    mode: "always"               # LLM 对已召回候选做语义支持判定；auto 仅在词面不确定时调用
    max_candidates: 5             # 单次最多判定的候选数
  definition_query_expansion:
    enabled: true
    max_added_queries: 1
  intent_query_expansion:
    enabled: true
    max_added_queries: 1
    aliases:
      核心用户: 目标用户
      用户群体: 目标用户
      服务对象: 目标用户
      目标人群: 目标用户
  exact_match:
    lexical_metadata_fields:
      - filename
      - file_stem
      - extension
      - doc_id
      - tenant_slug
      - block_kind
      - section_title
      - heading_path
      - table_headers
      - source_anchor
  reranker:
    enabled: true
    mode: "auto"                  # off | auto | on
    provider: "http"
    candidate_pool_size: 40
    probe_timeout_seconds: 10      # 整次后台/主动探测的总时限
    request_timeout: 5             # 在线精排请求超时；失败后自动降级
  multi_knowledge_base:
    strategy: "adaptive"         # global | parallel_candidates | adaptive
    max_selected_knowledge_bases: 6
    max_parallel_knowledge_bases: 3
    branch_timeout_seconds: 20
```

### llm / embedding — 模型配置

```yaml
llm:
  provider: "openai"            # 没有真实模型时可改成 "mock"
  api_base: "http://localhost:11434/v1"
  model_name: "qwen2.5:7b"
  temperature: 0.2
  max_tokens: 2048
  context_window: 8192
  request_timeout: 30            # 非流式逻辑请求的总超时预算（不是每次重试各 30 秒）
  stream_first_token_timeout_seconds: 60  # 流式请求等待首个服务端事件的上限
  stream_idle_timeout_seconds: 45         # 已开始后相邻服务端事件的最大静默间隔
  stream_total_timeout_seconds: 180       # 单个流式请求的总时长上限
  max_attempts: 3
  attempt_timeout_ratios: [0.5, 0.8, 1.0]  # 每次取总预算的比例和当时剩余时间中的较小值
  retry_backoff_seconds: 1
  thinking:
    mode: "off"                 # auto=服务默认，on=开启，off=关闭；设置页可持久化覆盖
    profiles:
      "qwen3.5-*":              # 精确名称优先，其后按配置顺序匹配 glob
        transport: "openai"     # openai | ollama | vllm | vllm_template | qwen
        effort: "medium"        # 开启时使用的推理档位
        efforts: ["none", "low", "medium", "high", "xhigh", "max"]
                                 # 输入框直接展示并提交的原生档位

embedding:
  provider: "openai"
  openai:
    api_base: "http://localhost:11434/v1"
    model_name: "qwen3-embedding-8b"
    max_tokens: 8192
    concurrent_requests: 3
    request_timeout: 60
    preflight_timeout_seconds: 15
    # 上传或重建索引前的 Embedding 可用性检查超时；失败会直接提示，不会留下 processing 任务。
    max_attempts: 3
    retry_backoff_seconds: 1
```

`thinking.profiles` 描述的是服务端协议，不是根据模型名猜出的能力。同一个 Qwen 模型由不同网关托管时，可能分别识别顶层 `reasoning_effort`、`chat_template_kwargs.enable_thinking` 或模板内的推理档位。项目只对命中配置表的模型发送思考参数，未知模型不注入私有字段。输入框的档位列表直接来自当前 profile 的 `efforts`，显示和提交的都是 `none`、`low` 等原生值；切换结果按会话保存。路由、查询改写、标题与连通性探测固定关闭思考，以保护它们很小的输出预算。

### chunking — 切片参数

```yaml
chunking:
  chunk_size: 512
  chunk_overlap: 100
  separators: ["\n\n", "\n", "。", ".", " "]
  semantic: false                # 语义切片开关
  overlap_sentences: 2           # 语义切片时的句子级 overlap
```

### storage / auth / server

```yaml
storage:
  upload_dir: "./data/uploads"
  metadata_db: "./data/metadata.db"
  max_file_size_mb: 50
  max_uncompressed_archive_size_mb: 250  # DOCX/XLSX 解压总大小上限
  max_archive_members: 10000             # DOCX/XLSX 内部文件数上限
  max_batch_files: 20
  allowed_extensions: [".pdf", ".txt", ".md", ".docx", ".html", ".htm"]

auth:
  enabled: true
  session_ttl_seconds: 604800    # 7 天
  session_cookie_name: "rag_session"

app:
  enable_startup_recovery: true
  startup_dependency_timeout_seconds: 15  # 已有数据时外部模型启动探测总超时
  auto_reindex_on_embedding_change: true
  index_generation_retention: 1
  index_generation_retention_days: 2
  index_retention_cleanup_interval_seconds: 0  # 0 表示启动/切换时清理
  vector_storage_orphan_grace_seconds: 86400
  metadata_backup_retention_days: 7
  metadata_backup_max_files: 5

queue:
  backend: "db"
  autostart_worker: true         # 启动后自动拉起后台 worker
  worker_poll_interval_seconds: 2
  knowledge_base_delete_wait_timeout_seconds: 60
  knowledge_base_delete_poll_interval_seconds: 0.5

server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: []               # 同源单端口部署保持为空；跨域时填实际 HTTPS 前端来源
  log_level: "INFO"
```

知识库删除等待参数只作用于目标知识库的关联写任务：最长等待 `60` 秒，每 `0.5` 秒检查一次。两项均可按部署负载调整，不需要改代码。

## 认证

- 浏览器认证使用 HttpOnly session cookie
- 首次初始化通过 Web 向导创建本地管理员密码
- 管理员密码仅保存哈希值，真值在 SQLite，不在 `.env`
- `SESSION_SECRET` 与密码解耦；本地源码开发时缺失会自动生成并写回 `config/.env`，生产/容器应显式配置固定随机值
- 修改密码后，当前会话保留，同一管理员的其他浏览器会话会全部失效
- API bearer token 仅接受显式配置的 `AUTH_BEARER_TOKEN` / `AUTH_BOOTSTRAP_ADMIN_TOKEN`
- 当 `auth.enabled: false` 时，系统仍会使用默认工作区 tenant，避免前后端和向量接口出现空 tenant 的分支差异。

## API 接口

### Auth

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/auth/bootstrap` | 返回认证三态：未初始化 / 未登录 / 已登录 |
| `POST` | `/api/v1/auth/setup` | 首次初始化管理员密码，并直接建立 session |
| `POST` | `/api/v1/auth/session` | 登录，创建 session |
| `GET` | `/api/v1/auth/me` | 检查登录态 |
| `POST` | `/api/v1/auth/password` | 修改管理员密码 |
| `POST` | `/api/v1/auth/logout` | 退出登录 |

### System

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 轻量存活检查 |
| `GET` | `/ready` | 本地持久化目录 readiness 检查 |
| `GET` | `/api/v1/system/status` | 状态汇总（含 LLM 连通性探测）；只展示模型名，不回传服务地址或本地路径 |
| `POST` | `/api/v1/system/checks` | 手动主动诊断（LLM / Embedding / Reranker / 向量库 / SQLite）；异常详情会脱敏 |

### Documents

文档接口按当前 tenant 过滤。上传、删除、重摄入都只作用于当前工作区的数据。

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/documents/upload` | 上传文件，返回 202，后台摄入 |
| `GET` | `/api/v1/documents` | 列出文档 |
| `GET` | `/api/v1/documents/revision` | 轻量文档版本，用于 Agent/前端发现外部变更 |
| `GET` | `/api/v1/documents/queue` | 查询摄入、重建和删除任务及可重试失败状态 |
| `GET` | `/api/v1/documents/{doc_id}` | 文档详情 |
| `DELETE` | `/api/v1/documents/{doc_id}` | 删除文档（含向量和文件） |
| `POST` | `/api/v1/documents/{doc_id}/reingest` | 将文档加入重建队列（保留原文件，后台重建向量索引） |
| `GET` | `/api/v1/knowledge-bases` | 列出可用于会话范围选择的知识库 |
| `GET` | `/api/v1/knowledge-bases/revision` | 轻量知识库版本，用于 Agent/前端发现外部变更 |
| `POST` | `/api/v1/knowledge-bases` | 新建知识库 |
| `DELETE` | `/api/v1/knowledge-bases/{kb_id}` | 异步删除知识库及其文档、索引、向量、缓存和上传文件；默认知识库受保护 |

### Chat

会话和检索同样按 tenant 作用域运行。默认工作区会在启动时自动补齐，因此本地单实例部署不需要手工初始化多套租户。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/conversations` | 列出会话 |
| `GET` | `/api/v1/conversations/{id}` | 获取会话和消息 |
| `PUT` | `/api/v1/conversations/{id}/retrieval-scope` | 设置会话知识库范围、可选全文文档和回答模式 |
| `DELETE` | `/api/v1/conversations/{id}` | 删除会话 |
| `DELETE` | `/api/v1/conversations` | 按当前 tenant 原子批量删除会话，body 为 `{"conversation_ids": ["..."]}` |
| `POST` | `/api/v1/chat` | RAG 对话（支持 SSE 流式） |
| `POST` | `/api/v1/tools/knowledge-search` | 规范的只读知识检索工具（不生成） |
| `POST` | `/api/v1/query` | 兼容的纯检索接口，与知识检索工具共用同一实现 |
| `POST` | `/api/v1/chat/completions` | OpenAI 兼容接口 |

### 独立 Vector API

内部系统可以直接调用 `/api/v1/vectors/...`，不需要走前端。这个接口和 RAG API 共享同一套 FastAPI 实例，但职责分开：

1. 普通 collection 可以由向量接口直接创建、写入和查询。
2. RAG 系统自用 collection 对 Vector API 完全隐藏，不能直接读取、写入或删除；Agent 检索知识库必须调用 `/api/v1/tools/knowledge-search`。
3. 系统自用 collection 的命名规则由后端统一生成，非默认 tenant 会带 `__rag_tenant__` 标记。
4. `api-keys` 管理接口是 tenant-scoped 的，显式传入不存在的 `tenant_id` 会返回 `404 Tenant not found`。

### Agent 调用契约

Agent 使用 Bearer API key，并按最小权限授予 scope：

| 能力 | 规范接口 | 最小 scope |
|------|----------|------------|
| 只读检索知识库证据 | `POST /api/v1/tools/knowledge-search` | `rag:read` |
| 上传、删除、重摄入文档 | `/api/v1/documents/...` | `rag:read`, `rag:write` |
| 删除知识库 | `DELETE /api/v1/knowledge-bases/{kb_id}` | `rag:write` |
| 调用纯 LLM | `POST /api/v1/chat/completions` | `llm:invoke` |
| 独立业务向量存取 | `/api/v1/vectors/...` | `vectors:read` / `vectors:write` |

`/api/v1/tools/knowledge-search` 只接受文本查询：`query` 为 1–4000 个字符，显式选择的知识库必须属于当前 tenant 且处于 active 状态。它只返回证据，不调用生成模型。

`/api/v1/chat/completions` 是有边界的文本版 OpenAI 兼容接口：最多 100 条消息、单条内容最多 16000 个字符、总内容最多 50000 个字符；`max_tokens` 不能超过服务配置的 LLM `context_window`。当前不支持 tools、函数调用、视觉内容或模型切换，传入未支持字段会返回 `422`。

服务 API key 的 `requests_per_minute` 和 `daily_quota` 会在 SQLite 中原子计数，跨进程生效；触发限制返回 `429` 和 `Retry-After`。未单独配置时，可通过 `rate_limit` / `quota` 的默认配置启用限制。

启用 `quota` 后，独立 Vector collection 和知识库 namespace 也分别受 `default_max_collections`、`default_max_namespaces` 限制；超限创建请求返回 `429`。

文档摄入、候选索引重建、文档删除和知识库删除可能返回 `202`；Agent 应轮询 `/api/v1/documents/queue`，不要假定请求返回后数据已经完成清理。知识库创建会在一个事务内完成名称冲突、namespace 配额和元数据写入；知识库进入删除态与创建删除任务也在同一事务内完成。知识库进入 `deleting` 后不再接受新的文档写入、重摄入或检索；清理完成后会删除其所有文档、上传文件、父块、索引代际、向量和 BM25 缓存，并从会话范围中移除。

上传、删除文档、换模型和手动重建共用同一个知识库候选索引生命周期。构建期间再次发生源文档变化时，当前任务会记录重跑请求，并在成功落账前基于最新快照再执行一次；删除失败会持久化为 `delete_failed`，即使任务历史被清空仍可重试。前端通过两个 revision 接口发现 Agent 侧的新建、删除和状态变化。完整机器可读契约见 `/openapi.json`。

### 运行时目录

| 目录 | 用途 | 是否需要持久化 |
|------|------|----------------|
| `data/chroma` | Chroma 向量库 | 是 |
| `data/metadata.db` | SQLite 元数据和会话 | 是 |
| `data/uploads` | 上传原文件 | 视业务需要 |
| `data/bm25_cache` | BM25 缓存 | 否，但保留可加快首次检索 |
| `data/metadata.*.bak` | 启动恢复自动备份 | 否，按保留策略自动清理 |
| `data/backups` | 手工发布备份归档 | 建议复制到独立备份位置 |

### 流式聊天协议

```json
// 请求
{
  "conversation_id": "optional",
  "edit_from_message_id": "optional",
  "message": { "content": "问题" },
  "stream": true,
  "grounding_mode": "auto",
  "knowledge_scope": "selected",
  "knowledge_base_ids": ["kb-a", "kb-b"]
}
```

`grounding_mode` 可选 `auto`（默认自动路由）、`knowledge`（严格知识库）或 `assistant`（通用助手）。聊天页会将它保存为当前会话设置；模式只影响之后的新消息，不会改写历史答案。

新会话可传 `knowledge_scope`：`all`（默认，`knowledge_base_ids` 必须为空）或 `selected`（`knowledge_base_ids` 至少一个）。每个选中的知识库都会成为硬检索边界；小范围选择时，系统按知识库并行取候选，并只做一次全局重排。`full_context_doc_id` 只允许在恰好选中一个知识库时使用，且文档必须属于该库；其完整父级证据文本必须落在当前模型的安全上下文预算内，否则接口会拒绝保存这个模式，而不会悄悄截断全文。

```txt
// SSE 事件流
event: meta       → 会话元数据（conversation_id, message_ids, title）
event: sources    → 检索来源（可能为空数组）
event: message    → 文字增量 {"content": "..."}
event: done       → 结束
event: error      → 错误（如有）
```

## 前端功能

- **单侧边栏**：导航 + 会话列表合并，文档/聊天两个页面
- **欢迎页**：居中输入框 + 动态推荐问题（根据文档库状态）
- **聊天**：用户气泡（右）+ 助手消息（左，带头像），流式输出
- **来源卡片**：助手消息下方折叠展示，流结束后显示
- **会话管理**：搜索、时间分组、批量删除（带确认弹窗）、单个删除
- **文档管理**：新建/删除知识库，上传、删除、重新摄入文档，状态轮询
- **认证**：首次初始化向导、登录页、设置页改密
- **设置页**：系统状态、主动诊断、遗留明文密码告警

## 目录结构

```txt
app/
├── api/            # 路由（auth, chat, documents, ingest, health）
├── chunkers/       # 切片器（递归、语义）
├── parsers/        # 文件解析器（PDF, TXT, MD, DOCX, HTML）
├── pipeline/       # 核心管道（route, evidence, retrieval, generation, recovery）
├── providers/      # LLM / Embedding provider（OpenAI 兼容）
├── settings/       # 配置加载（yaml + .env 合并）
├── store/          # 存储层（SQLite, ChromaDB, BM25）
├── utils/          # 工具（日志、文件操作、分词）
├── config.py       # 配置单例
└── main.py         # FastAPI 入口（托管前端 + API）

config/
├── config.yaml     # 非敏感配置
├── .env            # 敏感配置（不入版本控制）
└── .env.example    # .env 模板

web/
├── src/
│   ├── App.tsx     # 应用壳、页面与聊天编排
│   ├── components/ # 可独立加载的重型展示组件
│   ├── lib/        # API 封装、SSE 解析、工具函数
│   ├── types.ts    # TypeScript 类型
│   └── index.css   # Tailwind CSS 样式
├── dist/           # 构建产物（后端托管）
└── package.json

data/
├── chroma/         # ChromaDB 向量数据
├── uploads/        # 上传的原始文件
├── bm25_cache/     # BM25 索引缓存
└── metadata.db     # SQLite 元数据（文档、会话、消息、session）

```

## 鸣谢

Facet 受益于开源生态，也借鉴了 RAG 社区中许多项目的实现思路。项目以
[FastAPI](https://fastapi.tiangolo.com/)、[Chroma](https://www.trychroma.com/)、
[React](https://react.dev/)、[Vite](https://vite.dev/) 和
[Tailwind CSS](https://tailwindcss.com/) 为基础；文档解析与模型集成还使用了
[pypdf](https://pypdf.readthedocs.io/) 和
[OpenAI Python SDK](https://github.com/openai/openai-python) 等项目。

Facet 会保留随附依赖所需的许可证和版权声明。请参阅 [NOTICE](./NOTICE)、
`pyproject.toml` 与 `web/package-lock.json`。如果发现应补充或更正的署名，
请在重新分发前提交 issue。

## 许可证

Facet 使用 [Apache License 2.0](./LICENSE)。PDF 文本解析采用 BSD-3-Clause 的
`pypdf`；项目不再依赖 AGPL/商业双许可证的 PyMuPDF。
