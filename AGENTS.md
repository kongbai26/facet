# Repository Guidelines

## Project Structure & Module Organization

`app/` 是 FastAPI 后端。核心模块：
- `app/main.py` — 入口，托管前端静态文件 + API 路由
- `app/api/` — 路由（auth, chat, documents, ingest, health）和依赖注入
- `app/pipeline/` — 核心管道：ingest（摄入）、retrieval（检索）、generation（生成）、recovery（启动恢复）
- `app/parsers/` — 文件解析器（PDF, TXT, MD, DOCX, HTML）
- `app/chunkers/` — 切片器（递归、语义）
- `app/providers/` — LLM / Embedding provider（OpenAI 兼容协议）
- `app/store/` — 存储层（SQLite 元数据、ChromaDB 向量、BM25 索引、session）
- `app/settings/` — 配置加载（config.yaml + .env 合并）
- `app/utils/` — 工具（JSON 日志、文件操作、jieba 分词）

`web/` 是 React + TypeScript 前端。`web/dist/` 是构建产物，由后端直接托管。

`config/config.yaml` 存非敏感配置，`config/.env` 存 API key 和密码。`data/` 是运行时数据（向量库、上传文件、SQLite），不应视为源码。

## Build and Development Commands

```bash
# 后端
pip install -e ".[dev]"                    # 安装依赖
facet dev                                   # 后端开发启动（自动重载）
facet serve                                 # 正常启动（仅监听本机）

# 前端
cd web && npm install && npm run build     # 构建（首次或代码变更时）
cd web && npm run dev                      # 前端独立开发（需要 VITE_API_BASE_URL）

```

## Coding Style & Naming Conventions

- Python 3.10+，4 空格缩进，类型注解
- 函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`
- 路由文件按功能命名：`chat_router.py`、`documents_router.py`
- 小函数、显式逻辑、模块级 docstring、简洁行内注释
- 前端：单文件组件，Tailwind CSS 样式，无 CSS modules

## Architecture Decisions

**单端口部署**：后端 FastAPI 托管 `web/dist/` 静态文件，SPA catch-all 返回 `index.html`。用户只需启动后端。

**RAG / 直答自动切换**：检索结果最高分低于 `relevance_threshold` 时自动切换为直答模式，无需用户干预。

**启动恢复**：`recover_storage()` 在服务启动时扫描所有文档，自动修复中断状态、向量缺失、重复文档等异常。

**配置外部化**：所有关键参数（阈值、轮数、截断长度等）都在 `config/config.yaml` 可调，代码中无硬编码。

## Security & Configuration

- `config/.env` 不入版本控制
- session cookie 使用 HttpOnly + SameSite=Lax
- 密码比对用 `hmac.compare_digest` 防时序攻击
- 文件名消毒防路径穿越
- 日志不打印敏感数据
