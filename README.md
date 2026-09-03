# Facet

> Knowledge, in focus.

[English](./README.md) · [简体中文](./README.zh-CN.md)

Facet is a self-hosted knowledge-base application powered by retrieval-augmented generation (RAG). It combines a FastAPI backend, SQLite metadata store, Chroma vector store, and a bundled React interface in one local service.

Users can upload documents, organize them into knowledge bases, search across them, and receive streamed answers with source references. The application is designed for a single machine or single-process deployment.

## Highlights

- Upload and parse PDF, TXT, Markdown, DOCX, HTML, CSV, and XLSX files.
- Use hybrid vector and BM25 retrieval, optional reranking, and source-grounded answers.
- Keep documents, conversations, and vector collections isolated by tenant and knowledge base.
- Choose all knowledge bases or a selected set for each conversation.
- Run the frontend and API from one FastAPI process on one port.
- Store runtime data locally in SQLite and Chroma; no cloud service is required by Facet itself.

## Requirements

- Python 3.10 or later
- An OpenAI-compatible LLM and embedding API, or mock mode for an interface-only trial

The repository includes a prebuilt frontend in `web/dist`, so normal users do **not** need Node.js or other frontend build tools.

## Quick start

```bash
# Install the Python application
pip install -e .

# Start the application
facet serve
```

Open <http://localhost:8000>. The first-run wizard collects your LLM and embedding API settings, optional reranker settings, and administrator password. It saves private values to the ignored local file `config/.env`.

For auto-reload during backend development, run:

```bash
facet dev
```

To expose the service on a trusted LAN, explicitly set the bind address:

```bash
facet serve --host 0.0.0.0 --port 8000
```

## Configuration

Non-sensitive defaults are kept in [config/config.yaml](./config/config.yaml). The first-run wizard creates and saves local private settings in `config/.env`. You can also copy [config/.env.example](./config/.env.example) and configure it before starting the service. Never commit `config/.env`.

Common environment variables:

```bash
LLM_API_BASE=
LLM_API_KEY=
LLM_MODEL_NAME=

EMBEDDING_API_BASE=
EMBEDDING_API_KEY=
EMBEDDING_MODEL_NAME=

RERANKER_API_BASE=
RERANKER_EXPECTED_MODEL=

SESSION_SECRET=
```

`LLM_API_BASE` and `EMBEDDING_API_BASE` accept OpenAI-compatible API root URLs. Do not append endpoint paths such as `/chat/completions` or `/embeddings`.

The checked-in defaults are safe for a local first run: the server listens on `127.0.0.1`, uses development mode, and creates a session secret automatically. Before serving other devices or public traffic, set `app.env` to `production`, enable `auth.cookie_secure`, use HTTPS, and define a random `AUTH_BOOTSTRAP_ADMIN_TOKEN` for the first administrator setup.

## Runtime data

Facet creates `data/` at runtime. It contains uploaded originals, the SQLite metadata database, Chroma vectors, and BM25 caches. This directory is intentionally excluded from Git. Back it up before upgrades if you need to retain documents, accounts, or conversations.

## Frontend development

The checked-in `web/dist` directory is the frontend served to normal users. Rebuild it only after editing the React source:

```bash
cd web
npm ci
npm run build
cd ..
```

Commit the regenerated `web/dist` together with the corresponding frontend source change. `web/node_modules` remains local and is ignored by Git.

## Architecture

```text
Browser
   │
   ▼
FastAPI (one port: frontend + API)
   ├── Authentication and API routes
   ├── Ingestion, retrieval, and answer generation
   ├── SQLite metadata and conversations
   ├── Chroma vectors
   └── BM25 cache and uploaded files
```

The local Chroma persistence mode supports one application process per `data/` directory. Do not run multiple workers against the same data directory.

## Project layout

```text
app/                 Python backend
config/              Safe defaults and private-config template
web/src/             React source
web/dist/            Prebuilt frontend delivered with the repository
scripts/             Operational and evaluation utilities
data/                Local runtime state, created on first run and ignored
```

## Acknowledgements

Facet is made possible by the open-source ecosystem and incorporates ideas from
many projects in the RAG community. In particular, it is built with
[FastAPI](https://fastapi.tiangolo.com/), [Chroma](https://www.trychroma.com/),
[React](https://react.dev/), [Vite](https://vite.dev/), and
[Tailwind CSS](https://tailwindcss.com/); its document and model integrations
also rely on projects such as [pypdf](https://pypdf.readthedocs.io/) and the
[OpenAI Python SDK](https://github.com/openai/openai-python).

Facet retains the license notices required by its bundled dependencies. See
[NOTICE](./NOTICE), `pyproject.toml`, and `web/package-lock.json`. If you spot
an attribution that should be added or corrected, please open an issue before
redistributing the project.

## License

Facet is licensed under [Apache License 2.0](./LICENSE). See [NOTICE](./NOTICE) for attribution notices.
