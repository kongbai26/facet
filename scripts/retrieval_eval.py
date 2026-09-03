"""Local retrieval evaluation harness.

This script builds a temporary corpus, ingests it with the real embedding
backend, and reports retrieval quality for vector-only, hybrid baseline,
and hybrid exact-boosted retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from app.evaluation.manifest import build_evaluation_manifest
from app.evaluation.metrics import find_ranks as _quality_find_ranks
from app.evaluation.metrics import retrieval_metrics
from app.evaluation.release_gate import validate_minimum_metrics
from app.pipeline.ingest import ingest_document
from app.pipeline.retrieval import retrieve
from app.providers.embedding.registry import get_embedding_provider
from app.providers.reranker.registry import get_reranker
from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_dimension
from app.settings import loader
from app.settings.loader import load_config
from app.store.bm25_store import BM25Store
from app.store.document_store import DocumentStore
from app.store.vector_store import VectorStore


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    filename: str
    content: str


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_doc_ids: str | tuple[str, ...]
    category: str

    def __post_init__(self) -> None:
        if isinstance(self.expected_doc_ids, str):
            object.__setattr__(self, "expected_doc_ids", (self.expected_doc_ids,))

    @property
    def expected_doc_id(self) -> str:
        """Compatibility label for the first expected document."""
        return self.expected_doc_ids[0]


CORPUS: list[CorpusDoc] = [
    CorpusDoc(
        doc_id="session-secret",
        filename="session_secret.md",
        content=(
            "SESSION_SECRET 是浏览器会话签名密钥，和管理员密码解耦。\n"
            "如果配置缺失或仍然是占位值，系统首次启动会自动生成一个随机 secret 并写回 .env。\n"
            "会话签名与校验只依赖 SESSION_SECRET，和 LLM 密码没有直接关系。\n"
        ),
    ),
    CorpusDoc(
        doc_id="legacy-password",
        filename="legacy_password.md",
        content=(
            "AUTH_PASSWORD、ADMIN_PASSWORD、AUTH_API_KEY 这些旧字段只作为迁移输入。\n"
            "真正的管理员密码会写入 SQLite 的 auth_credentials 表，运行时不再依赖 .env 里的明文密码。\n"
            "如果系统检测到遗留明文配置，会在状态里发出告警，但不会继续使用这些字段做登录真值。\n"
        ),
    ),
    CorpusDoc(
        doc_id="mock-llm",
        filename="mock_llm.md",
        content=(
            "当没有可用的真实 LLM 时，系统可以切到 mock 模式。\n"
            "mock provider 会返回稳定、可预测的中文占位回答，用于本地演示和流程验证。\n"
            "这能让用户在模型不可达时仍然把文档上传、检索和界面流程跑通。\n"
        ),
    ),
    CorpusDoc(
        doc_id="embedding-base",
        filename="embedding_base.md",
        content=(
            "EMBEDDING_API_BASE 只应该填写 OpenAI 兼容接口的根地址。\n"
            "如果路径末尾带了 /embeddings，加载器会自动剥掉这个后缀，避免配置重复拼接。\n"
            "Embedding provider 负责把文本转成向量，供检索和摄入使用。\n"
        ),
    ),
    CorpusDoc(
        doc_id="delete-reingest",
        filename="delete_reingest.md",
        content=(
            "删除文档时，系统会删除向量和上传目录。\n"
            "重新摄入时必须仍然保留原始上传文件；如果源文件丢了，会返回 document_source_missing。\n"
            "失败重试时，原文件不能被提前删掉，否则用户没有办法重新摄入。\n"
        ),
    ),
    CorpusDoc(
        doc_id="system-status",
        filename="system_status.md",
        content=(
            "系统状态接口会汇总 auth.initialized、legacy_password_detected、session_secret_ok、llm.mode、embedding.configured 等信息。\n"
            "它的目标是让用户快速知道当前哪里正常、哪里只是处于模拟模式或缺少配置。\n"
            "状态页不访问外部网络，只做本地健康汇总。\n"
        ),
    ),
    CorpusDoc(
        doc_id="chat-routing",
        filename="chat_routing.md",
        content=(
            "聊天路由会在 RETRIEVE、REUSE 和 DIRECT 三条路径之间做判断。\n"
            "如果用户在追问上一轮已引用的资料，系统倾向于 REUSE；需要新事实时会走 RETRIEVE；纯闲聊或元问题会走 DIRECT。\n"
            "这类路由设计能减少不必要的检索和模型调用。\n"
        ),
    ),
    CorpusDoc(
        doc_id="hybrid-search",
        filename="hybrid_search.md",
        content=(
            "混合检索会把向量召回和 BM25 结果一起融合。\n"
            "当前实现默认使用 RRF 融合，也保留了加权融合的实现。\n"
            "BM25 对含有明确关键词的查询更有帮助，而向量检索更擅长语义相近的问法。\n"
        ),
    ),
    CorpusDoc(
        doc_id="retrieval-config",
        filename="config.yaml.md",
        content=(
            "retrieval.exact_match.enabled 控制精确命中增强。\n"
            "lexical_metadata_fields 默认包含 filename、file_stem、extension、doc_id、tenant_slug。\n"
            "这个配置主要用于技术资料和代码文档的精确召回。\n"
        ),
    ),
    CorpusDoc(
        doc_id="source-missing",
        filename="document_source_missing.md",
        content=(
            "当文档重新摄入时如果上传原文件已经丢失，系统会返回 document_source_missing。\n"
            "这类错误通常意味着用户需要重新上传原文件，而不是继续重试索引。\n"
            "删除和重新摄入是两条不同的流程。\n"
        ),
    ),
    CorpusDoc(
        doc_id="network-error",
        filename="econnrefused.md",
        content=(
            "ECONNREFUSED 常见于网络连接被拒绝，比如 Embedding 服务或 LLM 服务端口没有启动。\n"
            "这种错误和检索逻辑本身无关，通常是外部依赖不可达。\n"
            "诊断时应先看服务地址和端口是否可连通。\n"
        ),
    ),
    CorpusDoc(
        doc_id="hybrid-distractor",
        filename="hybrid_distractor.md",
        content=(
            "混合检索可以把多种信号一起用，但如果参数配置不当，也可能让错误答案排在前面。\n"
            "有些实现会把 BM25 权重开得过高，导致语义相近但事实不对的文档被误排到前面。\n"
            "因此我们要看的是稳定的召回指标，而不是只看某一个样本。\n"
        ),
    ),
    CorpusDoc(
        doc_id="file-validation",
        filename="file_validation.md",
        content=(
            "空文件上传会直接被拒绝，超出大小限制的文件会返回 413。\n"
            "不支持的扩展名也会被拦截，避免后续解析器白跑一趟。\n"
            "这些校验能让系统在异常输入下保持可解释的失败信息。\n"
        ),
    ),
    CorpusDoc(
        doc_id="large-ingest",
        filename="large_ingest.md",
        content=(
            "大文件摄入会先解析，再切片，最后批量写入向量库。\n"
            "一份很大的文本也可以被拆成许多 chunks，逐批请求 embedding，避免一次性占满内存。\n"
            "如果摄入成功，文档状态会从 processing 变成 ready。\n"
        ),
    ),
]


QUERIES: list[EvalQuery] = [
    EvalQuery("如果 SESSION_SECRET 没配会怎样", "session-secret", "exact_identifier"),
    EvalQuery("auth_credentials 表里存什么", "legacy-password", "exact_identifier"),
    EvalQuery("config.yaml 里 exact_match 怎么配置", "retrieval-config", "exact_filename"),
    EvalQuery("document_source_missing 怎么处理", "source-missing", "exact_error_code"),
    EvalQuery("ECONNREFUSED 代表什么问题", "network-error", "exact_error_code"),
    EvalQuery("没有真实模型时系统怎么还能跑起来", "mock-llm", "semantic_paraphrase"),
    EvalQuery("embedding 的 base url 为什么不能带 /embeddings", "embedding-base", "semantic_paraphrase"),
    EvalQuery("删除文档以后还能重新摄入吗", "delete-reingest", "semantic_paraphrase"),
    EvalQuery("状态页会显示哪些 auth 和 llm 信息", "system-status", "long_context"),
    EvalQuery("特别大的文本是怎么摄入的", "large-ingest", "long_context"),
    EvalQuery("聊天什么时候会复用上一轮资料", "chat-routing", "near_duplicate"),
    EvalQuery("向量检索和 BM25 是怎么一起工作的", "hybrid-search", "near_duplicate"),
    EvalQuery("混合检索默认用什么融合方式", "hybrid-search", "near_duplicate"),
    EvalQuery("这套系统怎样提示遗留明文密码", "legacy-password", "semantic_paraphrase"),
    EvalQuery("模型不可达时还能不能先给一个可预测的回答", "mock-llm", "semantic_paraphrase"),
]


def _prepare_temp_config(
    repo_root: Path,
    workdir: Path,
    *,
    hybrid_enabled: bool,
    exact_match_enabled: bool,
    reranker_enabled: bool,
) -> Path:
    config_dir = workdir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = repo_root / "config" / "config.yaml"
    env_path = repo_root / "config" / ".env"
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    config["auth"]["enabled"] = False
    config["retrieval"]["hybrid"]["enabled"] = hybrid_enabled
    config["retrieval"]["exact_match"]["enabled"] = exact_match_enabled
    config["retrieval"]["query_rewrite"]["enabled"] = False
    config["retrieval"]["decision"]["mode"] = "off"
    config["retrieval"]["reranker"]["enabled"] = reranker_enabled
    if not reranker_enabled:
        config["retrieval"]["reranker"]["mode"] = "off"
    config["storage"]["upload_dir"] = str(workdir / "uploads")
    config["storage"]["metadata_db"] = str(workdir / "metadata.db")
    config.setdefault("database", {})["sqlite_path"] = str(workdir / "metadata.db")
    config["vectorstore"]["persist_dir"] = str(workdir / "chroma")
    config["retrieval"]["hybrid"]["bm25_cache_dir"] = str(workdir / "bm25_cache")
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    env_text = env_path.read_text(encoding="utf-8")
    if "LLM_PROVIDER=" not in env_text:
        env_text += "\nLLM_PROVIDER=mock\n"
    else:
        lines = []
        replaced = False
        for line in env_text.splitlines():
            if line.startswith("LLM_PROVIDER="):
                lines.append("LLM_PROVIDER=mock")
                replaced = True
            else:
                lines.append(line)
        if not replaced:
            lines.append("LLM_PROVIDER=mock")
        env_text = "\n".join(lines) + "\n"
    (config_dir / ".env").write_text(env_text, encoding="utf-8")
    return config_dir


def _write_corpus_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def _ingest_corpus(
    settings,
    corpus: Iterable[CorpusDoc],
    *,
    tenant_id: str,
    tenant_slug: str,
):
    document_store = DocumentStore(settings.storage.metadata_db)
    embedding_provider = get_embedding_provider(settings.embedding, settings.vectorstore)
    vector_store = VectorStore(settings.vectorstore, settings.embedding.openai.model_name)
    bm25_store = BM25Store(
        settings.retrieval.hybrid.bm25_cache_dir,
        lexical_metadata_fields=settings.retrieval.exact_match.lexical_metadata_fields,
    )

    for doc in corpus:
        file_path = Path(settings.storage.upload_dir) / doc.doc_id / doc.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_corpus_file(file_path, doc.content)
        content_bytes = doc.content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        await document_store.create(
            doc.doc_id,
            doc.filename,
            len(content_bytes),
            content_hash=content_hash,
            embedding_model=settings.embedding.openai.model_name,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
        )
        await document_store.update_status(doc.doc_id, "processing")
        await ingest_document(
            file_path,
            doc.doc_id,
            settings,
            embedding_provider,
            vector_store,
            document_store,
            bm25_store,
        )

    if settings.retrieval.hybrid.enabled:
        dimension = await resolve_embedding_dimension(embedding_provider)
        await bm25_store.ensure_ready(
            vector_store,
            get_tenant_rag_collection_name(
                settings.vectorstore.collection_prefix,
                settings.embedding.openai.model_name,
                tenant_slug=tenant_slug,
                embedding_dimension=dimension,
            ),
        )

    return document_store, embedding_provider, vector_store, bm25_store


def _find_rank(results: list[dict], expected_doc_id: str) -> int | None:
    for index, result in enumerate(results, start=1):
        if result.get("metadata", {}).get("doc_id") == expected_doc_id:
            return index
    return None


def _find_ranks(results: list[dict], expected_doc_ids: tuple[str, ...]) -> list[int]:
    """Return one-based ranks for all expected documents."""
    return _quality_find_ranks(results, expected_doc_ids)


def _metrics(rows: list[dict]) -> dict[str, float]:
    return retrieval_metrics(rows, top_k=3)


async def _evaluate_mode(
    label: str,
    settings,
    document_store,
    embedding_provider,
    vector_store,
    bm25_store,
    *,
    tenant_id: str,
    tenant_slug: str,
    reranker=None,
):
    rows = []
    dimension = await resolve_embedding_dimension(embedding_provider)
    collection_name = get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        settings.embedding.openai.model_name,
        tenant_slug=tenant_slug,
        embedding_dimension=dimension,
    )

    for item in QUERIES:
        results = await retrieve(
            item.query,
            embedding_provider,
            vector_store,
            settings.retrieval,
            document_store,
            bm25_store,
            collection_name=collection_name,
            tenant_id=tenant_id,
            reranker=reranker,
        )
        ranks = _find_ranks(results, tuple(item.expected_doc_ids))
        rank = min(ranks) if ranks else None
        top_ids = [result.get("metadata", {}).get("doc_id", "") for result in results[:3]]
        rows.append(
            {
                "query": item.query,
                "expected": item.expected_doc_id,
                "expected_ids": list(item.expected_doc_ids),
                "rank": rank,
                "ranks": ranks,
                "category": item.category,
                "top3": top_ids,
                "top_ids": top_ids,
                "top_retrieval_score": results[0].get("retrieval_score", results[0].get("score", 0)) if results else None,
                "top_rank_score": results[0].get("rank_score") if results else None,
                "top_source": results[0].get("score_source") if results else None,
                "top_exact_bonus": results[0].get("exact_match_bonus") if results else None,
                "rerank_used": any(result.get("rerank_score") is not None for result in results),
            }
        )

    overall = _metrics(rows)
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        by_category[category] = _metrics(category_rows)

    return {
        "label": label,
        **overall,
        "by_category": by_category,
        "rows": rows,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval quality.")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep the temporary benchmark directory.")
    parser.add_argument(
        "--include-reranker",
        action="store_true",
        help="Add a real HTTP reranker A/B mode using the configured endpoint.",
    )
    parser.add_argument("--output", type=Path, help="Write the complete versioned JSON report to this path.")
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fail when the reranked mode (or final mode) is below a minimum metric; repeatable.",
    )
    args = parser.parse_args()

    minimums: dict[str, float] = {}
    for item in args.min_metric:
        name, separator, raw_value = item.partition("=")
        if not name or not separator:
            parser.error("--min-metric must use NAME=VALUE")
        try:
            minimums[name.strip()] = float(raw_value)
        except ValueError:
            parser.error("--min-metric value must be numeric")

    repo_root = Path(__file__).resolve().parent.parent
    workdir = Path(tempfile.mkdtemp(prefix="rag-retrieval-eval-", dir="/private/tmp"))
    try:
        tenant_id = "eval-tenant"
        tenant_slug = "default"

        results = []
        modes = [
            (False, False, "vector"),
            (True, False, "hybrid_baseline"),
            (True, True, "hybrid_exact_boosted"),
        ]
        if args.include_reranker:
            modes.append((True, True, "hybrid_reranked"))

        for hybrid_enabled, exact_match_enabled, label in modes:
            mode_dir = workdir / label
            if mode_dir.exists():
                shutil.rmtree(mode_dir, ignore_errors=True)
            mode_dir.mkdir(parents=True, exist_ok=True)
            config_dir = _prepare_temp_config(
                repo_root,
                mode_dir,
                hybrid_enabled=hybrid_enabled,
                exact_match_enabled=exact_match_enabled,
                reranker_enabled=label == "hybrid_reranked",
            )
            loader.CONFIG_DIR = config_dir
            settings = load_config()
            document_store, embedding_provider, vector_store, bm25_store = await _ingest_corpus(
                settings,
                CORPUS,
                tenant_id=tenant_id,
                tenant_slug=tenant_slug,
            )
            reranker = get_reranker(settings.retrieval.reranker) if label == "hybrid_reranked" else None
            if reranker is not None:
                initializer = getattr(reranker, "initialize", None)
                if callable(initializer):
                    await initializer()
            outcome = await _evaluate_mode(
                label,
                copy.deepcopy(settings),
                document_store,
                embedding_provider,
                vector_store,
                bm25_store,
                tenant_id=tenant_id,
                tenant_slug=tenant_slug,
                reranker=reranker,
            )
            dimension = await resolve_embedding_dimension(embedding_provider)
            status = reranker.status() if reranker is not None and hasattr(reranker, "status") else {}
            outcome["manifest"] = build_evaluation_manifest(
                settings,
                embedding_dimension=dimension,
                reranker_status=status,
            )
            outcome["reranker_status"] = status
            results.append(outcome)

        summary = [
            {
                "mode": item["label"],
                "hit@1": round(item["hit@1"], 4),
                "hit@3": round(item["hit@3"], 4),
                "mrr": round(item["mrr"], 4),
                "context_precision@3": round(item["context_precision@3"], 4),
                "context_recall@3": round(item["context_recall@3"], 4),
                "reranker_active": bool(item.get("reranker_status", {}).get("active")),
            }
            for item in results
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        for item in results:
            print(f"\n[{item['label']}]")
            for row in item["rows"]:
                print(
                    f"- rank={row['rank']!s:<4} expected={row['expected']:<18} "
                    f"retrieval={row['top_retrieval_score']!s:<8} rank_score={row['top_rank_score']!s:<8} "
                    f"source={row['top_source']!s:<8} exact={row['top_exact_bonus']!s:<5} "
                    f"category={row['category']:<18} query={row['query']} top3={row['top3']}"
                )

            print("  categories:")
            for category, metrics in item["by_category"].items():
                print(
                    f"    - {category}: hit@1={metrics['hit@1']:.3f} hit@3={metrics['hit@3']:.3f} "
                    f"mrr={metrics['mrr']:.3f} precision@3={metrics['context_precision@3']:.3f} "
                    f"recall@3={metrics['context_recall@3']:.3f}"
                )

        final_metrics = results[-1]
        failures = validate_minimum_metrics(final_metrics, minimums)
        report = {
            "schema_version": 1,
            "benchmark": "synthetic_retrieval_regression",
            "summary": summary,
            "results": results,
            "gate": {"minimums": minimums, "passed": not failures, "failures": failures},
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"report: {args.output}")
        if failures:
            print("release gate failed: " + "; ".join(failures))
            return 2

        return 0
    finally:
        if args.keep_workdir:
            print(f"kept workdir: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
