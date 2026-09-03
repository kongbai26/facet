"""Pydantic 配置模型"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsModel(BaseModel):
    """Reject misspelled settings instead of silently using a default."""

    model_config = ConfigDict(extra="forbid")


class OpenAIEmbeddingConfig(SettingsModel):
    api_base: str = ""
    api_key: str = ""
    model_name: str = "text-embedding-3-small"
    auto_detect_model_name: bool = False
    max_tokens: int = 8191          # 模型上下文窗口（token）
    concurrent_requests: int = 3    # 并发请求数（Ollama 建议 1）
    request_timeout: int = 60       # 批量 embedding 逻辑请求总预算（秒）
    query_timeout_seconds: int = Field(default=8, ge=1, le=120)
    tokenizer_timeout_seconds: int = Field(default=10, ge=1, le=120)
    preflight_timeout_seconds: int = Field(default=15, ge=1, le=120)
    max_attempts: int = Field(default=3, ge=1, le=5)
    retry_backoff_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    # OpenAI-compatible APIs do not expose a standard batch-token limit.  A
    # deployment can set the server's verified capacity here; unset means the
    # client only limits item count.
    max_batch_tokens: int | None = Field(default=None, ge=1)
    # Empty means auto-detect the llama.cpp-compatible /tokenize endpoint.
    tokenizer_endpoint: str = ""


class EmbeddingConfig(SettingsModel):
    provider: str = "openai"
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)


ThinkingEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
AnswerQualityMode = Literal["normal", "enhanced"]
DEFAULT_THINKING_EFFORTS: tuple[ThinkingEffort, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


class LLMThinkingProfile(SettingsModel):
    """Translate one model pattern into an endpoint-specific thinking dialect."""

    transport: Literal["openai", "ollama", "vllm", "vllm_template", "qwen"]
    effort: ThinkingEffort = "medium"
    # Native values exposed to the composer. This belongs to the endpoint
    # profile because different gateways may support different levels.
    efforts: List[ThinkingEffort] = Field(
        default_factory=lambda: list(DEFAULT_THINKING_EFFORTS)
    )

    @model_validator(mode="after")
    def validate_efforts(self):
        self.efforts = list(dict.fromkeys(self.efforts))
        if not self.efforts:
            raise ValueError("llm.thinking.profiles.*.efforts 不能为空")
        if "none" not in self.efforts:
            raise ValueError("llm.thinking.profiles.*.efforts 必须包含 none 以支持关闭思考")
        if self.effort not in self.efforts:
            raise ValueError("llm.thinking.profiles.*.effort 必须包含在 efforts 中")
        return self


class LLMThinkingConfig(SettingsModel):
    # ``auto`` omits all controls and lets the upstream server decide.
    mode: Literal["auto", "on", "off"] = "off"
    # Exact names win; ordered glob patterns are checked afterwards.  The
    # transport is configured explicitly because a model name cannot reveal
    # whether it is served by Ollama, vLLM, LM Studio, or another gateway.
    profiles: Dict[str, LLMThinkingProfile] = Field(default_factory=dict)


class LLMConfig(SettingsModel):
    provider: str = "openai"
    api_base: str = ""
    api_key: str = ""
    model_name: str = "qwen2.5:7b"
    # Models that users may select per conversation. The effective default
    # model_name is always included at runtime, so environment overrides do
    # not need to duplicate this list.
    selectable_models: List[str] = Field(default_factory=list)
    auto_detect_model_name: bool = False
    temperature: float = 0.2
    max_tokens: int = 2048
    request_timeout: int = Field(default=30, ge=1, le=600)
    # Streaming has different failure semantics from a one-shot completion:
    # a long answer that is actively arriving is healthy, while a stream that
    # never produces an event or goes silent is not.  Keep those budgets
    # explicit instead of treating ``request_timeout`` as a hard cap on the
    # entire visible answer.
    stream_first_token_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    stream_idle_timeout_seconds: float = Field(default=45.0, ge=1.0, le=600.0)
    stream_total_timeout_seconds: float = Field(default=180.0, ge=5.0, le=3600.0)
    max_attempts: int = Field(default=3, ge=1, le=5)
    attempt_timeout_ratios: List[float] = Field(default_factory=lambda: [0.5, 0.8, 1.0])
    retry_backoff_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    context_window: int = 4096      # 模型上下文窗口（token）
    connectivity_timeout_seconds: float = Field(default=10.0, ge=1.0, le=120.0)
    thinking: LLMThinkingConfig = Field(default_factory=LLMThinkingConfig)

    @model_validator(mode="after")
    def validate_attempt_timeout_ratios(self):
        """Keep retry count and per-attempt timeout policy explicit."""
        self.selectable_models = list(dict.fromkeys(
            model_name.strip()
            for model_name in self.selectable_models
            if model_name.strip()
        ))
        if len(self.attempt_timeout_ratios) != self.max_attempts:
            raise ValueError("llm.attempt_timeout_ratios 的数量必须与 llm.max_attempts 一致")
        if any(ratio <= 0 or ratio > 1 for ratio in self.attempt_timeout_ratios):
            raise ValueError("llm.attempt_timeout_ratios 的每项必须大于 0 且不超过 1")
        if list(self.attempt_timeout_ratios) != sorted(self.attempt_timeout_ratios):
            raise ValueError("llm.attempt_timeout_ratios 必须按从小到大排列")
        return self

    def available_model_names(self) -> List[str]:
        """Return the configured per-conversation model allowlist."""
        return list(dict.fromkeys(
            model_name
            for model_name in [self.model_name.strip(), *self.selectable_models]
            if model_name
        ))


class ChunkingConfig(SettingsModel):
    # Legacy flat-chunk options. They apply only when parent_child_enabled is
    # false and are kept so existing deployments continue to load.
    chunk_size: int = 512
    chunk_overlap: int = 100
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", "。", ".", " "])
    semantic: bool = False          # 语义切片开关
    overlap_sentences: int = 2      # 语义切片时的句子级 overlap 数量
    parent_child_enabled: bool = False
    parent_max_tokens: int = 1024
    # Target and continuity are expressed in tokens in both indexing modes.
    # child_max_tokens/child_overlap_tokens are legacy aliases retained for
    # configuration compatibility while installations migrate.
    child_target_tokens: int | None = Field(default=512, ge=1)
    child_continuity_tokens: int | None = Field(default=None, ge=0)
    child_max_tokens: int = 512
    child_overlap_tokens: int = 40
    # A true value is a deployment policy: ingestion must be supplied with a
    # tokenizer that is verified against the embedding service.  The current
    # OpenAI-compatible protocol does not standardise a token-count endpoint,
    # so the default remains explicit best-effort compatibility.
    require_exact_tokenizer: bool = False
    # Extra token budget reserved in compatible mode for model-added BOS/EOS
    # and similar special tokens that an HTTP API may add implicitly.
    fallback_token_reserve: int = Field(default=8, ge=0, le=128)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_child_target(cls, value):
        """Use the old child_max_tokens key when the new key is absent."""
        if isinstance(value, dict):
            normalized = dict(value)
            if "child_target_tokens" not in normalized and "child_max_tokens" in normalized:
                normalized["child_target_tokens"] = normalized["child_max_tokens"]
            return normalized
        return value


class HybridRetrievalConfig(SettingsModel):
    enabled: bool = False           # 混合检索开关
    vector_weight: float = 0.6      # 向量检索权重
    bm25_weight: float = 0.4        # BM25 检索权重
    fusion_method: Literal["weighted", "rrf"] = "rrf"
    rrf_k: int = 60                 # RRF 常数
    bm25_top_k: int = 20            # BM25 候选数量
    bm25_cache_dir: str = "./data/bm25_cache"  # BM25 缓存目录
    bm25_search_limit_multiplier: int = 3       # BM25 搜索候选扩展倍率


class QueryRewriteConfig(SettingsModel):
    enabled: bool = False           # 查询改写开关
    strategy: Literal["expand", "decompose"] = "expand"
    max_rewrites: int = 3           # 最大改写数
    timeout_seconds: int = 5        # 单条 LLM 改写调用超时（秒）
    expand_max_tokens: int = Field(default=200, ge=1)
    decompose_max_tokens: int = Field(default=300, ge=1)


class DefinitionQueryExpansionConfig(SettingsModel):
    # 为“实体是什么”补充产品定位意图，不调用 LLM，也不改变证据门控的原问题锚点。
    enabled: bool = True
    max_added_queries: int = Field(default=1, ge=0, le=1)
    title_candidate_limit: int = Field(default=5, ge=1, le=20)


class IntentQueryExpansionConfig(SettingsModel):
    # 受控的领域同义问法扩展，不调用 LLM，也不改变证据门控的原问题。
    enabled: bool = True
    max_added_queries: int = Field(default=1, ge=0, le=1)
    aliases: Dict[str, str] = Field(default_factory=lambda: {
        "核心用户": "目标用户",
        "用户群体": "目标用户",
        "服务对象": "目标用户",
        "目标人群": "目标用户",
    })


class RelationQueryConfig(SettingsModel):
    enabled: bool = True
    support_query_limit: int = 1        # 额外补充多少个非主实体支持查询
    diversify_top_n: int = 3            # 关系问法前 N 个结果内尽量覆盖多实体证据
    promote_missing_evidence: bool = True


class EvidenceSupportGraderConfig(SettingsModel):
    """Controls semantic support checks after retrieval."""

    mode: Literal["off", "auto", "always"] = "auto"
    timeout_seconds: float = Field(default=3.0, ge=0.1, le=30.0)
    max_candidates: int = Field(default=5, ge=1, le=8)
    max_tokens: int = Field(default=96, ge=32, le=256)
    max_candidate_chars: int = Field(default=1200, ge=100, le=4000)


class RerankerConfig(SettingsModel):
    enabled: bool = False
    # off: 始终使用旧链路；auto: 自动重探；on: 只在启动时探测。故障时都回退旧链路。
    mode: Literal["off", "auto", "on"] = "auto"
    provider: str = "http"
    api_base: str = ""
    expected_model: str = ""
    strict_model_match: bool = False
    # 4B 级 reranker 的首次加载/排队可能明显慢于普通 HTTP 健康检查。
    startup_timeout: int = Field(default=10, ge=1, le=60)
    # 整次能力探测的总时限，避免多个兼容端点各自消耗完整超时。
    probe_timeout_seconds: float = Field(default=10.0, ge=0.1, le=60.0)
    request_timeout: int = Field(default=5, ge=1, le=60)
    candidate_pool_size: int = Field(default=40, ge=1)
    # None 表示候选阶段不因分数淘汰，只由 candidate_pool_size 限制规模。
    candidate_prefilter_threshold: float | None = None
    # HTTP adapter 内部统一输出概率分数；若上游返回 raw logit 则设置为 logit。
    score_mode: Literal["probability", "logit", "auto"] = "probability"
    # Rerank 后低于该分数的候选不会进入 LLM 上下文。0 保留旧的“只看
    # 最佳候选”行为；部署配置可按实际模型校准更高门槛。
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Optional quality stages should fail fast: after one online request
    # failure, use the baseline until a background capability probe succeeds.
    circuit_breaker_failures: int = Field(default=1, ge=1)
    reprobe_interval_seconds: int = Field(default=60, ge=1)


class MultiKnowledgeBaseRetrievalConfig(SettingsModel):
    """Controls bounded retrieval across an explicitly selected KB set."""

    # global: one filtered candidate pool; parallel_candidates: one candidate
    # pool per selected KB; adaptive chooses the latter only for a bounded
    # selected set.  "all" intentionally always remains global.
    strategy: Literal["global", "parallel_candidates", "adaptive"] = "adaptive"
    max_selected_knowledge_bases: int = Field(default=6, ge=1, le=32)
    max_parallel_knowledge_bases: int = Field(default=3, ge=1, le=12)
    branch_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)


class RetrievalDecisionConfig(SettingsModel):
    mode: Literal["auto", "heuristic", "llm_gate", "always", "off"] = "auto"
    reuse_last_sources: bool = True     # 是否复用上一轮 grounded sources
    llm_timeout_seconds: int = 5        # LLM gate 超时时间
    llm_max_tokens: int = Field(default=48, ge=8, le=128)
    fallback_mode: Literal["retrieve", "heuristic", "direct"] = "heuristic"
    log_decisions: bool = True          # 是否记录决策日志


class AnswerQualityProfileConfig(SettingsModel):
    """Bounded semantic quality budget for one user-selectable mode."""

    max_corrective_retrievals: int = Field(default=1, ge=0, le=3)
    max_context_expansions: int = Field(default=1, ge=0, le=3)
    context_expansion_radius: int = Field(default=1, ge=1, le=4)
    evidence_judge_max_retries: int = Field(default=1, ge=0, le=3)
    evidence_timeout_seconds: float = Field(default=12.0, ge=1.0, le=120.0)
    evidence_max_candidates: int = Field(default=5, ge=1, le=12)
    evidence_max_tokens: int = Field(default=256, ge=96, le=1024)
    evidence_max_candidate_chars: int = Field(default=1400, ge=200, le=6000)
    semantic_answer_verification: bool = True
    answer_verification_max_retries: int = Field(default=1, ge=0, le=3)
    answer_verification_timeout_seconds: float = Field(default=20.0, ge=1.0, le=180.0)
    answer_verification_max_tokens: int = Field(default=256, ge=64, le=1024)
    max_answer_repairs: int = Field(default=1, ge=0, le=3)
    turn_timeout_seconds: float = Field(default=180.0, ge=5.0, le=3600.0)


class AnswerQualityConfig(SettingsModel):
    """Quality modes share semantics and differ only in bounded recovery depth."""

    default_mode: AnswerQualityMode = "normal"
    normal: AnswerQualityProfileConfig = Field(default_factory=AnswerQualityProfileConfig)
    enhanced: AnswerQualityProfileConfig = Field(
        default_factory=lambda: AnswerQualityProfileConfig(
            max_corrective_retrievals=2,
            max_context_expansions=2,
            context_expansion_radius=2,
            evidence_judge_max_retries=2,
            evidence_timeout_seconds=20.0,
            evidence_max_candidates=8,
            evidence_max_tokens=384,
            evidence_max_candidate_chars=1800,
            answer_verification_max_retries=2,
            answer_verification_timeout_seconds=30.0,
            answer_verification_max_tokens=384,
            max_answer_repairs=2,
            turn_timeout_seconds=300.0,
        )
    )

    def profile(self, mode: AnswerQualityMode) -> AnswerQualityProfileConfig:
        return self.enhanced if mode == "enhanced" else self.normal


class RetrievalExactMatchConfig(SettingsModel):
    enabled: bool = True
    identifier_bonus: float = 0.12
    error_code_bonus: float = 0.18
    filename_bonus: float = 0.18
    doc_id_bonus: float = 0.12
    phrase_bonus: float = 0.1
    lexical_metadata_fields: List[str] = Field(
        default_factory=lambda: [
            "filename",
            "file_stem",
            "extension",
            "doc_id",
            "tenant_slug",
            "block_kind",
            "section_title",
            "heading_path",
            "table_headers",
            "source_anchor",
        ]
    )


class ChatConfig(SettingsModel):
    history_limit: int = 8              # 带入上下文的历史消息轮数
    history_truncate: int = 4000        # 每条历史消息最大字符数
    title_max_length: int = 32          # 会话标题最大字符数
    auto_title_enabled: bool = True
    auto_title_timeout_seconds: int = 5
    auto_title_max_input_length: int = 200
    rewrite_context_truncate: int = 1000  # 查询改写时历史上下文截断字符数
    history_rewrite_max_tokens: int = Field(default=256, ge=1)
    answer_quality: AnswerQualityConfig = Field(default_factory=AnswerQualityConfig)
    # 格式校验默认只记录并做确定性清理；需要人工选择才会额外调用模型重写。
    answer_validation_max_retries: int = 0
    # 仅保留给无资料直答和旧调用；知识回答由质量模式强制先校验再输出。
    stream_validate_before_emit: bool = False
    stream_output_chunk_chars: int = Field(default=24, ge=1, le=256)  # 校验后每次向前端发送的字符数
    stream_output_chunk_delay_ms: int = Field(default=20, ge=0, le=500)  # 校验后分片发送间隔，单位毫秒
    # Local OpenAI-compatible servers commonly have one effective inference
    # slot. Operators with verified parallel capacity can raise this value.
    max_concurrent_streams: int = Field(default=1, ge=1, le=32)
    # A busy model should return a retryable result instead of leaving a
    # browser stream in an unbounded semaphore queue.
    generation_queue_wait_timeout_seconds: float = Field(default=12.0, ge=1.0, le=300.0)
    # Includes slot wait, routing, retrieval, answer generation and optional
    # validation. Provider request budgets remain a lower-level safeguard.
    turn_timeout_seconds: float = Field(default=180.0, ge=5.0, le=3600.0)


class RetrievalConfig(SettingsModel):
    top_k: int = 5
    score_threshold: float = 0.2        # 向量检索分数阈值
    relevance_threshold: float = 0.35   # 过滤低分候选；knowledge 证据不足时拒答，auto 仅保留非引用的部分上下文后回答
    candidate_multiplier: int = 3       # 向量检索候选扩展倍率
    max_effective_queries: int = 2      # 每次检索最多实际执行的 query 数
    skip_standalone_rewrite_for_short_query: bool = True  # 短 query 直接跳过 standalone 改写
    standalone_rewrite_min_length: int = 12               # 判定为短 query 的最小字符数
    log_retrieval_timings: bool = True                    # 是否记录检索耗时
    decision: RetrievalDecisionConfig = Field(default_factory=RetrievalDecisionConfig)
    exact_match: RetrievalExactMatchConfig = Field(default_factory=RetrievalExactMatchConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)
    definition_query_expansion: DefinitionQueryExpansionConfig = Field(default_factory=DefinitionQueryExpansionConfig)
    intent_query_expansion: IntentQueryExpansionConfig = Field(default_factory=IntentQueryExpansionConfig)
    relation_query: RelationQueryConfig = Field(default_factory=RelationQueryConfig)
    support_grader: EvidenceSupportGraderConfig = Field(default_factory=EvidenceSupportGraderConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    multi_knowledge_base: MultiKnowledgeBaseRetrievalConfig = Field(
        default_factory=MultiKnowledgeBaseRetrievalConfig,
    )


class IngestConfig(SettingsModel):
    batch_size: int = 100


class ObservabilityConfig(SettingsModel):
    log_vector_query_timing: bool = True
    log_bm25_timing: bool = True
    log_bm25_rebuild_reason: bool = True
    log_query_rewrite_skips: bool = True
    log_retrieval_trace: bool = False


class VectorStoreConfig(SettingsModel):
    backend: str = "chroma"             # 当前仅实现 chroma
    mode: str = "persistent"            # 当前仅实现本地 persistent
    persist_dir: str = "./data/chroma"
    collection_prefix: str = "rag"


class StorageConfig(SettingsModel):
    upload_dir: str = "./data/uploads"
    metadata_db: str = "./data/metadata.db"
    max_file_size_mb: int = 50
    max_uncompressed_archive_size_mb: int = Field(default=250, ge=1, le=2048)
    max_archive_members: int = Field(default=10_000, ge=1, le=100_000)
    max_batch_files: int = Field(default=20, ge=1, le=1000)
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".csv", ".xlsx"]
    )


class AuthConfig(SettingsModel):
    enabled: bool = True
    password_hash: str = ""           # legacy env migration input only
    admin_password: str = ""
    session_secret: str = ""
    bootstrap_admin_token: str = ""
    password: str = ""                # legacy migration input only
    bearer_token: str = ""
    session_ttl_seconds: int = 86400
    session_sliding_expiration_enabled: bool = True
    session_cookie_name: str = "rag_session"
    cookie_secure: bool = False
    cookie_samesite: str = "lax"
    login_rate_limit_attempts: int = Field(default=10, ge=1, le=100)
    login_rate_limit_window_seconds: int = Field(default=300, ge=1, le=3600)


class AppRuntimeConfig(SettingsModel):
    env: str = "development"            # development | production
    enable_startup_recovery: bool = True
    startup_dependency_timeout_seconds: int = Field(default=15, ge=1, le=120)
    auto_reindex_on_embedding_change: bool = True
    auto_rebuild_index_on_profile_change: bool = True
    index_generation_retention: int = Field(default=1, ge=0)
    index_generation_retention_days: int = Field(default=7, ge=0)
    index_retention_cleanup_interval_seconds: int = Field(default=0, ge=0)
    vector_storage_orphan_grace_seconds: int = Field(default=86_400, ge=0)
    ingest_job_history_retention_days: int = Field(default=30, ge=0)
    metadata_backup_retention_days: int = 7
    metadata_backup_max_files: int = 5


class DatabaseConfig(SettingsModel):
    backend: str = "sqlite"             # 当前仅实现 sqlite
    sqlite_path: str = "./data/metadata.db"


class RateLimitConfig(SettingsModel):
    enabled: bool = False
    backend: str = "sqlite"
    default_requests_per_minute: int = 120


class QueueConfig(SettingsModel):
    backend: str = "db"                 # 当前仅实现 db
    autostart_worker: bool = True
    worker_poll_interval_seconds: int = Field(default=2, ge=1)
    max_attempts: int = Field(default=5, ge=1)
    lock_timeout_seconds: int = Field(default=300, ge=3)
    knowledge_base_delete_wait_timeout_seconds: int = Field(default=60, ge=1)
    knowledge_base_delete_poll_interval_seconds: float = Field(default=0.5, ge=0.1, le=10)


class QuotaConfig(SettingsModel):
    enabled: bool = False
    default_daily_quota: int = 100000
    default_max_collections: int = 50
    default_max_namespaces: int = 20


class ServerConfig(SettingsModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=list)
    log_level: str = "INFO"


class AppConfig(SettingsModel):
    app: AppRuntimeConfig = Field(default_factory=AppRuntimeConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
