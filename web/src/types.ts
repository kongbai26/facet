export type DocumentStatus =
  | 'processing'
  | 'ready'
  | 'failed'
  | 'reindex_queued'
  | 'reindexing'
  | 'deleting'
  | 'delete_failed'

export type RagDocument = {
  doc_id: string
  kb_id?: string | null
  filename: string
  file_size?: number
  chunks_count?: number
  status: DocumentStatus
  status_reason?: string
  error_message?: string
  embedding_model?: string
  index_pipeline_version?: string
  created_at?: string
  updated_at?: string
  full_context_available?: boolean
  full_context_reason?: string
  full_context_tokens?: number
  full_context_token_budget?: number
}

export type KnowledgeBase = {
  kb_id: string
  name: string
  slug?: string
  status: string
  ready_documents_count: number
}

export type KnowledgeBaseIndex = {
  index_id: string
  profile_hash: string
  collection_name: string
  source_fingerprint: string
  status: 'building' | 'ready' | 'active' | 'failed'
  chunk_count?: number
  error_message?: string
  created_at?: string
  updated_at?: string
}

export type DocumentQueueItem = {
  key: string
  kind: 'document' | 'bm25' | 'knowledge_base'
  status: 'queued' | 'running' | 'failed' | 'succeeded' | 'cancelled'
  label: string
  detail: string
  title: string
  origin?: 'active' | 'history'
  doc_id?: string
  kb_id?: string
  document_status?: DocumentStatus
  job_type?: string
  reason?: string
  job_id?: string
  collection_name?: string
  summary?: string
  related_documents?: string[]
  created_at?: string
  updated_at?: string
}

export type DocumentQueueResponse = {
  items: DocumentQueueItem[]
  history: DocumentQueueItem[]
  counts: {
    documents: number
    bm25: number
    active: number
    history: number
  }
}

export type AuthPrincipal = {
  principal_id: string
  name: string
  principal_type: string
  is_admin: boolean
}

export type AuthBootstrap = {
  auth_enabled: boolean
  setup_required: boolean
  bootstrap_token_required: boolean
  authenticated: boolean
  principal: AuthPrincipal | null
  warnings: string[]
}

export type ApiKeyInfo = {
  key_id: string
  tenant_id: string
  principal_id: string
  name: string
  scopes: string[]
  is_admin: boolean
  is_active: boolean
  requests_per_minute?: number | null
  daily_quota?: number | null
  expires_at?: string | null
  last_used_at?: string | null
  created_at: string
  updated_at: string
}

export type CreatedApiKey = ApiKeyInfo & {
  raw_key: string
  warning: string
}

export type SystemStatus = {
  auth: {
    initialized: boolean
    legacy_password_detected: boolean
    session_secret_ok: boolean
  }
  llm: {
    configured: boolean
    mode: string
    mock: boolean
    // Status reads are passive; connectivity is exercised only by the
    // explicit diagnostic action so normal polling never competes with chat.
    reachable: boolean | null
    connectivity_check: 'on_demand'
    thinking: LLMThinkingState
  }
  embedding: {
    configured: boolean
    reachable: boolean | null
    connectivity_check: 'on_demand'
    current_model: string
  }
  retrieval: {
    hybrid_enabled: boolean
    exact_match_enabled: boolean
    bm25_ready: boolean
    bm25_rebuilding: boolean
    bm25_state: 'disabled' | 'empty' | 'ready' | 'waiting_documents' | 'building' | 'unavailable'
    bm25_target_count: number
    bm25_ready_count: number
    reranker: {
      configured: boolean
      active: boolean
      available: boolean
      mode: string
      expected_model: string
      model_name: string
      last_error: string
      failure_count: number
    }
  }
  vectorstore: { mode: string }
  database: { backend: string }
  queue: {
    backend: string
    autostart_worker: boolean
    worker: {
      state: 'running' | 'stopped' | 'disabled' | 'unknown'
      queued_count: number
      running_count: number
      last_heartbeat_age_seconds: number | null
      detail: string
    }
  }
  reindex: {
    enabled: boolean
    pending_count: number
    running_count: number
    blocked_count: number
  }
  prompt: {
    profile: 'auto' | 'local' | 'cloud'
    effective_profile: 'local' | 'cloud'
    source: 'auto' | 'manual'
    local_endpoint_detected: boolean
  }
  frontend: { built: boolean }
  warnings: string[]
}

export type ChatCapabilities = {
  thinking: LLMThinkingState
  models: {
    default_model: string
    options: LLMModelOption[]
  }
  answer_quality: {
    default_mode: 'normal' | 'enhanced'
    modes: Array<'normal' | 'enhanced'>
  }
}

export type LLMModelOption = {
  model_name: string
  display_name: string
  thinking: LLMThinkingState
}

export type PromptProfile = 'auto' | 'local' | 'cloud'

export type PromptProfileState = {
  profile: PromptProfile
  effective_profile: 'local' | 'cloud'
  source: 'auto' | 'manual'
  local_endpoint_detected: boolean
}

export type LLMThinkingMode = 'auto' | 'on' | 'off'

export type ThinkingEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export type LLMThinkingState = {
  mode: LLMThinkingMode
  supported: boolean
  transport: string | null
  effort: ThinkingEffort | null
  efforts: ThinkingEffort[]
  matched_pattern: string | null
  model_name: string
  source: 'config' | 'manual'
}

export type SystemCheck = {
  key: string
  status: 'ok' | 'warn' | 'error'
  code: string
  message: string
  detail?: string
}

export type SourceMetadata = {
  block_kind?: string
  section_title?: string
  heading_path?: string
  table_headers?: string
  source_anchor?: string
  page?: number | string
  [key: string]: unknown
}

export type Source = {
  index: number
  doc_id: string
  kb_id?: string
  knowledge_base_name?: string
  filename: string
  chunk_id: string
  chunk_index: number | null
  score?: number
  score_source?: 'hybrid' | 'vector' | 'bm25' | 'reranker' | 'context_expansion' | 'reuse' | 'full_context' | 'unknown' | string
  text: string
  metadata?: SourceMetadata
}

export type Conversation = {
  conversation_id: string
  title: string
  created_at: string
  updated_at: string
  last_message_at?: string
  messages_count?: number
  knowledge_base_id?: string | null
  knowledge_scope?: 'all' | 'selected'
  knowledge_base_ids?: string[]
  full_context_doc_id?: string | null
  grounding_mode?: 'auto' | 'knowledge' | 'assistant'
  answer_quality_mode?: 'normal' | 'enhanced'
  stream_validation_mode?: 'validated' | 'realtime'
  llm_model?: string | null
  thinking_effort?: ThinkingEffort | null
}

export type ChatMessage = {
  message_id: string
  conversation_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  status: 'completed' | 'streaming' | 'stopped' | 'error'
  sources?: Source[]
  grounding_mode?: 'auto' | 'knowledge' | 'assistant'
  answer_quality_mode?: 'normal' | 'enhanced'
  evidence_status?: 'pending' | 'grounded' | 'partial' | 'conflict' | 'no_evidence' | 'direct' | 'unavailable'
  error_message?: string
  seq: number
  created_at: string
  updated_at: string
}

export type ChatStreamMeta = {
  conversation_id: string
  user_message_id: string
  assistant_message_id: string
  title: string
  decision?: 'PENDING' | 'RETRIEVE' | 'REUSE' | 'DIRECT' | 'LIVE_UNSUPPORTED' | 'NO_EVIDENCE'
  reason?: string
  fallback_used?: boolean
  grounding_mode?: 'auto' | 'knowledge' | 'assistant'
  answer_quality_mode?: 'normal' | 'enhanced'
  llm_model?: string
  thinking_effort?: ThinkingEffort | null
  response_mode?: 'auto' | 'auto_fallback' | 'auto_partial' | 'evidence_partial' | 'evidence_conflict' | 'verification_unavailable' | 'knowledge_no_evidence' | 'live_unsupported'
  evidence_status?: ChatMessage['evidence_status']
}
