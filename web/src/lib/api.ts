import type {
  AuthBootstrap,
  ApiKeyInfo,
  ChatMessage,
  ChatCapabilities,
  Conversation,
  KnowledgeBase,
  KnowledgeBaseIndex,
  DocumentQueueResponse,
  PromptProfile,
  PromptProfileState,
  LLMThinkingMode,
  LLMThinkingState,
  ThinkingEffort,
  RagDocument,
  SystemCheck,
  SystemStatus,
  CreatedApiKey,
} from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

type RequestOptions = RequestInit & { json?: unknown }

export type InitialProviderConfig = {
  llm_api_base: string
  llm_api_key: string
  llm_model_name: string
  embedding_api_base: string
  embedding_api_key: string
  embedding_model_name: string
  reranker_api_base: string
  reranker_expected_model: string
}

export class ApiError extends Error {
  status: number
  body: unknown
  code?: string

  constructor(status: number, message: string, body: unknown, code?: string) {
    super(message)
    this.status = status
    this.body = body
    this.code = code
  }
}

export function apiErrorFromPayload(
  status: number,
  statusText: string,
  payload: unknown,
): ApiError {
  const payloadObj = typeof payload === 'object' && payload ? payload as Record<string, unknown> : null
  const detailObj =
    payloadObj && typeof payloadObj.detail === 'object' && payloadObj.detail
      ? payloadObj.detail as Record<string, unknown>
      : null
  const errorObj =
    payloadObj && typeof payloadObj.error === 'object' && payloadObj.error
      ? payloadObj.error as Record<string, unknown>
      : null
  // Foreground endpoints expose a specific, safe ``error`` alongside a
  // compatibility ``message`` such as “生成失败”.  Prefer the actionable text
  // so queue/deadline failures do not collapse back into the generic toast.
  const message =
    typeof payloadObj?.error === 'string' ? payloadObj.error
      : typeof errorObj?.message === 'string' ? errorObj.message
        : typeof payloadObj?.message === 'string' ? payloadObj.message
          : typeof detailObj?.message === 'string' ? detailObj.message
            : typeof payloadObj?.detail === 'string' ? payloadObj.detail
              : statusText
  const code =
    typeof payloadObj?.error_code === 'string' ? payloadObj.error_code
      : typeof payloadObj?.code === 'string' ? payloadObj.code
        : typeof errorObj?.code === 'string' ? errorObj.code
          : typeof detailObj?.code === 'string' ? detailObj.code
            : undefined
  return new ApiError(status, message, payload, code)
}

async function readBody(response: Response) {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers)
  let body = options.body
  if (options.json !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(options.json)
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    body,
    credentials: 'include',
  })
  const payload = await readBody(response)
  if (!response.ok) {
    throw apiErrorFromPayload(response.status, response.statusText, payload)
  }
  return payload as T
}

export function streamFetch(path: string, options: RequestOptions = {}) {
  const headers = new Headers(options.headers)
  let body = options.body
  if (options.json !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(options.json)
  }
  return fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    body,
    credentials: 'include',
  })
}

export const api = {
  bootstrap: () => apiFetch<AuthBootstrap>('/api/v1/auth/bootstrap'),
  setupAuth: (
    password: string,
    confirmPassword: string,
    bootstrapToken?: string,
    providerConfig?: InitialProviderConfig,
  ) =>
    apiFetch<{ authenticated: boolean; principal: AuthBootstrap['principal'] }>('/api/v1/auth/setup', {
      method: 'POST',
      headers: bootstrapToken ? { Authorization: `Bearer ${bootstrapToken}` } : undefined,
      json: {
        password,
        confirm_password: confirmPassword,
        provider_config: providerConfig,
      },
    }),
  login: (password: string) =>
    apiFetch<{ authenticated: boolean }>('/api/v1/auth/session', {
      method: 'POST',
      json: { password },
    }),
  me: () => apiFetch<{ authenticated: boolean; method: string }>('/api/v1/auth/me'),
  changePassword: (currentPassword: string, newPassword: string, confirmPassword: string) =>
    apiFetch<{ updated: boolean }>('/api/v1/auth/password', {
      method: 'POST',
      json: {
        current_password: currentPassword,
        new_password: newPassword,
        confirm_password: confirmPassword,
      },
    }),
  logout: () => apiFetch<{ authenticated: boolean }>('/api/v1/auth/logout', { method: 'POST' }),
  systemStatus: () => apiFetch<SystemStatus>('/api/v1/system/status'),
  systemChecks: () => apiFetch<{ checks: SystemCheck[] }>('/api/v1/system/checks', { method: 'POST' }),
  apiKeys: () => apiFetch<{ api_keys: ApiKeyInfo[] }>('/api/v1/vectors/api-keys'),
  createApiKey: (payload: {
    name: string
    scopes: string[]
    is_admin?: boolean
    expires_at?: string | null
    requests_per_minute?: number | null
    daily_quota?: number | null
  }) => apiFetch<CreatedApiKey>('/api/v1/vectors/api-keys', {
    method: 'POST',
    json: payload,
  }),
  revokeApiKey: (keyId: string) => apiFetch<{ key_id: string; status: string }>(
    `/api/v1/vectors/api-keys/${keyId}`,
    { method: 'DELETE' },
  ),
  updateApiKeyScopes: (keyId: string, scopes: string[]) => apiFetch<ApiKeyInfo>(
    `/api/v1/vectors/api-keys/${keyId}`,
    { method: 'PATCH', json: { scopes } },
  ),
  deleteRevokedApiKeys: (keyIds?: string[]) => apiFetch<{ deleted_count: number }>(
    '/api/v1/vectors/api-keys/revoked',
    { method: 'DELETE', json: keyIds?.length ? { key_ids: keyIds } : {} },
  ),
  deleteRevokedApiKey: (keyId: string) => apiFetch<{ deleted_count: number }>(
    `/api/v1/vectors/api-keys/${keyId}/permanent`,
    { method: 'DELETE' },
  ),
  updatePromptProfile: (profile: PromptProfile) =>
    apiFetch<PromptProfileState>('/api/v1/system/prompt-profile', {
      method: 'PUT',
      json: { profile },
    }),
  updateLLMThinkingMode: (mode: LLMThinkingMode) =>
    apiFetch<LLMThinkingState>('/api/v1/system/llm-thinking', {
      method: 'PUT',
      json: { mode },
    }),
  documents: (kbId?: string | null) => apiFetch<{ documents: RagDocument[] }>(
    kbId ? `/api/v1/documents?kb_id=${encodeURIComponent(kbId)}` : '/api/v1/documents',
  ),
  documentRevision: () =>
    apiFetch<{ item_count: number; updated_at: string }>('/api/v1/documents/revision'),
  documentQueue: () => apiFetch<DocumentQueueResponse>('/api/v1/documents/queue'),
  clearDocumentQueueHistory: () =>
    apiFetch<{ deleted: number }>('/api/v1/documents/queue/history/clear', {
      method: 'POST',
    }),
  uploadDocument: (file: File, kbId?: string | null) => {
    const form = new FormData()
    form.append('file', file)
    if (kbId) form.append('kb_id', kbId)
    return apiFetch<{ doc_id: string; filename: string; status?: string; message?: string }>(
      '/api/v1/documents/upload',
      { method: 'POST', body: form },
    )
  },
  batchUploadDocuments: (files: File[], kbId?: string | null) => {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    if (kbId) form.append('kb_id', kbId)
    return apiFetch<{
      batch_job_id?: string
      status: string
      message?: string
      prepared: Array<{
        doc_id?: string
        filename: string
        status: string
        message?: string
        error?: string
        errors?: string[]
      }>
    }>('/api/v1/documents/batch-upload', {
      method: 'POST',
      body: form,
    })
  },
  deleteDocument: (docId: string) =>
    apiFetch<{ status: string; job_id?: string; message?: string }>(
      `/api/v1/documents/${docId}`,
      { method: 'DELETE' },
    ),
  reingestDocument: (docId: string) =>
    apiFetch<{ doc_id: string; filename: string; status: string; message: string }>(
      `/api/v1/documents/${docId}/reingest`,
      { method: 'POST' },
    ),
  knowledgeBases: () => apiFetch<{ knowledge_bases: KnowledgeBase[] }>('/api/v1/knowledge-bases'),
  knowledgeBaseRevision: () =>
    apiFetch<{ item_count: number; updated_at: string }>('/api/v1/knowledge-bases/revision'),
  createKnowledgeBase: (name: string) =>
    apiFetch<KnowledgeBase>('/api/v1/knowledge-bases', { method: 'POST', json: { name } }),
  deleteKnowledgeBase: (kbId: string) =>
    apiFetch<{ status: string; job_id: string; message: string }>(
      `/api/v1/knowledge-bases/${kbId}`,
      { method: 'DELETE' },
    ),
  knowledgeBaseDocuments: (kbId: string) =>
    apiFetch<{ documents: RagDocument[] }>(`/api/v1/knowledge-bases/${kbId}/documents`),
  knowledgeBaseIndexes: (kbId: string) =>
    apiFetch<{ indexes: KnowledgeBaseIndex[] }>(`/api/v1/knowledge-bases/${kbId}/indexes`),
  buildKnowledgeBaseIndexCandidate: (kbId: string) =>
    apiFetch<{ job_id: string; status: string; message: string }>(
      `/api/v1/knowledge-bases/${kbId}/index-candidates`,
      { method: 'POST' },
    ),
  activateKnowledgeBaseIndex: (kbId: string, indexId: string) =>
    apiFetch<{ index: KnowledgeBaseIndex; message: string }>(
      `/api/v1/knowledge-bases/${kbId}/indexes/${indexId}/activate`,
      { method: 'POST' },
    ),
  reclaimKnowledgeBaseIndex: (kbId: string, indexId: string) =>
    apiFetch<{ deleted_index_id: string; status: string }>(
      `/api/v1/knowledge-bases/${kbId}/indexes/${indexId}`,
      { method: 'DELETE' },
    ),
  conversations: () => apiFetch<{ conversations: Conversation[] }>('/api/v1/conversations'),
  chatCapabilities: () => apiFetch<ChatCapabilities>('/api/v1/chat/capabilities'),
  conversation: (conversationId: string) =>
    apiFetch<{ conversation: Conversation; messages: ChatMessage[] }>(
      `/api/v1/conversations/${conversationId}`,
    ),
  deleteConversation: (conversationId: string) =>
    apiFetch<{ status: string }>(`/api/v1/conversations/${conversationId}`, { method: 'DELETE' }),
  deleteConversations: (conversationIds: string[]) =>
    apiFetch<{ status: string; deleted: number }>('/api/v1/conversations', {
      method: 'DELETE',
      json: { conversation_ids: conversationIds },
    }),
  updateConversationSessionOptions: (
    conversationId: string,
    knowledgeScope: 'all' | 'selected',
    knowledgeBaseIds: string[],
    fullContextDocId: string | null,
    groundingMode: 'auto' | 'knowledge' | 'assistant',
    streamValidationMode: 'validated' | 'realtime',
    thinkingEffort: ThinkingEffort | null,
    answerQualityMode: 'normal' | 'enhanced' = 'normal',
    llmModel?: string,
  ) => apiFetch<{ conversation: Conversation }>(`/api/v1/conversations/${conversationId}/retrieval-scope`, {
    method: 'PUT',
    json: {
      knowledge_scope: knowledgeScope,
      knowledge_base_ids: knowledgeBaseIds,
      full_context_doc_id: fullContextDocId,
      grounding_mode: groundingMode,
      stream_validation_mode: streamValidationMode,
      llm_model: llmModel,
      thinking_effort: thinkingEffort,
      answer_quality_mode: answerQualityMode,
    },
  }),
}
