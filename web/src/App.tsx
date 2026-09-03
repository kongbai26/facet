import {
  lazy,
  Suspense,
  type ChangeEvent,
  type DragEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  Bot,
  Check,
  CheckSquare,
  Copy,
  Database,
  FileText,
  KeyRound,
  Clock3,
  ListChecks,
  LogOut,
  MessageSquare,
  Pencil,
  Play,
  RefreshCw,
  Send,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Trash2,
  Upload,
  Wrench,
} from 'lucide-react'
import { ApiError, api, apiErrorFromPayload, streamFetch } from './lib/api'
import {
  buildOptimisticConversationMessages,
  buildOptimisticMessages,
  chatLocationStorageKey,
  conversationStreamKey,
  createNewConversationViewId,
  parseStoredChatLocation,
  resolveOutgoingEditFromMessageId,
  shouldActivateStreamConversation,
  shouldRestoreEditingMessageIdAfterSendFailure,
  upsertConversation,
} from './lib/chat'
import { parseJsonEvent, readSseStream } from './lib/sse'
import { cn, formatBytes, shortDate } from './lib/utils'
import type {
  AuthBootstrap,
  ApiKeyInfo,
  ChatMessage,
  ChatStreamMeta,
  Conversation,
  DocumentQueueItem,
  KnowledgeBaseIndex,
  PromptProfile,
  LLMThinkingMode,
  LLMThinkingState,
  ThinkingEffort,
  Source,
  SourceMetadata,
  SystemStatus,
  SystemCheck,
  CreatedApiKey,
} from './types'

type Toast = { id: number; text: string; tone?: 'ok' | 'error' }
type ConfirmDialogState = {
  title: string
  message: string
  confirmLabel: string
  tone?: 'danger' | 'primary'
  onConfirm: () => void
}
type GroundingMode = 'auto' | 'knowledge' | 'assistant'
type AnswerQualityMode = 'normal' | 'enhanced'
const STREAM_RENDER_INTERVAL_MS = 40
const MarkdownContent = lazy(() => import('./components/MarkdownContent'))

function MessageMarkdown({ content }: { content: string }) {
  return (
    <Suspense fallback={<div className="markdown-body whitespace-pre-wrap">{content}</div>}>
      <MarkdownContent content={content} />
    </Suspense>
  )
}

function configuredThinkingEfforts(thinking?: LLMThinkingState): ThinkingEffort[] {
  const efforts = thinking?.efforts
  return efforts || []
}

function defaultThinkingEffort(thinking?: LLMThinkingState): ThinkingEffort {
  if (!thinking?.supported || thinking.mode === 'off') return 'none'
  return thinking.effort || 'medium'
}

function resolveThinkingEffort(
  value: string | null | undefined,
  thinking?: LLMThinkingState,
): ThinkingEffort {
  const efforts = configuredThinkingEfforts(thinking)
  if (value && efforts.includes(value as ThinkingEffort)) return value as ThinkingEffort
  return defaultThinkingEffort(thinking)
}

function LLMModelSelect({
  value,
  models,
  disabled,
  onChange,
}: {
  value: string
  models: Array<{ model_name: string; display_name: string }>
  disabled?: boolean
  onChange: (value: string) => void
}) {
  if (!value || models.length === 0) return null
  return (
    <label className="composer-model-control" title={`本会话模型：${value}`}>
      <span>模型</span>
      <select
        className="composer-model-select"
        aria-label="本会话模型"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {models.map((model) => (
          <option key={model.model_name} value={model.model_name}>
            {model.display_name || model.model_name}
          </option>
        ))}
      </select>
    </label>
  )
}

function ThinkingEffortSelect({
  value,
  efforts,
  supported,
  disabled,
  onChange,
}: {
  value: ThinkingEffort
  efforts: ThinkingEffort[]
  supported: boolean
  disabled?: boolean
  onChange: (value: ThinkingEffort) => void
}) {
  if (!supported || efforts.length === 0) return null
  return (
    <label className="composer-thinking-control" title={`思考强度：${value}`}>
      <span>思考</span>
      <select
        className="composer-thinking-select"
        aria-label="思考强度"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value as ThinkingEffort)}
      >
        {efforts.map((effort) => (
          <option key={effort} value={effort}>{effort}</option>
        ))}
      </select>
    </label>
  )
}

function AnswerQualityToggle({
  value,
  disabled,
  onChange,
}: {
  value: AnswerQualityMode
  disabled?: boolean
  onChange: (value: AnswerQualityMode) => void
}) {
  return (
    <div
      className="answer-quality-toggle"
      role="group"
      aria-label="回答质量"
      title={value === 'enhanced'
        ? '增强：增加纠错检索、证据聚合和答案核验，响应较慢'
        : '普通：完成完整语义判断和答案核验，以较低延迟回答'}
    >
      <button
        type="button"
        className={cn('answer-quality-option', value === 'normal' && 'is-active')}
        aria-pressed={value === 'normal'}
        disabled={disabled}
        onClick={() => onChange('normal')}
      >普通</button>
      <button
        type="button"
        className={cn('answer-quality-option answer-quality-enhanced', value === 'enhanced' && 'is-active')}
        aria-pressed={value === 'enhanced'}
        disabled={disabled}
        onClick={() => onChange('enhanced')}
      ><ShieldCheck className="h-3 w-3" />增强</button>
    </div>
  )
}

const DOCUMENT_ACCEPT = '.pdf,.txt,.md,.markdown,.docx,.html,.htm'
const PASSWORD_MIN_LENGTH = 6
const API_KEY_SCOPE_OPTIONS = [
  { value: 'rag:read', label: '知识库检索', description: '允许 Agent 调用 knowledge-search' },
  { value: 'llm:invoke', label: '模型调用', description: '允许 Agent 调用 chat/completions' },
  { value: 'rag:write', label: '知识库写入', description: '允许 Agent 写入或管理知识库' },
  { value: 'vectors:read', label: '向量读取', description: '允许访问独立向量集合' },
  { value: 'vectors:write', label: '向量写入', description: '允许修改独立向量集合' },
]

function formatApiKeyScope(scope: string) {
  if (scope === 'admin:*') return '全部管理员权限'
  return API_KEY_SCOPE_OPTIONS.find((option) => option.value === scope)?.label || scope
}
function getApiMessage(error: unknown, fallback: string) {
  if (error instanceof ApiError) return error.message || fallback
  if (error instanceof Error) return error.message || fallback
  return fallback
}

function getDocumentActionError(error: unknown, fallback: string) {
  if (error instanceof ApiError) {
    switch (error.code) {
      case 'document_processing':
        return '文档仍在处理中，暂时不能操作。'
      case 'document_source_missing':
        return '上传原文件已经缺失，当前无法重新摄入。'
      case 'document_reingest_not_allowed':
      case 'document_status_conflict':
        return error.message
      case 'retry_cleanup_failed':
      case 'document_delete_failed':
      case 'document_metadata_delete_failed':
      case 'reingest_job_create_failed':
        return error.message
      default:
        return error.message || fallback
    }
  }
  return fallback
}

function getLoginError(error: unknown) {
  if (error instanceof ApiError) {
    if (error.code === 'setup_required') return '系统尚未初始化，请先完成首次设置。'
    if (error.code === 'invalid_password') return '密码不正确。'
    return error.message || '登录失败。'
  }
  if (error instanceof Error && error.message.includes('fetch')) return '无法连接到后端服务，请确认服务已启动。'
  return '登录失败。'
}

function getBm25StatusNotice(statusData: SystemStatus | undefined) {
  if (!statusData?.retrieval?.hybrid_enabled) return null
  const state = statusData.retrieval.bm25_state
  if (state === 'building') {
    return {
      text: 'BM25 索引正在建立中，完成后会自动刷新。',
      pending: true,
    }
  }
  if (state === 'waiting_documents') {
    return {
      text: 'BM25 索引正在等待文档处理完成，随后会自动生成。',
      pending: true,
    }
  }
  if (state === 'unavailable') {
    return {
      text: 'BM25 索引未就绪，且当前没有构建任务；检索会暂时使用向量召回。',
      pending: false,
    }
  }
  return null
}

function filterBm25Warnings(warnings: string[]) {
  return warnings.filter((warning) => !warning.includes('BM25'))
}

function normalizeSourceText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) {
    return value.map((item) => normalizeSourceText(item)).filter(Boolean).join(' > ')
  }
  return ''
}

function normalizeSourcePage(value: unknown): string {
  const text = normalizeSourceText(value)
  if (!text) return ''
  if (/^\d+$/.test(text)) return `第 ${text} 页`
  return text
}

function displayModelName(value: unknown): string {
  const text = normalizeSourceText(value)
  if (!text) return ''
  const parts = text.replaceAll('\\', '/').split('/').filter(Boolean)
  return parts.at(-1) || text
}

function formatSourceKind(kind: string) {
  const normalized = kind.trim().toLowerCase()
  if (!normalized) return ''
  const labels: Record<string, string> = {
    heading: '标题',
    title: '标题',
    paragraph: '段落',
    list_item: '列表',
    list: '列表',
    table: '表格',
    code: '代码',
    quote: '引用',
  }
  return labels[normalized] || normalized.replace(/_/g, ' ')
}

function getSourceMetadata(source: Source): SourceMetadata {
  return source.metadata || {}
}

function getSourceStructure(source: Source) {
  const metadata = getSourceMetadata(source)
  const kind = normalizeSourceText(metadata.block_kind || metadata.kind || '')
  const sectionTitle = normalizeSourceText(metadata.section_title || '')
  const headingPath = normalizeSourceText(metadata.heading_path || '')
  const tableHeaders = normalizeSourceText(metadata.table_headers || '')
  const sourceAnchor = normalizeSourceText(metadata.source_anchor || '')
  const page = normalizeSourcePage(metadata.page)
  const kindLabel = formatSourceKind(kind)

  return {
    kind,
    kindLabel,
    sectionTitle,
    headingPath,
    tableHeaders,
    sourceAnchor,
    page,
  }
}

function Bm25StatusNotice({ statusData }: { statusData?: SystemStatus }) {
  const notice = getBm25StatusNotice(statusData)
  if (!notice) return null
  return (
    <div className={cn('bm25-banner', notice.pending ? 'bm25-banner-pending' : 'bm25-banner-warning')}>
      {notice.pending
        ? <RefreshCw className="h-4 w-4 shrink-0 animate-spin" />
        : <AlertTriangle className="h-4 w-4 shrink-0" />}
      <p>{notice.text}</p>
    </div>
  )
}

async function readStreamError(response: Response) {
  const text = await response.text()
  if (!text) return new ApiError(response.status, response.statusText, null)
  try {
    const payload = JSON.parse(text) as Record<string, unknown>
    return apiErrorFromPayload(response.status, response.statusText, payload)
  } catch {
    return new ApiError(response.status, response.statusText, text)
  }
}

function App() {
  const [toasts, setToasts] = useState<Toast[]>([])
  const toastTimers = useRef<Map<string, number>>(new Map())
  const queryClient = useQueryClient()
  const bootstrap = useQuery({ queryKey: ['bootstrap'], queryFn: api.bootstrap, retry: 1, retryDelay: 800 })

  const notify = (text: string, tone: Toast['tone'] = 'ok', durationMs = 3200) => {
    const existing = toastTimers.current.get(text)
    if (existing) window.clearTimeout(existing)

    setToasts((items) => {
      const withoutSame = items.filter((item) => item.text !== text)
      return [...withoutSame, { id: Date.now(), text, tone }]
    })

    const timer = window.setTimeout(() => {
      setToasts((items) => items.filter((item) => item.text !== text))
      toastTimers.current.delete(text)
    }, durationMs)
    toastTimers.current.set(text, timer)
  }

  const refreshAuth = async () => {
    await queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
    await queryClient.invalidateQueries({ queryKey: ['me'] })
  }

  if (bootstrap.isLoading) return <ShellLoading />
  if (bootstrap.isError || !bootstrap.data) return <ServiceUnavailableScreen onRetry={() => bootstrap.refetch()} />

  const bootstrapState = bootstrap.data

  return (
    <>
      {bootstrapState.setup_required ? (
        <SetupScreen
          onComplete={refreshAuth}
          bootstrapTokenRequired={bootstrapState.bootstrap_token_required}
        />
      ) : !bootstrapState.authenticated ? (
        <LoginScreen onLogin={refreshAuth} warnings={bootstrapState.warnings} />
      ) : (
        <AppShell
          bootstrap={bootstrapState}
          notify={notify}
          onLogout={refreshAuth}
        />
      )}
      <div className="toast-stack">
        {toasts.map((toast) => (
          <div key={toast.id} className={cn('toast', toast.tone === 'error' && 'toast-error')}>
            {toast.text}
          </div>
        ))}
      </div>
    </>
  )
}

function ShellLoading() {
  return (
    <main className="auth-screen">
      <div className="auth-panel">
        <Database className="h-6 w-6 animate-pulse text-[var(--c-accent)]" />
        <p className="text-sm text-[var(--c-text-secondary)]">正在检查服务状态...</p>
      </div>
    </main>
  )
}

function ServiceUnavailableScreen({ onRetry }: { onRetry: () => void }) {
  return (
    <main className="auth-screen">
      <div className="auth-panel auth-card">
        <AlertTriangle className="h-7 w-7 text-[var(--c-warning)]" />
        <h1>服务暂时不可达</h1>
        <p className="auth-copy">后端服务还没有响应，我们先确认它是否已经启动。</p>
        <button type="button" onClick={onRetry}>
          <RefreshCw className="h-4 w-4" /> 重新检查
        </button>
      </div>
    </main>
  )
}

function SetupScreen({
  onComplete,
  bootstrapTokenRequired,
}: {
  onComplete: () => void
  bootstrapTokenRequired: boolean
}) {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [bootstrapToken, setBootstrapToken] = useState('')
  const [llmApiBase, setLlmApiBase] = useState('')
  const [llmApiKey, setLlmApiKey] = useState('')
  const [llmModelName, setLlmModelName] = useState('')
  const [embeddingApiBase, setEmbeddingApiBase] = useState('')
  const [embeddingApiKey, setEmbeddingApiKey] = useState('')
  const [embeddingModelName, setEmbeddingModelName] = useState('')
  const [rerankerApiBase, setRerankerApiBase] = useState('')
  const [rerankerExpectedModel, setRerankerExpectedModel] = useState('')
  const status = useQuery({ queryKey: ['system-status', 'setup'], queryFn: api.systemStatus, retry: 1 })
  const setup = useMutation({
    mutationFn: () => api.setupAuth(password, confirmPassword, bootstrapToken, {
      llm_api_base: llmApiBase,
      llm_api_key: llmApiKey,
      llm_model_name: llmModelName,
      embedding_api_base: embeddingApiBase,
      embedding_api_key: embeddingApiKey,
      embedding_model_name: embeddingModelName,
      reranker_api_base: rerankerApiBase,
      reranker_expected_model: rerankerExpectedModel,
    }),
    onSuccess: onComplete,
  })
  const statusData = status.data

  return (
    <main className="auth-screen">
      <div className="setup-shell">
        <form
          className="auth-panel auth-card setup-form"
          onSubmit={(event) => {
            event.preventDefault()
            setup.mutate()
          }}
        >
          <ShieldCheck className="h-7 w-7 text-[var(--c-accent)]" />
          <h1>首次初始化</h1>
          <p className="auth-copy">连接模型并设置管理员密码。密钥只会写入这台电脑的 `config/.env`。</p>
          <section className="setup-stage">
            <div className="setup-stage-heading">
              <span>01</span>
              <div>
                <h2>连接模型服务</h2>
                <p>填写 OpenAI 兼容 API 的根地址和实际加载的模型名称。</p>
              </div>
            </div>
            <div className="setup-provider-grid">
              <fieldset className="setup-provider-card">
                <legend>对话模型</legend>
                <label className="setup-field">
                  <span>API 地址</span>
                  <input value={llmApiBase} onChange={(event) => setLlmApiBase(event.target.value)} type="url" placeholder="https://api.example.com/v1" autoFocus />
                </label>
                <label className="setup-field">
                  <span>模型名称</span>
                  <input value={llmModelName} onChange={(event) => setLlmModelName(event.target.value)} placeholder="your-chat-model" />
                </label>
                <label className="setup-field">
                  <span>API Key <em>可留空</em></span>
                  <input value={llmApiKey} onChange={(event) => setLlmApiKey(event.target.value)} type="password" placeholder="服务不需要时留空" autoComplete="off" />
                </label>
              </fieldset>
              <fieldset className="setup-provider-card">
                <legend>向量模型</legend>
                <label className="setup-field">
                  <span>API 地址</span>
                  <input value={embeddingApiBase} onChange={(event) => setEmbeddingApiBase(event.target.value)} type="url" placeholder="https://api.example.com/v1" />
                </label>
                <label className="setup-field">
                  <span>模型名称</span>
                  <input value={embeddingModelName} onChange={(event) => setEmbeddingModelName(event.target.value)} placeholder="your-embedding-model" />
                </label>
                <label className="setup-field">
                  <span>API Key <em>可留空</em></span>
                  <input value={embeddingApiKey} onChange={(event) => setEmbeddingApiKey(event.target.value)} type="password" placeholder="服务不需要时留空" autoComplete="off" />
                </label>
              </fieldset>
            </div>
            <details className="setup-reranker">
              <summary>可选：配置重排模型</summary>
              <p>留空时使用混合检索；填写后系统会自动探测重排服务。</p>
              <div className="setup-reranker-fields">
                <label className="setup-field">
                  <span>重排 API 地址</span>
                  <input value={rerankerApiBase} onChange={(event) => setRerankerApiBase(event.target.value)} type="url" placeholder="http://localhost:6061" />
                </label>
                <label className="setup-field">
                  <span>重排模型名称 <em>可留空</em></span>
                  <input value={rerankerExpectedModel} onChange={(event) => setRerankerExpectedModel(event.target.value)} placeholder="your-reranker-model" />
                </label>
              </div>
            </details>
          </section>
          <section className="setup-stage">
            <div className="setup-stage-heading">
              <span>02</span>
              <div>
                <h2>保护工作台</h2>
                <p>管理员密码仅保存哈希值，之后可在设置页修改。</p>
              </div>
            </div>
          {bootstrapTokenRequired && (
            <>
              <label className="setup-field">
                <span>首次初始化令牌</span>
                <input
                  value={bootstrapToken}
                  onChange={(event) => setBootstrapToken(event.target.value)}
                  type="password"
                  placeholder="生产环境部署时设置的令牌"
                  autoComplete="off"
                />
              </label>
            </>
          )}
          <div className="setup-password-grid">
            <label className="setup-field">
              <span>管理员密码</span>
              <input
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                type="password"
                placeholder={`至少 ${PASSWORD_MIN_LENGTH} 位`}
                autoComplete="new-password"
              />
            </label>
            <label className="setup-field">
              <span>确认密码</span>
              <input
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                type="password"
                placeholder="再次输入管理员密码"
                autoComplete="new-password"
              />
            </label>
          </div>
          </section>
          {setup.isError && <p className="form-error">{getApiMessage(setup.error, '初始化失败。')}</p>}
          <button type="submit" disabled={
            !llmApiBase.trim()
            || !llmModelName.trim()
            || !embeddingApiBase.trim()
            || !embeddingModelName.trim()
            || password.length < PASSWORD_MIN_LENGTH
            || confirmPassword.length < PASSWORD_MIN_LENGTH
            || (bootstrapTokenRequired && !bootstrapToken.trim())
            || setup.isPending
          }>
            <Play className="h-4 w-4" /> 完成初始化
          </button>
        </form>
        <section className="tool-panel setup-status-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">System</p>
              <h1>系统准备情况</h1>
            </div>
          </div>
          {status.isLoading ? (
            <p className="panel-copy">正在检查本地状态...</p>
          ) : status.isError ? (
            <p className="form-error">系统状态读取失败，请确认后端已完成启动。</p>
          ) : statusData ? (
            <>
              <StatusList
                items={[
                  { label: 'LLM 配置', ok: statusData.llm.configured },
                  { label: 'Embedding 配置', ok: statusData.embedding.configured },
                  { label: 'Embedding 已连接', ok: statusData.embedding.reachable },
                  { label: 'Session Secret', ok: statusData.auth.session_secret_ok },
                  { label: '前端构建', ok: statusData.frontend.built },
                ]}
              />
              <div className="meta-grid">
                <div><span>LLM 模式</span><strong>{statusData.llm.mock ? '模拟模式' : statusData.llm.mode}</strong></div>
                <div><span>Embedding 模式</span><strong>{statusData.embedding.configured ? 'openai' : '未配置'}</strong></div>
              </div>
              {statusData.warnings.length > 0 && (
                <ul className="warning-list">
                  {statusData.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              )}
              <p className="panel-note">完成后会将模型设置安全写入本机的 `config/.env`，该文件不会提交到 Git。</p>
            </>
          ) : null}
        </section>
      </div>
    </main>
  )
}

function LoginScreen({ onLogin, warnings }: { onLogin: () => void; warnings: string[] }) {
  const [password, setPassword] = useState('')
  const login = useMutation({ mutationFn: api.login, onSuccess: onLogin })

  return (
    <main className="auth-screen">
      <form
        className="auth-panel auth-card"
        onSubmit={(event) => {
          event.preventDefault()
          login.mutate(password)
        }}
      >
        <Database className="h-7 w-7 text-slate-950" />
        <h1>Facet</h1>
        <p className="auth-brandline">Knowledge, in focus.</p>
        <p className="auth-copy">输入管理员密码进入工作台。</p>
        <input
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          type="password"
          placeholder="管理员密码"
          autoFocus
        />
        {login.isError && <p className="form-error">{getLoginError(login.error)}</p>}
        {warnings.length > 0 && (
          <ul className="warning-list compact-warnings">
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        )}
        <button type="submit" disabled={!password || login.isPending}>
          <Play className="h-4 w-4" /> 进入工作台
        </button>
      </form>
    </main>
  )
}

function AppShell({
  bootstrap,
  onLogout,
  notify,
}: {
  bootstrap: AuthBootstrap
  onLogout: () => void
  notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void
}) {
  const navigate = useNavigate()
  const locationStorageKey = chatLocationStorageKey(bootstrap.principal?.principal_id)
  const [initialChatLocation] = useState(() => (
    parseStoredChatLocation(localStorage.getItem(locationStorageKey))
  ))
  const logout = useMutation({
    mutationFn: api.logout,
    onSuccess: async () => {
      localStorage.removeItem(locationStorageKey)
      await onLogout()
      navigate('/documents')
    },
  })
  const [activeConvId, setActiveConvId] = useState<string | null>(initialChatLocation.conversationId)
  const [newConversationViewId, setNewConversationViewId] = useState(() => (
    initialChatLocation.newConversationViewId || createNewConversationViewId()
  ))
  const startNewConversation = () => {
    setActiveConvId(null)
    setNewConversationViewId(createNewConversationViewId())
  }

  useEffect(() => {
    localStorage.setItem(locationStorageKey, JSON.stringify({
      conversationId: activeConvId,
      newConversationViewId,
    }))
  }, [activeConvId, locationStorageKey, newConversationViewId])

  return (
    <div className="app-shell">
      <LifecycleTaskMonitor notify={notify} />
      <KnowledgeBaseRevisionMonitor />
      <DocumentRevisionMonitor />
      <aside className="sidebar">
        <div className="brand">
          <Database className="h-5 w-5" />
          <span>Facet</span>
        </div>
        <div className="sidebar-user">
          <span className="sidebar-user-name">{bootstrap.principal?.name || '管理员'}</span>
          <span className="sidebar-user-meta">{bootstrap.principal?.is_admin ? '本地管理员' : '已登录'}</span>
        </div>
        <nav>
          <SideLink to="/documents" icon={<FileText />}>文档</SideLink>
          <SideLink to="/chat" icon={<MessageSquare />}>聊天</SideLink>
          <SideLink to="/api-keys" icon={<KeyRound />}>Agent 密钥</SideLink>
          <SideLink to="/settings" icon={<Settings />}>设置</SideLink>
        </nav>
        <ConversationSidebarPanel
          activeId={activeConvId}
          setActiveId={setActiveConvId}
          onNewConversation={startNewConversation}
          notify={notify}
        />
        <button className="ghost-button mt-auto" onClick={() => logout.mutate()}>
          <LogOut className="h-4 w-4" /> 退出
        </button>
      </aside>
      <Routes>
        <Route path="/documents" element={<DocumentsPage notify={notify} />} />
        <Route path="/chat" element={(
          <ChatPage
            notify={notify}
            conversationId={activeConvId}
            newConversationViewId={newConversationViewId}
            setConversationId={setActiveConvId}
          />
        )} />
        <Route path="/api-keys" element={<AgentKeysPage notify={notify} />} />
        <Route path="/settings" element={<SettingsPage notify={notify} />} />
        <Route path="*" element={<Navigate to="/documents" replace />} />
      </Routes>
    </div>
  )
}

function LifecycleTaskMonitor({
  notify,
}: {
  notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void
}) {
  const queryClient = useQueryClient()
  const startedAt = useRef(Date.now())
  const seenTerminalKeys = useRef(new Set<string>())
  const queue = useQuery({
    queryKey: ['document-queue'],
    queryFn: api.documentQueue,
    refetchInterval: (query) => query.state.data?.items.some(
      (item) => item.status === 'queued' || item.status === 'running',
    ) ? 2500 : false,
  })

  useEffect(() => {
    const lifecycleItems = [...(queue.data?.items ?? []), ...(queue.data?.history ?? [])]
    for (const item of lifecycleItems) {
      const isKnowledgeBaseDelete = item.job_type === 'knowledge_base_delete'
      const isDocumentDelete = item.job_type === 'index_candidate' && item.reason === 'document_delete'
      const isIndexCandidate = item.job_type === 'index_candidate'
      if (isIndexCandidate && item.kb_id) {
        // 候选任务不修改文档行：索引卡必须直接跟随任务状态，而不能等文档下一次变化。
        queryClient.invalidateQueries({ queryKey: ['knowledge-base-indexes', item.kb_id] })
      }
      if (!isKnowledgeBaseDelete && !isDocumentDelete && !isIndexCandidate) continue
      if (item.status !== 'succeeded' && item.status !== 'failed') continue

      const terminalKey = item.job_id
        ? `${item.job_id}:${item.status}`
        : `${item.kind}:${item.kb_id || item.doc_id || item.title}:${item.updated_at || item.status}`
      if (seenTerminalKeys.current.has(terminalKey)) continue
      seenTerminalKeys.current.add(terminalKey)

      const updatedAt = Date.parse(item.updated_at || '')
      if (Number.isFinite(updatedAt) && updatedAt + 1000 < startedAt.current) continue

      const snapshotRetrying = isIndexCandidate && item.status === 'failed'
        && item.detail.includes('文档已变化')
      if (snapshotRetrying) continue

      if (isIndexCandidate && !isDocumentDelete) {
        if (item.status === 'succeeded') {
          notify(`知识库索引已更新：${item.title}`)
        } else {
          notify(`知识库索引更新失败：${item.title}。当前会继续使用上一个可用索引，请在处理队列查看原因。`, 'error', 6000)
        }
      } else if (item.status === 'succeeded') {
        notify(`${isKnowledgeBaseDelete ? '知识库' : '文档'}删除完成：${item.title}`)
      } else {
        notify(
          `${isKnowledgeBaseDelete ? '知识库' : '文档'}删除失败：${item.title}，请在处理队列中重试。`,
          'error',
          6000,
        )
      }
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] })
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-indexes'] })
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    }
  }, [notify, queue.data, queryClient])

  return null
}

function KnowledgeBaseRevisionMonitor() {
  const queryClient = useQueryClient()
  const previousRevision = useRef<string | null>(null)
  const revision = useQuery({
    queryKey: ['knowledge-base-revision'],
    queryFn: api.knowledgeBaseRevision,
    refetchInterval: 15000,
  })

  useEffect(() => {
    if (!revision.data) return
    const current = `${revision.data.item_count}:${revision.data.updated_at}`
    if (previousRevision.current !== null && previousRevision.current !== current) {
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] })
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-indexes'] })
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
    }
    previousRevision.current = current
  }, [queryClient, revision.data])

  return null
}

function DocumentRevisionMonitor() {
  const queryClient = useQueryClient()
  const previousRevision = useRef<string | null>(null)
  const revision = useQuery({
    queryKey: ['document-revision'],
    queryFn: api.documentRevision,
    refetchInterval: 15000,
  })

  useEffect(() => {
    if (!revision.data) return
    const current = `${revision.data.item_count}:${revision.data.updated_at}`
    if (previousRevision.current !== null && previousRevision.current !== current) {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-indexes'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    }
    previousRevision.current = current
  }, [queryClient, revision.data])

  return null
}

function SideLink({ to, icon, children }: { to: string; icon: ReactNode; children: string }) {
  return (
    <NavLink to={to} className={({ isActive }) => cn('side-link', isActive && 'active')}>
      <span className="side-icon">{icon}</span>
      {children}
    </NavLink>
  )
}

function ConversationSidebarPanel({
  activeId,
  setActiveId,
  onNewConversation,
  notify,
}: {
  activeId: string | null
  setActiveId: (id: string | null) => void
  onNewConversation: () => void
  notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void
}) {
  const location = useLocation()
  if (location.pathname !== '/chat') return null
  return (
    <ConversationListContent
      activeId={activeId}
      setActiveId={setActiveId}
      onNewConversation={onNewConversation}
      notify={notify}
    />
  )
}

function groupConversations(conversations: Conversation[]) {
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterdayStart = new Date(todayStart.getTime() - 86400000)
  const weekStart = new Date(todayStart.getTime() - 7 * 86400000)
  const monthStart = new Date(todayStart.getTime() - 30 * 86400000)
  const groups: { label: string; items: Conversation[] }[] = [
    { label: '今天', items: [] },
    { label: '昨天', items: [] },
    { label: '最近 7 天', items: [] },
    { label: '最近 30 天', items: [] },
    { label: '更早', items: [] },
  ]
  for (const conversation of conversations) {
    const timestamp = new Date(conversation.last_message_at || conversation.updated_at)
    if (timestamp >= todayStart) groups[0].items.push(conversation)
    else if (timestamp >= yesterdayStart) groups[1].items.push(conversation)
    else if (timestamp >= weekStart) groups[2].items.push(conversation)
    else if (timestamp >= monthStart) groups[3].items.push(conversation)
    else groups[4].items.push(conversation)
  }
  return groups.filter((group) => group.items.length > 0)
}

function ConversationListContent({
  activeId,
  setActiveId,
  onNewConversation,
  notify,
}: {
  activeId: string | null
  setActiveId: (id: string | null) => void
  onNewConversation: () => void
  notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void
}) {
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectMode, setSelectMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [lastClickedId, setLastClickedId] = useState<string | null>(null)
  const [showConfirm, setShowConfirm] = useState(false)
  const [conversationToDelete, setConversationToDelete] = useState<string | null>(null)

  const conversations = useQuery({ queryKey: ['conversations'], queryFn: api.conversations })
  const allConversations = conversations.data?.conversations || []
  const filtered = searchQuery
    ? allConversations.filter((conversation) => conversation.title.toLowerCase().includes(searchQuery.toLowerCase()))
    : allConversations
  const grouped = groupConversations(filtered)

  const deleteConversation = useMutation({
    mutationFn: (id: string) => api.deleteConversation(id),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
      if (activeId === id) onNewConversation()
      setConversationToDelete(null)
      notify('会话已删除。')
    },
    onError: (error) => notify(getApiMessage(error, '删除会话失败。'), 'error'),
  })

  const batchDelete = useMutation({
    mutationFn: (ids: string[]) => api.deleteConversations(ids),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
      if (activeId && selectedIds.has(activeId)) onNewConversation()
      setSelectedIds(new Set())
      setSelectMode(false)
      notify('选中的会话已删除。')
    },
    onError: (error) => notify(getApiMessage(error, '批量删除会话失败。'), 'error'),
  })

  const toggleSelect = (id: string, shiftKey: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (shiftKey && lastClickedId) {
        const flat = filtered.map((conversation) => conversation.conversation_id)
        const start = flat.indexOf(lastClickedId)
        const end = flat.indexOf(id)
        if (start !== -1 && end !== -1) {
          const [lo, hi] = start < end ? [start, end] : [end, start]
          for (let index = lo; index <= hi; index += 1) next.add(flat[index])
        }
      } else if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
    setLastClickedId(id)
  }

  return (
    <div className="sidebar-conversations">
      <div className="sidebar-conv-header">
        <span className="text-[11px] font-semibold text-[var(--c-text-muted)]">
          {selectMode ? `已选 ${selectedIds.size}` : '会话'}
        </span>
        <div className="flex items-center gap-1">
          {selectMode && selectedIds.size > 0 && (
            <button
              className="conversation-row-btn text-[var(--c-danger)]"
              onClick={() => setShowConfirm(true)}
              title={`删除 ${selectedIds.size} 个`}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
          <button
            className={cn('conversation-row-btn', selectMode && 'text-[var(--c-accent)]')}
            onClick={() => {
              setSelectMode(!selectMode)
              setSelectedIds(new Set())
            }}
            title={selectMode ? '取消选择' : '批量管理'}
          >
            {selectMode ? <Square className="h-3.5 w-3.5" /> : <ListChecks className="h-3.5 w-3.5" />}
          </button>
          <button
            className="conversation-row-btn"
            onClick={() => {
              onNewConversation()
              queryClient.invalidateQueries({ queryKey: ['conversations'] })
            }}
            title="新对话"
          >
            <Pencil className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      <div className="px-3 pb-2">
        <input className="sidebar-search" placeholder="搜索会话..." value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} />
      </div>
      <div className="sidebar-conv-items">
        {selectMode && (
          <button
            className="conversation-row w-full"
            onClick={() => {
              const allIds = filtered.map((conversation) => conversation.conversation_id)
              setSelectedIds(selectedIds.size === allIds.length ? new Set() : new Set(allIds))
            }}
          >
            {selectedIds.size === filtered.length && filtered.length > 0 ? (
              <CheckSquare className="h-3.5 w-3.5 shrink-0 text-[var(--c-accent)]" />
            ) : (
              <Square className="h-3.5 w-3.5 shrink-0 text-[var(--c-text-muted)]" />
            )}
            <span className="conversation-row-title text-[var(--c-text-muted)]">
              {selectedIds.size === filtered.length && filtered.length > 0 ? '取消全选' : '全选'}
            </span>
          </button>
        )}
        {grouped.map((group) => (
          <div key={group.label}>
            <div className="conversation-group-label">{group.label}</div>
            {group.items.map((conversation) => (
              <div
                key={conversation.conversation_id}
                className={cn(
                  'conversation-row group',
                  activeId === conversation.conversation_id && !selectMode && 'active',
                  selectMode && selectedIds.has(conversation.conversation_id) && 'bg-[var(--c-accent-light)]',
                )}
              >
                <button
                  type="button"
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  onClick={(event) => {
                    if (selectMode) toggleSelect(conversation.conversation_id, event.shiftKey)
                    else setActiveId(conversation.conversation_id)
                  }}
                >
                  {selectMode && (
                    selectedIds.has(conversation.conversation_id) ? (
                      <CheckSquare className="h-3.5 w-3.5 shrink-0 text-[var(--c-accent)]" />
                    ) : (
                      <Square className="h-3.5 w-3.5 shrink-0 text-[var(--c-text-muted)]" />
                    )
                  )}
                  <span className="conversation-row-title">{conversation.title}</span>
                </button>
                {!selectMode && (
                  <div className="conversation-row-actions">
                    <button
                      type="button"
                      className="conversation-row-btn"
                      onClick={() => setConversationToDelete(conversation.conversation_id)}
                      title="删除会话"
                      aria-label={`删除会话：${conversation.title}`}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
        {filtered.length === 0 && (
          <p className="px-4 py-8 text-center text-xs text-[var(--c-text-muted)]">
            {searchQuery ? '没有匹配的会话' : '还没有会话'}
          </p>
        )}
      </div>
      {selectMode && selectedIds.size > 0 && (
        <div className="border-t border-[var(--c-border-light)] px-4 py-2.5">
          <button className="w-full rounded-lg bg-[var(--c-danger-light)] px-3 py-2 text-xs font-medium text-[var(--c-danger)] transition hover:bg-red-100" onClick={() => setShowConfirm(true)}>
            删除选中的 {selectedIds.size} 个会话
          </button>
        </div>
      )}
      <ConfirmDialog
        open={showConfirm}
        title="批量删除会话"
        message={`确定要删除选中的 ${selectedIds.size} 个会话吗？此操作不可撤销。`}
        confirmLabel="删除"
        onConfirm={() => {
          setShowConfirm(false)
          batchDelete.mutate([...selectedIds])
        }}
        onCancel={() => setShowConfirm(false)}
      />
      <ConfirmDialog
        open={conversationToDelete !== null}
        title="删除会话"
        message="确定要删除这个会话吗？会话中的消息也会一并删除，此操作不可撤销。"
        confirmLabel="删除"
        onConfirm={() => {
          if (conversationToDelete) deleteConversation.mutate(conversationToDelete)
        }}
        onCancel={() => setConversationToDelete(null)}
      />
    </div>
  )
}

function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  onConfirm: () => void
  onCancel: () => void
}) {
  if (!open) return null
  return (
    <div className="dialog-overlay" onClick={onCancel}>
      <div className="dialog-panel" onClick={(event) => event.stopPropagation()}>
        <h3 className="dialog-title">{title}</h3>
        <p className="dialog-message">{message}</p>
        <div className="dialog-actions">
          <button className="ghost-button" onClick={onCancel}>取消</button>
          <button className="danger-button" onClick={onConfirm}>{confirmLabel || '确认'}</button>
        </div>
      </div>
    </div>
  )
}

function KnowledgeBaseDialog({
  open,
  submitting,
  onCancel,
  onCreate,
}: {
  open: boolean
  submitting: boolean
  onCancel: () => void
  onCreate: (name: string) => void
}) {
  const [name, setName] = useState('')

  useEffect(() => {
    if (open) setName('')
  }, [open])

  if (!open) return null
  const normalizedName = name.trim()
  return (
    <div className="dialog-overlay" onClick={() => !submitting && onCancel()}>
      <form
        className="dialog-panel"
        onClick={(event) => event.stopPropagation()}
        onSubmit={(event) => {
          event.preventDefault()
          if (normalizedName && !submitting) onCreate(normalizedName)
        }}
      >
        <h3 className="dialog-title">新建知识库</h3>
        <p className="dialog-message">知识库用于独立管理文档，并可在会话中单独限定检索范围。</p>
        <label className="mt-4 block text-sm font-medium text-[var(--c-text)]" htmlFor="knowledge-base-name">名称</label>
        <input
          autoFocus
          id="knowledge-base-name"
          className="mt-2 w-full"
          maxLength={80}
          placeholder="例如：产品资料"
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
        <div className="dialog-actions">
          <button className="ghost-button" disabled={submitting} type="button" onClick={onCancel}>取消</button>
          <button className="primary-button" disabled={!normalizedName || submitting} type="submit">
            {submitting ? '创建中...' : '创建知识库'}
          </button>
        </div>
      </form>
    </div>
  )
}

function DocumentsPage({ notify }: { notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void }) {
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const dragDepthRef = useRef(0)
  const [isDragActive, setIsDragActive] = useState(false)
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState<string | null>(null)
  const [createKnowledgeBaseOpen, setCreateKnowledgeBaseOpen] = useState(false)
  const [deleteKnowledgeBaseOpen, setDeleteKnowledgeBaseOpen] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState<{ doc_id: string; filename: string } | null>(null)
  const knowledgeBases = useQuery({ queryKey: ['knowledge-bases'], queryFn: api.knowledgeBases })
  const selectedKnowledgeBase = knowledgeBases.data?.knowledge_bases.find(
    (knowledgeBase) => knowledgeBase.kb_id === selectedKnowledgeBaseId,
  )
  useEffect(() => {
    const available = knowledgeBases.data?.knowledge_bases ?? []
    if (!available.length) return
    if (selectedKnowledgeBaseId && available.some((item) => item.kb_id === selectedKnowledgeBaseId)) return
    // 上传始终有明确归属：首次进入默认选中默认知识库，删除当前库后也回到它。
    setSelectedKnowledgeBaseId(available.find((item) => item.slug === 'default')?.kb_id ?? available[0].kb_id)
  }, [knowledgeBases.data, selectedKnowledgeBaseId])
  const documents = useQuery({
    queryKey: ['documents', selectedKnowledgeBaseId],
    queryFn: () => api.documents(selectedKnowledgeBaseId),
    enabled: Boolean(selectedKnowledgeBaseId),
    refetchInterval: (query) =>
      query.state.data?.documents.some((doc) =>
        doc.status === 'processing'
        || doc.status === 'reindex_queued'
        || doc.status === 'reindexing'
        || doc.status === 'deleting') ? 2500 : false,
  })
  const queue = useQuery({
    queryKey: ['document-queue'],
    queryFn: api.documentQueue,
    refetchInterval: (query) => query.state.data?.items.some(
      (item) => item.status === 'queued' || item.status === 'running',
    ) ? 2500 : false,
  })
  const hasPendingIndexCandidate = Boolean(selectedKnowledgeBaseId && queue.data?.items.some(
    (item) => (
      item.job_type === 'index_candidate'
      && item.kb_id === selectedKnowledgeBaseId
      && (item.status === 'queued' || item.status === 'running')
    ),
  ))
  const knowledgeBaseIndexes = useQuery({
    queryKey: ['knowledge-base-indexes', selectedKnowledgeBaseId],
    queryFn: () => api.knowledgeBaseIndexes(selectedKnowledgeBaseId!),
    enabled: Boolean(selectedKnowledgeBaseId),
    refetchInterval: (query) => {
      const sourceUpdating = documents.data?.documents.some((doc) =>
        doc.status === 'processing' || doc.status === 'reindex_queued' || doc.status === 'reindexing')
      return sourceUpdating || hasPendingIndexCandidate || query.state.data?.indexes.some((index) => index.status === 'building') ? 2500 : false
    },
  })
  const upload = useMutation({
    mutationFn: (file: File) => api.uploadDocument(file, selectedKnowledgeBaseId),
    onSuccess: (data) => {
      notify(data.message || '文档已进入后台处理')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error) => notify(getDocumentActionError(error, '上传失败。'), 'error'),
  })
  const batchUpload = useMutation({
    mutationFn: (files: File[]) => api.batchUploadDocuments(files, selectedKnowledgeBaseId),
    onSuccess: (data) => {
      const prepared = data.prepared ?? []
      const acceptedCount = prepared.filter((item) => item.status === 'processing' || item.status === 'ready').length
      const failedCount = prepared.filter((item) => item.status === 'failed' || item.status === 'conflict').length
      const summary = [
        data.message || '文档已加入后台处理',
        acceptedCount > 0 ? `成功 ${acceptedCount}` : '',
        failedCount > 0 ? `异常 ${failedCount}` : '',
      ].filter(Boolean).join('，')
      notify(summary, failedCount > 0 && acceptedCount === 0 ? 'error' : 'ok')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error) => notify(getDocumentActionError(error, '批量上传失败。'), 'error'),
  })
  const createKnowledgeBase = useMutation({
    mutationFn: api.createKnowledgeBase,
    onSuccess: (knowledgeBase) => {
      setSelectedKnowledgeBaseId(knowledgeBase.kb_id)
      setCreateKnowledgeBaseOpen(false)
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] })
      notify(`已创建知识库：${knowledgeBase.name}`)
    },
    onError: (error) => notify(getApiMessage(error, '创建知识库失败。'), 'error'),
  })
  const deleteKnowledgeBase = useMutation({
    mutationFn: api.deleteKnowledgeBase,
    onSuccess: (data) => {
      setDeleteKnowledgeBaseOpen(false)
      setSelectedKnowledgeBaseId(null)
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] })
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['knowledge-base-indexes'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      notify(data.message || '知识库已进入删除队列。')
    },
    onError: (error) => notify(getApiMessage(error, '删除知识库失败。'), 'error'),
  })
  const remove = useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: (data) => {
      setDocumentToDelete(null)
      notify(data.message || (data.status === 'deleted' ? '文档已删除。' : '文档已进入安全删除队列。'))
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error) => notify(getDocumentActionError(error, '删除失败。'), 'error'),
  })
  const reingest = useMutation({
    mutationFn: api.reingestDocument,
    onSuccess: (data) => {
      notify(data.message || '正在重新摄入')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error) => notify(getDocumentActionError(error, '重新摄入失败。'), 'error'),
  })
  const uploadBusy = upload.isPending || batchUpload.isPending

  const submitFiles = (incoming: File[] | FileList | null | undefined) => {
    const files = Array.from(incoming ?? []).filter((file) => file.size > 0)
    if (!files.length) return
    if (files.length === 1) {
      upload.mutate(files[0])
      return
    }
    batchUpload.mutate(files)
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    submitFiles(event.target.files)
    event.currentTarget.value = ''
  }

  const openFilePicker = () => {
    if (uploadBusy) return
    fileInputRef.current?.click()
  }

  const handleDragEnter = (event: DragEvent<HTMLElement>) => {
    if (uploadBusy || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepthRef.current += 1
    setIsDragActive(true)
  }

  const handleDragOver = (event: DragEvent<HTMLElement>) => {
    if (uploadBusy || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event: DragEvent<HTMLElement>) => {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setIsDragActive(false)
  }

  const handleDrop = (event: DragEvent<HTMLElement>) => {
    if (uploadBusy) return
    event.preventDefault()
    dragDepthRef.current = 0
    setIsDragActive(false)
    submitFiles(event.dataTransfer.files)
  }

  return (
    <main className="page">
      <div className="documents-layout">
        <section className="documents-main">
          <header className="page-header">
            <div>
              <p className="eyebrow">Documents</p>
              <h1>文档库</h1>
              <p className="page-copy">文件会上传到下方选择的知识库；首次默认使用“默认知识库”。</p>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              <label className="flex items-center gap-2 text-sm text-[var(--c-text-secondary)]">
                <span className="whitespace-nowrap">上传到</span>
                <select
                  className="rounded-lg border border-[var(--c-border)] bg-[var(--c-surface)] px-3 py-2 text-sm text-[var(--c-text)]"
                  value={selectedKnowledgeBaseId || ''}
                  disabled={uploadBusy || knowledgeBases.isLoading}
                  onChange={(event) => setSelectedKnowledgeBaseId(event.target.value || null)}
                  aria-label="上传目标知识库"
                >
                  {!selectedKnowledgeBaseId && <option value="">正在选择默认知识库…</option>}
                  {knowledgeBases.data?.knowledge_bases.map((knowledgeBase) => (
                    <option key={knowledgeBase.kb_id} value={knowledgeBase.kb_id}>{knowledgeBase.name}</option>
                  ))}
                </select>
              </label>
              <button
                className="ghost-button compact"
                disabled={knowledgeBases.isFetching}
                type="button"
                title="刷新知识库列表"
                onClick={() => {
                  void knowledgeBases.refetch()
                  void documents.refetch()
                }}
              >
                <RefreshCw className={cn('h-4 w-4', knowledgeBases.isFetching && 'animate-spin')} />
              </button>
              <button
                className="ghost-button"
                disabled={uploadBusy || createKnowledgeBase.isPending}
                type="button"
                onClick={() => setCreateKnowledgeBaseOpen(true)}
              >新建知识库</button>
              {selectedKnowledgeBase && selectedKnowledgeBase.slug !== 'default' && (
                <button
                  className="danger-button"
                  disabled={uploadBusy || deleteKnowledgeBase.isPending}
                  type="button"
                  onClick={() => setDeleteKnowledgeBaseOpen(true)}
                >
                  <Trash2 className="h-4 w-4" /> 删除知识库
                </button>
              )}
              <button className="primary-button" disabled={uploadBusy || !selectedKnowledgeBaseId} type="button" onClick={openFilePicker}>
                <Upload className="h-4 w-4" /> 上传文档
              </button>
            </div>
          </header>
          <input
            ref={fileInputRef}
            accept={DOCUMENT_ACCEPT}
            hidden
            multiple
            type="file"
            onChange={handleFileChange}
          />
          <section
            className={cn(
              'upload-dropzone',
              isDragActive && 'drag-active',
              uploadBusy && 'is-disabled',
            )}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="upload-dropzone-copy">
              <div className="upload-dropzone-icon">
                <Upload className="h-5 w-5" />
              </div>
              <div>
                <h2>拖拽文档到这里，或直接选择文件</h2>
                <p>支持 PDF、TXT、Markdown、DOCX 和 HTML，可一次选择多个文件。</p>
              </div>
            </div>
            <div className="upload-dropzone-actions">
              <button className="ghost-button" disabled={uploadBusy} type="button" onClick={openFilePicker}>
                <Upload className="h-4 w-4" /> 选择文件
              </button>
            </div>
          </section>
          {selectedKnowledgeBaseId && (
            <KnowledgeBaseIndexesPanel
              indexes={knowledgeBaseIndexes.data?.indexes ?? []}
              loading={knowledgeBaseIndexes.isLoading}
              pending={hasPendingIndexCandidate}
            />
          )}
          {!documents.isLoading && selectedKnowledgeBaseId && (documents.data?.documents.length ?? 0) === 0 ? (
            <section className="empty-state guided-empty">
              <div>
                <FileText className="mx-auto mb-3 h-8 w-8 text-[var(--c-text-muted)]" />
                <h2>这个知识库还没有文档</h2>
                <p>上传的资料只会进入当前选中的知识库，并独立参与后续会话检索。</p>
                <button className="primary-button mt-4" disabled={uploadBusy} type="button" onClick={openFilePicker}>
                  <Upload className="h-4 w-4" /> 上传第一个文档
                </button>
              </div>
            </section>
          ) : (
            <section className="document-grid">
              {documents.data?.documents.map((doc) => (
                <article className="item-card" key={doc.doc_id}>
                  <div className="item-card-head">
                    <FileText className="h-5 w-5" />
                    <StatusBadge status={doc.status} />
                  </div>
                  <h2>{doc.filename}</h2>
                  <div className="item-meta">
                    {doc.kb_id && <span>{knowledgeBases.data?.knowledge_bases.find((item) => item.kb_id === doc.kb_id)?.name || '知识库'}</span>}
                    <span>{formatBytes(doc.file_size)}</span>
                    <span>{doc.chunks_count ?? 0} chunks</span>
                    {doc.index_pipeline_version && <span>{doc.index_pipeline_version === 'parent_child_v1' ? '结构化索引' : '待升级索引'}</span>}
                    <span>{shortDate(doc.updated_at || doc.created_at)}</span>
                  </div>
                  {doc.error_message && <p className="error-text">{doc.error_message}</p>}
                  <p className="card-note">
                    {doc.status === 'reindex_queued' ? '检测到模型变化，已加入后台重建队列。'
                      : doc.status === 'reindexing' ? '正在用当前 embedding 模型重建索引。'
                        : doc.status_reason === 'model_mismatch' ? '原文件已保留，系统会基于当前模型重建索引。'
                          : doc.status_reason === 'empty_content' ? '文档没有解析出可检索文本，建议换可复制文本版本或补充 OCR 后再重新摄入。'
                          : doc.status === 'failed' ? '原文件已保留，可直接重试摄入。'
                            : '文档已可用于检索。'}
                  </p>
                  <div className="flex gap-2">
                    {(doc.status === 'failed' || doc.status === 'ready') && (
                      <button className="ghost-button" disabled={reingest.isPending} onClick={() => reingest.mutate(doc.doc_id)}>
                        <RefreshCw className="h-4 w-4" /> 重新摄入
                      </button>
                    )}
                    <button
                      className="danger-button"
                      disabled={doc.status === 'processing' || doc.status === 'reindexing' || remove.isPending}
                      onClick={() => setDocumentToDelete({ doc_id: doc.doc_id, filename: doc.filename })}
                    >
                      <Trash2 className="h-4 w-4" /> 删除
                    </button>
                  </div>
                </article>
              ))}
            </section>
          )}
        </section>
        <DocumentsQueuePanel notify={notify} />
      </div>
      <KnowledgeBaseDialog
        open={createKnowledgeBaseOpen}
        submitting={createKnowledgeBase.isPending}
        onCancel={() => setCreateKnowledgeBaseOpen(false)}
        onCreate={(name) => createKnowledgeBase.mutate(name)}
      />
      <ConfirmDialog
        open={deleteKnowledgeBaseOpen && Boolean(selectedKnowledgeBase)}
        title="删除知识库"
        message={`确定要删除“${selectedKnowledgeBase?.name || '当前知识库'}”吗？该操作会异步清理其中的文档、文件、索引、向量和缓存，且不可撤销。`}
        confirmLabel={deleteKnowledgeBase.isPending ? '删除中…' : '确认删除'}
        onConfirm={() => {
          if (!selectedKnowledgeBaseId || deleteKnowledgeBase.isPending) return
          deleteKnowledgeBase.mutate(selectedKnowledgeBaseId)
        }}
        onCancel={() => !deleteKnowledgeBase.isPending && setDeleteKnowledgeBaseOpen(false)}
      />
      <ConfirmDialog
        open={Boolean(documentToDelete)}
        title="删除文档"
        message={`确定要删除“${documentToDelete?.filename || '当前文档'}”吗？系统会安全更新索引并清理文件、向量和缓存，此操作不可撤销。`}
        confirmLabel={remove.isPending ? '删除中…' : '确认删除'}
        onConfirm={() => {
          if (!documentToDelete || remove.isPending) return
          remove.mutate(documentToDelete.doc_id)
        }}
        onCancel={() => !remove.isPending && setDocumentToDelete(null)}
      />
    </main>
  )
}

function KnowledgeBaseIndexesPanel({
  indexes,
  loading,
  pending,
}: {
  indexes: KnowledgeBaseIndex[]
  loading: boolean
  pending: boolean
}) {
  const activeIndex = indexes.find((index) => index.status === 'active')
  const building = indexes.some((index) => index.status === 'building')
  const failedIndex = indexes.find((index) => index.status === 'failed')
  return (
    <section className="mb-5 rounded-xl border border-[var(--c-border)] bg-[var(--c-surface)] p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-[var(--c-text)]">检索索引</h2>
          <p className="mt-1 text-xs text-[var(--c-text-muted)]">文档变更后会自动重建并安全切换，过程中不影响正在使用的检索。</p>
        </div>
        {loading && <span className="text-xs text-[var(--c-text-muted)]">读取中…</span>}
      </div>
      <p className="mt-3 text-sm text-[var(--c-text-secondary)]">
        {pending || building
          ? '正在更新索引；完成后会自动切换。'
          : failedIndex
            ? activeIndex
              ? `最近一次更新失败；当前仍使用 ${activeIndex.chunk_count ?? 0} 个切片的可用索引。请在处理队列查看原因。`
              : '索引构建失败；请在处理队列查看原因并处理后重试。'
          : activeIndex
            ? `已就绪 · ${activeIndex.chunk_count ?? 0} 个切片`
            : '等待首个文档处理完成后自动建立。'}
      </p>
    </section>
  )
}

function StatusBadge({ status }: { status: string }) {
  const labelMap: Record<string, string> = {
    ready: '已就绪',
    processing: '处理中',
    reindex_queued: '等待重建',
    reindexing: '重建中',
    failed: '失败',
    deleting: '删除中',
    delete_failed: '删除失败',
  }
  return <span className={cn('status-badge', `status-${status}`)}>{labelMap[status] || status}</span>
}

function DocumentsQueuePanel({ notify }: { notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void }) {
  const queryClient = useQueryClient()
  const queue = useQuery({
    queryKey: ['document-queue'],
    queryFn: api.documentQueue,
    refetchInterval: (query) => query.state.data?.items.some(
      (item) => item.status === 'queued' || item.status === 'running',
    ) ? 2500 : false,
  })
  const queueItems = queue.data?.items
  const queueHistory = queue.data?.history
  const items = useMemo(() => queueItems ?? [], [queueItems])
  const history = useMemo(() => queueHistory ?? [], [queueHistory])
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [selectionMode, setSelectionMode] = useState<'auto' | 'manual'>('auto')
  const [queueTab, setQueueTab] = useState<'active' | 'history'>('active')
  const clearHistory = useMutation({
    mutationFn: api.clearDocumentQueueHistory,
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      notify(`已清空 ${result.deleted} 条历史记录。`, 'ok')
    },
  })
  const visibleItems = useMemo(() => (queueTab === 'active' ? items : history), [history, items, queueTab])
  const activeDocumentItems = useMemo(() => items.filter((item) => item.kind === 'document'), [items])
  const activeBm25Items = useMemo(() => items.filter((item) => item.kind === 'bm25'), [items])
  const activeKnowledgeBaseItems = useMemo(() => items.filter((item) => item.kind === 'knowledge_base'), [items])

  const retryKnowledgeBaseDelete = useMutation({
    mutationFn: (kbId: string) => api.deleteKnowledgeBase(kbId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      notify(data.message || '已重新加入知识库删除队列。')
    },
    onError: (error) => notify(getApiMessage(error, '重试删除知识库失败。'), 'error'),
  })
  const retryDocumentDelete = useMutation({
    mutationFn: (docId: string) => api.deleteDocument(docId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      queryClient.invalidateQueries({ queryKey: ['document-queue'] })
      notify(data.message || '已重新加入文档删除队列。')
    },
    onError: (error) => notify(getDocumentActionError(error, '重试删除文档失败。'), 'error'),
  })

  useEffect(() => {
    if (queueTab === 'history') {
      void queue.refetch()
    }
  }, [queue, queueTab])

  useEffect(() => {
    const currentItems = visibleItems
    if ((items.length ?? 0) === 0 && (history.length ?? 0) === 0) {
      if (selectedKey !== null) setSelectedKey(null)
      if (selectionMode !== 'auto') setSelectionMode('auto')
      return
    }
    const selectedExists = selectedKey ? currentItems.some((item) => item.key === selectedKey) : false
    if (currentItems.length > 0) {
      if (!selectedExists) {
        setSelectedKey(currentItems[0].key)
        setSelectionMode('auto')
      }
      return
    }
    if (selectedKey !== null) setSelectedKey(null)
    if (selectionMode !== 'auto') setSelectionMode('auto')
  }, [history.length, items.length, queueTab, selectedKey, selectionMode, visibleItems])

  const selectedItem = selectedKey ? visibleItems.find((item) => item.key === selectedKey) ?? null : null
  // 历史记录的重点是连续浏览；把详情卡留给当前任务，避免它占掉
  // 历史列表的大部分可视高度。历史行本身已包含状态、说明和时间。
  const detailItem = queueTab === 'active' ? selectedItem : null

  return (
    <aside className="tool-panel queue-panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Queue</p>
          <h1>处理队列</h1>
        </div>
        <div className="panel-actions">
          <button
            className="ghost-button compact"
            onClick={() => clearHistory.mutate()}
            disabled={history.length === 0 || clearHistory.isPending}
            title="清空历史记录"
          >
            <Trash2 className="h-4 w-4" /> 清空历史
          </button>
          <ListChecks className="h-5 w-5 text-[var(--c-text-muted)]" />
        </div>
      </div>
      <p className="panel-copy">这里会显示文档摄入、知识库删除、索引重建和 BM25 任务。</p>
      <div className="queue-tabs" role="tablist" aria-label="处理队列视图切换">
        <button
          className={cn('queue-tab', queueTab === 'active' && 'queue-tab-active')}
          type="button"
          role="tab"
          aria-selected={queueTab === 'active'}
          onClick={() => setQueueTab('active')}
        >
          <span>当前任务</span>
          <strong>{items.length}</strong>
        </button>
        <button
          className={cn('queue-tab', queueTab === 'history' && 'queue-tab-active')}
          type="button"
          role="tab"
          aria-selected={queueTab === 'history'}
          onClick={() => setQueueTab('history')}
        >
          <Clock3 className="h-3.5 w-3.5" />
          <span>历史记录</span>
          <strong>{history.length}</strong>
        </button>
      </div>
      {queue.isLoading ? (
        <p className="panel-copy">正在读取队列状态...</p>
      ) : queue.isError ? (
        <p className="form-error">队列状态读取失败，请稍后再试。</p>
      ) : (
        <div className="queue-body">
          {detailItem ? (
            <QueueDetailCard
              item={detailItem}
              retrying={retryKnowledgeBaseDelete.isPending || retryDocumentDelete.isPending}
              onRetryKnowledgeBase={(kbId) => retryKnowledgeBaseDelete.mutate(kbId)}
              onRetryDocument={(docId) => retryDocumentDelete.mutate(docId)}
            />
          ) : null}
          {visibleItems.length === 0 ? (
            <div className="queue-empty">
              <Check className="h-4 w-4 text-[var(--c-success)]" />
              <span>{queueTab === 'active' ? '当前没有进行中的任务。' : '当前没有历史记录。'}</span>
            </div>
          ) : (
            <div className="queue-sections">
              {queueTab === 'active' ? (
                <>
                  {activeDocumentItems.length > 0 && (
                    <QueueSection title="文档处理" count={activeDocumentItems.length}>
                      {activeDocumentItems.map((item) => (
                        <QueueItemRow
                          key={item.key}
                          item={item}
                          showOrigin
                          selected={item.key === selectedKey}
                          onSelect={() => {
                            setSelectedKey(item.key)
                            setSelectionMode('manual')
                          }}
                        />
                      ))}
                    </QueueSection>
                  )}
                  {activeBm25Items.length > 0 && (
                    <QueueSection title="BM25 索引" count={activeBm25Items.length}>
                      {activeBm25Items.map((item) => (
                        <QueueItemRow
                          key={item.key}
                          item={item}
                          showOrigin
                          selected={item.key === selectedKey}
                          onSelect={() => {
                            setSelectedKey(item.key)
                            setSelectionMode('manual')
                          }}
                        />
                      ))}
                    </QueueSection>
                  )}
                  {activeKnowledgeBaseItems.length > 0 && (
                    <QueueSection title="知识库任务" count={activeKnowledgeBaseItems.length}>
                      {activeKnowledgeBaseItems.map((item) => (
                        <QueueItemRow
                          key={item.key}
                          item={item}
                          showOrigin
                          selected={item.key === selectedKey}
                          onSelect={() => {
                            setSelectedKey(item.key)
                            setSelectionMode('manual')
                          }}
                        />
                      ))}
                    </QueueSection>
                  )}
                </>
              ) : (
                <QueueSection title="历史记录" count={history.length}>
                  {history.map((item) => (
                    <QueueItemRow
                      key={item.key}
                      item={item}
                      showOrigin={false}
                      selected={item.key === selectedKey}
                      onSelect={() => {
                        setSelectedKey(item.key)
                        setSelectionMode('manual')
                      }}
                    />
                  ))}
                </QueueSection>
              )}
            </div>
          )}
        </div>
      )}
    </aside>
  )
}

function QueueSection({
  title,
  count,
  children,
}: {
  title: string
  count: number
  children: ReactNode
}) {
  return (
    <section className="queue-section">
      <div className="queue-section-head">
        <span>{title}</span>
        <strong>{count}</strong>
      </div>
      <div className="queue-list">{children}</div>
    </section>
  )
}

function QueueItemRow({
  item,
  showOrigin = true,
  selected,
  onSelect,
}: {
  item: DocumentQueueItem
  showOrigin?: boolean
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button className={cn('queue-item', selected && 'queue-item-selected')} type="button" onClick={onSelect}>
      <div className="queue-item-top">
        <div className="queue-item-title-wrap">
          <span className="queue-item-title">{item.title}</span>
          {showOrigin ? (
            <span className="queue-item-origin">{item.origin === 'history' ? '最近记录' : '当前任务'}</span>
          ) : null}
          <span
            className={cn(
              'queue-pill',
              item.status === 'running' && 'queue-pill-running',
              item.status === 'failed' && 'queue-pill-failed',
              item.status === 'succeeded' && 'queue-pill-success',
              item.status === 'cancelled' && 'queue-pill-cancelled',
            )}
          >
            {item.label}
          </span>
        </div>
        {item.status === 'running' && <RefreshCw className="h-3.5 w-3.5 shrink-0 animate-spin text-[var(--c-accent)]" />}
      </div>
      <p className="queue-item-detail">{item.detail}</p>
      <div className="queue-item-meta">
        <span>{item.kind === 'document' ? '文档任务' : item.kind === 'knowledge_base' ? '知识库任务' : 'BM25 任务'}</span>
        <span>{item.origin === 'history' ? '历史' : '进行中'}</span>
        <span>{shortDate(item.updated_at || item.created_at)}</span>
      </div>
    </button>
  )
}

function QueueDetailCard({
  item,
  retrying,
  onRetryKnowledgeBase,
  onRetryDocument,
}: {
  item: DocumentQueueItem
  retrying: boolean
  onRetryKnowledgeBase: (kbId: string) => void
  onRetryDocument: (docId: string) => void
}) {
  const relatedDocuments = item.related_documents ?? []
  return (
    <section className="queue-detail">
      <div className="queue-detail-head">
        <div className="queue-detail-title-wrap">
          <p className="eyebrow">Detail</p>
          <h2>{item.title}</h2>
        </div>
        <span
          className={cn(
            'queue-pill',
            item.status === 'running' && 'queue-pill-running',
            item.status === 'failed' && 'queue-pill-failed',
            item.status === 'succeeded' && 'queue-pill-success',
            item.status === 'cancelled' && 'queue-pill-cancelled',
          )}
        >
          {item.label}
        </span>
      </div>
      <p className="queue-detail-copy">{item.detail}</p>
      <div className="queue-detail-origin">
        <span>{item.origin === 'history' ? '最近记录' : '当前任务'}</span>
      </div>
      {item.summary && <p className="queue-detail-summary">{item.summary}</p>}
      {relatedDocuments.length > 0 && (
        <div className="queue-detail-docs">
          {relatedDocuments.map((name) => (
            <span key={name}>{name}</span>
          ))}
        </div>
      )}
      <dl className="queue-detail-grid">
        <div>
          <dt>类型</dt>
          <dd>{item.kind === 'document' ? '文档任务' : item.kind === 'knowledge_base' ? '知识库任务' : 'BM25 任务'}</dd>
        </div>
        <div>
          <dt>状态</dt>
          <dd>{item.label}</dd>
        </div>
        {item.doc_id && (
          <div>
            <dt>文档 ID</dt>
            <dd>{item.doc_id}</dd>
          </div>
        )}
        {item.kb_id && (
          <div>
            <dt>知识库 ID</dt>
            <dd>{item.kb_id}</dd>
          </div>
        )}
        {item.collection_name && (
          <div>
            <dt>集合</dt>
            <dd>{item.collection_name}</dd>
          </div>
        )}
        {item.job_type && (
          <div>
            <dt>任务类型</dt>
            <dd>{item.job_type}</dd>
          </div>
        )}
        {item.job_id && (
          <div>
            <dt>任务 ID</dt>
            <dd>{item.job_id}</dd>
          </div>
        )}
        {item.created_at && (
          <div>
            <dt>创建时间</dt>
            <dd>{shortDate(item.created_at)}</dd>
          </div>
        )}
        {item.updated_at && (
          <div>
            <dt>更新时间</dt>
            <dd>{shortDate(item.updated_at)}</dd>
          </div>
        )}
      </dl>
      {item.kind === 'knowledge_base' && item.status === 'failed' && item.kb_id && (
        <div className="mt-4 flex justify-end">
          <button
            className="primary-button"
            disabled={retrying}
            type="button"
            onClick={() => onRetryKnowledgeBase(item.kb_id!)}
          >
            {retrying ? '重新排队中…' : '重试删除'}
          </button>
        </div>
      )}
      {(item.reason === 'document_delete' || item.document_status === 'delete_failed')
        && item.status === 'failed' && item.doc_id && (
        <div className="mt-4 flex justify-end">
          <button
            className="primary-button"
            disabled={retrying}
            type="button"
            onClick={() => onRetryDocument(item.doc_id!)}
          >
            {retrying ? '重新排队中…' : '重试删除文档'}
          </button>
        </div>
      )}
    </section>
  )
}

function AgentKeysPage({ notify }: { notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void }) {
  const [apiKeyName, setApiKeyName] = useState('')
  const [apiKeyScopes, setApiKeyScopes] = useState<string[]>(['rag:read', 'llm:invoke'])
  const [apiKeyIsAdmin, setApiKeyIsAdmin] = useState(false)
  const [apiKeyExpiryDays, setApiKeyExpiryDays] = useState('90')
  const [apiKeyNeverExpires, setApiKeyNeverExpires] = useState(false)
  const [apiKeyRequestsPerMinute, setApiKeyRequestsPerMinute] = useState('60')
  const [apiKeyDailyQuota, setApiKeyDailyQuota] = useState('10000')
  const [createdApiKey, setCreatedApiKey] = useState<CreatedApiKey | null>(null)
  const [selectedKeyIds, setSelectedKeyIds] = useState<string[]>([])
  const [editingKeyId, setEditingKeyId] = useState<string | null>(null)
  const [editingScopes, setEditingScopes] = useState<string[]>([])
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState | null>(null)
  const queryClient = useQueryClient()
  const apiKeys = useQuery({
    queryKey: ['api-keys'],
    queryFn: api.apiKeys,
  })
  const createApiKey = useMutation({
    mutationFn: () => api.createApiKey({
      name: apiKeyName.trim(),
      scopes: apiKeyIsAdmin ? ['admin:*'] : apiKeyScopes,
      is_admin: apiKeyIsAdmin,
      expires_at: apiKeyNeverExpires
        ? null
        : new Date(Date.now() + Number(apiKeyExpiryDays) * 24 * 60 * 60 * 1000).toISOString(),
      requests_per_minute: Number(apiKeyRequestsPerMinute),
      daily_quota: Number(apiKeyDailyQuota),
    }),
    onSuccess: (key) => {
      setCreatedApiKey(key)
      setApiKeyName('')
      notify('API Key 创建成功。完整密钥只显示这一次，请立即复制。')
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
    onError: (error) => notify(getApiMessage(error, 'API Key 创建失败。'), 'error'),
  })
  const revokeApiKey = useMutation({
    mutationFn: (keyId: string) => api.revokeApiKey(keyId),
    onSuccess: () => {
      notify('API Key 已撤销，已失效但仍保留审计记录。')
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
    onError: (error) => notify(getApiMessage(error, 'API Key 撤销失败。'), 'error'),
  })
  const updateApiKey = useMutation({
    mutationFn: ({ keyId, scopes }: { keyId: string; scopes: string[] }) => api.updateApiKeyScopes(keyId, scopes),
    onSuccess: () => {
      setEditingKeyId(null)
      setEditingScopes([])
      notify('API Key 权限已更新。')
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
    onError: (error) => notify(getApiMessage(error, 'API Key 权限更新失败。'), 'error'),
  })
  const deleteRevokedApiKeys = useMutation({
    mutationFn: (keyIds?: string[]) => api.deleteRevokedApiKeys(keyIds),
    onSuccess: (result) => {
      setSelectedKeyIds([])
      notify(`已永久删除 ${result.deleted_count} 个已撤销密钥。`)
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
    onError: (error) => notify(getApiMessage(error, '删除已撤销密钥失败。'), 'error'),
  })
  const deleteRevokedApiKey = useMutation({
    mutationFn: (keyId: string) => api.deleteRevokedApiKey(keyId),
    onSuccess: () => {
      notify('已永久删除该密钥。')
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
    },
    onError: (error) => notify(getApiMessage(error, '删除密钥失败。'), 'error'),
  })

  const keys = apiKeys.data?.api_keys || []
  const revokedKeys = keys.filter((key) => !key.is_active)
  const activeKeys = keys.filter((key) => key.is_active)
  const allRevokedSelected = revokedKeys.length > 0 && revokedKeys.every((key) => selectedKeyIds.includes(key.key_id))
  const validApiKeyPolicy = Number.isInteger(Number(apiKeyRequestsPerMinute))
    && Number(apiKeyRequestsPerMinute) >= 1
    && Number(apiKeyRequestsPerMinute) <= 100000
    && Number.isInteger(Number(apiKeyDailyQuota))
    && Number(apiKeyDailyQuota) >= 1
    && Number(apiKeyDailyQuota) <= 100000000
    && (apiKeyNeverExpires || (Number.isInteger(Number(apiKeyExpiryDays)) && Number(apiKeyExpiryDays) >= 1 && Number(apiKeyExpiryDays) <= 3650))

  const toggleApiKeyScope = (scope: string) => {
    setApiKeyScopes((current) => current.includes(scope)
      ? current.filter((item) => item !== scope)
      : [...current, scope])
  }
  const toggleSelectedKey = (keyId: string) => {
    setSelectedKeyIds((current) => current.includes(keyId)
      ? current.filter((item) => item !== keyId)
      : [...current, keyId])
  }
  const startEditingKey = (key: ApiKeyInfo) => {
    setEditingKeyId(key.key_id)
    setEditingScopes(key.scopes)
  }
  const toggleEditingScope = (scope: string) => {
    setEditingScopes((current) => current.includes(scope)
      ? current.filter((item) => item !== scope)
      : [...current, scope])
  }
  const submitApiKey = () => {
    if (apiKeyIsAdmin) {
      setConfirmDialog({
        title: '创建管理员密钥？',
        message: '该密钥拥有全部 Agent 和管理权限，只应交给受信任的内部服务。',
        confirmLabel: '确认创建',
        tone: 'danger',
        onConfirm: () => createApiKey.mutate(),
      })
      return
    }
    createApiKey.mutate()
  }
  const askConfirm = (dialog: Omit<ConfirmDialogState, 'onConfirm'>, onConfirm: () => void) => {
    setConfirmDialog({ ...dialog, onConfirm })
  }

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Agent Access</p>
          <h1>Agent 密钥</h1>
          <p className="page-copy">为外部 Agent 创建、授权、撤销和清理独立 API Key。</p>
        </div>
        <KeyRound className="h-6 w-6 text-[var(--c-text-muted)]" />
      </header>

      <section className="api-key-page-grid">
        <article className="tool-panel settings-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Create</p>
              <h1>创建密钥</h1>
            </div>
          </div>
          <p className="panel-copy">完整密钥只在创建成功后显示一次，请立即复制并安全保存。</p>
          <div className="api-key-editor">
            <div className="api-key-create-grid">
              <input
                value={apiKeyName}
                onChange={(event) => setApiKeyName(event.target.value)}
                placeholder="密钥名称，例如：生产 Agent"
                maxLength={128}
              />
              <button
                className="primary-button"
                disabled={!apiKeyName.trim() || (!apiKeyIsAdmin && apiKeyScopes.length === 0) || !validApiKeyPolicy || createApiKey.isPending}
                onClick={submitApiKey}
              >
                <KeyRound className="h-4 w-4" /> 创建
              </button>
            </div>
            <div className="api-key-policy-grid">
              <div className="api-key-expiry-field">
                <span>有效期（天）</span>
                <div className="api-key-expiry-input-row">
                  <input
                    type="number"
                    min={1}
                    max={3650}
                    step={1}
                    value={apiKeyExpiryDays}
                    disabled={apiKeyNeverExpires}
                    onChange={(event) => setApiKeyExpiryDays(event.target.value)}
                  />
                  <label className="api-key-never-option">
                    <input type="checkbox" checked={apiKeyNeverExpires} onChange={(event) => setApiKeyNeverExpires(event.target.checked)} />
                    <span>永不过期</span>
                  </label>
                </div>
              </div>
              <label>
                <span>每分钟请求数</span>
                <input type="number" min={1} max={100000} value={apiKeyRequestsPerMinute} onChange={(event) => setApiKeyRequestsPerMinute(event.target.value)} />
              </label>
              <label>
                <span>每日配额</span>
                <input type="number" min={1} max={100000000} value={apiKeyDailyQuota} onChange={(event) => setApiKeyDailyQuota(event.target.value)} />
              </label>
            </div>
            {apiKeyIsAdmin ? (
              <div className="api-key-admin-summary">
                <ShieldCheck className="h-4 w-4 shrink-0" />
                <span><strong>管理员密钥</strong><small>将生成拥有全部权限的密钥，下方普通 Agent 权限不再单独限制它。</small></span>
              </div>
            ) : (
              <div className="api-key-scope-grid">
                {API_KEY_SCOPE_OPTIONS.map((option) => (
                  <label key={option.value} className="api-key-scope-option">
                    <input type="checkbox" checked={apiKeyScopes.includes(option.value)} onChange={() => toggleApiKeyScope(option.value)} />
                    <span><strong>{option.label}</strong><small>{option.description}</small></span>
                  </label>
                ))}
              </div>
            )}
            <label className="api-key-admin-option">
              <input type="checkbox" checked={apiKeyIsAdmin} onChange={(event) => setApiKeyIsAdmin(event.target.checked)} />
              <span><strong>创建管理员密钥</strong><small>拥有全部管理权限，仅用于受信任的内部服务。</small></span>
            </label>
            {createdApiKey && (
              <div className="api-key-created">
                <div><strong>新密钥仅显示这一次</strong><p>{createdApiKey.warning}</p></div>
                <code>{createdApiKey.raw_key}</code>
                <button className="secondary-button" onClick={() => {
                  void navigator.clipboard.writeText(createdApiKey.raw_key)
                  notify('API Key 已复制。')
                }}>
                  <Copy className="h-4 w-4" /> 复制
                </button>
              </div>
            )}
          </div>
        </article>

        <article className="tool-panel settings-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Lifecycle</p>
              <h1>密钥列表</h1>
            </div>
            <span className="api-key-count">{keys.length}</span>
          </div>
          <p className="panel-copy">撤销会立即失效；删除只作用于已撤销密钥，用于清理历史记录。</p>
          <div className="api-key-toolbar">
            <label className="api-key-select-all">
              <input
                type="checkbox"
                checked={allRevokedSelected}
                disabled={revokedKeys.length === 0}
                onChange={() => setSelectedKeyIds(allRevokedSelected ? [] : revokedKeys.map((key) => key.key_id))}
              />
              <span>选择已撤销密钥（{revokedKeys.length}）</span>
            </label>
            <button
              className="danger-button compact"
              disabled={revokedKeys.length === 0 || deleteRevokedApiKeys.isPending}
              onClick={() => {
                const targetIds = selectedKeyIds.length > 0 ? selectedKeyIds : undefined
                const count = targetIds?.length || revokedKeys.length
                askConfirm({
                  title: '永久删除已撤销密钥？',
                  message: `将永久删除 ${count} 个已撤销密钥，删除后不可恢复。`,
                  confirmLabel: '永久删除',
                  tone: 'danger',
                }, () => deleteRevokedApiKeys.mutate(targetIds))
              }}
            >
              <Trash2 className="h-4 w-4" /> {selectedKeyIds.length > 0 ? '删除选中' : '清理全部已撤销'}
            </button>
          </div>
          <div className="api-key-list">
            {apiKeys.isLoading ? (
              <p className="panel-copy">正在读取 API Key...</p>
            ) : apiKeys.isError ? (
              <p className="form-error">API Key 列表读取失败。</p>
            ) : keys.length === 0 ? (
              <p className="panel-copy">还没有 API Key。</p>
            ) : (
              keys.map((key: ApiKeyInfo) => (
                <div key={key.key_id} className={cn('api-key-row', !key.is_active && 'api-key-row-revoked')}>
                  {!key.is_active && (
                    <input type="checkbox" checked={selectedKeyIds.includes(key.key_id)} onChange={() => toggleSelectedKey(key.key_id)} />
                  )}
                  <div className="api-key-row-main">
                    <strong>{key.name}</strong>
                    <p>
                      {key.scopes.map(formatApiKeyScope).join('、') || '无额外权限'} · 创建于 {shortDate(key.created_at)} · {' '}
                      {key.expires_at ? `有效至 ${shortDate(key.expires_at)}` : '永不过期'}
                    </p>
                    {editingKeyId === key.key_id && key.is_active && !key.is_admin && (
                      <div className="api-key-edit-scopes">
                        {API_KEY_SCOPE_OPTIONS.map((option) => (
                          <label key={option.value}>
                            <input type="checkbox" checked={editingScopes.includes(option.value)} onChange={() => toggleEditingScope(option.value)} />
                            <span>{option.label}</span>
                          </label>
                        ))}
                        <button className="secondary-button compact" disabled={editingScopes.length === 0 || updateApiKey.isPending} onClick={() => updateApiKey.mutate({ keyId: key.key_id, scopes: editingScopes })}>保存权限</button>
                        <button className="ghost-button compact" onClick={() => setEditingKeyId(null)}>取消</button>
                      </div>
                    )}
                  </div>
                  <div className="api-key-row-actions">
                    <span className={cn('status-badge', key.is_active ? 'status-ready' : 'status-failed')}>
                      {key.is_active ? '启用' : '已撤销'}
                    </span>
                    {key.is_active ? (
                      <>
                        {!key.is_admin && <button className="secondary-button compact" onClick={() => startEditingKey(key)}>权限</button>}
                        <button className="danger-button compact" disabled={revokeApiKey.isPending} onClick={() => {
                          askConfirm({
                            title: '撤销这个 API Key？',
                            message: `撤销后「${key.name}」会立即失效，但记录仍会保留，之后可以再永久删除。`,
                            confirmLabel: '确认撤销',
                            tone: 'danger',
                          }, () => revokeApiKey.mutate(key.key_id))
                        }}>
                          <Trash2 className="h-4 w-4" /> 撤销
                        </button>
                      </>
                    ) : (
                      <button className="danger-button compact" disabled={deleteRevokedApiKey.isPending} onClick={() => {
                        askConfirm({
                          title: '永久删除这个密钥？',
                          message: `「${key.name}」删除后不可恢复。`,
                          confirmLabel: '永久删除',
                          tone: 'danger',
                        }, () => deleteRevokedApiKey.mutate(key.key_id))
                      }}>
                        <Trash2 className="h-4 w-4" /> 删除
                      </button>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
          {activeKeys.length > 0 && <p className="panel-note">当前有 {activeKeys.length} 个有效密钥。管理员密钥不能只改 scopes，如需收回请直接撤销。</p>}
        </article>
      </section>
      {confirmDialog && (
        <ConfirmDialog
          open
          title={confirmDialog.title}
          message={confirmDialog.message}
          confirmLabel={confirmDialog.confirmLabel}
          onConfirm={() => {
            const onConfirm = confirmDialog.onConfirm
            setConfirmDialog(null)
            onConfirm()
          }}
          onCancel={() => setConfirmDialog(null)}
        />
      )}
    </main>
  )
}

function SettingsPage({ notify }: { notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void }) {
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const queryClient = useQueryClient()
  const status = useQuery({
    queryKey: ['system-status'],
    queryFn: api.systemStatus,
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data?.retrieval?.hybrid_enabled) return false
      return ['building', 'waiting_documents'].includes(data.retrieval.bm25_state) ? 3000 : false
    },
  })
  const checks = useMutation({
    mutationFn: api.systemChecks,
    onSuccess: () => {
      // The diagnostic request has just exercised LLM, Embedding, vector DB
      // and SQLite. Refresh the status card immediately so it reflects that
      // latest probe instead of waiting for another page action or poll.
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error) => notify(getApiMessage(error, '诊断失败。'), 'error'),
  })
  const promptProfileMutation = useMutation({
    mutationFn: (profile: PromptProfile) => api.updatePromptProfile(profile),
    onSuccess: (state) => {
      const label = state.profile === 'auto' ? '自动' : state.profile === 'local' ? '本地' : '云端'
      const effective = state.effective_profile === 'local' ? '本地提示词' : '云端提示词'
      notify(`提示词模式已更新为${label}，当前生效为${effective}。`)
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error) => notify(getApiMessage(error, '更新提示词模式失败。'), 'error'),
  })
  const thinkingModeMutation = useMutation({
    mutationFn: (mode: LLMThinkingMode) => api.updateLLMThinkingMode(mode),
    onSuccess: (state) => {
      const label = state.mode === 'on' ? '开启' : state.mode === 'off' ? '关闭' : '服务默认'
      notify(`模型思考已设为${label}。`)
      queryClient.invalidateQueries({ queryKey: ['system-status'] })
    },
    onError: (error) => notify(getApiMessage(error, '更新模型思考失败。'), 'error'),
  })
  const changePassword = useMutation({
    mutationFn: () => api.changePassword(currentPassword, newPassword, confirmPassword),
    onSuccess: () => {
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      notify('密码已更新，其他会话已失效。')
      queryClient.invalidateQueries({ queryKey: ['bootstrap'] })
    },
    onError: (error) => notify(getApiMessage(error, '修改密码失败。'), 'error'),
  })
  const checkResults = checks.data?.checks || []
  const statusData = status.data
  const currentPromptProfile = statusData?.prompt.profile ?? 'auto'
  const currentThinkingMode = statusData?.llm.thinking?.mode ?? 'off'
  const warnings = filterBm25Warnings(statusData?.warnings ?? [])
  const rerankerStatus = statusData?.retrieval.reranker
  return (
    <main className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Settings</p>
          <h1>系统设置</h1>
          <p className="page-copy">这里处理管理员密码、系统状态和主动诊断。</p>
        </div>
      </header>
      <section className="settings-grid">
        <article className="tool-panel settings-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Password</p>
              <h1>修改密码</h1>
            </div>
            <KeyRound className="h-5 w-5 text-[var(--c-text-muted)]" />
          </div>
          <div className="stack-gap">
            <input value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} type="password" placeholder="当前密码" />
            <input value={newPassword} onChange={(event) => setNewPassword(event.target.value)} type="password" placeholder={`新密码（至少 ${PASSWORD_MIN_LENGTH} 位）`} />
            <input value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} type="password" placeholder="再次输入新密码" />
            <button
              className="primary-button"
              disabled={!currentPassword || newPassword.length < PASSWORD_MIN_LENGTH || confirmPassword.length < PASSWORD_MIN_LENGTH || changePassword.isPending}
              onClick={() => changePassword.mutate()}
            >
              <ShieldCheck className="h-4 w-4" /> 保存新密码
            </button>
          </div>
        </article>

        <article className="tool-panel settings-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Reasoning</p>
              <h1>模型思考</h1>
            </div>
            <Bot className="h-5 w-5 text-[var(--c-text-muted)]" />
          </div>
          <div className="stack-gap">
            <select
              value={currentThinkingMode}
              onChange={(event) => thinkingModeMutation.mutate(event.target.value as LLMThinkingMode)}
              disabled={thinkingModeMutation.isPending || status.isLoading || !statusData?.llm.thinking?.supported}
            >
              <option value="off">关闭（更快、更稳定）</option>
              <option value="on">开启</option>
              <option value="auto">服务默认</option>
            </select>
            <p className="panel-copy">
              这里设置新会话默认值；输入框里的原生档位可按会话覆盖。路由、查询改写和标题生成始终关闭思考，避免短输出预算被推理过程耗尽。
            </p>
            {statusData?.llm.thinking && (
              <div className="meta-grid">
                <div><span>当前模型</span><strong>{displayModelName(statusData.llm.thinking.model_name)}</strong></div>
                <div><span>适配状态</span><strong>{statusData.llm.thinking.supported ? '已匹配配置表' : '未匹配'}</strong></div>
                <div><span>参数协议</span><strong>{statusData.llm.thinking.transport || '不发送参数'}</strong></div>
              </div>
            )}
          </div>
        </article>

        <article className="tool-panel settings-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Prompt</p>
              <h1>提示词模式</h1>
            </div>
            <SlidersHorizontal className="h-5 w-5 text-[var(--c-text-muted)]" />
          </div>
          <div className="stack-gap">
            <select
              value={currentPromptProfile}
              onChange={(event) => promptProfileMutation.mutate(event.target.value as PromptProfile)}
              disabled={promptProfileMutation.isPending || status.isLoading || !statusData}
            >
              <option value="auto">自动</option>
              <option value="local">本地</option>
              <option value="cloud">云端</option>
            </select>
            <p className="panel-copy">
              自动模式会根据 LLM 地址判断。本地地址优先使用更短、更克制的提示词，云端模型则使用更完整的提示词。
            </p>
            {statusData && (
              <div className="meta-grid">
                <div><span>当前选择</span><strong>{statusData.prompt.profile === 'auto' ? '自动' : statusData.prompt.profile === 'local' ? '本地' : '云端'}</strong></div>
                <div><span>当前生效</span><strong>{statusData.prompt.effective_profile === 'local' ? '本地提示词' : '云端提示词'}</strong></div>
                <div><span>自动判断</span><strong>{statusData.prompt.local_endpoint_detected ? '本地地址' : '云端地址'}</strong></div>
              </div>
            )}
          </div>
        </article>

        <article className="tool-panel settings-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Status</p>
              <h1>系统状态</h1>
            </div>
            <Database className="h-5 w-5 text-[var(--c-text-muted)]" />
          </div>
          {status.isLoading ? (
            <p className="panel-copy">正在汇总本地状态...</p>
          ) : status.isError ? (
            <p className="form-error">系统状态读取失败。</p>
          ) : statusData ? (
            <>
              <Bm25StatusNotice statusData={statusData} />
              <StatusList
                items={[
                  { label: '管理员已初始化', ok: statusData.auth.initialized },
                  { label: 'Session Secret 正常', ok: statusData.auth.session_secret_ok },
                  { label: 'LLM 已配置', ok: statusData.llm.configured },
                  { label: 'LLM 可连接', ok: statusData.llm.reachable },
                  { label: 'Embedding 已配置', ok: statusData.embedding.configured },
                  { label: 'Embedding 已连接', ok: statusData.embedding.reachable },
                  { label: '文档任务执行器运行中', ok: statusData.queue.worker.state === 'running' },
                  { label: 'Reranker 可用', ok: !rerankerStatus?.configured || rerankerStatus.active },
                  { label: '前端已构建', ok: statusData.frontend.built },
                ]}
              />
              <div className="meta-grid">
                <div><span>LLM 模式</span><strong>{statusData.llm.mock ? '模拟模式' : statusData.llm.mode}</strong></div>
                <div><span>提示词模式</span><strong>{statusData.prompt.profile === 'auto' ? '自动' : statusData.prompt.profile === 'local' ? '本地' : '云端'}</strong></div>
                <div><span>生效提示词</span><strong>{statusData.prompt.effective_profile === 'local' ? '本地提示词' : '云端提示词'}</strong></div>
                <div><span>向量模式</span><strong>{statusData.vectorstore.mode}</strong></div>
                <div><span>当前 Embedding</span><strong>{displayModelName(statusData.embedding.current_model)}</strong></div>
                <div><span>重排模型</span><strong>
                  {!rerankerStatus?.configured
                    ? '未启用（旧链路）'
                    : rerankerStatus.active
                      ? displayModelName(rerankerStatus.model_name) || '已启用'
                      : '不可用（已回退）'}
                </strong></div>
                <div><span>数据库</span><strong>{statusData.database.backend}</strong></div>
                <div><span>队列后端</span><strong>{statusData.queue.backend}</strong></div>
                <div><span>排队任务</span><strong>{statusData.queue.worker.queued_count}</strong></div>
                <div><span>执行中任务</span><strong>{statusData.queue.worker.running_count}</strong></div>
                <div><span>自动重建</span><strong>{statusData.reindex.enabled ? '已启用' : '未启用'}</strong></div>
                <div><span>等待重建</span><strong>{statusData.reindex.pending_count}</strong></div>
                <div><span>重建中</span><strong>{statusData.reindex.running_count}</strong></div>
                <div><span>阻塞文档</span><strong>{statusData.reindex.blocked_count}</strong></div>
                <div><span>遗留明文密码</span><strong>{statusData.auth.legacy_password_detected ? '检测到' : '未检测到'}</strong></div>
              </div>
              {warnings.length > 0 && (
                <ul className="warning-list">
                  {warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              )}
            </>
          ) : null}
        </article>

        <article className="tool-panel settings-panel settings-panel-checks">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Checks</p>
              <h1>主动诊断</h1>
            </div>
            <Wrench className="h-5 w-5 text-[var(--c-text-muted)]" />
          </div>
          <div className="panel-actions">
            <button
              className="ghost-button"
              onClick={() => checks.mutate()}
              disabled={checks.isPending}
              aria-busy={checks.isPending}
            >
              <RefreshCw className={cn('h-4 w-4', checks.isPending && 'animate-spin')} />
              {checks.isPending ? '检查中…' : '运行检查'}
            </button>
          </div>
          {checkResults.length > 0 ? (
            <div className="checks-list">
              {checkResults.map((item) => (
                <CheckRow key={item.key} item={item} />
              ))}
            </div>
          ) : (
              <p className="panel-copy">状态页不会占用模型；这里才会主动检查 LLM、Embedding、向量库和 SQLite 写入能力。</p>
          )}
        </article>
      </section>
    </main>
  )
}

function StatusList({ items }: { items: Array<{ label: string; ok: boolean | null }> }) {
  return (
    <div className="status-list">
      {items.map((item) => (
        <div key={item.label} className="status-list-row">
          <span>{item.label}</span>
          <strong className={
            item.ok === true
              ? 'text-[var(--c-success)]'
              : item.ok === false
                ? 'text-[var(--c-danger)]'
                : 'text-[var(--c-text-muted)]'
          }>
            {item.ok === true ? '正常' : item.ok === false ? '缺失' : '未检查'}
          </strong>
        </div>
      ))}
    </div>
  )
}

function CheckRow({ item }: { item: SystemCheck }) {
  return (
    <div className="check-row">
      <div>
        <strong>{item.message}</strong>
        <p>{item.detail || item.code}</p>
      </div>
      <span className={cn('status-badge', item.status === 'ok' ? 'status-ready' : item.status === 'warn' ? 'status-processing' : 'status-failed')}>
        {item.status === 'ok' ? '通过' : item.status === 'warn' ? '待处理' : '失败'}
      </span>
    </div>
  )
}

function AutoResizeTextarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`
  }, [props.value])
  return <textarea ref={ref} rows={1} {...props} />
}

function WelcomeScreen({
  onStart,
  onOpenDocuments,
  streaming,
  llmModel,
  llmModels,
  onLLMModelChange,
  thinkingEffort,
  thinkingEfforts,
  thinkingSupported,
  onThinkingEffortChange,
  answerQualityMode,
  onAnswerQualityModeChange,
}: {
  onStart: (text: string) => void
  onOpenDocuments: () => void
  streaming: boolean
  llmModel: string
  llmModels: Array<{ model_name: string; display_name: string }>
  onLLMModelChange: (value: string) => void
  thinkingEffort: ThinkingEffort
  thinkingEfforts: ThinkingEffort[]
  thinkingSupported: boolean
  onThinkingEffortChange: (value: ThinkingEffort) => void
  answerQualityMode: AnswerQualityMode
  onAnswerQualityModeChange: (value: AnswerQualityMode) => void
}) {
  const documents = useQuery({ queryKey: ['documents'], queryFn: () => api.documents() })
  const readyDocuments = documents.data?.documents.filter((doc) => doc.status === 'ready') ?? []
  const [input, setInput] = useState('')
  const isComposingRef = useRef(false)

  const starters = readyDocuments.length > 0
    ? [
        { label: '总结已上传文档', prompt: '请总结已上传文档的主要内容' },
        { label: '这些文档说了什么？', prompt: '这些文档的主要观点是什么？' },
        { label: '帮我找重点结论', prompt: '请帮我提炼这些资料里的重点结论' },
      ]
    : [
        { label: '如何上传文档？', prompt: '怎么上传文档让你基于文档回答？' },
        { label: '你能做什么？', prompt: '你能帮我做什么？' },
        { label: '先介绍一下自己', prompt: '你好，简单介绍一下你自己' },
      ]

  const handleSend = () => {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')
    onStart(text)
  }

  return (
    <div className="welcome-screen">
      <Database className="h-12 w-12 text-[var(--c-accent)]" />
      <h2>Facet</h2>
      <p className="welcome-subtitle">
        {readyDocuments.length > 0 ? `已加载 ${readyDocuments.length} 个文档，可以直接提问。` : '当前还没有可检索文档，建议先上传资料。'}
      </p>
      {readyDocuments.length === 0 && (
        <button className="ghost-button compact" onClick={onOpenDocuments}>
          <Upload className="h-4 w-4" /> 去上传文档
        </button>
      )}
      <div className="welcome-composer">
        <AutoResizeTextarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="输入问题..."
          maxLength={4000}
          onCompositionStart={() => {
            isComposingRef.current = true
          }}
          onCompositionEnd={() => {
            isComposingRef.current = false
          }}
          onKeyDown={(event) => {
            const nativeEvent = event.nativeEvent as KeyboardEvent
            if (event.key === 'Enter' && !event.shiftKey) {
              if (isComposingRef.current || nativeEvent.isComposing || nativeEvent.key === 'Process') return
              event.preventDefault()
              handleSend()
            }
          }}
        />
        <div className="composer-toolbar">
          <div className="composer-leading">
            <AnswerQualityToggle
              value={answerQualityMode}
              disabled={streaming}
              onChange={onAnswerQualityModeChange}
            />
            <span className="composer-hint">Enter 发送，Shift+Enter 换行</span>
          </div>
          <div className="composer-actions">
            <LLMModelSelect
              value={llmModel}
              models={llmModels}
              disabled={streaming}
              onChange={onLLMModelChange}
            />
            <ThinkingEffortSelect
              value={thinkingEffort}
              efforts={thinkingEfforts}
              supported={thinkingSupported}
              disabled={streaming}
              onChange={onThinkingEffortChange}
            />
            <button className="composer-send" onClick={handleSend} disabled={!input.trim() || streaming} title="发送">
              <Send className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>
      <div className="welcome-starters">
        {starters.map((starter) => (
          <button key={starter.label} className="starter-btn" onClick={() => onStart(starter.prompt)}>
            {starter.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function ChatPage({
  notify,
  conversationId,
  newConversationViewId,
  setConversationId,
}: {
  notify: (text: string, tone?: Toast['tone'], durationMs?: number) => void
  conversationId: string | null
  newConversationViewId: string
  setConversationId: (id: string | null) => void
}) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [drafts, setDrafts] = useState<Record<string, string>>(() => {
    const stored = localStorage.getItem('facet.chat.drafts')
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        if (parsed && typeof parsed === 'object') return parsed
      } catch {
        // Ignore an older malformed browser value.
      }
    }
    const legacyDraft = localStorage.getItem('rag.chat.draft') || ''
    return legacyDraft ? { legacy: legacyDraft } : {}
  })
  const [newGroundingMode, setNewGroundingMode] = useState<GroundingMode>('auto')
  const [savingSessionOptions, setSavingSessionOptions] = useState(false)
  const [newKnowledgeScope, setNewKnowledgeScope] = useState<'all' | 'selected'>('all')
  const [newKnowledgeBaseIds, setNewKnowledgeBaseIds] = useState<string[]>([])
  const [newFullContextDocId, setNewFullContextDocId] = useState<string | null>(null)
  const [newStreamValidationMode, setNewStreamValidationMode] = useState<'validated' | 'realtime'>('realtime')
  const [newAnswerQualityMode, setNewAnswerQualityMode] = useState<AnswerQualityMode>('normal')
  const answerQualityInitializedRef = useRef(false)
  const [newLLMModel, setNewLLMModel] = useState<string | null>(null)
  const [newThinkingEffort, setNewThinkingEffort] = useState<ThinkingEffort | null>(null)
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [optimisticMessagesByView, setOptimisticMessagesByView] = useState<Record<string, ChatMessage[]>>({})
  const [streamingKeys, setStreamingKeys] = useState<Set<string>>(() => new Set())
  const isComposingRef = useRef(false)
  const streamingKeysRef = useRef<Set<string>>(new Set())
  const streamControllersRef = useRef<Map<string, AbortController>>(new Map())
  const activeConversationIdRef = useRef<string | null>(conversationId)
  const activeNewConversationViewIdRef = useRef(newConversationViewId)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const titleRefreshTimersRef = useRef<Map<string, number[]>>(new Map())

  const draftKey = conversationId ? `conversation:${conversationId}` : `new:${newConversationViewId}`
  const draft = drafts[draftKey] || ''
  const setDraft = (value: string) => {
    setDrafts((previous) => ({ ...previous, [draftKey]: value }))
  }
  const optimisticMessages = useMemo(
    () => optimisticMessagesByView[newConversationViewId] || [],
    [newConversationViewId, optimisticMessagesByView],
  )
  const activeStreamKey = conversationStreamKey(conversationId, newConversationViewId)

  const active = useQuery({
    queryKey: ['conversation', conversationId],
    queryFn: () => api.conversation(conversationId!),
    enabled: Boolean(conversationId),
  })
  useEffect(() => {
    if (!conversationId || !active.isError) return
    if (!(active.error instanceof ApiError) || ![403, 404].includes(active.error.status)) return
    setConversationId(null)
    notify('之前打开的会话已不存在或不可访问，已回到新会话。')
  }, [active.error, active.isError, conversationId, notify, setConversationId])
  const serverStreaming = Boolean(
    conversationId && active.data?.messages.some((message) => message.status === 'streaming'),
  )
  const streaming = streamingKeys.has(activeStreamKey) || serverStreaming
  const canStopCurrentStream = streamingKeys.has(activeStreamKey)
  const knowledgeBases = useQuery({ queryKey: ['knowledge-bases'], queryFn: api.knowledgeBases })
  const chatCapabilities = useQuery({
    queryKey: ['chat-capabilities'],
    queryFn: api.chatCapabilities,
    refetchOnMount: 'always',
    refetchOnWindowFocus: 'always',
  })
  const llmModels = chatCapabilities.data?.models?.options || (
    chatCapabilities.data?.thinking
      ? [{
          model_name: chatCapabilities.data.thinking.model_name,
          display_name: displayModelName(chatCapabilities.data.thinking.model_name),
          thinking: chatCapabilities.data.thinking,
        }]
      : []
  )
  const defaultLLMModel = chatCapabilities.data?.models?.default_model
    || chatCapabilities.data?.thinking?.model_name
    || ''
  const selectedKnowledgeScope = conversationId
    ? active.data?.conversation.knowledge_scope || (active.data?.conversation.knowledge_base_id ? 'selected' : 'all')
    : newKnowledgeScope
  const selectedKnowledgeBaseIds = conversationId
    ? active.data?.conversation.knowledge_base_ids
      || (active.data?.conversation.knowledge_base_id ? [active.data.conversation.knowledge_base_id] : [])
    : newKnowledgeBaseIds
  const selectedKnowledgeBaseId = selectedKnowledgeScope === 'selected' && selectedKnowledgeBaseIds.length === 1
    ? selectedKnowledgeBaseIds[0]
    : null
  const selectedFullContextDocId = conversationId
    ? active.data?.conversation.full_context_doc_id || null
    : newFullContextDocId
  const selectedGroundingMode = conversationId
    ? active.data?.conversation.grounding_mode || 'auto'
    : newGroundingMode
  const selectedStreamValidationMode = conversationId
    ? active.data?.conversation.stream_validation_mode || 'realtime'
    : newStreamValidationMode
  const selectedAnswerQualityMode = conversationId
    ? active.data?.conversation.answer_quality_mode || 'normal'
    : newAnswerQualityMode
  const preferredLLMModel = conversationId
    ? active.data?.conversation.llm_model || defaultLLMModel
    : newLLMModel || defaultLLMModel
  const selectedLLMModel = (
    llmModels.length === 0
    || llmModels.some((model) => model.model_name === preferredLLMModel)
  )
    ? preferredLLMModel
    : defaultLLMModel
  const selectedLLMOption = llmModels.find((model) => model.model_name === selectedLLMModel)
  const thinkingCapability = selectedLLMOption?.thinking
  const thinkingEfforts = configuredThinkingEfforts(thinkingCapability)
  const thinkingSupported = thinkingCapability?.supported ?? false
  const selectedThinkingEffort = resolveThinkingEffort(
    conversationId ? active.data?.conversation.thinking_effort : newThinkingEffort,
    thinkingCapability,
  )
  const outgoingThinkingEffort = chatCapabilities.data && thinkingSupported
    ? selectedThinkingEffort
    : undefined

  useEffect(() => {
    const configuredDefault = chatCapabilities.data?.answer_quality.default_mode
    if (!configuredDefault || answerQualityInitializedRef.current) return
    answerQualityInitializedRef.current = true
    setNewAnswerQualityMode(configuredDefault)
  }, [chatCapabilities.data?.answer_quality.default_mode])
  const knowledgeBaseDocuments = useQuery({
    queryKey: ['knowledge-base-documents', selectedKnowledgeBaseId],
    queryFn: () => api.knowledgeBaseDocuments(selectedKnowledgeBaseId!),
    enabled: Boolean(selectedKnowledgeBaseId),
  })

  useEffect(() => {
    localStorage.setItem('facet.chat.drafts', JSON.stringify(drafts))
  }, [drafts])
  useEffect(() => {
    setEditingMessageId(null)
  }, [conversationId])
  useEffect(() => {
    setNewLLMModel(null)
    setNewThinkingEffort(null)
  }, [newConversationViewId])
  useEffect(() => {
    activeConversationIdRef.current = conversationId
  }, [conversationId])
  useEffect(() => {
    activeNewConversationViewIdRef.current = newConversationViewId
  }, [newConversationViewId])
  useEffect(() => {
    if (!conversationId || !serverStreaming) return undefined
    const timer = window.setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ['conversation', conversationId] })
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
    }, 1500)
    return () => window.clearInterval(timer)
  }, [conversationId, queryClient, serverStreaming])
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [active.data?.messages, optimisticMessages, streaming])
  const titleRefreshTimers = titleRefreshTimersRef.current
  useEffect(() => {
    return () => {
      titleRefreshTimers.forEach((timers) => {
        timers.forEach((timer) => window.clearTimeout(timer))
      })
      titleRefreshTimers.clear()
    }
  }, [titleRefreshTimers])

  const scheduleConversationTitleRefresh = (conversationIdToRefresh: string | null) => {
    if (!conversationIdToRefresh) return
    titleRefreshTimersRef.current.get(conversationIdToRefresh)?.forEach((timer) => window.clearTimeout(timer))
    const delays = [1200, 3200]
    const timers = delays.map((delay) => {
      const timer = window.setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['conversations'] })
        queryClient.invalidateQueries({ queryKey: ['conversation', conversationIdToRefresh] })
      }, delay)
      return timer
    })
    titleRefreshTimersRef.current.set(conversationIdToRefresh, timers)
  }

  const startStream = (streamKey: string, controller: AbortController) => {
    streamingKeysRef.current.add(streamKey)
    streamControllersRef.current.set(streamKey, controller)
    setStreamingKeys(new Set(streamingKeysRef.current))
  }

  const promoteStream = (fromKey: string, toKey: string) => {
    if (fromKey === toKey) return
    const controller = streamControllersRef.current.get(fromKey)
    streamingKeysRef.current.delete(fromKey)
    streamingKeysRef.current.add(toKey)
    streamControllersRef.current.delete(fromKey)
    if (controller) streamControllersRef.current.set(toKey, controller)
    setStreamingKeys(new Set(streamingKeysRef.current))
  }

  const finishStream = (streamKey: string) => {
    streamingKeysRef.current.delete(streamKey)
    streamControllersRef.current.delete(streamKey)
    setStreamingKeys(new Set(streamingKeysRef.current))
  }

  const updateOptimisticMessages = (
    viewId: string,
    updater: ChatMessage[] | ((previous: ChatMessage[]) => ChatMessage[]),
  ) => {
    setOptimisticMessagesByView((previous) => {
      const current = previous[viewId] || []
      const next = typeof updater === 'function' ? updater(current) : updater
      return { ...previous, [viewId]: next }
    })
  }

  const applySessionOptions = async (
    knowledgeScope: 'all' | 'selected',
    knowledgeBaseIds: string[],
    fullContextDocId: string | null,
    groundingMode: GroundingMode,
    streamValidationMode: 'validated' | 'realtime' = selectedStreamValidationMode,
    thinkingEffort: ThinkingEffort | null = outgoingThinkingEffort ?? null,
    answerQualityMode: AnswerQualityMode = selectedAnswerQualityMode,
    llmModel: string = selectedLLMModel,
  ) => {
    if (streaming) return
    const targetLLMModel = llmModel || defaultLLMModel
    const targetThinking = llmModels.find((model) => model.model_name === targetLLMModel)?.thinking
    const normalizedThinkingEffort = targetThinking?.supported
      ? resolveThinkingEffort(thinkingEffort, targetThinking)
      : null
    const normalizedIds = knowledgeScope === 'selected'
      ? [...new Set(knowledgeBaseIds.filter(Boolean))]
      : []
    const normalizedFullContextDocId = normalizedIds.length === 1 ? fullContextDocId : null
    if (!conversationId) {
      setNewKnowledgeScope(knowledgeScope)
      setNewKnowledgeBaseIds(normalizedIds)
      setNewFullContextDocId(normalizedFullContextDocId)
      setNewGroundingMode(groundingMode)
      setNewStreamValidationMode(streamValidationMode)
      setNewLLMModel(targetLLMModel || null)
      setNewThinkingEffort(normalizedThinkingEffort)
      setNewAnswerQualityMode(answerQualityMode)
      return
    }
    setSavingSessionOptions(true)
    try {
      const response = await api.updateConversationSessionOptions(
        conversationId, knowledgeScope, normalizedIds, normalizedFullContextDocId, groundingMode,
        streamValidationMode, normalizedThinkingEffort,
        answerQualityMode, targetLLMModel || undefined,
      )
      queryClient.setQueryData(['conversation', conversationId], (previous: any) => (
        previous ? { ...previous, conversation: response.conversation } : previous
      ))
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
      notify('已更新本会话设置。')
    } catch (error) {
      notify(getApiMessage(error, '更新检索范围失败。'), 'error')
    } finally {
      setSavingSessionOptions(false)
    }
  }

  const send = async (overrideContent?: string) => {
    const content = (overrideContent ?? draft).trim()
    const startedNewConversationViewId = newConversationViewId
    const initialStreamKey = conversationStreamKey(conversationId, startedNewConversationViewId)
    if (
      !content
      || serverStreaming
      || streamingKeysRef.current.has(initialStreamKey)
      || (Boolean(conversationId) && active.isLoading)
    ) return
    const startedWithoutConversation = !conversationId
    const startedConversationId = conversationId
    const editingFirstUserMessage = Boolean(
      startedConversationId
      && editingMessageId
      && active.data?.messages?.[0]?.message_id === editingMessageId
    )
    const shouldRefreshTitle = startedWithoutConversation || editingFirstUserMessage
    const editFromMessageId = resolveOutgoingEditFromMessageId(conversationId, editingMessageId)
    const submittedEditMessageId = editFromMessageId ?? null
    const createdAt = new Date().toISOString()
    const pendingUserMessageId = `pending-user-${Date.now()}`

    const controller = new AbortController()
    let streamKey = initialStreamKey
    startStream(streamKey, controller)
    setDraft('')

    let streamConversationId: string | null = conversationId
    let assistantMessageId = `pending-assistant-${Date.now()}`
    let accumulated = ''
    let streamErrored = false
    let transportInterrupted = false
    let lastFlushedContent = ''
    let streamFlushTimer: ReturnType<typeof setTimeout> | null = null

    const applyAssistantError = (errorText: string) => {
      const updateAssistant = (message: ChatMessage): ChatMessage => (
        message.message_id === assistantMessageId
          ? {
              ...message,
              content: accumulated || message.content,
              status: 'error' as const,
              error_message: errorText,
            }
          : message
      )

      if (!streamConversationId) {
        updateOptimisticMessages(startedNewConversationViewId, (previous) => previous.map(updateAssistant))
        return
      }

      queryClient.setQueryData(['conversation', streamConversationId], (prev: any) => {
        if (!prev) return prev
        return {
          ...prev,
          messages: prev.messages.map(updateAssistant),
        }
      })
    }

    if (startedWithoutConversation) {
      updateOptimisticMessages(startedNewConversationViewId, buildOptimisticMessages(
        content,
        createdAt,
        pendingUserMessageId,
        assistantMessageId,
      ))
    } else {
      queryClient.setQueryData(['conversation', conversationId], (prev: any) => {
        const existingMessages: ChatMessage[] = prev?.messages || active.data?.messages || []
        const nextMessages = buildOptimisticConversationMessages(
          existingMessages,
          content,
          createdAt,
          pendingUserMessageId,
          assistantMessageId,
          editFromMessageId,
        )
        return prev
          ? { ...prev, messages: nextMessages }
          : {
              conversation: {
                conversation_id: conversationId,
                title: active.data?.conversation.title || '新对话',
                created_at: createdAt,
                updated_at: createdAt,
              },
              messages: nextMessages,
          }
      })
    }
    setEditingMessageId(null)

    const flushAssistantToCache = () => {
      if (streamFlushTimer !== null) {
        clearTimeout(streamFlushTimer)
        streamFlushTimer = null
      }
      if (lastFlushedContent === accumulated) return
      lastFlushedContent = accumulated
      const cid = streamConversationId
      if (!cid) {
        updateOptimisticMessages(startedNewConversationViewId, (previous) => previous.map((message) => (
          message.role === 'assistant'
            ? { ...message, message_id: assistantMessageId, content: accumulated, updated_at: new Date().toISOString() }
            : message
        )))
        return
      }
      const snapshot: ChatMessage = {
        message_id: assistantMessageId,
        conversation_id: cid,
        role: 'assistant',
        content: accumulated,
        status: 'streaming',
        sources: [],
        seq: 999999,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
      queryClient.setQueryData(['conversation', cid], (prev: any) => {
        if (!prev) return prev
        const others = prev.messages.filter((message: ChatMessage) => message.message_id !== assistantMessageId && !message.message_id.startsWith('pending-'))
        return { ...prev, messages: [...others, snapshot] }
      })
    }

    const scheduleAssistantCacheFlush = () => {
      if (streamFlushTimer !== null) return
      streamFlushTimer = setTimeout(() => {
        streamFlushTimer = null
        flushAssistantToCache()
      }, STREAM_RENDER_INTERVAL_MS)
    }

    try {
      const response = await streamFetch('/api/v1/chat', {
        method: 'POST',
        signal: controller.signal,
        json: {
          conversation_id: conversationId,
          edit_from_message_id: editFromMessageId,
          message: { content },
          stream: true,
          grounding_mode: selectedGroundingMode,
          stream_validation_mode: selectedStreamValidationMode,
          answer_quality_mode: selectedAnswerQualityMode,
          llm_model: selectedLLMModel || undefined,
          thinking_effort: outgoingThinkingEffort,
          ...(conversationId ? {} : {
            knowledge_scope: selectedKnowledgeScope,
            knowledge_base_ids: selectedKnowledgeBaseIds,
            full_context_doc_id: selectedFullContextDocId,
          }),
        },
      })
      if (!response.ok) throw await readStreamError(response)

      await readSseStream(response, (event) => {
        if (event.event === 'meta') {
          const meta = parseJsonEvent<ChatStreamMeta>(event)
          if (meta?.conversation_id) {
            streamConversationId = meta.conversation_id
            assistantMessageId = meta.assistant_message_id
            const conversationKey = conversationStreamKey(meta.conversation_id, startedNewConversationViewId)
            promoteStream(streamKey, conversationKey)
            streamKey = conversationKey
            if (shouldActivateStreamConversation(
              startedConversationId,
              activeConversationIdRef.current,
              startedNewConversationViewId,
              activeNewConversationViewIdRef.current,
            )) {
              setConversationId(meta.conversation_id)
            }
            updateOptimisticMessages(startedNewConversationViewId, [])
            queryClient.setQueryData(['conversations'], (prev: { conversations: Conversation[] } | undefined) => {
              const nextConversation: Conversation = {
                conversation_id: meta.conversation_id,
                title: meta.title,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                last_message_at: new Date().toISOString(),
                llm_model: meta.llm_model || selectedLLMModel,
                thinking_effort: meta.thinking_effort || selectedThinkingEffort,
                answer_quality_mode: meta.answer_quality_mode || selectedAnswerQualityMode,
              }
              return {
                conversations: upsertConversation(prev?.conversations || [], nextConversation),
              }
            })
            queryClient.setQueryData(['conversation', meta.conversation_id], (prev: any) => {
              const userMessage: ChatMessage = {
                message_id: meta.user_message_id,
                conversation_id: meta.conversation_id,
                role: 'user',
                content,
                status: 'completed',
                sources: [],
                seq: 1,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
              }
              const assistantMessage: ChatMessage = {
                message_id: meta.assistant_message_id,
                conversation_id: meta.conversation_id,
                role: 'assistant',
                content: '',
                status: 'streaming',
                sources: [],
                seq: 2,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
              }
              if (prev) {
                const others = prev.messages.filter((message: ChatMessage) => !message.message_id.startsWith('pending-'))
                return {
                  ...prev,
                  conversation: {
                    ...prev.conversation,
                    llm_model: meta.llm_model || selectedLLMModel,
                    thinking_effort: meta.thinking_effort || selectedThinkingEffort,
                  },
                  messages: [...others, userMessage, assistantMessage],
                }
              }
              return {
                conversation: {
                  conversation_id: meta.conversation_id,
                  title: meta.title,
                  created_at: new Date().toISOString(),
                  updated_at: new Date().toISOString(),
                  llm_model: meta.llm_model || selectedLLMModel,
                  thinking_effort: meta.thinking_effort || selectedThinkingEffort,
                },
                messages: [userMessage, assistantMessage],
              }
            })
          }
          return
        }
        if (event.event === 'sources') return
        if (event.event === 'message') {
          const payload = parseJsonEvent<{ content: string }>(event)
          accumulated += payload?.content || ''
          scheduleAssistantCacheFlush()
          return
        }
        if (event.event === 'error') {
          streamErrored = true
          const payload = parseJsonEvent<{
            error?: string
            message?: string
            partial_output?: boolean
          }>(event)
          const baseErrorText = payload?.error || payload?.message || '生成失败。'
          const errorText = payload?.partial_output
            ? `生成中断，以下内容可能不完整。${baseErrorText}`
            : baseErrorText
          flushAssistantToCache()
          applyAssistantError(errorText)
        }
      })
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        notify('已停止生成。')
      } else {
        const baseErrorText = getApiMessage(error, '发送失败。')
        const errorText = accumulated
          ? `生成中断，以下内容可能不完整。${baseErrorText}`
          : baseErrorText
        transportInterrupted = true
        notify(errorText, 'error')
        // We received a conversation identity but not its terminal SSE event.
        // Preserve any partial answer while making the interruption visible
        // immediately; the subsequent cache refresh reconciles server state.
        if (streamConversationId && assistantMessageId && !streamErrored) {
          flushAssistantToCache()
          applyAssistantError(errorText)
        }
      }
      if (shouldRestoreEditingMessageIdAfterSendFailure(
        startedConversationId,
        activeConversationIdRef.current,
        submittedEditMessageId,
        controller.signal.aborted,
        streamErrored,
      )) {
        setEditingMessageId(submittedEditMessageId)
      }
      if (startedWithoutConversation && !streamConversationId) {
        if (activeConversationIdRef.current === null && activeNewConversationViewIdRef.current === startedNewConversationViewId) {
          setDraft(content)
        } else {
          setDrafts((previous) => ({ ...previous, [`new:${startedNewConversationViewId}`]: content }))
        }
        if (!streamErrored) updateOptimisticMessages(startedNewConversationViewId, [])
      }
    } finally {
      flushAssistantToCache()
      finishStream(streamKey)
      if (!streamConversationId && !streamErrored && !transportInterrupted) updateOptimisticMessages(startedNewConversationViewId, [])
      queryClient.invalidateQueries({ queryKey: ['conversations'] })
      if (streamConversationId) queryClient.invalidateQueries({ queryKey: ['conversation', streamConversationId] })
      if (shouldRefreshTitle && streamConversationId) {
        scheduleConversationTitleRefresh(streamConversationId)
      }
    }
  }

  const messages = conversationId ? (active.data?.messages || []) : optimisticMessages
  const hasMessages = messages.length > 0
  const scopeDisabled = streaming || savingSessionOptions || (Boolean(conversationId) && active.isLoading)
  const fullContextDocuments = knowledgeBaseDocuments.data?.documents.filter((document) => document.full_context_available) || []
  const unavailableFullContextDocuments = knowledgeBaseDocuments.data?.documents.filter((document) => !document.full_context_available) || []
  const sessionControls = (
    <section className="chat-session-bar">
      <div className="chat-session-controls">
      <label className="chat-session-label">
        <span>会话资料</span>
      <select
        className="rounded-md border border-[var(--c-border)] bg-[var(--c-surface)] px-2 py-1 text-[var(--c-text)]"
        value={selectedKnowledgeScope}
        disabled={scopeDisabled || knowledgeBases.isLoading}
        onChange={(event) => applySessionOptions(
          event.target.value as 'all' | 'selected',
          event.target.value === 'all'
            ? []
            : (selectedKnowledgeBaseIds.length
              ? selectedKnowledgeBaseIds
              : (knowledgeBases.data?.knowledge_bases[0] ? [knowledgeBases.data.knowledge_bases[0].kb_id] : [])),
          null,
          selectedGroundingMode,
        )}
      >
        <option value="all">全部知识库</option>
        <option value="selected">指定知识库</option>
      </select>
      {selectedKnowledgeScope === 'selected' && (
        <select
          multiple
          size={Math.min(4, Math.max(2, knowledgeBases.data?.knowledge_bases.length || 0))}
          className="min-w-40 rounded-md border border-[var(--c-border)] bg-[var(--c-surface)] px-2 py-1 text-[var(--c-text)]"
          value={selectedKnowledgeBaseIds}
          disabled={scopeDisabled || knowledgeBases.isLoading}
          aria-label="指定知识库"
          onChange={(event) => applySessionOptions(
            'selected',
            Array.from(event.currentTarget.selectedOptions, (option) => option.value),
            null,
            selectedGroundingMode,
          )}
        >
          {knowledgeBases.data?.knowledge_bases.map((knowledgeBase) => (
            <option key={knowledgeBase.kb_id} value={knowledgeBase.kb_id}>{knowledgeBase.name}</option>
          ))}
        </select>
      )}
      {selectedKnowledgeBaseId && selectedGroundingMode !== 'assistant' && (
        <select
          className="rounded-md border border-[var(--c-border)] bg-[var(--c-surface)] px-2 py-1 text-[var(--c-text)]"
          value={selectedFullContextDocId || ''}
          disabled={scopeDisabled || knowledgeBaseDocuments.isLoading}
          onChange={(event) => applySessionOptions(
            'selected',
            selectedKnowledgeBaseIds,
            event.target.value || null,
            selectedGroundingMode,
          )}
        >
          <option value="">智能检索</option>
          {selectedFullContextDocId && !fullContextDocuments.some((document) => document.doc_id === selectedFullContextDocId) && (
            <option value={selectedFullContextDocId}>当前全文模式已不可用，请切换为智能检索</option>
          )}
          {fullContextDocuments.map((document) => (
            <option key={document.doc_id} value={document.doc_id}>全文阅读：{document.filename}</option>
          ))}
        </select>
      )}
      </label>
      <label className="chat-session-label">
        <span>回答模式</span>
        <select
          className="rounded-md border border-[var(--c-border)] bg-[var(--c-surface)] px-2 py-1 text-[var(--c-text)]"
          value={selectedGroundingMode}
          disabled={scopeDisabled}
          onChange={(event) => applySessionOptions(
            selectedKnowledgeScope,
            selectedKnowledgeBaseIds,
            event.target.value === 'assistant' ? null : selectedFullContextDocId,
            event.target.value as GroundingMode,
          )}
        >
          <option value="auto">自动路由</option>
          <option value="knowledge">知识库模式</option>
          <option value="assistant">通用助手</option>
        </select>
      </label>
      {selectedGroundingMode === 'assistant' && <span className="chat-session-note">不检索资料</span>}
      {selectedGroundingMode !== 'assistant' && selectedFullContextDocId && <span className="chat-session-note">全文阅读</span>}
      {selectedGroundingMode !== 'assistant' && selectedKnowledgeBaseId && !knowledgeBaseDocuments.isLoading && fullContextDocuments.length === 0 && (
        <span className="chat-session-note" title={unavailableFullContextDocuments[0]?.full_context_reason}>全文：暂无可用短文</span>
      )}
      {selectedGroundingMode !== 'assistant' && selectedKnowledgeBaseId && fullContextDocuments.length > 0 && unavailableFullContextDocuments.length > 0 && (
        <span className="chat-session-note" title={unavailableFullContextDocuments.map((document) => `${document.filename}：${document.full_context_reason || '不可用'}`).join('\n')}>全文：{unavailableFullContextDocuments.length} 篇不适用</span>
      )}
      </div>
    </section>
  )

  return (
    <main className="chat-main">
      {sessionControls}
      {hasMessages ? (
        <>
          <div className="messages">
            <div className="messages-inner">
              {messages.map((message) =>
                message.role === 'user' ? (
                  <UserBubble
                    key={message.message_id}
                    message={message}
                    onEdit={() => {
                      setEditingMessageId(message.message_id)
                      setDraft(message.content)
                    }}
                  />
                ) : (
                  <AssistantMessage key={message.message_id} message={message} streaming={streaming && message.status === 'streaming'} />
                ),
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
          <div className="composer">
            <div className="composer-box">
              {editingMessageId && <span className="edit-pill">正在编辑历史问题</span>}
              <AutoResizeTextarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="输入问题..."
                maxLength={4000}
                onCompositionStart={() => {
                  isComposingRef.current = true
                }}
                onCompositionEnd={() => {
                  isComposingRef.current = false
                }}
                onKeyDown={(event) => {
                  const nativeEvent = event.nativeEvent as KeyboardEvent
                  if (event.key === 'Enter' && !event.shiftKey) {
                    if (isComposingRef.current || nativeEvent.isComposing || nativeEvent.key === 'Process') return
                    event.preventDefault()
                    send()
                  }
                }}
              />
              <div className="composer-toolbar">
                <div className="composer-leading">
                  <AnswerQualityToggle
                    value={selectedAnswerQualityMode}
                    disabled={scopeDisabled}
                    onChange={(mode) => applySessionOptions(
                      selectedKnowledgeScope,
                      selectedKnowledgeBaseIds,
                      selectedFullContextDocId,
                      selectedGroundingMode,
                      selectedStreamValidationMode,
                      selectedThinkingEffort,
                      mode,
                    )}
                  />
                  <span className="composer-hint">{streaming ? '正在生成...' : 'Enter 发送，Shift+Enter 换行'}</span>
                </div>
                <div className="composer-actions">
                  <LLMModelSelect
                    value={selectedLLMModel}
                    models={llmModels}
                    disabled={scopeDisabled}
                    onChange={(llmModel) => {
                      const targetThinking = llmModels.find(
                        (model) => model.model_name === llmModel,
                      )?.thinking
                      applySessionOptions(
                        selectedKnowledgeScope,
                        selectedKnowledgeBaseIds,
                        selectedFullContextDocId,
                        selectedGroundingMode,
                        selectedStreamValidationMode,
                        targetThinking?.supported
                          ? resolveThinkingEffort(null, targetThinking)
                          : null,
                        selectedAnswerQualityMode,
                        llmModel,
                      )
                    }}
                  />
                  <ThinkingEffortSelect
                    value={selectedThinkingEffort}
                    efforts={thinkingEfforts}
                    supported={thinkingSupported}
                    disabled={scopeDisabled}
                    onChange={(effort) => applySessionOptions(
                      selectedKnowledgeScope,
                      selectedKnowledgeBaseIds,
                      selectedFullContextDocId,
                      selectedGroundingMode,
                      selectedStreamValidationMode,
                      effort,
                    )}
                  />
                  {streaming && canStopCurrentStream ? (
                    <button
                      className="composer-stop"
                      onClick={() => activeStreamKey && streamControllersRef.current.get(activeStreamKey)?.abort()}
                      title="停止"
                    ><Square className="h-3.5 w-3.5" /></button>
                  ) : streaming ? (
                    <span className="text-xs text-[var(--c-text-muted)]" title="该会话正在后台生成，返回后会自动同步结果">后台生成中</span>
                  ) : (
                    <button className="composer-send" onClick={() => send()} disabled={!draft.trim() || scopeDisabled} title="发送"><Send className="h-3.5 w-3.5" /></button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <WelcomeScreen
          onStart={(text) => send(text)}
          onOpenDocuments={() => navigate('/documents')}
          streaming={streaming}
          llmModel={selectedLLMModel}
          llmModels={llmModels}
          onLLMModelChange={(llmModel) => {
            const targetThinking = llmModels.find(
              (model) => model.model_name === llmModel,
            )?.thinking
            applySessionOptions(
              selectedKnowledgeScope,
              selectedKnowledgeBaseIds,
              selectedFullContextDocId,
              selectedGroundingMode,
              selectedStreamValidationMode,
              targetThinking?.supported
                ? resolveThinkingEffort(null, targetThinking)
                : null,
              selectedAnswerQualityMode,
              llmModel,
            )
          }}
          thinkingEffort={selectedThinkingEffort}
          thinkingEfforts={thinkingEfforts}
          thinkingSupported={thinkingSupported}
          answerQualityMode={selectedAnswerQualityMode}
          onAnswerQualityModeChange={(mode) => applySessionOptions(
            selectedKnowledgeScope,
            selectedKnowledgeBaseIds,
            selectedFullContextDocId,
            selectedGroundingMode,
            selectedStreamValidationMode,
            selectedThinkingEffort,
            mode,
          )}
          onThinkingEffortChange={(effort) => applySessionOptions(
            selectedKnowledgeScope,
            selectedKnowledgeBaseIds,
            selectedFullContextDocId,
            selectedGroundingMode,
            selectedStreamValidationMode,
            effort,
          )}
        />
      )}
    </main>
  )
}

function UserBubble({ message, onEdit }: { message: ChatMessage; onEdit: () => void }) {
  return (
    <div className="bubble-row-user group">
      <div className="user-bubble">
        <MessageMarkdown content={message.content} />
      </div>
      <button className="ml-1 mt-auto flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[var(--c-text-muted)] opacity-0 transition hover:bg-[var(--c-border-light)] hover:text-[var(--c-text)] group-hover:opacity-100" onClick={onEdit} title="编辑">
        <Pencil className="h-3 w-3" />
      </button>
    </div>
  )
}

function AssistantMessage({ message, streaming }: { message: ChatMessage; streaming?: boolean }) {
  const [copied, setCopied] = useState(false)
  const evidenceNotice = message.evidence_status
    ? ({
        partial: '部分证据',
        conflict: '证据冲突',
        no_evidence: '证据不足',
        unavailable: '校验未完成',
      } as Partial<Record<NonNullable<ChatMessage['evidence_status']>, string>>)[message.evidence_status]
    : undefined

  const copyText = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="assistant-row group">
      <div className="assistant-avatar"><Bot className="h-4 w-4" /></div>
      <div className="assistant-content">
        {evidenceNotice && message.status !== 'error' && (
          <span className="mb-2 inline-flex rounded-full border border-[var(--c-warning)]/30 bg-[var(--c-warning)]/10 px-2 py-0.5 text-[11px] font-medium text-[var(--c-text-secondary)]">
            {evidenceNotice}
          </span>
        )}
        {message.status === 'error' && (
          <div className="mb-2 rounded-md border border-[var(--c-danger)]/25 bg-[var(--c-danger)]/10 px-3 py-2 text-sm text-[var(--c-danger)]">
            {message.error_message || '生成失败，请稍后重试。'}
          </div>
        )}
        {message.status === 'stopped' && (
          <div className="mb-2 rounded-md border border-[var(--c-warning)]/25 bg-[var(--c-warning)]/10 px-3 py-2 text-sm text-[var(--c-text-muted)]">
            生成已停止，以下内容可能不完整。
          </div>
        )}
        {streaming && !message.content ? (
          <div className="thinking-dots"><span /><span /><span /></div>
        ) : (
          <>
            {message.content ? (
              <MessageMarkdown content={message.content} />
            ) : message.status !== 'error' ? (
              <MessageMarkdown content="..." />
            ) : null}
            {message.sources && message.sources.length > 0 && <SourceList sources={message.sources} />}
            {message.status === 'completed' && message.content && (
              <div className="mt-2 flex items-center gap-1 opacity-0 transition group-hover:opacity-100">
                <button className="flex h-6 items-center gap-1 rounded-md px-2 text-[11px] text-[var(--c-text-muted)] transition hover:bg-[var(--c-border-light)] hover:text-[var(--c-text)]" onClick={copyText}>
                  {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />} {copied ? '已复制' : '复制'}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function SourceList({ sources }: { sources: Source[] }) {
  const [openKeys, setOpenKeys] = useState<string[]>([])

  const collapseSource = (key: string) => {
    setOpenKeys((items) => items.filter((item) => item !== key))
  }

  const collapseAll = () => {
    setOpenKeys([])
  }

  return (
    <div className="source-list">
      {openKeys.length > 0 && (
        <div className="source-list-toolbar">
          <button type="button" className="source-list-collapse-all" onClick={collapseAll}>
            收起全部
          </button>
        </div>
      )}
      {sources.map((source) => {
        const sourceKey = `${source.index}-${source.chunk_id}`
        const structure = getSourceStructure(source)
        const metaItems = [
          structure.sectionTitle && { label: '章节', value: structure.sectionTitle },
          structure.headingPath && structure.headingPath !== structure.sectionTitle && { label: '路径', value: structure.headingPath },
          structure.tableHeaders && { label: '表头', value: structure.tableHeaders },
          structure.page && { label: '页码', value: structure.page },
          structure.sourceAnchor && { label: '锚点', value: structure.sourceAnchor },
        ].filter(Boolean) as Array<{ label: string; value: string }>
        const showComparableScore = typeof source.score === 'number'
          && !['reuse', 'full_context', 'context_expansion', 'unknown'].includes(source.score_source || 'unknown')

        return (
          <details
            className="source-card"
            key={sourceKey}
            open={openKeys.includes(sourceKey)}
            onToggle={(event) => {
              const isOpen = event.currentTarget.open
              setOpenKeys((items) => {
                if (isOpen) {
                  return items.includes(sourceKey) ? items : [...items, sourceKey]
                }
                return items.filter((item) => item !== sourceKey)
              })
            }}
          >
            <summary className="source-card-summary">
              <div className="source-card-head">
                <span className="source-card-index">[{source.index}]</span>
                <span className="source-card-title">{source.filename || source.doc_id}</span>
                {structure.kindLabel && <span className="source-card-kind">{structure.kindLabel}</span>}
              </div>
              {showComparableScore && (
                <small className="source-card-score" title="检索匹配度，不代表答案正确率">
                  匹配 {source.score?.toFixed(3)}
                </small>
              )}
            </summary>
            <div className="source-card-body">
              {metaItems.length > 0 && (
                <div className="source-card-meta">
                  {metaItems.map((item) => (
                    <div className="source-card-meta-item" key={`${source.chunk_id}-${item.label}`}>
                      <span className="source-card-meta-label">{item.label}</span>
                      <span className="source-card-meta-value" title={item.value}>{item.value}</span>
                    </div>
                  ))}
                </div>
              )}
              <p className="source-card-snippet">{source.text}</p>
              <div className="source-card-footer">
                <button type="button" className="source-card-collapse" onClick={() => collapseSource(sourceKey)}>
                  收起
                </button>
              </div>
            </div>
          </details>
        )
      })}
    </div>
  )
}

export default App
