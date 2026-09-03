import type { ChatMessage, Conversation } from '../types'

export type StoredChatLocation = {
  conversationId: string | null
  newConversationViewId: string | null
}

const SAFE_CHAT_ID = /^[A-Za-z0-9_-]{1,160}$/

export function chatLocationStorageKey(principalId?: string | null): string {
  const scope = principalId && SAFE_CHAT_ID.test(principalId) ? principalId : 'local'
  return `facet.chat.location:${scope}`
}

export function createNewConversationViewId(): string {
  return `new-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function parseStoredChatLocation(raw: string | null): StoredChatLocation {
  if (!raw) return { conversationId: null, newConversationViewId: null }
  try {
    const value = JSON.parse(raw) as Record<string, unknown> | null
    if (!value || typeof value !== 'object') {
      return { conversationId: null, newConversationViewId: null }
    }
    const conversationId = typeof value.conversationId === 'string'
      && SAFE_CHAT_ID.test(value.conversationId)
      ? value.conversationId
      : null
    const newConversationViewId = typeof value.newConversationViewId === 'string'
      && value.newConversationViewId.startsWith('new-')
      && SAFE_CHAT_ID.test(value.newConversationViewId)
      ? value.newConversationViewId
      : null
    return { conversationId, newConversationViewId }
  } catch {
    return { conversationId: null, newConversationViewId: null }
  }
}

export function resolveOutgoingEditFromMessageId(
  conversationId: string | null,
  editingMessageId: string | null,
): string | undefined {
  if (!conversationId) return undefined
  return editingMessageId || undefined
}

export function buildOptimisticMessages(
  content: string,
  createdAt: string,
  userMessageId: string,
  assistantMessageId: string,
): ChatMessage[] {
  return [
    {
      message_id: userMessageId,
      conversation_id: 'pending',
      role: 'user',
      content,
      status: 'completed',
      sources: [],
      seq: 1,
      created_at: createdAt,
      updated_at: createdAt,
    },
    {
      message_id: assistantMessageId,
      conversation_id: 'pending',
      role: 'assistant',
      content: '',
      status: 'streaming',
      sources: [],
      seq: 2,
      created_at: createdAt,
      updated_at: createdAt,
    },
  ]
}

export function buildOptimisticConversationMessages(
  existingMessages: ChatMessage[],
  content: string,
  createdAt: string,
  userMessageId: string,
  assistantMessageId: string,
  editFromMessageId?: string | null,
): ChatMessage[] {
  const baseMessages = editFromMessageId
    ? (() => {
        const editIndex = existingMessages.findIndex((message) => message.message_id === editFromMessageId)
        return editIndex >= 0 ? existingMessages.slice(0, editIndex) : existingMessages
      })()
    : existingMessages

  return [
    ...baseMessages,
    {
      message_id: userMessageId,
      conversation_id: baseMessages[0]?.conversation_id || 'pending',
      role: 'user',
      content,
      status: 'completed',
      sources: [],
      seq: baseMessages.length + 1,
      created_at: createdAt,
      updated_at: createdAt,
    },
    {
      message_id: assistantMessageId,
      conversation_id: baseMessages[0]?.conversation_id || 'pending',
      role: 'assistant',
      content: '',
      status: 'streaming',
      sources: [],
      seq: baseMessages.length + 2,
      created_at: createdAt,
      updated_at: createdAt,
    },
  ]
}

export function shouldActivateStreamConversation(
  startedConversationId: string | null,
  currentConversationId: string | null,
  startedNewConversationViewId?: string,
  currentNewConversationViewId?: string,
): boolean {
  if (startedConversationId) return startedConversationId === currentConversationId
  return currentConversationId === null && startedNewConversationViewId === currentNewConversationViewId
}

export function conversationStreamKey(conversationId: string | null, newConversationViewId: string): string {
  return conversationId ? `conversation:${conversationId}` : `new:${newConversationViewId}`
}

export function shouldRestoreEditingMessageIdAfterSendFailure(
  startedConversationId: string | null,
  currentConversationId: string | null,
  submittedEditMessageId: string | null,
  aborted: boolean,
  streamErrored: boolean,
): boolean {
  if (!submittedEditMessageId) return false
  if (aborted || streamErrored) return false
  return startedConversationId === currentConversationId
}

export function upsertConversation(
  conversations: Conversation[],
  nextConversation: Conversation,
): Conversation[] {
  return [
    nextConversation,
    ...conversations.filter((conversation) => conversation.conversation_id !== nextConversation.conversation_id),
  ]
}
