"""Chat, retrieval, and conversation routes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

from app.api.authz import enforce_chat_access, require_llm_invoke, require_rag_read, require_rag_write
from app.api.failure_responses import public_failure_payload, public_failure_response
from app.chunkers.recursive import estimate_tokens
from app.api.deps import (
    _get_settings,
    get_app_settings_store_instance,
    get_conversation_store_instance,
    get_bm25_store_instance,
    get_document_store_instance,
    get_embedding_provider_instance,
    get_embedding_provider_for_index_profile,
    get_index_profile_store_instance,
    get_knowledge_base_store_instance,
    get_llm_provider_instance,
    get_reranker_instance,
    get_vector_store_instance,
    resolve_identity_tenant,
    sync_llm_thinking_preference,
    verify_auth,
)
from app.pipeline.chat_flow import prepare_chat_turn, prepare_direct_chat_turn, prepare_full_context_turn, prepare_retrieval_only
from app.pipeline.retrieval_target import RetrievalTarget, resolve_active_retrieval_targets
from app.pipeline.conversation_title import build_fallback_title, generate_llm_title
from app.pipeline.generation import filter_sources_by_answer_citations, generate, generate_stream
from app.pipeline.grounding_contract import should_semantically_verify
from app.providers.llm.base import bind_llm_model
from app.providers.llm.thinking import describe_thinking_configuration, resolve_thinking_profile
from app.prompt_profile import resolve_prompt_profile
from app.utils.conversation_turns import (
    conversation_turn_lock,
    foreground_generation_budget,
    generation_slot,
)
from app.utils.model_labels import display_model_name
from app.utils.runtime_errors import (
    AnswerVerificationFailedError,
    AnswerVerificationUnavailableError,
    ConversationTurnQueueTimeoutError,
    GenerationQueueTimeoutError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])
ThinkingEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
AnswerQualityMode = Literal["normal", "enhanced"]


def _available_llm_models(settings) -> list[str]:
    return settings.llm.available_model_names()


def _validate_llm_model(settings, model_name: str | None) -> str:
    """Resolve one configured model without accepting arbitrary upstream IDs."""
    if model_name is None:
        return settings.llm.model_name
    requested = model_name.strip()
    matches = [
        candidate
        for candidate in _available_llm_models(settings)
        if requested in {candidate, display_model_name(candidate)}
    ]
    if len(matches) != 1:
        raise HTTPException(
            status_code=422,
            detail="该模型未在 llm.selectable_models 中配置。",
        )
    return matches[0]


def _effective_stored_llm_model(settings, model_name: object) -> str:
    stored = str(model_name or "").strip()
    return stored if stored in _available_llm_models(settings) else settings.llm.model_name


def _validate_thinking_effort(
    settings,
    effort: ThinkingEffort | None,
    *,
    model_name: str | None = None,
) -> ThinkingEffort | None:
    """Accept only native effort values advertised by the active profile."""
    if effort is None:
        return None
    _pattern, profile = resolve_thinking_profile(
        settings.llm.thinking,
        model_name or settings.llm.model_name,
    )
    if profile is None:
        raise HTTPException(
            status_code=422,
            detail="当前模型未配置原生思考能力，不能指定思考等级。",
        )
    if effort not in profile.efforts:
        raise HTTPException(
            status_code=422,
            detail="当前模型不支持该思考等级，请从配置表提供的选项中选择。",
        )
    return effort


def _effective_stored_thinking_effort(
    settings,
    effort: object,
    *,
    model_name: str | None = None,
) -> ThinkingEffort | None:
    """Ignore stale persisted values when a model/profile was changed."""
    normalized = str(effort or "").strip().lower()
    _pattern, profile = resolve_thinking_profile(
        settings.llm.thinking,
        model_name or settings.llm.model_name,
    )
    if profile is None or normalized not in profile.efforts:
        return None
    return normalized  # type: ignore[return-value]


def _stream_error_text(payload: dict[str, object]) -> str:
    """Make stored partial output visibly different from a failed empty turn."""
    error = str(payload.get("error") or "生成失败，请稍后重试。")
    if payload.get("partial_output"):
        return f"生成中断，以下内容可能不完整。{error}"
    return error


class ChatMessageInput(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    query: Optional[str] = Field(default=None, max_length=4000)
    message: Optional[ChatMessageInput] = None
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    edit_from_message_id: Optional[str] = Field(default=None, max_length=128)
    stream: bool = False
    # None means "do not mutate an existing conversation".  New
    # conversations resolve it to auto below.
    grounding_mode: Literal["auto", "knowledge", "assistant"] | None = None
    answer_quality_mode: AnswerQualityMode | None = None
    stream_validation_mode: Literal["validated", "realtime"] | None = None
    llm_model: Optional[str] = Field(default=None, max_length=256)
    thinking_effort: ThinkingEffort | None = None
    knowledge_scope: Literal["all", "selected"] | None = None
    knowledge_base_ids: list[Annotated[str, Field(min_length=1, max_length=128)]] | None = Field(
        default=None,
        max_length=32,
    )
    full_context_doc_id: Optional[str] = Field(default=None, max_length=128)


class ConversationScopeRequest(BaseModel):
    knowledge_scope: Literal["all", "selected"]
    knowledge_base_ids: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list,
        max_length=32,
    )
    full_context_doc_id: Optional[str] = Field(default=None, max_length=128)
    grounding_mode: Literal["auto", "knowledge", "assistant"]
    answer_quality_mode: AnswerQualityMode | None = None
    stream_validation_mode: Literal["validated", "realtime"] | None = None
    llm_model: Optional[str] = Field(default=None, max_length=256)
    thinking_effort: ThinkingEffort | None = None


class ConversationBatchDeleteRequest(BaseModel):
    conversation_ids: list[str] = Field(min_length=1, max_length=100)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    knowledge_scope: Literal["all", "selected"] = "all"
    knowledge_base_ids: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(
        default_factory=list,
        max_length=32,
    )


class OpenAIChatMessage(BaseModel):
    """The text-only message subset supported by the local Agent endpoint."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str = Field(min_length=1, max_length=16000)


class ChatCompletionsRequest(BaseModel):
    """Bounded OpenAI-compatible request accepted by ``/chat/completions``."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, max_length=128)
    messages: list[OpenAIChatMessage] = Field(min_length=1, max_length=100)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    reasoning_effort: ThinkingEffort | None = None

    @model_validator(mode="after")
    def validate_total_message_size(self):
        total_chars = sum(len(message.content or "") for message in self.messages)
        if total_chars > 50000:
            raise ValueError("messages content exceeds the 50000-character limit")
        return self


def _history_for_context(messages: list[dict], current_message_id: str | None = None) -> list[dict]:
    history = []
    for message in messages:
        if message.get("message_id") == current_message_id:
            break
        if message.get("role") == "assistant" and message.get("status") != "completed":
            continue
        if message.get("role") in {"user", "assistant"}:
            history.append(message)
    return history


@router.get("/chat/capabilities")
async def get_chat_capabilities(_identity: dict = Depends(require_rag_read)):
    """Return composer capabilities without probing model or embedding services."""
    settings = _get_settings()
    thinking = await sync_llm_thinking_preference()
    models = []
    for model_name in _available_llm_models(settings):
        model_thinking = describe_thinking_configuration(
            settings.llm.thinking,
            model_name,
            thinking["mode"],
        )
        model_thinking["source"] = thinking["source"]
        models.append({
            "model_name": model_name,
            "display_name": display_model_name(model_name),
            "thinking": model_thinking,
        })
    return {
        # Keep the default-model field for older clients while new clients use
        # the model-scoped table below.
        "thinking": thinking,
        "models": {
            "default_model": settings.llm.model_name,
            "options": models,
        },
        "answer_quality": {
            "default_mode": settings.chat.answer_quality.default_mode,
            "modes": ["normal", "enhanced"],
        },
    }


def _resolve_cited_sources(
    answer: str,
    results: list[dict],
    sources: list[dict],
    settings,
) -> list[dict]:
    return filter_sources_by_answer_citations(
        answer,
        sources,
        results,
        context_window=settings.llm.context_window,
        max_output_tokens=settings.llm.max_tokens,
        relevance_threshold=settings.retrieval.relevance_threshold,
    )


async def _safe_update_message(message_id: str, *, conversation_store=None, **updates) -> None:
    """Treat a concurrent conversation deletion as an intentional no-op."""
    conversation_store = conversation_store or get_conversation_store_instance()
    try:
        await conversation_store.update_message(message_id, **updates)
    except KeyError:
        logger.info("conversation deleted while message was being finalized: message_id=%s", message_id)


def _response_mode_for_turn(turn) -> str:
    """Use the orchestration policy instead of inferring it from DIRECT."""
    return turn.response_mode


def _quality_profile_for_turn(settings, turn):
    return settings.chat.answer_quality.profile(turn.answer_quality_mode)


def _quality_generation_kwargs(settings, turn) -> dict[str, object]:
    profile = _quality_profile_for_turn(settings, turn)
    verifier_candidate_chars = profile.evidence_max_candidate_chars
    # Full-context mode has already proved that the complete document fits the
    # generation budget. Truncating it again here makes facts near the end look
    # unsupported even though generation saw them.
    full_context_lengths = [
        len(str(result.get("text") or ""))
        for result in turn.results
        if (result.get("metadata") or {}).get("full_context")
    ]
    if full_context_lengths:
        verifier_candidate_chars = max(verifier_candidate_chars, *full_context_lengths)
    return {
        "semantic_verify": (
            profile.semantic_answer_verification
            and should_semantically_verify(
                turn.response_mode,
                has_sources=bool(turn.results),
                evidence_status=str((turn.evidence or {}).get("status") or ""),
            )
        ),
        "semantic_verification_timeout_seconds": profile.answer_verification_timeout_seconds,
        "semantic_verification_max_tokens": profile.answer_verification_max_tokens,
        "semantic_verification_max_candidate_chars": verifier_candidate_chars,
        "semantic_verification_max_retries": profile.answer_verification_max_retries,
        "semantic_verification_max_repairs": profile.max_answer_repairs,
        "evidence_guidance": turn.evidence_guidance,
    }


def _evidence_status_for_turn(turn) -> str:
    """Map the typed evidence result to the persisted public message state."""
    status = str((turn.evidence or {}).get("status") or "")
    if status in {"grounded", "boundary"}:
        return "grounded"
    if status in {"partial", "conflict", "unavailable"}:
        return status
    if status == "missing":
        return "no_evidence"
    if status == "not_checked" and turn.decision == "DIRECT":
        return "direct"
    return "unavailable"


async def _annotate_knowledge_base_sources(sources: list[dict], tenant_id: str) -> list[dict]:
    """Expose a stable KB label with every citation without leaking paths."""
    kb_ids = list(dict.fromkeys(
        (source.get("kb_id") or (source.get("metadata") or {}).get("kb_id") or "")
        for source in sources
    ))
    kb_ids = [kb_id for kb_id in kb_ids if kb_id]
    if not kb_ids:
        return sources
    store = get_knowledge_base_store_instance()
    names: dict[str, str] = {}
    for kb_id in kb_ids:
        kb = await store.get(kb_id)
        if kb and kb.get("tenant_id") == tenant_id:
            names[kb_id] = kb.get("name") or "知识库"
    for source in sources:
        kb_id = source.get("kb_id") or (source.get("metadata") or {}).get("kb_id") or ""
        if kb_id:
            source["kb_id"] = kb_id
        if kb_id in names:
            source["knowledge_base_name"] = names[kb_id]
    return sources


async def _resolve_current_prompt_profile(settings) -> str:
    configured = await get_app_settings_store_instance().get_prompt_profile()
    return resolve_prompt_profile(settings, configured)


def _first_user_message(messages: list[dict]) -> dict | None:
    return next((message for message in messages if message.get("role") == "user"), None)


async def _auto_title_conversation(
    *,
    conversation_id: str,
    tenant_id: str | None,
    assistant_message_id: str,
    expected_title: str,
    first_user_message_id: str,
    content: str,
    llm_provider,
    settings,
    conversation_store,
) -> None:
    try:
        assistant_message = await conversation_store.get_message(assistant_message_id, tenant_id=tenant_id)
        if not assistant_message or assistant_message.get("status") != "completed":
            return

        conversation = await conversation_store.get_conversation(conversation_id, tenant_id=tenant_id)
        if not conversation or conversation.get("title") != expected_title:
            return

        messages = await conversation_store.list_messages(conversation_id, tenant_id=tenant_id)
        first_user = _first_user_message(messages)
        if not first_user or first_user.get("message_id") != first_user_message_id:
            return

        # Titles are cosmetic. Never let a background title request steal a
        # scarce local-model slot from an interactive turn or wait behind it.
        try:
            async with generation_slot(
                settings.chat.max_concurrent_streams,
                wait_timeout_seconds=min(
                    1.0,
                    settings.chat.generation_queue_wait_timeout_seconds,
                ),
            ):
                generated_title = await generate_llm_title(content, llm_provider, settings)
        except GenerationQueueTimeoutError:
            logger.debug(
                "auto title skipped because the generation queue is busy: conversation=%s",
                conversation_id,
            )
            return
        if not generated_title or generated_title == expected_title:
            return

        conversation = await conversation_store.get_conversation(conversation_id, tenant_id=tenant_id)
        if not conversation or conversation.get("title") != expected_title:
            return

        messages = await conversation_store.list_messages(conversation_id, tenant_id=tenant_id)
        first_user = _first_user_message(messages)
        if not first_user or first_user.get("message_id") != first_user_message_id:
            return

        await conversation_store.update_title(
            conversation_id,
            generated_title,
            tenant_id=tenant_id,
        )
    except Exception:
        logger.warning("auto title generation skipped for conversation=%s", conversation_id, exc_info=True)


def _build_auto_title_background_task(
    *,
    should_auto_title: bool,
    conversation_id: str,
    tenant_id: str | None,
    assistant_message_id: str,
    expected_title: str,
    first_user_message_id: str,
    content: str,
    llm_provider,
    settings,
    conversation_store,
) -> None:
    if not should_auto_title or not settings.chat.auto_title_enabled:
        return None
    return BackgroundTask(
        _auto_title_conversation,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        assistant_message_id=assistant_message_id,
        expected_title=expected_title,
        first_user_message_id=first_user_message_id,
        content=content,
        llm_provider=llm_provider,
        settings=settings,
        conversation_store=conversation_store,
    )


async def _validate_retrieval_scope(
    *,
    knowledge_scope: str,
    knowledge_base_ids: list[str] | None,
    full_context_doc_id: str | None,
    tenant_id: str,
) -> list[str]:
    """Validate a tenant-bound all/selected knowledge-base scope."""
    selected_ids = list(dict.fromkeys(kb_id for kb_id in (knowledge_base_ids or []) if kb_id))
    if knowledge_scope == "all":
        if selected_ids:
            raise HTTPException(400, "全部知识库范围不能同时指定知识库")
        if full_context_doc_id:
            raise HTTPException(400, "全文模式需要恰好选择一个知识库")
        return []
    if knowledge_scope != "selected" or not selected_ids:
        raise HTTPException(400, "请至少选择一个知识库")
    settings = _get_settings()
    if len(selected_ids) > settings.retrieval.multi_knowledge_base.max_selected_knowledge_bases:
        raise HTTPException(400, "选择的知识库数量超过当前上限")
    knowledge_base_store = get_knowledge_base_store_instance()
    for kb_id in selected_ids:
        knowledge_base = await knowledge_base_store.get(kb_id)
        if not knowledge_base or knowledge_base.get("tenant_id") != tenant_id or knowledge_base.get("status") != "active":
            raise HTTPException(404, "Knowledge base not found")
    document_store = get_document_store_instance()
    if full_context_doc_id:
        if len(selected_ids) != 1:
            raise HTTPException(400, "全文模式只能选择一个知识库")
        document = await document_store.get(full_context_doc_id, tenant_id=tenant_id)
        if not document or document.get("kb_id") != selected_ids[0] or document.get("status") != "ready":
            raise HTTPException(400, "全文模式文档不属于当前知识库或尚未就绪")
    return selected_ids


async def _resolve_retrieval_targets(
    *,
    settings,
    tenant_id: str,
    tenant_slug: str | None,
    selected_kb_ids: list[str] | None,
    embedding_provider,
) -> list[RetrievalTarget]:
    """Compatibility wrapper retained for existing chat-route callers."""
    return await resolve_active_retrieval_targets(
        settings=settings,
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
        selected_kb_ids=selected_kb_ids,
        embedding_provider=embedding_provider,
        knowledge_base_store=get_knowledge_base_store_instance(),
        index_profile_store=get_index_profile_store_instance(),
        profile_embedding_provider_factory=get_embedding_provider_for_index_profile,
    )


async def _has_ready_documents_in_scope(
    document_store,
    *,
    tenant_id: str,
    selected_kb_ids: list[str] | None,
) -> bool:
    """Check local metadata before any Embedding/profile probe.

    A chat with no ready documents must remain available even while the
    Embedding service is down.  Collection dimensions are only needed once
    there is a corpus that could actually be queried.
    """
    if selected_kb_ids:
        list_ready_doc_ids = getattr(document_store, "list_ready_doc_ids", None)
        if callable(list_ready_doc_ids):
            for kb_id in selected_kb_ids:
                if await list_ready_doc_ids(kb_id, tenant_id=tenant_id):
                    return True
            return False

    has_ready_documents = getattr(document_store, "has_ready_documents", None)
    if callable(has_ready_documents):
        return bool(await has_ready_documents(tenant_id=tenant_id))

    # Older stores do not expose metadata readiness; retain their legacy
    # target resolution rather than falsely declaring an empty library.
    return True


async def _prepare_persistent_chat(req: ChatRequest, tenant_id: str | None = None):
    settings = _get_settings()
    content = (req.message.content if req.message else "").strip()
    if not content:
        raise HTTPException(400, "message.content is required")

    conversation_store = get_conversation_store_instance()
    conversation_id = req.conversation_id
    created_conversation = False
    edited_first_user = False
    if conversation_id:
        conversation = await conversation_store.get_conversation(conversation_id, tenant_id=tenant_id)
        if not conversation:
            raise HTTPException(404, "Conversation not found")
        stored_model = _effective_stored_llm_model(settings, conversation.get("llm_model"))
        selected_model = (
            _validate_llm_model(settings, req.llm_model)
            if req.llm_model is not None
            else stored_model
        )
        requested_thinking_effort = (
            _validate_thinking_effort(
                settings,
                req.thinking_effort,
                model_name=selected_model,
            )
            if req.thinking_effort is not None
            else _effective_stored_thinking_effort(
                settings,
                conversation.get("thinking_effort"),
                model_name=selected_model,
            )
        )
        model_changed = selected_model != str(conversation.get("llm_model") or "")
        thinking_changed = (
            req.thinking_effort is not None
            and (requested_thinking_effort or "") != str(conversation.get("thinking_effort") or "")
        )
        requested_ids = list(dict.fromkeys(kb_id for kb_id in (req.knowledge_base_ids or []) if kb_id))
        if (
            req.knowledge_scope is not None
            and (
                req.knowledge_scope != conversation.get("knowledge_scope", "all")
                or requested_ids != conversation.get("knowledge_base_ids", [])
            )
        ) or (
            req.full_context_doc_id is not None and req.full_context_doc_id != conversation.get("full_context_doc_id")
        ):
            raise HTTPException(409, "请先更新会话的知识库范围")
        if (
            req.grounding_mode is not None
            and req.grounding_mode != conversation.get("grounding_mode", "auto")
        ) or (
            req.answer_quality_mode is not None
            and req.answer_quality_mode != conversation.get(
                "answer_quality_mode",
                settings.chat.answer_quality.default_mode,
            )
        ) or (
            req.stream_validation_mode is not None
            and req.stream_validation_mode != conversation.get("stream_validation_mode", "realtime")
        ) or (
            model_changed or thinking_changed
        ):
            conversation = await conversation_store.update_retrieval_scope(
                conversation_id,
                knowledge_scope=conversation.get("knowledge_scope", "all"),
                knowledge_base_ids=conversation.get("knowledge_base_ids", []),
                full_context_doc_id=None if req.grounding_mode == "assistant" else conversation.get("full_context_doc_id"),
                grounding_mode=req.grounding_mode or conversation.get("grounding_mode", "auto"),
                answer_quality_mode=(
                    req.answer_quality_mode
                    or conversation.get("answer_quality_mode")
                    or settings.chat.answer_quality.default_mode
                ),
                llm_model=selected_model if model_changed else None,
                thinking_effort=(
                    requested_thinking_effort or ""
                    if model_changed or req.thinking_effort is not None
                    else None
                ),
                stream_validation_mode=req.stream_validation_mode,
                tenant_id=tenant_id,
            )
    else:
        selected_model = _validate_llm_model(settings, req.llm_model)
        requested_thinking_effort = _validate_thinking_effort(
            settings,
            req.thinking_effort,
            model_name=selected_model,
        )
        grounding_mode = req.grounding_mode or "auto"
        answer_quality_mode = req.answer_quality_mode or settings.chat.answer_quality.default_mode
        knowledge_scope = req.knowledge_scope or "all"
        knowledge_base_ids = list(dict.fromkeys(kb_id for kb_id in (req.knowledge_base_ids or []) if kb_id))
        if grounding_mode == "assistant" and req.full_context_doc_id:
            raise HTTPException(400, "通用助手不使用全文资料，请先切换为智能检索")
        await _validate_retrieval_scope(
            knowledge_scope=knowledge_scope,
            knowledge_base_ids=knowledge_base_ids,
            full_context_doc_id=req.full_context_doc_id,
            tenant_id=tenant_id or "",
        )
        conversation = await conversation_store.create_conversation(
            build_fallback_title(content, settings.chat.title_max_length),
            tenant_id=tenant_id,
            knowledge_scope=knowledge_scope,
            knowledge_base_ids=knowledge_base_ids,
            full_context_doc_id=req.full_context_doc_id,
            grounding_mode=grounding_mode,
            answer_quality_mode=answer_quality_mode,
            llm_model=selected_model,
            thinking_effort=requested_thinking_effort or "",
            stream_validation_mode=(
                req.stream_validation_mode
                or ("validated" if settings.chat.stream_validate_before_emit else "realtime")
            ),
        )
        conversation_id = conversation["conversation_id"]
        created_conversation = True

    # A profile edit can make an old conversation's stored effort invalid.
    # Keep the record for audit/history but never forward that stale value to
    # the provider or return it as the effective selection for this turn.
    conversation["llm_model"] = _effective_stored_llm_model(
        settings,
        conversation.get("llm_model"),
    )
    conversation["thinking_effort"] = _effective_stored_thinking_effort(
        settings,
        conversation.get("thinking_effort"),
        model_name=conversation["llm_model"],
    ) or ""

    if req.edit_from_message_id:
        target = await conversation_store.get_message(req.edit_from_message_id, tenant_id=tenant_id)
        if not target or target["conversation_id"] != conversation_id:
            raise HTTPException(404, "Message not found")
        if target["role"] != "user":
            raise HTTPException(400, "Only user messages can be edited")
        existing_messages = await conversation_store.list_messages(conversation_id, tenant_id=tenant_id)
        first_user = _first_user_message(existing_messages)
        edited_first_user = bool(first_user and first_user.get("message_id") == req.edit_from_message_id)
        await conversation_store.delete_from_message(conversation_id, req.edit_from_message_id)
        if edited_first_user:
            fallback_title = build_fallback_title(content, settings.chat.title_max_length)
            await conversation_store.update_title(
                conversation_id,
                fallback_title,
                tenant_id=tenant_id,
            )
            conversation = await conversation_store.get_conversation(conversation_id, tenant_id=tenant_id)
            if not conversation:
                raise HTTPException(404, "Conversation not found")

    user_message = await conversation_store.append_message(
        conversation_id,
        role="user",
        content=content,
        status="completed",
        grounding_mode=conversation.get("grounding_mode", "auto"),
        answer_quality_mode=(
            conversation.get("answer_quality_mode")
            or settings.chat.answer_quality.default_mode
        ),
        tenant_id=tenant_id,
    )
    messages = await conversation_store.list_messages(conversation_id, tenant_id=tenant_id)
    history = _history_for_context(messages, user_message["message_id"])
    return (
        conversation,
        user_message,
        history,
        {
            "should_auto_title": created_conversation or edited_first_user,
            "expected_title": conversation["title"],
            "content": content,
        },
    )


@router.get("/conversations")
async def list_conversations(identity: dict = Depends(require_rag_read)):
    conversation_store = get_conversation_store_instance()
    tenant = await resolve_identity_tenant(identity)
    return {
        "conversations": await conversation_store.list_conversations(
            tenant_id=tenant["tenant_id"],
        )
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, identity: dict = Depends(require_rag_read)):
    conversation_store = get_conversation_store_instance()
    tenant = await resolve_identity_tenant(identity)
    tenant_id = tenant["tenant_id"]
    conversation = await conversation_store.get_conversation(conversation_id, tenant_id=tenant_id)
    if not conversation:
        raise HTTPException(404, "Conversation not found")
    messages = await conversation_store.list_messages(conversation_id, tenant_id=tenant_id)
    return {"conversation": conversation, "messages": messages}


@router.put("/conversations/{conversation_id}/retrieval-scope")
async def update_conversation_retrieval_scope(
    conversation_id: str,
    req: ConversationScopeRequest,
    identity: dict = Depends(require_rag_write),
):
    tenant = await resolve_identity_tenant(identity)
    if req.grounding_mode == "assistant" and req.full_context_doc_id:
        raise HTTPException(400, "通用助手不使用全文资料，请先切换为智能检索")
    settings = _get_settings()
    conversation_store = get_conversation_store_instance()
    existing = await conversation_store.get_conversation(
        conversation_id,
        tenant_id=tenant["tenant_id"],
    )
    if not existing:
        raise HTTPException(404, "Conversation not found")
    stored_model = _effective_stored_llm_model(settings, existing.get("llm_model"))
    selected_model = (
        _validate_llm_model(settings, req.llm_model)
        if req.llm_model is not None
        else stored_model
    )
    requested_thinking_effort = (
        _validate_thinking_effort(
            settings,
            req.thinking_effort,
            model_name=selected_model,
        )
        if req.thinking_effort is not None
        else _effective_stored_thinking_effort(
            settings,
            existing.get("thinking_effort"),
            model_name=selected_model,
        )
    )
    model_changed = selected_model != str(existing.get("llm_model") or "")
    await _validate_retrieval_scope(
        knowledge_scope=req.knowledge_scope,
        knowledge_base_ids=req.knowledge_base_ids,
        full_context_doc_id=req.full_context_doc_id,
        tenant_id=tenant["tenant_id"],
    )
    try:
        conversation = await conversation_store.update_retrieval_scope(
            conversation_id,
            knowledge_scope=req.knowledge_scope,
            knowledge_base_ids=req.knowledge_base_ids,
            full_context_doc_id=req.full_context_doc_id,
            grounding_mode=req.grounding_mode,
            answer_quality_mode=req.answer_quality_mode,
            llm_model=selected_model if model_changed else None,
            thinking_effort=(
                requested_thinking_effort or ""
                if model_changed or req.thinking_effort is not None
                else None
            ),
            stream_validation_mode=req.stream_validation_mode,
            tenant_id=tenant["tenant_id"],
        )
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None
    return {"conversation": conversation}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, identity: dict = Depends(require_rag_write)):
    conversation_store = get_conversation_store_instance()
    tenant = await resolve_identity_tenant(identity)
    try:
        await conversation_store.delete_conversation(
            conversation_id,
            tenant_id=tenant["tenant_id"],
        )
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None
    return {"status": "deleted"}


@router.delete("/conversations")
async def delete_conversations(
    req: ConversationBatchDeleteRequest,
    identity: dict = Depends(require_rag_write),
):
    """Atomically delete a tenant-scoped conversation batch."""
    tenant = await resolve_identity_tenant(identity)
    deleted = await get_conversation_store_instance().delete_conversations(
        req.conversation_ids,
        tenant_id=tenant["tenant_id"],
    )
    return {"status": "deleted", "deleted": deleted}


@router.post("/chat")
async def chat(req: ChatRequest, identity: dict = Depends(verify_auth)):
    """RAG chat. New clients use persistent conversation fields; query remains compatibility mode."""
    enforce_chat_access(identity, has_message=bool(req.message))

    settings = _get_settings()
    requested_llm_model = _validate_llm_model(settings, req.llm_model)
    requested_thinking_effort = (
        None
        if req.message
        else _validate_thinking_effort(
            settings,
            req.thinking_effort,
            model_name=requested_llm_model,
        )
    )
    base_llm_provider = get_llm_provider_instance()
    await sync_llm_thinking_preference(base_llm_provider)
    llm_provider = bind_llm_model(base_llm_provider, requested_llm_model)
    # Resolve the effective mode only after a persistent conversation has
    # been loaded. Retrieval dependencies stay lazy so a general-assistant
    # turn never touches embedding/vector/reranker providers.
    assistant_mode = False
    effective_grounding_mode = req.grounding_mode or "auto"
    effective_answer_quality_mode = (
        req.answer_quality_mode or settings.chat.answer_quality.default_mode
    )
    embedding_provider = None
    vector_store = None
    document_store = None
    bm25_store = None
    reranker = None
    conversation_store = get_conversation_store_instance()
    prompt_profile = await _resolve_current_prompt_profile(settings)
    tenant = await resolve_identity_tenant(identity)
    tenant_id = tenant["tenant_id"]
    tenant_slug = tenant.get("slug")
    allowed_doc_ids: list[str] | None = None
    allowed_kb_ids: list[str] | None = None
    retrieval_targets: list[RetrievalTarget] | None = None

    def ensure_retrieval_dependencies() -> None:
        nonlocal embedding_provider, vector_store, document_store, bm25_store, reranker
        if embedding_provider is not None:
            return
        embedding_provider = get_embedding_provider_instance()
        vector_store = get_vector_store_instance()
        document_store = get_document_store_instance()
        bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
        reranker = get_reranker_instance()

    async def ensure_retrieval_targets() -> list[RetrievalTarget]:
        nonlocal retrieval_targets
        ensure_retrieval_dependencies()
        if retrieval_targets is None:
            if not await _has_ready_documents_in_scope(
                document_store,
                tenant_id=tenant_id,
                selected_kb_ids=allowed_kb_ids,
            ):
                retrieval_targets = []
            else:
                retrieval_targets = await _resolve_retrieval_targets(
                    settings=settings,
                    tenant_id=tenant_id,
                    tenant_slug=tenant_slug,
                    selected_kb_ids=allowed_kb_ids,
                    embedding_provider=embedding_provider,
                )
        return retrieval_targets

    async def prepare_request_turn(query: str, history: list[dict]):
        if assistant_mode:
            return prepare_direct_chat_turn(
                query,
                grounding_mode="assistant",
                answer_quality_mode=effective_answer_quality_mode,
            )
        ensure_retrieval_dependencies()
        targets = await ensure_retrieval_targets()
        return await prepare_chat_turn(
            query,
            history,
            settings,
            llm_provider,
            embedding_provider,
            vector_store,
            document_store,
            bm25_store,
            reranker,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
            grounding_mode=effective_grounding_mode,
            answer_quality_mode=effective_answer_quality_mode,
            allowed_doc_ids=allowed_doc_ids,
            allowed_kb_ids=allowed_kb_ids,
            retrieval_targets=targets,
        )

    if req.message:
        query = req.message.content.strip()

        async def prepare_persistent_request_turn(
            conversation: dict,
            history: list[dict],
            allowed_kb_ids: list[str],
            request_llm_provider,
        ):
            effective_grounding_mode = conversation.get("grounding_mode", "auto")
            effective_answer_quality_mode = (
                conversation.get("answer_quality_mode")
                or settings.chat.answer_quality.default_mode
            )
            if effective_grounding_mode == "assistant":
                return prepare_direct_chat_turn(
                    query,
                    grounding_mode="assistant",
                    answer_quality_mode=effective_answer_quality_mode,
                )
            ensure_retrieval_dependencies()
            targets = await ensure_retrieval_targets()
            full_context_doc_id = conversation.get("full_context_doc_id")
            if full_context_doc_id:
                active_target = targets[0] if len(targets) == 1 else next(
                    (target for target in targets if target.kb_ids == tuple(allowed_kb_ids)),
                    None,
                )
                return await prepare_full_context_turn(
                    query,
                    settings,
                    document_store,
                    doc_id=full_context_doc_id,
                    tenant_id=tenant_id,
                    profile_hash=active_target.index_id if active_target else "legacy",
                    llm_provider=request_llm_provider,
                    answer_quality_mode=effective_answer_quality_mode,
                )
            return await prepare_chat_turn(
                query, history, settings, request_llm_provider, embedding_provider, vector_store,
                document_store, bm25_store, reranker, tenant_id=tenant_id, tenant_slug=tenant_slug,
                grounding_mode=effective_grounding_mode,
                answer_quality_mode=effective_answer_quality_mode,
                allowed_doc_ids=allowed_doc_ids,
                allowed_kb_ids=allowed_kb_ids,
                retrieval_targets=targets,
            )

        if req.stream:
            title_job: dict | None = None

            async def run_auto_title_when_ready() -> None:
                if title_job:
                    await _auto_title_conversation(**title_job)

            async def event_generator():
                nonlocal title_job, allowed_kb_ids
                assistant_message: dict | None = None
                chunks: list[str] = []
                sources: list[dict] = []
                results: list[dict] = []
                message_evidence_status = "pending"
                try:
                    async with conversation_turn_lock(
                        tenant_id,
                        req.conversation_id,
                        wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
                    ):
                        conversation, user_message, history, title_context = await _prepare_persistent_chat(
                            req,
                            tenant_id=tenant_id,
                        )
                        request_llm_provider = bind_llm_model(
                            base_llm_provider,
                            conversation["llm_model"],
                        )
                        effective_grounding_mode = conversation.get("grounding_mode", "auto")
                        effective_answer_quality_mode = (
                            conversation.get("answer_quality_mode")
                            or settings.chat.answer_quality.default_mode
                        )
                        allowed_kb_ids = await _validate_retrieval_scope(
                            knowledge_scope=conversation.get("knowledge_scope", "all"),
                            knowledge_base_ids=conversation.get("knowledge_base_ids", []),
                            full_context_doc_id=conversation.get("full_context_doc_id"),
                            tenant_id=tenant_id,
                        )
                        assistant_message = await conversation_store.append_message(
                            conversation["conversation_id"],
                            role="assistant",
                            content="",
                            status="streaming",
                            sources=[],
                            grounding_mode=effective_grounding_mode,
                            answer_quality_mode=effective_answer_quality_mode,
                            evidence_status="pending",
                            tenant_id=tenant_id,
                        )
                        # Identity is emitted before potentially slow retrieval or model work.
                        yield {
                            "event": "meta",
                            "data": json.dumps({
                                "conversation_id": conversation["conversation_id"],
                                "user_message_id": user_message["message_id"],
                                "assistant_message_id": assistant_message["message_id"],
                                "title": conversation["title"],
                                "decision": "PENDING",
                                "reason": "preparing",
                                "fallback_used": False,
                                "grounding_mode": effective_grounding_mode,
                                "answer_quality_mode": effective_answer_quality_mode,
                                "llm_model": conversation["llm_model"],
                                "thinking_effort": conversation.get("thinking_effort") or None,
                            }, ensure_ascii=False),
                        }

                        completed = False
                        async with foreground_generation_budget(
                            settings.chat.max_concurrent_streams,
                            queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
                            turn_timeout_seconds=settings.chat.answer_quality.profile(
                                effective_answer_quality_mode
                            ).turn_timeout_seconds,
                        ):
                            turn = await prepare_persistent_request_turn(
                                conversation,
                                history,
                                allowed_kb_ids,
                                request_llm_provider,
                            )
                            message_evidence_status = _evidence_status_for_turn(turn)
                            await _annotate_knowledge_base_sources(turn.sources, tenant_id)
                            results = turn.results
                            sources = turn.sources
                            failed = False
                            async for event in generate_stream(
                                query,
                                results,
                                request_llm_provider,
                                history_messages=history,
                                prompt_profile=prompt_profile,
                                response_mode=_response_mode_for_turn(turn),
                                context_window=settings.llm.context_window,
                                max_output_tokens=settings.llm.max_tokens,
                                relevance_threshold=settings.retrieval.relevance_threshold,
                                history_limit=settings.chat.history_limit,
                                history_truncate=settings.chat.history_truncate,
                                validation_max_retries=settings.chat.answer_validation_max_retries,
                                validate_before_emit=(
                                    bool(results)
                                    or conversation.get("stream_validation_mode", "realtime") == "validated"
                                ),
                                output_chunk_chars=settings.chat.stream_output_chunk_chars,
                                output_chunk_delay_ms=settings.chat.stream_output_chunk_delay_ms,
                                thinking_effort=conversation.get("thinking_effort") or None,
                                unverified_context=turn.unverified_context,
                                **_quality_generation_kwargs(settings, turn),
                            ):
                                if event.get("event") == "message":
                                    payload = json.loads(event.get("data") or "{}")
                                    chunks.append(payload.get("content", ""))
                                    yield event
                                elif event.get("event") == "error":
                                    failed = True
                                    cited_sources = _resolve_cited_sources("".join(chunks), results, sources, settings)
                                    error_payload = json.loads(event.get("data") or "{}")
                                    if str(error_payload.get("code") or "").startswith("verification_"):
                                        message_evidence_status = "unavailable"
                                    await _safe_update_message(
                                        assistant_message["message_id"], content="".join(chunks), status="error",
                                        sources=cited_sources,
                                        evidence_status=message_evidence_status,
                                        error_message=_stream_error_text(error_payload),
                                    )
                                    yield event
                                elif event.get("event") == "done":
                                    if not failed:
                                        cited_sources = _resolve_cited_sources("".join(chunks), results, sources, settings)
                                        await _safe_update_message(
                                            assistant_message["message_id"], content="".join(chunks),
                                            status="completed", sources=cited_sources,
                                            evidence_status=message_evidence_status,
                                        )
                                        completed = True
                                        yield {
                                            "event": "sources",
                                            "data": json.dumps({"sources": cited_sources}, ensure_ascii=False),
                                        }
                                    yield event
                        if completed and title_context["should_auto_title"] and settings.chat.auto_title_enabled:
                            title_job = {
                                "conversation_id": conversation["conversation_id"],
                                "tenant_id": tenant_id,
                                "assistant_message_id": assistant_message["message_id"],
                                "expected_title": title_context["expected_title"],
                                "first_user_message_id": user_message["message_id"],
                                "content": title_context["content"],
                                "llm_provider": request_llm_provider,
                                "settings": settings,
                                "conversation_store": conversation_store,
                            }
                except asyncio.CancelledError:
                    if assistant_message:
                        cited_sources = _resolve_cited_sources("".join(chunks), results, sources, settings)
                        await _safe_update_message(
                            assistant_message["message_id"], content="".join(chunks),
                            status="stopped", sources=cited_sources,
                            evidence_status=(
                                message_evidence_status
                                if message_evidence_status != "pending"
                                else "unavailable"
                            ),
                        )
                    raise
                except Exception as exc:
                    logger.exception("persistent chat stream failed")
                    error_payload = public_failure_payload(
                        exc,
                        fallback="生成失败，请稍后重试。",
                        partial_output=bool(chunks),
                    )
                    error_text = _stream_error_text(error_payload)
                    if assistant_message:
                        cited_sources = _resolve_cited_sources("".join(chunks), results, sources, settings)
                        await _safe_update_message(
                            assistant_message["message_id"], content="".join(chunks), status="error",
                            sources=cited_sources,
                            error_message=error_text,
                            evidence_status=(
                                message_evidence_status
                                if message_evidence_status != "pending"
                                else "unavailable"
                            ),
                        )
                    yield {"event": "error", "data": json.dumps(error_payload, ensure_ascii=False)}
                    yield {"event": "done", "data": ""}

            return EventSourceResponse(
                event_generator(),
                background=BackgroundTask(run_auto_title_when_ready),
            )

        try:
            async with conversation_turn_lock(
                tenant_id,
                req.conversation_id,
                wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
            ):
                assistant_message: dict | None = None
                sources: list[dict] = []
                message_evidence_status = "pending"
                try:
                    conversation, user_message, history, title_context = await _prepare_persistent_chat(req, tenant_id=tenant_id)
                    request_llm_provider = bind_llm_model(
                        base_llm_provider,
                        conversation["llm_model"],
                    )
                    effective_grounding_mode = conversation.get("grounding_mode", "auto")
                    effective_answer_quality_mode = (
                        conversation.get("answer_quality_mode")
                        or settings.chat.answer_quality.default_mode
                    )
                    allowed_kb_ids = await _validate_retrieval_scope(
                        knowledge_scope=conversation.get("knowledge_scope", "all"),
                        knowledge_base_ids=conversation.get("knowledge_base_ids", []),
                        full_context_doc_id=conversation.get("full_context_doc_id"),
                        tenant_id=tenant_id,
                    )
                    assistant_message = await conversation_store.append_message(
                        conversation["conversation_id"], role="assistant", content="", status="streaming",
                        sources=[], grounding_mode=effective_grounding_mode, evidence_status="pending", tenant_id=tenant_id,
                        answer_quality_mode=effective_answer_quality_mode,
                    )
                    async with foreground_generation_budget(
                        settings.chat.max_concurrent_streams,
                        queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
                        turn_timeout_seconds=settings.chat.answer_quality.profile(
                            effective_answer_quality_mode
                        ).turn_timeout_seconds,
                    ):
                        turn = await prepare_persistent_request_turn(
                            conversation,
                            history,
                            allowed_kb_ids,
                            request_llm_provider,
                        )
                        message_evidence_status = _evidence_status_for_turn(turn)
                        await _annotate_knowledge_base_sources(turn.sources, tenant_id)
                        results = turn.results
                        sources = turn.sources
                        answer = await generate(
                            query, results, request_llm_provider, history_messages=history, prompt_profile=prompt_profile,
                            response_mode=_response_mode_for_turn(turn), context_window=settings.llm.context_window,
                            max_output_tokens=settings.llm.max_tokens,
                            relevance_threshold=settings.retrieval.relevance_threshold,
                            history_limit=settings.chat.history_limit, history_truncate=settings.chat.history_truncate,
                            validation_max_retries=settings.chat.answer_validation_max_retries,
                            thinking_effort=conversation.get("thinking_effort") or None,
                            unverified_context=turn.unverified_context,
                            **_quality_generation_kwargs(settings, turn),
                        )
                    cited_sources = _resolve_cited_sources(answer, results, sources, settings)
                    await _safe_update_message(
                        assistant_message["message_id"],
                        content=answer,
                        status="completed",
                        sources=cited_sources,
                        evidence_status=message_evidence_status,
                    )
                    meta = {
                        "conversation_id": conversation["conversation_id"],
                        "user_message_id": user_message["message_id"],
                        "assistant_message_id": assistant_message["message_id"],
                        "title": conversation["title"],
                        "decision": turn.decision,
                        "reason": turn.reason,
                        "fallback_used": turn.fallback_used,
                        "grounding_mode": turn.grounding_mode,
                        "answer_quality_mode": turn.answer_quality_mode,
                        "llm_model": conversation["llm_model"],
                        "thinking_effort": conversation.get("thinking_effort") or None,
                        "response_mode": turn.response_mode,
                    }
                    background = _build_auto_title_background_task(
                        should_auto_title=title_context["should_auto_title"],
                        conversation_id=conversation["conversation_id"], tenant_id=tenant_id,
                        assistant_message_id=assistant_message["message_id"],
                        expected_title=title_context["expected_title"], first_user_message_id=user_message["message_id"],
                        content=title_context["content"], llm_provider=request_llm_provider, settings=settings,
                        conversation_store=conversation_store,
                    )
                    return JSONResponse(content={**meta, "answer": answer, "sources": cited_sources}, background=background)
                except HTTPException:
                    raise
                except Exception as exc:
                    logger.exception("persistent chat generation failed")
                    error_payload = public_failure_payload(exc, fallback="生成失败，请稍后重试。")
                    safe_error = str(error_payload["error"])
                    if isinstance(
                        exc,
                        (AnswerVerificationUnavailableError, AnswerVerificationFailedError),
                    ):
                        message_evidence_status = "unavailable"
                    if assistant_message:
                        await _safe_update_message(
                            assistant_message["message_id"], content="", status="error",
                            error_message=safe_error,
                            sources=sources,
                            evidence_status=(
                                message_evidence_status
                                if message_evidence_status != "pending"
                                else "unavailable"
                            ),
                        )
                    return public_failure_response(exc, fallback="生成失败，请稍后重试。")
        except ConversationTurnQueueTimeoutError as exc:
            return public_failure_response(
                exc,
                fallback="生成失败，请稍后重试。",
            )

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "query or message.content is required")

    effective_grounding_mode = req.grounding_mode or "auto"
    assistant_mode = effective_grounding_mode == "assistant"
    if not assistant_mode:
        allowed_kb_ids = await _validate_retrieval_scope(
            knowledge_scope=req.knowledge_scope or "all",
            knowledge_base_ids=req.knowledge_base_ids or [],
            full_context_doc_id=req.full_context_doc_id,
            tenant_id=tenant_id,
        )

    if req.stream:
        async def legacy_stream():
            chunks: list[str] = []
            failed = False
            try:
                async with foreground_generation_budget(
                    settings.chat.max_concurrent_streams,
                    queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
                    turn_timeout_seconds=settings.chat.answer_quality.profile(
                        effective_answer_quality_mode
                    ).turn_timeout_seconds,
                ):
                    turn = await prepare_request_turn(query, [])
                    await _annotate_knowledge_base_sources(turn.sources, tenant_id)
                    results = turn.results
                    sources = turn.sources
                    async for event in generate_stream(
                        query,
                        results,
                        llm_provider,
                        prompt_profile=prompt_profile,
                        response_mode=_response_mode_for_turn(turn),
                        context_window=settings.llm.context_window,
                        max_output_tokens=settings.llm.max_tokens,
                        relevance_threshold=settings.retrieval.relevance_threshold,
                        history_limit=settings.chat.history_limit,
                        history_truncate=settings.chat.history_truncate,
                        validation_max_retries=settings.chat.answer_validation_max_retries,
                        validate_before_emit=bool(results) or settings.chat.stream_validate_before_emit,
                        output_chunk_chars=settings.chat.stream_output_chunk_chars,
                        output_chunk_delay_ms=settings.chat.stream_output_chunk_delay_ms,
                        thinking_effort=requested_thinking_effort,
                        unverified_context=turn.unverified_context,
                        **_quality_generation_kwargs(settings, turn),
                    ):
                        if event.get("event") == "message":
                            payload = json.loads(event.get("data") or "{}")
                            chunks.append(payload.get("content", ""))
                            yield event
                        elif event.get("event") == "error":
                            failed = True
                            yield event
                        elif event.get("event") == "done":
                            if not failed:
                                cited_sources = _resolve_cited_sources(
                                    "".join(chunks),
                                    results,
                                    sources,
                                    settings,
                                )
                                yield {"event": "sources", "data": json.dumps({"sources": cited_sources}, ensure_ascii=False)}
                            yield event
            except Exception as exc:
                logger.exception("legacy chat stream failed")
                error_payload = public_failure_payload(
                    exc,
                    fallback="生成失败，请稍后重试。",
                    partial_output=bool(chunks),
                )
                yield {"event": "error", "data": json.dumps(error_payload, ensure_ascii=False)}
                yield {"event": "done", "data": ""}

        return EventSourceResponse(legacy_stream())

    try:
        async with foreground_generation_budget(
            settings.chat.max_concurrent_streams,
            queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
            turn_timeout_seconds=settings.chat.answer_quality.profile(
                effective_answer_quality_mode
            ).turn_timeout_seconds,
        ):
            turn = await prepare_request_turn(query, [])
            await _annotate_knowledge_base_sources(turn.sources, tenant_id)
            answer = await generate(
                query,
                turn.results,
                llm_provider,
                prompt_profile=prompt_profile,
                response_mode=_response_mode_for_turn(turn),
                context_window=settings.llm.context_window,
                max_output_tokens=settings.llm.max_tokens,
                relevance_threshold=settings.retrieval.relevance_threshold,
                history_limit=settings.chat.history_limit,
                history_truncate=settings.chat.history_truncate,
                validation_max_retries=settings.chat.answer_validation_max_retries,
                thinking_effort=requested_thinking_effort,
                unverified_context=turn.unverified_context,
                **_quality_generation_kwargs(settings, turn),
            )
        cited_sources = _resolve_cited_sources(answer, turn.results, turn.sources, settings)
        return {"answer": answer, "sources": cited_sources}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("legacy chat generation failed")
        return public_failure_response(exc, fallback="生成失败，请稍后重试。")


async def _run_knowledge_search(req: QueryRequest, identity: dict):
    """Run the canonical read-only knowledge-search tool."""
    settings = _get_settings()
    tenant = await resolve_identity_tenant(identity)
    selected_kb_ids = await _validate_retrieval_scope(
        knowledge_scope=req.knowledge_scope,
        knowledge_base_ids=req.knowledge_base_ids,
        full_context_doc_id=None,
        tenant_id=tenant["tenant_id"],
    )
    # Tool/search calls can still invoke the LLM router or query rewriter.
    # They share the foreground budget with chat rather than bypassing the
    # local model limiter through a secondary API surface.
    async with foreground_generation_budget(
        settings.chat.max_concurrent_streams,
        queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
        turn_timeout_seconds=settings.chat.answer_quality.profile(
            settings.chat.answer_quality.default_mode
        ).turn_timeout_seconds,
    ):
        embedding_provider = get_embedding_provider_instance()
        turn = await prepare_retrieval_only(
            req.query,
            [],
            settings,
            get_llm_provider_instance(),
            embedding_provider,
            get_vector_store_instance(),
            get_document_store_instance(),
            get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None,
            get_reranker_instance(),
            tenant_id=tenant["tenant_id"],
            tenant_slug=tenant.get("slug"),
            allowed_kb_ids=selected_kb_ids or None,
            retrieval_targets=await _resolve_retrieval_targets(
                settings=settings,
                tenant_id=tenant["tenant_id"],
                tenant_slug=tenant.get("slug"),
                selected_kb_ids=selected_kb_ids,
                embedding_provider=embedding_provider,
            ),
            answer_quality_mode=settings.chat.answer_quality.default_mode,
        )
    await _annotate_knowledge_base_sources(turn.sources, tenant["tenant_id"])
    return turn, selected_kb_ids


@router.post("/tools/knowledge-search")
async def knowledge_search_endpoint(req: QueryRequest, identity: dict = Depends(require_rag_read)):
    """Tenant-scoped retrieval tool.  It never generates an answer."""
    try:
        turn, selected_kb_ids = await _run_knowledge_search(req, identity)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("knowledge-search failed")
        return public_failure_response(exc, fallback="检索暂时不可用，请稍后重试。")
    evidence_status = _evidence_status_for_turn(turn)
    status = {
        "grounded": "evidence_found",
        "partial": "partial_evidence",
        "conflict": "conflicting_evidence",
        "unavailable": "verification_unavailable",
    }.get(evidence_status, "no_evidence")
    return {
        "status": status,
        "evidence_status": evidence_status,
        "scope": {
            "type": req.knowledge_scope,
            "knowledge_base_ids": selected_kb_ids,
        },
        "retrieval_query": turn.retrieval_query,
        "decision": turn.decision,
        "reason": turn.reason,
        "evidence": turn.sources,
        "sources": turn.sources,
    }


@router.post("/query")
async def query_endpoint(req: QueryRequest, identity: dict = Depends(require_rag_read)):
    """Compatibility read-only retrieval endpoint backed by knowledge-search."""
    try:
        turn, _selected_kb_ids = await _run_knowledge_search(req, identity)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("compatibility retrieval query failed")
        return public_failure_response(exc, fallback="检索暂时不可用，请稍后重试。")
    return {"results": turn.sources}


@router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionsRequest,
    _: dict = Depends(require_llm_invoke),
):
    """Bounded text-only OpenAI-compatible endpoint for Agents."""
    messages = [message.model_dump(exclude_none=True) for message in req.messages]
    stream = req.stream
    settings = _get_settings()
    selected_model = _validate_llm_model(settings, req.model)
    requested_reasoning_effort = _validate_thinking_effort(
        settings,
        req.reasoning_effort,
        model_name=selected_model,
    )
    if req.max_tokens is not None and req.max_tokens > settings.llm.context_window:
        raise HTTPException(
            status_code=422,
            detail=f"max_tokens cannot exceed the configured context window ({settings.llm.context_window})",
        )
    estimated_input_tokens = estimate_tokens(
        "\n".join(message.get("content") or "" for message in messages)
    )
    requested_output_tokens = req.max_tokens or settings.llm.max_tokens
    if estimated_input_tokens + requested_output_tokens > settings.llm.context_window:
        raise HTTPException(
            status_code=422,
            detail="messages and max_tokens exceed the configured context window",
        )
    base_llm_provider = get_llm_provider_instance()
    await sync_llm_thinking_preference(base_llm_provider)
    llm_provider = bind_llm_model(base_llm_provider, selected_model)
    model_name = display_model_name(selected_model)
    created = int(time.time())
    response_id = f"chatcmpl-{uuid4().hex}"
    llm_kwargs = {}
    if req.max_tokens is not None:
        llm_kwargs["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        llm_kwargs["temperature"] = req.temperature
    if requested_reasoning_effort is not None:
        llm_kwargs["thinking_effort"] = requested_reasoning_effort

    if stream:
        async def openai_stream():
            first_chunk = True
            try:
                async with foreground_generation_budget(
                    settings.chat.max_concurrent_streams,
                    queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
                    turn_timeout_seconds=settings.chat.turn_timeout_seconds,
                ):
                    async for chunk in llm_provider.chat_stream(messages, **llm_kwargs):
                        delta = {"content": chunk}
                        if first_chunk:
                            delta = {"role": "assistant", "content": chunk}
                            first_chunk = False
                        yield {
                            "data": json.dumps({
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None,
                                }],
                            })
                        }
            except Exception as exc:
                logger.exception("Agent chat completion stream failed")
                failure_payload = public_failure_payload(
                    exc,
                    fallback="模型服务暂时不可用，请稍后重试。",
                )
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": {
                            "message": failure_payload["error"],
                            "type": "upstream_error",
                            "code": failure_payload["code"],
                        }
                    }, ensure_ascii=False),
                }
                yield "[DONE]"
                return
            yield {
                "data": json.dumps({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                })
            }
            yield "[DONE]"

        return EventSourceResponse(openai_stream())

    try:
        async with foreground_generation_budget(
            settings.chat.max_concurrent_streams,
            queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
            turn_timeout_seconds=settings.chat.turn_timeout_seconds,
        ):
            content = await llm_provider.chat(messages, **llm_kwargs)
    except Exception as exc:
        logger.exception("Agent chat completion failed")
        failure_payload = public_failure_payload(
            exc,
            fallback="模型服务暂时不可用，请稍后重试。",
        )
        return JSONResponse(
            status_code=503 if failure_payload["retryable"] else 502,
            content={
                "error": {
                    "message": failure_payload["error"],
                    "type": "upstream_error",
                    "code": failure_payload["code"],
                }
            },
        )
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }
