"""Evidence requirements, deterministic query expansion, and lexical gating.

This module owns the complete evidence-policy layer.  It deliberately has no
knowledge of routing, storage, or answer generation so the chat orchestrator
only coordinates typed stage results.
"""

from __future__ import annotations

import logging
import re
from math import ceil

from app.pipeline.evidence_contract import (
    EvidenceRequirement,
    extract_ability_claim_terms,
    extract_open_location_contract,
)
from app.utils.text_utils import STOP_WORDS, tokenize_mixed

logger = logging.getLogger(__name__)

_COLLECTION_OVERVIEW_RE = re.compile(
    r"(?:"
    r"(?:总结|概述|归纳|概括|提炼|梳理|介绍|说说|讲讲).{0,16}(?:文档|资料|知识库|内容)"
    r"|(?:这些|已上传|当前|全部|所有).{0,8}(?:文档|资料|内容).{0,16}"
    r"(?:主要|重点|结论|观点|说了什么|讲了什么)"
    r"|\b(?:summari[sz]e|overview|key\s+(?:points?|takeaways?))\b"
    r")",
    re.IGNORECASE,
)

_EVIDENCE_GATE_GENERIC_TERMS = {
    "什么", "怎么", "如何", "哪些", "多少", "为什么", "是否", "包含",
    "采用", "制作", "使用", "系统", "内容", "资料", "文档", "问题",
    "情况", "信息", "名字", "这个", "那个", "当前", "相关", "具体",
}
_EVIDENCE_GATE_QUESTION_TERMS = _EVIDENCE_GATE_GENERIC_TERMS | {
    "东西", "玩意", "意思", "介绍", "请问", "一下", "一下子", "到底",
    "叫", "属于", "类型", "种类", "方面", "有关", "了解", "说说",
    "方案",
}
_EVIDENCE_GATE_RELATION_TERMS = {
    "关系", "关联", "联系", "区别", "不同", "差异", "对比", "比较",
}
_EVIDENCE_GATE_RELATION_RE = re.compile(
    r"((和|与|跟|及|以及).*(关系|关联|联系|区别|不同|差异|对比|比较))"
    r"|((关系|关联|联系|区别|不同|差异|对比|比较).*(和|与|跟|及|以及))"
)
_EVIDENCE_GATE_TYPE_RE = re.compile(
    r"(?:什么|哪种|哪类|何种|哪个|哪一)(平台|产品|项目|服务|应用|系统|工具|软件|网站)"
    r"|(?:属于|算|是)(?:什么|哪种|哪类|何种)(平台|产品|项目|服务|应用|系统|工具|软件|网站)"
)
_EVIDENCE_GATE_DEFINITION_RE = re.compile(
    r"^(?:[a-z][a-z0-9_.-]{1,31}|[\u4e00-\u9fff]{2,12})\s*"
    r"(?:是(?:一个|一款|一种)?|叫|指)?\s*(?:什么(?:东西|玩意)?|什么意思|是啥|指什么)[？?。！!]*$",
    re.IGNORECASE,
)
_EVIDENCE_GATE_ABILITY_RE = re.compile(
    r"(?:是否会|都?会(?:不会|做)?|擅长|能不能|是否能|能够|能做|"
    r"可不可以|是否可以|可以|技能|能力|本领|专长).*(?:什么|哪些|哪几|吗|么|事)?$",
    re.IGNORECASE,
)
_EVIDENCE_GATE_ABILITY_SUFFIX_RE = re.compile(
    r"(?:是否会|都?会(?:不会|做)?|擅长|能不能|是否能|能够|能做|"
    r"可不可以|是否可以|可以|技能|能力|本领|专长).*$",
    re.IGNORECASE,
)
_EVIDENCE_GATE_BINDING_RE = re.compile(
    r"(?:在|位于|来自|就职于|属于|负责|支持|使用|提供|包含)"
    r"|(?:和|与).*(?:是同一个|同一)",
    re.IGNORECASE,
)
_NEGATIVE_FACT_QUERY_RE = re.compile(
    r"(?:包含|包括|包不|有没有|是否|能不能|可以不可以|可不可以|支持不支持|做不做).+"
    r"(?:吗|么|？|\?)$",
    re.IGNORECASE,
)
_NEGATIVE_FACT_TERMS = ("批量", "私信", "建联", "自动", "骚扰", "绕过", "触达")
_NEGATIVE_FACT_NOISE_RE = re.compile(
    r"(?:包不包含|是否包含|是否包括|包含不包含|包括不包括|有没有|能不能|可以不可以|可不可以|支持不支持|做不做)",
    re.IGNORECASE,
)
_NEGATIVE_FACT_EVIDENCE_RE = re.compile(
    r"(?:不包含|不包括|不做|不支持|不提供|不具备|不允许|禁止|未提供|"
    r"不涉及|不覆盖|不在(?:本|该)?范围|包含|包括|支持|提供|具备|允许)",
    re.IGNORECASE,
)
_NEGATIVE_FACT_CAPABILITY_PATTERNS = (
    ("批量私信", ("批量", "私信")),
    ("自动建联", ("自动", "建联")),
    ("自动触达", ("自动", "触达")),
    ("批量", ("批量",)),
    ("私信", ("私信",)),
    ("建联", ("建联",)),
    ("骚扰", ("骚扰",)),
    ("绕过", ("绕过",)),
    ("触达", ("触达",)),
)


def _is_negative_fact_query(query: str) -> bool:
    normalized = (query or "").strip().lower()
    return bool(
        _NEGATIVE_FACT_QUERY_RE.search(normalized)
        and any(term in normalized for term in _NEGATIVE_FACT_TERMS)
    )


_DOCUMENT_REFERENCE_RE = re.compile(
    r"[a-z0-9_\-\u4e00-\u9fff]+(?:\.[a-z0-9]{1,8})+",
    re.IGNORECASE,
)
_EVIDENCE_GATE_NUMERIC_RE = re.compile(
    r"多少|几(?:个|次|天|年|月|小时|分钟)?|数值|参数|门槛|价格|费用|成本|"
    r"比例|百分比|时长|日期|版本|错误码|error\s*code|\bcode\b",
    re.IGNORECASE,
)
_EVIDENCE_GATE_NUMERIC_FACETS = {
    "cpm", "cpc", "ctr", "roi",
    "gmv", "sku", "api", "url", "id",
}
_EVIDENCE_GATE_NUMERIC_FACET_GROUPS = {
    "参数": ("参数", "配置", "设置", "选项"),
    "门槛": ("门槛", "阈值", "最低", "最高", "上限", "下限", "<=", ">="),
    "价格": ("价格", "售价", "定价", "费用", "成本", "元", "$"),
    "费用": ("费用", "价格", "收费", "成本", "元", "$"),
    "成本": ("成本", "费用", "价格", "元", "$"),
    "比例": ("比例", "占比", "百分比", "%"),
    "百分比": ("百分比", "比例", "占比", "%"),
    "时长": ("时长", "小时", "分钟", "秒", "天", "周期"),
    "日期": ("日期", "时间", "年", "月", "日", "发布"),
    "版本": ("版本", "v", "release", "发行"),
    "错误码": ("错误码", "错误", "error", "code", "异常"),
}
_EVIDENCE_GATE_TYPE_GROUPS = {
    "平台": ("平台", "服务", "应用", "系统", "工具", "软件", "网站", "产品"),
    "产品": ("产品", "服务", "应用", "工具", "软件", "系统", "平台"),
    "项目": ("项目", "工程", "方案", "系统", "服务", "应用", "产品"),
    "服务": ("服务", "平台", "应用", "系统", "产品", "工具"),
    "应用": ("应用", "app", "软件", "服务", "平台", "系统", "工具"),
    "系统": ("系统", "平台", "服务", "软件", "应用", "工具"),
    "工具": ("工具", "软件", "应用", "服务", "平台", "系统"),
    "软件": ("软件", "应用", "工具", "系统", "服务", "平台"),
    "网站": ("网站", "平台", "服务", "应用", "系统"),
}
_DEFINITION_RETRIEVAL_INTENT = "产品定位 目标用户 核心能力"
_ABILITY_RETRIEVAL_INTENT = "技能 能力 擅长 职责"
_DEFINITION_EVIDENCE_TERMS = (
    "是",
    "指",
    "定位",
    "用于",
    "面向",
    "平台",
    "产品",
    "项目",
    "服务",
    "应用",
    "系统",
    "工具",
    "软件",
    "网站",
)
_OPEN_ABILITY_EVIDENCE_TERMS = (
    "会",
    "能",
    "擅长",
    "精通",
    "熟悉",
    "掌握",
    "负责",
    "从事",
    "技能",
    "能力",
    "本领",
    "专长",
)


def _unique_terms(terms: list[str]) -> list[str]:
    return list(dict.fromkeys(term for term in terms if term))


def _evidence_anchor_terms(query: str, *, ignored: set[str]) -> list[str]:
    terms: list[str] = []
    # Filenames are retrieval scope hints, not answer facts.  Tokenising
    # ``测试小说.txt`` into ``测试``, ``小说`` and ``txt`` would otherwise
    # make a correct result fail merely because its body omits the extension.
    query_without_document_refs = _DOCUMENT_REFERENCE_RE.sub(" ", query or "")
    for raw in tokenize_mixed(query_without_document_refs):
        term = raw.strip().lower()
        if (
            len(term) <= 1
            or term in STOP_WORDS
            or term in ignored
        ):
            continue
        terms.append(term)
    return _unique_terms(terms)


def _negative_fact_capability_groups(query: str) -> tuple[tuple[str, ...], ...]:
    """Extract capability phrases without treating each character as context.

    A scope question such as ``MVP 包含批量私信或自动建联吗`` has two facts
    to verify.  The words ``批量``/``私信``/``自动``/``建联`` are not product
    identity anchors; they are grouped here so each requested capability must
    be represented by the same evidence unit.
    """
    normalized = (query or "").strip().lower()
    groups: list[tuple[str, ...]] = []
    used_terms: set[str] = set()
    for phrase, terms in _NEGATIVE_FACT_CAPABILITY_PATTERNS:
        if phrase not in normalized and not all(term in normalized for term in terms):
            continue
        if any(term in used_terms for term in terms):
            continue
        groups.append((phrase, *terms))
        used_terms.update(terms)
    return tuple(groups)


def _build_evidence_requirement(query: str) -> EvidenceRequirement:
    """Parse a small, safe subset of Chinese question forms without an LLM."""
    normalized_query = (query or "").strip().lower()
    # A collection overview is a request to synthesize the currently scoped
    # source set, not a claim that must repeat literal words from the query.
    # Applying entity/facet anchors here made valid uploaded-document
    # summaries disappear before the LLM received any evidence.
    if _COLLECTION_OVERVIEW_RE.search(normalized_query):
        return EvidenceRequirement(kind="collection_overview")
    open_location = extract_open_location_contract(normalized_query)
    if open_location is not None:
        subject, relation_groups = open_location
        entities = _evidence_anchor_terms(
            subject,
            ignored=_EVIDENCE_GATE_QUESTION_TERMS | _EVIDENCE_GATE_RELATION_TERMS,
        )
        return EvidenceRequirement(
            kind="binding",
            required_entities=tuple(entities),
            alternative_fact_groups=relation_groups,
        )
    type_match = _EVIDENCE_GATE_TYPE_RE.search(normalized_query)
    base_ignored = _EVIDENCE_GATE_QUESTION_TERMS | _EVIDENCE_GATE_RELATION_TERMS
    ability_question = bool(_EVIDENCE_GATE_ABILITY_RE.search(normalized_query))
    # Some segmenters emit ``林小北会`` as one token for “林小北会做什么”.
    # Remove the capability predicate before extracting the entity so the
    # question is grounded on “林小北”, not on a non-existent “林小北会”.
    negative_fact_query = _is_negative_fact_query(normalized_query)
    anchor_query = (
        _EVIDENCE_GATE_ABILITY_SUFFIX_RE.sub(" ", normalized_query)
        if ability_question
        else normalized_query
    )
    if negative_fact_query:
        # “包不包含/是否支持”等是提问句式，不是文档中必须逐字出现的
        # 实体锚点；只保留产品名和被核验的能力词。
        anchor_query = _NEGATIVE_FACT_NOISE_RE.sub(" ", anchor_query)
        # Capability words are checked as grouped facts below, not as product
        # identity anchors.  This allows a parent titled only “MVP 范围” to
        # support a question that also mentions the broader “KOL 采集” topic.
        negative_terms = set(_NEGATIVE_FACT_TERMS)
        anchors = _evidence_anchor_terms(
            anchor_query,
            ignored=base_ignored | negative_terms,
        )
        return EvidenceRequirement(
            kind="factual",
            required_entities=tuple(anchors),
            alternative_fact_groups=_negative_fact_capability_groups(normalized_query),
        )
    anchors = _evidence_anchor_terms(anchor_query, ignored=base_ignored)

    # Type questions are checked before generic definitions: ``是什么平台`` is
    # both a definition form and a type constraint.
    if type_match:
        requested_type = next((group for group in type_match.groups() if group), "")
        entities = [term for term in anchors if term != requested_type]
        synonyms = _EVIDENCE_GATE_TYPE_GROUPS.get(requested_type, (requested_type,))
        return EvidenceRequirement(
            kind="type",
            required_entities=tuple(entities),
            alternative_fact_groups=(tuple(synonyms),),
        )

    if _EVIDENCE_GATE_RELATION_RE.search(normalized_query):
        return EvidenceRequirement(kind="relation", required_entities=tuple(anchors))

    if _EVIDENCE_GATE_NUMERIC_RE.search(normalized_query):
        alternative_groups = [
            _EVIDENCE_GATE_NUMERIC_FACET_GROUPS[term]
            for term in anchors
            if term in _EVIDENCE_GATE_NUMERIC_FACET_GROUPS
        ]
        facets = [term for term in anchors if term in _EVIDENCE_GATE_NUMERIC_FACETS]
        # Uppercase abbreviations and error identifiers are often the fact
        # dimension even when they are not in the small known-facet set.
        for raw in re.findall(r"[a-z][a-z0-9_.-]{1,31}", normalized_query, re.IGNORECASE):
            term = raw.lower()
            if term in anchors and (len(term) >= 2 or any(char.isdigit() for char in term)):
                if term in _EVIDENCE_GATE_NUMERIC_FACETS or any(char.isdigit() for char in term):
                    facets.append(term)
        facets = _unique_terms(facets)
        entities = [
            term for term in anchors
            if term not in facets and term not in _EVIDENCE_GATE_NUMERIC_FACET_GROUPS
        ]
        # Do not let a numeric-looking question with no stable fact dimension
        # silently become weaker than the old gate.
        if not facets and not alternative_groups:
            return EvidenceRequirement(kind="factual", required_entities=tuple(anchors))
        return EvidenceRequirement(
            kind="numeric",
            required_entities=tuple(entities),
            required_facets=tuple(facets),
            alternative_fact_groups=tuple(tuple(_unique_terms(list(group))) for group in alternative_groups),
        )

    # These question forms bind an entity to a location, identity, owner or
    # capability.  Do not let evidence from separate documents be combined
    # into a fact that neither document states on its own.
    if _EVIDENCE_GATE_BINDING_RE.search(normalized_query) and not negative_fact_query:
        return EvidenceRequirement(kind="binding", required_entities=tuple(anchors))

    # A question such as "林小北都会什么" is about a person's capabilities,
    # not a product-definition question.  Detect it before the broad
    # definition pattern that also ends in "什么".
    if ability_question:
        claim_terms = extract_ability_claim_terms(normalized_query)
        alternative_groups = () if claim_terms else (_OPEN_ABILITY_EVIDENCE_TERMS,)
        return EvidenceRequirement(
            kind="ability",
            required_entities=tuple(anchors),
            required_claim_terms=claim_terms,
            alternative_fact_groups=alternative_groups,
        )

    if _EVIDENCE_GATE_DEFINITION_RE.search(normalized_query):
        return EvidenceRequirement(
            kind="definition",
            required_entities=tuple(anchors),
            alternative_fact_groups=(_DEFINITION_EVIDENCE_TERMS,),
        )

    return EvidenceRequirement(kind="factual", required_entities=tuple(anchors))


def _definition_retrieval_expansion(query: str) -> str:
    """Return a deterministic retrieval intent for an entity definition question.

    The expansion adds no facts and retains the user's entity verbatim.  It
    only gives documents that describe a product in a positioning section a
    useful retrieval target when the entity itself appears only in a title.
    """
    requirement = _build_evidence_requirement(query)
    if requirement.kind not in {"definition", "type"}:
        return ""
    if len(requirement.required_entities) != 1:
        return ""
    entity = requirement.required_entities[0]
    if len(entity) < 2:
        return ""
    return f"{entity} {_DEFINITION_RETRIEVAL_INTENT}"


def _ability_retrieval_expansion(query: str) -> str:
    """Add stable skill-oriented terms for capability questions.

    This keeps the named entity intact while avoiding product-positioning
    vocabulary, which is useful for a person, role, or fictional character.
    """
    requirement = _build_evidence_requirement(query)
    if requirement.kind != "ability" or len(requirement.required_entities) != 1:
        return ""
    entity = requirement.required_entities[0]
    return f"{entity} {_ABILITY_RETRIEVAL_INTENT}" if len(entity) >= 2 else ""


_INTENT_QUERY_ALIASES = (
    ("核心用户", "目标用户"),
    ("用户群体", "目标用户"),
    ("服务对象", "目标用户"),
    ("目标人群", "目标用户"),
)


def _intent_retrieval_expansion(query: str, aliases: dict[str, str] | None = None) -> str:
    """Expand product synonyms while retaining the user's original anchors."""
    additions: list[str] = []
    configured_aliases = aliases or dict(_INTENT_QUERY_ALIASES)
    alias_items = (
        configured_aliases.items()
        if isinstance(configured_aliases, dict)
        else _INTENT_QUERY_ALIASES
    )
    for source, target in alias_items:
        source = str(source).strip()
        target = str(target).strip()
        if not source or not target:
            continue
        if source in query and target not in query:
            additions.append(target)
    return f"{query} {' '.join(dict.fromkeys(additions))}".strip() if additions else ""


def _negative_fact_retrieval_expansion(query: str) -> str:
    """Add boundary terms for yes/no questions about excluded capabilities."""
    normalized = (query or "").strip()
    if not normalized or not _NEGATIVE_FACT_QUERY_RE.search(normalized):
        return ""
    if not any(term in normalized for term in _NEGATIVE_FACT_TERMS):
        return ""
    return f"{normalized} 范围 边界 不包含 不做"


def _evidence_haystack(results: list[dict]) -> str:
    return "\n".join(
        "\n".join(
            [
                str(item.get("text") or ""),
                str((item.get("metadata") or {}).get("section_title") or ""),
                str((item.get("metadata") or {}).get("heading_path") or ""),
                str((item.get("metadata") or {}).get("table_headers") or ""),
                str((item.get("metadata") or {}).get("filename") or ""),
                str((item.get("metadata") or {}).get("file_stem") or ""),
            ]
        ).lower()
        for item in results
    )


def _contains_evidence_term(haystack: str, term: str) -> bool:
    """Match an evidence anchor without allowing ASCII prefix collisions.

    Chinese anchors are naturally matched as substrings, while identifiers and
    English names need token boundaries (``hook`` must not match ``atlas``).
    """
    normalized_haystack = (haystack or "").lower()
    normalized_term = (term or "").strip().lower()
    if not normalized_term:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", normalized_term, re.IGNORECASE):
        pattern = rf"(?<![a-z0-9_.-]){re.escape(normalized_term)}(?![a-z0-9_.-])"
        return re.search(pattern, normalized_haystack, re.IGNORECASE) is not None
    return normalized_term in normalized_haystack


def _requirement_matches_haystack(requirement: EvidenceRequirement, haystack: str) -> bool:
    """Check all structured anchors inside one evidence unit."""
    if requirement.kind == "collection_overview":
        return bool(haystack.strip())
    if requirement.kind in {"definition", "type"} and not requirement.required_entities:
        return False
    if requirement.kind == "relation" and len(requirement.required_entities) < 2:
        return False
    if (
        requirement.kind == "factual"
        and not requirement.required_entities
        and not requirement.required_facets
        and not requirement.alternative_fact_groups
    ):
        # A question made only of generic wording (for example “是什么东西”)
        # has no stable anchor that can ground a knowledge-base answer.
        return False
    if requirement.kind == "type":
        # Descriptive type questions often contain several loose modifiers
        # ("内容团队、跨渠道工具、本质").  Requiring every modifier verbatim
        # rejects a valid synonym such as “跨平台内容创作系统”.  A named
        # entity still remains strict; a description only needs one stable
        # anchor plus the type synonym below.
        if not any(_contains_evidence_term(haystack, term) for term in requirement.required_entities):
            return False
    elif any(not _contains_evidence_term(haystack, term) for term in requirement.required_entities):
        return False
    if any(not _contains_evidence_term(haystack, term) for term in requirement.required_facets):
        return False
    if any(not _contains_evidence_term(haystack, term) for term in requirement.required_claim_terms):
        return False
    return all(
        any(_contains_evidence_term(haystack, term) for term in group)
        for group in requirement.alternative_fact_groups
    )


def _has_sufficient_structured_parent_evidence(query: str, haystack: str) -> bool:
    requirement = _build_evidence_requirement(query)
    if requirement.kind == "collection_overview":
        return bool(haystack.strip())
    missing_entities = [term for term in requirement.required_entities if not _contains_evidence_term(haystack, term)]
    missing_facets = [term for term in requirement.required_facets if not _contains_evidence_term(haystack, term)]
    missing_claims = [term for term in requirement.required_claim_terms if not _contains_evidence_term(haystack, term)]
    unmatched_groups = [
        group for group in requirement.alternative_fact_groups
        if not any(_contains_evidence_term(haystack, term) for term in group)
    ]

    if requirement.kind in {"definition", "type"} and not requirement.required_entities:
        return False
    if requirement.kind == "relation" and len(requirement.required_entities) < 2:
        return False
    if (
        requirement.kind == "factual"
        and not requirement.required_entities
        and not requirement.required_facets
        and not requirement.required_claim_terms
        and not requirement.alternative_fact_groups
    ):
        # A question made only of generic wording (for example “是什么东西”)
        # has no stable anchor that can ground a knowledge-base answer.
        return False

    # Scope questions such as “是否包含批量私信和自动建联” are a
    # conjunction check, not a loose topical question.  Requiring every
    # extracted anchor prevents a chunk mentioning only “私信” from being
    # accepted as evidence for the full exclusion boundary.
    if _is_negative_fact_query(query):
        # Identity/context wording can be broader than the section heading
        # (for example “KOL 采集 MVP” is answered by a parent headed “MVP
        # 范围”).  At least one context anchor plus every requested capability
        # and an explicit scope/stance marker is sufficient; requiring every
        # loose context token rejects valid table rows.
        context_match = (
            not requirement.required_entities
            or any(
                _contains_evidence_term(haystack, term)
                for term in requirement.required_entities
            )
        )
        capability_match = all(
            _contains_evidence_term(haystack, group[0])
            or all(_contains_evidence_term(haystack, term) for term in group[1:])
            for group in requirement.alternative_fact_groups
        )
        stance_match = _NEGATIVE_FACT_EVIDENCE_RE.search(haystack) is not None
        return context_match and capability_match and stance_match

    if not missing_entities and not missing_facets and not missing_claims and not unmatched_groups:
        return True

    # Generic factual questions are intentionally conservative, but should not
    # demand a literal replay of every non-fact wording term.  Definition,
    # type, numeric, and relation forms above use their exact requirement.
    if requirement.kind == "factual" and requirement.required_entities:
        entities = requirement.required_entities
        matched = [term for term in entities if _contains_evidence_term(haystack, term)]
        required = 1 if len(entities) == 1 else min(len(entities), max(2, ceil(len(entities) * 0.5)))
        if len(matched) >= required:
            return True
        missing_entities = [term for term in entities if not _contains_evidence_term(haystack, term)]

    logger.info(
        "knowledge evidence gate rejected mode=structured kind=%s query=%r "
        "required_entities=%s required_facets=%s required_claims=%s alternative_groups=%s "
        "missing_entities=%s missing_facets=%s missing_claims=%s unmatched_groups=%s",
        requirement.kind,
        query,
        requirement.required_entities,
        requirement.required_facets,
        requirement.required_claim_terms,
        requirement.alternative_fact_groups,
        missing_entities,
        missing_facets,
        missing_claims,
        unmatched_groups,
    )
    return False


def _has_sufficient_parent_evidence(
    query: str,
    results: list[dict],
) -> bool:
    """Reject obvious semantic false positives in strict knowledge mode.

    Vector similarity is always relative: with a small corpus it can rank an
    unrelated chunk highly.  For the structured parent-child index, validate
    the evidence anchors required by the question form.  The same validation
    applies to structured parent-child and legacy flat indexes.
    """
    if not results:
        return False
    requirement = _build_evidence_requirement(query)
    if requirement.kind in {"definition", "type", "numeric", "relation", "binding"}:
        # Do not assemble a type/metric/relation answer from unrelated chunks.
        # The same parent (or flat result) must carry every required anchor.
        return any(
            _requirement_matches_haystack(requirement, _evidence_haystack([result]))
            for result in results
        )
    if _is_negative_fact_query(query):
        # Negative scope facts are conjunctive.  Never combine the product
        # anchor from one parent with one capability/stance from another;
        # otherwise a multi-document candidate pool can manufacture an
        # answer that no individual source actually states.
        return any(
            _has_sufficient_structured_parent_evidence(
                query,
                _evidence_haystack([result]),
            )
            for result in results
        )
    haystack = _evidence_haystack(results)
    return _has_sufficient_structured_parent_evidence(query, haystack)


def _has_sufficient_evidence_for_queries(
    queries: list[str],
    results: list[dict],
) -> bool:
    """Accept original or safely rewritten wording for grounded follow-ups.

    A pronoun such as ``他`` has no usable entity anchor in the original
    message, while the standalone rewrite contains the entity from history.
    Checking both keeps the gate strict for new facts without rejecting valid
    contextual follow-ups.
    """
    return any(
        query and _has_sufficient_parent_evidence(query, results)
        for query in dict.fromkeys(queries)
    )


def _filter_results_to_question_evidence(query: str, results: list[dict]) -> list[dict]:
    """Remove context candidates that cannot support the question's anchors.

    A reranker can retain a semantically nearby name (for example “林晓北”
    for “林小北”) with a low but non-zero score.  For question forms whose
    evidence must be self-contained, those candidates add noise and can never
    support a valid answer, so do not send them to the LLM at all.
    """
    requirement = _build_evidence_requirement(query)
    self_contained_kinds = {
        "ability", "definition", "type", "numeric", "relation", "binding",
    }
    # Multi-anchor factual questions such as “杭州办公室周三有什么安排”
    # also need one self-contained evidence unit; otherwise a travel policy
    # mentioning “杭州” can be mixed into an office schedule answer.
    self_contained = requirement.kind in self_contained_kinds or _is_negative_fact_query(query) or (
        requirement.kind == "factual" and len(requirement.required_entities) >= 2
    )
    if not self_contained:
        return results
    if requirement.kind == "factual":
        return [
            result
            for result in results
            if _has_sufficient_structured_parent_evidence(
                query, _evidence_haystack([result])
            )
        ]
    return [
        result
        for result in results
        if _requirement_matches_haystack(requirement, _evidence_haystack([result]))
    ]


def _extend_with_supplementary_evidence(
    requirement: EvidenceRequirement,
    accepted: list[dict],
    candidates: list[dict],
) -> list[dict]:
    """Add entity context only after a self-contained relation was accepted.

    This is intended for retrieval/search responses, not the default answer
    context.  It cannot manufacture a relationship from split documents
    because an accepted core evidence unit is required first.
    """
    if requirement.kind != "relation" or not accepted:
        return accepted
    accepted_ids = {id(result) for result in accepted}
    return [
        result
        for result in candidates
        if id(result) in accepted_ids
        or any(
            _contains_evidence_term(_evidence_haystack([result]), entity)
            for entity in requirement.required_entities
        )
    ]


def _select_unverified_context_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Keep only partial context that is visibly related to the question.

    These candidates have already failed the full evidence gate, so they must
    never become citations.  They can still help the LLM explain what the
    uploaded material does and does not cover (for example, a character's
    occupation is present but their location is absent).  Requiring at least
    one stable query anchor keeps unrelated semantic false positives out of
    this weaker context as well.
    """
    if not candidates:
        return []
    requirement = _build_evidence_requirement(query)
    if requirement.kind == "collection_overview":
        return candidates

    # Prefer identity anchors whenever the question has them.  Type or
    # capability synonyms alone (for example “服务”) are too generic to make
    # an unrelated result safe even as weak context.
    anchors = list(requirement.required_entities)
    if not anchors:
        anchors = _unique_terms([
            *requirement.required_facets,
            *(term for group in requirement.alternative_fact_groups for term in group),
        ])
    if not anchors:
        return []
    return [
        candidate
        for candidate in candidates
        if any(_contains_evidence_term(_evidence_haystack([candidate]), anchor) for anchor in anchors)
    ]


def _select_support_grader_candidates(
    _query: str,
    candidates: list[dict],
    *,
    limit: int,
) -> list[dict]:
    """Return a bounded ranked set for the semantic support grader.

    Candidate selection has already applied knowledge-base/document scope,
    retrieval ranking, and request-size limits.  Reapplying question-word
    overlap here used to hide the very paraphrases the semantic grader exists
    to evaluate: for example, an ending written as “新的起点” could never be
    considered for a question containing “结局”.  The LLM receives only this
    small, untrusted candidate batch and must return a strict source-index
    verdict; it, rather than a hand-maintained vocabulary, decides support.
    """

    return list(candidates[:max(1, limit)])
