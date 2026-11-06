"""
自主决策 Agent、Function Calling 工具与强制状态机。

设计原则：
1. 模型可以决定何时查询、校验、建草稿或转人工；
2. 服务端通过签名 Token 和一次性消费记录强制执行工具依赖；
3. 系统提示词负责“告知”，服务端校验负责“阻止绕过”。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Sequence

import httpx

try:
    from .db import SQLiteDatabase
    from .rag_runtime import Citation, RagRuntime
    from .security import AgentTokenService
except ImportError:
    from employee_ticket_routing.core.db import SQLiteDatabase
    from employee_ticket_routing.core.rag_runtime import Citation, RagRuntime
    from employee_ticket_routing.core.security import AgentTokenService


class AgentToolError(Exception):
    """工具拒绝执行时返回给模型和 SSE 客户端的安全错误。"""

    def __init__(self, code: str, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class ModelToolCall:
    """模型返回的一次工具调用。"""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    """一次模型回合，要么产生文本，要么产生工具调用。"""

    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()
    raw_assistant_message: dict[str, Any] | None = None


@dataclass
class ToolExecution:
    """工具执行结果。"""

    name: str
    call_id: str
    model_payload: dict[str, Any]
    public_payload: dict[str, Any]
    citations: list[Citation] = field(default_factory=list)
    draft: dict[str, Any] | None = None


@dataclass
class AgentRunState:
    """
    单次 Agent 运行状态。

    该状态只存在于当前请求内；持久化事实写入 SQLite。签名 Token 绑定
    case_id 和用户身份，不能跨请求或跨员工复用。
    """

    profile_loaded: bool = False
    profile: dict[str, Any] | None = None
    evidence: dict[int, Citation] = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)
    policy_token: str | None = None
    consultation_token: str | None = None
    route_token: str | None = None
    route_data: dict[str, Any] | None = None
    draft: dict[str, Any] | None = None
    human_handoff: dict[str, Any] | None = None


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_profile",
            "description": (
                "获取当前登录员工最小必要的组织上下文。"
                "员工身份由服务端 JWT 决定，不能传入 employee_id。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "employment_type",
                                "region",
                                "policy_group",
                            ],
                        },
                    }
                },
                "required": ["include"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索薪酬福利、社保公积金、假期或入职调薪制度。返回带版本和有效期的证据。"
                "涉及政策事实时必须使用此工具。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "domain": {
                        "type": "string",
                        "enum": [
                            "salary_benefit",
                            "attendance",
                            "insurance",
                            "employment_change",
                            "other",
                        ],
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["query", "domain", "top_k"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_policy",
            "description": (
                "校验制度适用性并签发短期 policy validation token。"
                "必须引用 search_knowledge 返回的 evidence_id。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "question": {"type": "string", "minLength": 1},
                },
                "required": ["evidence_ids", "question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_consultation",
            "description": (
                "校验员工薪酬福利咨询工单是否具备提交条件。"
                "必须基于已通过的政策校验，检查咨询主题、具体诉求和适用信息；"
                "资料不足时返回缺失项，高风险事项返回人工复核标记。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_validation_token": {"type": "string"},
                    "topic": {
                        "type": "string",
                        "enum": [
                            "new_hire_compensation",
                            "salary_change",
                            "bonus",
                            "benefits",
                            "social_insurance",
                            "housing_fund",
                            "leave",
                            "other",
                        ],
                    },
                    "details": {"type": "string"},
                    "requested_effective_date": {
                        "type": "string"
                    },
                },
                "required": [
                    "policy_validation_token",
                    "topic",
                    "details",
                    "requested_effective_date",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_ticket",
            "description": (
                "基于已通过的制度校验和咨询资料校验，确定员工服务队列、"
                "优先级及是否需要人工处理。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "case_type": {
                        "type": "string",
                        "enum": [
                            "compensation",
                            "benefits",
                            "social_insurance",
                            "employment_change",
                            "other",
                        ],
                    },
                    "policy_validation_token": {"type": "string"},
                    "consultation_validation_token": {"type": "string"},
                    "risk_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "case_type",
                    "policy_validation_token",
                    "consultation_validation_token",
                    "risk_flags",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket_draft",
            "description": (
                "拟创建薪酬福利咨询工单草稿。Harness 会在真正执行前暂停，"
                "将标题和说明流式展示给员工确认；确认后才写入数据库。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "routing_token": {"type": "string"},
                    "title": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 1},
                    "attachment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "routing_token",
                    "title",
                    "description",
                    "attachment_ids",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_human_review",
            "description": (
                "资料不足、制度冲突、高风险或用户主动要求人工时创建人工交接。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason_code": {"type": "string"},
                    "reason": {"type": "string", "minLength": 1},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["reason_code", "reason", "evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "查询当前员工自己的工单状态。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    },
]


def canonical_hash(payload: dict[str, Any]) -> str:
    """生成稳定摘要，用于绑定校验时的输入内容。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def public_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    移除返回给前端的内部校验 Token。

    Token 只交给模型继续调用工具，不进入 SSE 展示和浏览器内存。
    """

    def scrub(value):
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if not key.endswith("_token")
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(payload)


class AgentToolExecutor:
    """
    所有 Agent 工具的受控执行器。

    输入参数由模型提供，但用户、案件、数据库连接和校验凭证均由服务端注入。
    """

    def __init__(
        self,
        *,
        database: SQLiteDatabase,
        runtime: RagRuntime,
        token_service: AgentTokenService,
        user: dict[str, Any],
        conversation_id: str,
        request_question: str = "",
    ):
        self.database = database
        self.runtime = runtime
        self.token_service = token_service
        self.user = user
        self.conversation_id = conversation_id
        self.request_question = request_question
        self.state = AgentRunState()

    def _success(
        self,
        *,
        call_id: str,
        name: str,
        data: dict[str, Any],
        citations: list[Citation] | None = None,
        draft: dict[str, Any] | None = None,
    ) -> ToolExecution:
        payload = {
            "ok": True,
            "code": "OK",
            "message": "工具调用成功。",
            "data": data,
        }
        return ToolExecution(
            name=name,
            call_id=call_id,
            model_payload=payload,
            public_payload=public_tool_payload(payload),
            citations=citations or [],
            draft=draft,
        )

    def _failure(
        self,
        *,
        call_id: str,
        name: str,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ToolExecution:
        payload = {
            "ok": False,
            "code": code,
            "message": message,
            "details": details or {},
        }
        return ToolExecution(
            name=name,
            call_id=call_id,
            model_payload=payload,
            public_payload=payload,
        )

    async def execute(
        self,
        *,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """路由到具体工具，并统一将异常转换为工具失败结果。"""

        try:
            if name == "get_employee_profile":
                return await self._get_employee_profile(call_id, arguments)
            if name == "search_knowledge":
                return await self._search_knowledge(call_id, arguments)
            if name == "validate_policy":
                return await self._validate_policy(call_id, arguments)
            if name == "validate_consultation":
                return await self._validate_consultation(call_id, arguments)
            if name == "route_ticket":
                return await self._route_ticket(call_id, arguments)
            if name == "create_ticket_draft":
                return await self._create_ticket_draft(call_id, arguments)
            if name == "request_human_review":
                return await self._request_human_review(call_id, arguments)
            if name == "get_ticket_status":
                return await self._get_ticket_status(call_id, arguments)
            return self._failure(
                call_id=call_id,
                name=name,
                code="UNKNOWN_TOOL",
                message=f"不存在的工具：{name}",
            )
        except AgentToolError as exc:
            return self._failure(
                call_id=call_id,
                name=name,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        except Exception as exc:
            return self._failure(
                call_id=call_id,
                name=name,
                code="TOOL_EXECUTION_FAILED",
                message="工具执行失败。",
                details={"safe_error": str(exc)[:300]},
            )

    async def _get_employee_profile(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """返回 JWT 对应用户的可信最小上下文。"""

        profile = await self.database.get_user(str(self.user["id"]))
        if profile is None or profile.get("status") != "active":
            raise AgentToolError("EMPLOYEE_NOT_FOUND", "未找到有效员工身份。")
        data = {
            "employee_ref": profile["id"],
            "employee_no": profile["employee_no"],
            "name": profile["name"],
            "employment_type": "employee",
            "region": os.getenv("DEFAULT_REGION", "CN"),
            "policy_group": "default",
            "active": True,
            "context_version": int(profile.get("token_version") or 1),
        }
        self.state.profile = data
        self.state.profile_loaded = True
        return self._success(call_id=call_id, name="get_employee_profile", data=data)

    async def _search_knowledge(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """调用现有混合检索，并保存证据供后续校验引用。"""

        query = str(arguments.get("query", "")).strip()
        if not self.state.profile_loaded:
            raise AgentToolError(
                "PROFILE_REQUIRED",
                "检索制度前必须先调用 get_employee_profile。",
            )
        if not query:
            raise AgentToolError("EMPTY_SEARCH_QUERY", "知识检索关键词不能为空。")

        hits, citations = await self.runtime.retrieve(query, self.database)
        evidence = []
        for hit, citation in zip(hits, citations):
            self.state.evidence[citation.chunk_id] = citation
            evidence.append(
                {
                    "evidence_id": str(citation.chunk_id),
                    "document_id": citation.document_id,
                    "file_name": citation.file_name,
                    "document_version": citation.document_version,
                    "chunk_id": citation.chunk_id,
                    "excerpt": citation.excerpt,
                    "score": round(hit.hybrid_score, 6),
                }
            )
        self.state.citations = citations
        return self._success(
            call_id=call_id,
            name="search_knowledge",
            data={
                "evidence": evidence,
                "has_evidence": bool(evidence),
                "query": query,
            },
            citations=citations,
        )

    async def _validate_policy(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """验证所选证据属于当前检索结果，并签发 policy Token。"""

        if not self.state.profile_loaded:
            raise AgentToolError(
                "PROFILE_REQUIRED",
                "制度校验前必须先获取员工上下文。",
            )
        evidence_ids = arguments.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise AgentToolError("EVIDENCE_REQUIRED", "制度校验必须引用有效证据。")
        selected = []
        for raw_id in evidence_ids:
            try:
                chunk_id = int(str(raw_id))
            except ValueError as exc:
                raise AgentToolError("INVALID_EVIDENCE", "证据编号格式不正确。") from exc
            citation = self.state.evidence.get(chunk_id)
            if citation is None:
                raise AgentToolError(
                    "EVIDENCE_NOT_FROM_SEARCH",
                    "校验证据必须来自本轮 search_knowledge 返回结果。",
                )
            selected.append(citation)
        if not selected:
            raise AgentToolError("EVIDENCE_REQUIRED", "没有可用于制度校验的证据。")

        payload = {
            "evidence_ids": [citation.chunk_id for citation in selected],
            "document_versions": [
                {
                    "document_id": citation.document_id,
                    "version": citation.document_version,
                }
                for citation in selected
            ],
            "question_hash": canonical_hash(
                {"question": str(arguments.get("question", ""))}
            ),
        }
        token = self.token_service.create(
            token_type="policy",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            payload=payload,
        )
        decoded = self.token_service.decode(
            token,
            expected_type="policy",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
        )
        self.state.policy_token = token
        return self._success(
            call_id=call_id,
            name="validate_policy",
            data={
                "passed": True,
                "validation_id": decoded["jti"],
                "validation_token": token,
                "evidence_ids": [str(citation.chunk_id) for citation in selected],
                "reason": "制度证据与当前员工上下文已完成一致性校验。",
            },
            citations=selected,
        )

    async def _validate_consultation(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """
        校验薪酬福利咨询工单的“可提交性”。

        该工具不判断员工最终能否获得某项待遇，只检查工单是否具备：
        1. 本轮有效的制度依据；
        2. 可执行的咨询主题；
        3. 足以让服务人员处理的具体诉求；
        4. 是否需要因劳动争议、个人例外等风险转人工。
        """

        policy_token = str(arguments.get("policy_validation_token") or "")
        try:
            policy_claims = self.token_service.decode(
                policy_token,
                expected_type="policy",
                user_id=str(self.user["id"]),
                case_id=self.conversation_id,
            )
        except ValueError as exc:
            raise AgentToolError(
                "POLICY_VALIDATION_REQUIRED",
                "咨询资料校验前必须先通过 validate_policy。",
            ) from exc

        topic = str(arguments.get("topic") or "").strip()
        details = str(arguments.get("details") or "").strip()
        effective_date = str(
            arguments.get("requested_effective_date") or ""
        ).strip()
        missing: list[str] = []
        checks: list[dict[str, Any]] = []

        if not topic:
            missing.append("咨询主题")
            checks.append({"name": "咨询主题", "passed": False})
        else:
            # “other” 仍可作为合法主题，但具体诉求必须足够清楚，避免
            # 把无法分流的空泛工单直接送入队列。
            checks.append({"name": "咨询主题", "passed": True})

        if len(details) < 6:
            missing.append("具体诉求")
            checks.append({"name": "具体诉求", "passed": False})
        else:
            checks.append({"name": "具体诉求", "passed": True})

        if effective_date:
            try:
                datetime.fromisoformat(effective_date)
                checks.append({"name": "期望生效日期", "passed": True})
            except ValueError:
                missing.append("期望生效日期")
                checks.append({"name": "期望生效日期", "passed": False})

        # 这些场景涉及个人权益、劳动争议或制度例外，不能由智能体直接承诺。
        risk_rules = {
            "salary_exception": ("调薪", "薪资调整", "例外", "个人特殊"),
            "labor_dispute": ("劳动仲裁", "投诉", "申诉", "争议"),
            "separation_compensation": ("离职补偿", "裁员", "解除劳动"),
            "work_injury": ("工伤", "医疗期"),
            "social_insurance_retroactive": ("社保补缴", "公积金补缴"),
        }
        combined_text = f"{self.request_question}\n{details}"
        risk_flags = sorted(
            {
                flag
                for flag, keywords in risk_rules.items()
                if any(keyword in combined_text for keyword in keywords)
            }
        )

        if missing:
            return self._success(
                call_id=call_id,
                name="validate_consultation",
                data={
                    "passed": False,
                    "validation_id": None,
                    "validation_token": None,
                    "checks": checks,
                    "missing_fields": missing,
                    "risk_flags": risk_flags,
                    "reason": "咨询资料不完整，暂不能创建工单草稿。",
                },
                citations=list(self.state.evidence.values()),
            )

        payload = {
            "policy_jti": policy_claims["jti"],
            "topic": topic,
            "details": details[:1000],
            "requested_effective_date": effective_date,
            "risk_flags": risk_flags,
        }
        token = self.token_service.create(
            token_type="consultation",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            payload=payload,
        )
        decoded = self.token_service.decode(
            token,
            expected_type="consultation",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
        )
        self.state.consultation_token = token
        return self._success(
            call_id=call_id,
            name="validate_consultation",
            data={
                "passed": True,
                "validation_id": decoded["jti"],
                "validation_token": token,
                "checks": checks,
                "missing_fields": [],
                "risk_flags": risk_flags,
                "reason": "咨询主题、具体诉求和制度适用性已完成校验。",
            },
            citations=list(self.state.evidence.values()),
        )

    async def _route_ticket(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """
        依据制度与咨询资料双校验结果签发 routing Token。

        routing Token 是本轮唯一可以进入 create_ticket_draft 的凭证。
        policy 和 consultation 凭证在路由时立即消费，防止跨案件重放。
        """

        case_type = str(arguments.get("case_type") or "other")
        policy_token = str(arguments.get("policy_validation_token") or "")
        consultation_token = str(
            arguments.get("consultation_validation_token") or ""
        )
        risk_flags = arguments.get("risk_flags")
        if not isinstance(risk_flags, list):
            risk_flags = []
        risk_flags = [str(item) for item in risk_flags]

        try:
            policy_claims = self.token_service.decode(
                policy_token,
                expected_type="policy",
                user_id=str(self.user["id"]),
                case_id=self.conversation_id,
            )
        except ValueError as exc:
            raise AgentToolError(
                "POLICY_VALIDATION_REQUIRED",
                "路由前必须提供有效的制度校验凭证。",
            ) from exc

        try:
            consultation_claims = self.token_service.decode(
                consultation_token,
                expected_type="consultation",
                user_id=str(self.user["id"]),
                case_id=self.conversation_id,
            )
        except ValueError as exc:
            raise AgentToolError(
                "CONSULTATION_VALIDATION_REQUIRED",
                "路由前必须通过 validate_consultation 补齐咨询事实。",
            ) from exc

        consultation_payload = consultation_claims.get("payload", {})
        if str(consultation_payload.get("policy_jti") or "") != str(
            policy_claims["jti"]
        ):
            raise AgentToolError(
                "CONSULTATION_POLICY_MISMATCH",
                "咨询资料校验与当前制度依据不属于同一轮处理。",
            )

        consultation_risk = consultation_payload.get("risk_flags", [])
        if not isinstance(consultation_risk, list):
            consultation_risk = []
        all_risk = sorted(set(risk_flags + [str(item) for item in consultation_risk]))
        requires_human = bool(all_risk)
        queue_map = {
            "compensation": "compensation_benefits",
            "benefits": "compensation_benefits",
            "social_insurance": "social_insurance",
            "employment_change": "employee_services",
            "other": "employee_services",
        }
        queue = queue_map.get(case_type, "employee_services")
        priority = "high" if requires_human else "normal"

        # 校验 Token 在进入路由后立即消费，不能用于另一案件或第二次路由。
        if not await self.database.consume_tool_token(
            jti=str(policy_claims["jti"]),
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            token_type="policy",
            purpose="route_ticket",
        ):
            raise AgentToolError("TOKEN_REPLAYED", "制度校验凭证已被使用。")
        if not await self.database.consume_tool_token(
            jti=str(consultation_claims["jti"]),
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            token_type="consultation",
            purpose="route_ticket",
        ):
            raise AgentToolError("TOKEN_REPLAYED", "咨询资料校验凭证已被使用。")

        route_payload = {
            "case_type": case_type,
            "policy_jti": policy_claims["jti"],
            "consultation_jti": consultation_claims["jti"],
            "topic": consultation_payload.get("topic"),
            "details": consultation_payload.get("details"),
            "queue": queue,
            "priority": priority,
            "requires_human": requires_human,
            "risk_flags": all_risk,
        }
        token = self.token_service.create(
            token_type="routing",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            payload=route_payload,
        )
        decoded = self.token_service.decode(
            token,
            expected_type="routing",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
        )
        self.state.route_token = token
        self.state.route_data = route_payload
        return self._success(
            call_id=call_id,
            name="route_ticket",
            data={
                "routing_id": decoded["jti"],
                "routing_token": token,
                **route_payload,
            },
            citations=list(self.state.evidence.values()),
        )

    def prepare_ticket_draft(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        对 create_ticket_draft 做“零副作用预检查”。

        Harness 在真正执行工具前调用本方法：
        - 验证 routing Token 和员工归属；
        - 阻止高风险工单进入普通建单；
        - 生成员工可阅读的安全预览；
        - 不消费 Token、不写数据库。
        """

        routing_token = str(arguments.get("routing_token") or "")
        try:
            claims = self.token_service.decode(
                routing_token,
                expected_type="routing",
                user_id=str(self.user["id"]),
                case_id=self.conversation_id,
            )
        except ValueError as exc:
            raise AgentToolError(
                "ROUTING_REQUIRED",
                "创建工单草稿前必须完成路由校验。",
            ) from exc

        payload = claims.get("payload", {})
        if not payload.get("policy_jti") or not payload.get("consultation_jti"):
            raise AgentToolError(
                "CONSULTATION_VALIDATION_REQUIRED",
                "路由凭证缺少制度或咨询资料校验，不能创建工单。",
            )
        if payload.get("requires_human"):
            raise AgentToolError(
                "HUMAN_REVIEW_REQUIRED",
                "当前事项存在高风险，只能转人工处理，不能创建普通工单。",
            )
        title = str(arguments.get("title") or "").strip()
        description = str(arguments.get("description") or "").strip()
        if not title or not description:
            raise AgentToolError(
                "DRAFT_CONTENT_REQUIRED",
                "工单标题和说明不能为空。",
            )
        attachment_ids = [
            str(item)
            for item in arguments.get("attachment_ids", [])
            if str(item).strip()
        ]
        return {
            "call_id": call_id,
            "title": title,
            "description": description,
            "category": str(payload.get("case_type") or "other"),
            "topic": payload.get("topic"),
            "queue": payload.get("queue"),
            "priority": payload.get("priority"),
            "attachment_ids": attachment_ids,
        }

    async def _create_ticket_draft(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """
        创建薪酬福利咨询工单草稿。

        本方法只会在员工确认后由确认接口执行。routing Token 同时绑定
        policy 与 consultation 校验，缺少任一前置步骤都无法取得。
        """

        routing_token = str(arguments.get("routing_token") or "")
        try:
            claims = self.token_service.decode(
                routing_token,
                expected_type="routing",
                user_id=str(self.user["id"]),
                case_id=self.conversation_id,
            )
        except ValueError as exc:
            raise AgentToolError(
                "ROUTING_REQUIRED",
                "创建工单草稿前必须完成路由校验。",
            ) from exc

        payload = claims.get("payload", {})
        if not payload.get("policy_jti") or not payload.get("consultation_jti"):
            raise AgentToolError(
                "CONSULTATION_VALIDATION_REQUIRED",
                "创建工单前必须同时通过制度校验和咨询资料校验。",
            )
        if payload.get("requires_human"):
            raise AgentToolError(
                "HUMAN_REVIEW_REQUIRED",
                "当前工单存在高风险，必须转人工处理，不能自动创建草稿。",
            )
        title = str(arguments.get("title") or "").strip()
        description = str(arguments.get("description") or "").strip()
        if not title or not description:
            raise AgentToolError("DRAFT_CONTENT_REQUIRED", "工单标题和描述不能为空。")
        if not await self.database.consume_tool_token(
            jti=str(claims["jti"]),
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            token_type="routing",
            purpose="create_ticket_draft",
        ):
            raise AgentToolError("TOKEN_REPLAYED", "路由凭证已被使用。")
        attachment_ids = [
            str(item)
            for item in arguments.get("attachment_ids", [])
            if str(item).strip()
        ]
        validation_summary = {
            "policy_jti": payload.get("policy_jti"),
            "consultation_jti": payload.get("consultation_jti"),
            "routing_jti": str(claims["jti"]),
            "queue": payload.get("queue"),
            "priority": payload.get("priority"),
            "validation_digest": canonical_hash(payload),
        }
        draft = await self.database.create_ticket_draft(
            user_id=str(self.user["id"]),
            conversation_id=self.conversation_id,
            title=title,
            summary=description[:500],
            details=description,
            category=str(payload.get("case_type") or "other"),
            priority=str(payload.get("priority") or "normal"),
            fields={
                "queue": payload.get("queue"),
                "attachment_ids": attachment_ids,
            },
            validation=validation_summary,
            expires_at=(
                datetime.now(timezone.utc) + timedelta(minutes=15)
            ).replace(microsecond=0).isoformat(),
        )
        # 校验摘要供确认接口再次核对，不把内部凭证交给浏览器。
        self.state.draft = draft
        return self._success(
            call_id=call_id,
            name="create_ticket_draft",
            data={
                **draft,
                "requires_user_confirmation": True,
                "validation_digest": validation_summary["validation_digest"],
            },
            citations=list(self.state.evidence.values()),
            draft={
                "draft_id": draft["draft_id"],
                "draft_no": draft["draft_no"],
                "draft_version": draft["draft_version"],
                "status": draft["status"],
                "title": title,
                "description": description,
                "topic": payload.get("topic"),
                "priority": payload.get("priority"),
                "queue": payload.get("queue"),
                "expires_at": draft["expires_at"],
            },
        )

    async def _request_human_review(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """创建人工交接草稿；这是允许在校验失败时使用的安全出口。"""

        if not self.state.profile_loaded:
            raise AgentToolError(
                "PROFILE_REQUIRED",
                "转人工前必须先获取员工上下文。",
            )
        reason = str(arguments.get("reason") or "").strip()
        if not reason:
            raise AgentToolError("REVIEW_REASON_REQUIRED", "转人工原因不能为空。")
        evidence_ids = arguments.get("evidence_ids")
        evidence_text = ""
        if isinstance(evidence_ids, list):
            evidence_text = "、".join(str(item) for item in evidence_ids)
        draft = await self.database.create_ticket_draft(
            user_id=str(self.user["id"]),
            conversation_id=self.conversation_id,
            title="人工复核：薪酬福利咨询",
            summary=reason[:500],
            details=(
                f"转人工原因：{reason}\n"
                f"关联证据：{evidence_text or '无'}"
            ),
            category="human_review",
            priority="high",
            fields={
                "reason_code": str(arguments.get("reason_code") or "MANUAL_REVIEW"),
                "evidence_ids": evidence_ids or [],
            },
            validation={"manual_review": True},
            expires_at=(
                datetime.now(timezone.utc) + timedelta(minutes=15)
            ).replace(microsecond=0).isoformat(),
        )
        self.state.human_handoff = draft
        return self._success(
            call_id=call_id,
            name="request_human_review",
            data={
                "handoff_id": draft["draft_id"],
                "status": "waiting_human",
                "reason": reason,
            },
            citations=list(self.state.evidence.values()),
        )

    async def _get_ticket_status(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """只允许当前员工查询自己的工单。"""

        if not self.state.profile_loaded:
            raise AgentToolError(
                "PROFILE_REQUIRED",
                "查询工单前必须先获取员工上下文。",
            )
        ticket_id = str(arguments.get("ticket_id") or "").strip()
        if not ticket_id:
            raise AgentToolError("TICKET_ID_REQUIRED", "工单编号不能为空。")
        ticket = await self.database.get_ticket_status(
            ticket_id=ticket_id,
            user_id=str(self.user["id"]),
        )
        if ticket is None:
            raise AgentToolError("TICKET_NOT_FOUND", "未找到该员工名下的工单。")
        return self._success(
            call_id=call_id,
            name="get_ticket_status",
            data=ticket,
        )


class AgentModelClient:
    """模型工具调用客户端统一接口，具体供应商可替换。"""

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelTurn:
        raise NotImplementedError


class OpenAICompatibleToolClient:
    """
    OpenAI 兼容的 Function Calling 适配器。

    工具决策使用非流式响应，确保工具名和 JSON 参数完整、可校验；Agent
    完成工具循环并得到最终回答后，统一按 SSE delta 切片，因此前端仍显示
    打字机效果。
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelTurn:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            # 允许多工具规划；Harness 仍按依赖顺序串行执行，不并发。
            "parallel_tool_calls": True,
            "temperature": 0.1,
        }
        timeout = httpx.Timeout(
            connect=min(10.0, self.timeout),
            read=self.timeout,
            write=30.0,
            pool=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"模型接口返回 HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        result = response.json()
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("模型接口未返回 choices。")
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("模型返回的消息结构无效。")

        tool_calls: list[ModelToolCall] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function", {})
            if not isinstance(function, dict):
                continue
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError("模型返回了无法解析的工具参数。") from exc
            if not isinstance(arguments, dict):
                raise RuntimeError("模型工具参数必须是 JSON 对象。")
            tool_calls.append(
                ModelToolCall(
                    call_id=str(raw_call.get("id") or f"call_{index}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )

        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        if not isinstance(content, str):
            content = str(content)
        return ModelTurn(
            content=content.strip(),
            tool_calls=tuple(tool_calls),
            raw_assistant_message=message,
        )


def _latest_tool_result(
    messages: Sequence[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    """读取最近一次工具结果，供离线 Mock 根据真实执行结果推进。"""

    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        return str(payload.get("tool_name") or ""), payload.get("result", {})
    return None, None


def _extract_question(messages: Sequence[dict[str, Any]]) -> str:
    """读取当前轮问题；历史回答不参与 Mock 的当前意图判断。"""

    for message in reversed(messages):
        if message.get("role") == "user":
            content = str(message.get("content") or "")
            match = re.search(r"【当前问题】\s*(.+)$", content, flags=re.S)
            return match.group(1).strip() if match else content.strip()
    return ""


def _find_tool_result(
    messages: Sequence[dict[str, Any]],
    tool_name: str,
) -> dict[str, Any]:
    """从消息历史中找到指定工具的最近结果。"""

    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        if payload.get("tool_name") == tool_name:
            result = payload.get("result", {})
            return result if isinstance(result, dict) else {}
    return {}


class MockToolCallingClient:
    """
    离线 Function Calling 模拟器。

    它不预设完整流程，而是根据上一轮真实工具结果选择下一步，用于在
    没有 API Key 时验证工具依赖、Token 签发、高风险转人工和草稿确认。
    """

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelTurn:
        await asyncio.sleep(0.03)
        question = _extract_question(messages)
        latest_name, latest_result = _latest_tool_result(messages)
        latest_result = latest_result if isinstance(latest_result, dict) else {}

        def call(name: str, arguments: dict[str, Any]) -> ModelTurn:
            return ModelTurn(
                content="",
                tool_calls=(
                    ModelToolCall(
                        call_id=f"mock_{name}_{len(messages)}",
                        name=name,
                        arguments=arguments,
                    ),
                ),
            )

        def infer_domain() -> str:
            if any(word in question for word in ("社保", "公积金", "医保", "养老")):
                return "insurance"
            if any(word in question for word in ("假期", "考勤", "请假")):
                return "attendance"
            if any(word in question for word in ("入职", "转正", "调岗", "离职")):
                return "employment_change"
            return "salary_benefit"

        def infer_topic() -> str:
            if "入职" in question:
                return "new_hire_compensation"
            if any(word in question for word in ("调薪", "涨薪", "薪资调整")):
                return "salary_change"
            if any(word in question for word in ("奖金", "绩效")):
                return "bonus"
            if "公积金" in question:
                return "housing_fund"
            if any(word in question for word in ("社保", "医保", "养老")):
                return "social_insurance"
            if any(word in question for word in ("假期", "请假", "年假")):
                return "leave"
            if any(word in question for word in ("福利", "补贴", "体检", "年金")):
                return "benefits"
            return "other"

        def infer_case_type() -> str:
            topic = infer_topic()
            if topic in {"social_insurance", "housing_fund", "leave"}:
                return "social_insurance"
            if topic in {"benefits", "bonus"}:
                return "benefits"
            if topic in {"new_hire_compensation", "salary_change"}:
                return "compensation"
            return "other"

        if latest_name is None:
            # 第一轮同时规划“身份上下文 + 制度检索”。Harness 会按安全
            # 顺序执行，因此这是两个顺序工具调用，不是并行读取。
            return ModelTurn(
                content="",
                tool_calls=(
                    ModelToolCall(
                        call_id=f"mock_profile_{len(messages)}",
                        name="get_employee_profile",
                        arguments={
                            "include": [
                                "employment_type",
                                "region",
                                "policy_group",
                            ]
                        },
                    ),
                    ModelToolCall(
                        call_id=f"mock_search_{len(messages)}",
                        name="search_knowledge",
                        arguments={
                            "query": question or "薪酬福利制度",
                            "domain": infer_domain(),
                            "top_k": 3,
                        },
                    ),
                ),
            )

        data = latest_result.get("data", {})
        data = data if isinstance(data, dict) else {}
        if latest_name == "search_knowledge":
            evidence = data.get("evidence", [])
            if not isinstance(evidence, list) or not evidence:
                return ModelTurn(
                    content=(
                        "制度中未找到足够依据，建议补充制度文件，"
                        "或明确选择转人工确认。"
                    )
                )
            return call(
                "validate_policy",
                {
                    "evidence_ids": [
                        str(item["evidence_id"])
                        for item in evidence[:3]
                        if isinstance(item, dict) and item.get("evidence_id")
                    ],
                    "question": question,
                },
            )

        if latest_name == "validate_policy":
            if not data.get("passed"):
                return ModelTurn(
                    content="制度校验未通过，需要补充资料或转人工确认。"
                )

            # 只有明确的“执行型”表达才进入建单链路；“能不能提交”
            # 属于制度咨询，不能误解为员工已经授权创建工单。
            action_phrases = (
                "帮我提交",
                "请帮我提交",
                "帮我建单",
                "请创建工单",
                "我要提交",
                "发起工单",
                "请提交工单",
            )
            wants_ticket = any(phrase in question for phrase in action_phrases)
            if not wants_ticket:
                search_result = _find_tool_result(messages, "search_knowledge")
                search_data = search_result.get("data", {})
                evidence_lines = []
                for index, item in enumerate(
                    search_data.get("evidence", [])[:2],
                    start=1,
                ):
                    if not isinstance(item, dict):
                        continue
                    evidence_lines.append(
                        f"【资料{index}】{str(item.get('excerpt', ''))[:220]}"
                    )
                return ModelTurn(
                    content=(
                        f"根据当前有效制度，关于“{question}”可确认：\n"
                        + "\n".join(evidence_lines)
                        + "\n以上信息仅供参考；涉及个人待遇或例外情况请转人工确认。"
                    )
                )

            # “帮我提交我的咨询工单”本身没有说明要咨询什么，必须让员工
            # 补充具体诉求，不能仅凭一句建单指令生成空泛工单。
            stripped = question
            for phrase in action_phrases:
                stripped = stripped.replace(phrase, "")
            stripped = re.sub(
                r"(我的)?(咨询)?工单|请|帮我|提交|创建|发起|[:：,，。\s]",
                "",
                stripped,
            )
            has_specific_need = len(stripped) >= 8 and any(
                word in question
                for word in (
                    "入职",
                    "薪酬",
                    "工资",
                    "调薪",
                    "奖金",
                    "福利",
                    "保险",
                    "补贴",
                    "体检",
                    "年金",
                    "社保",
                    "公积金",
                    "医保",
                    "年假",
                    "假期",
                    "考勤",
                )
            )
            if not has_specific_need:
                return ModelTurn(
                    content=(
                        "可以。为了准确分流，请补充：\n"
                        "1. 要咨询的薪酬或福利事项；\n"
                        "2. 你的具体诉求或希望确认的结果；\n"
                        "3. 与事项相关的日期（如入职日、调薪生效日，可不填）。"
                    )
                )

            date_match = re.search(r"20\d{2}-\d{2}-\d{2}", question)
            return call(
                "validate_consultation",
                {
                    "policy_validation_token": str(
                        data.get("validation_token") or ""
                    ),
                    "topic": infer_topic(),
                    "details": stripped,
                    "requested_effective_date": (
                        date_match.group(0) if date_match else ""
                    ),
                },
            )

        if latest_name == "validate_consultation":
            if not data.get("passed"):
                missing = "、".join(
                    str(item) for item in data.get("missing_fields", [])
                )
                return ModelTurn(
                    content=(
                        "咨询信息尚不完整，暂时不能创建工单。"
                        f"请补充：{missing or '具体诉求'}。"
                    )
                )
            policy_result = _find_tool_result(messages, "validate_policy")
            policy_data = policy_result.get("data", {})
            return call(
                "route_ticket",
                {
                    "case_type": infer_case_type(),
                    "policy_validation_token": str(
                        policy_data.get("validation_token") or ""
                    ),
                    "consultation_validation_token": str(
                        data.get("validation_token") or ""
                    ),
                    "risk_flags": data.get("risk_flags", []),
                },
            )

        if latest_name == "route_ticket":
            if data.get("requires_human"):
                return call(
                    "request_human_review",
                    {
                        "reason_code": "HIGH_RISK",
                        "reason": "该事项涉及个人权益、劳动争议或制度例外，需要人工复核。",
                        "evidence_ids": [],
                    },
                )
            return call(
                "create_ticket_draft",
                {
                    "routing_token": str(data.get("routing_token") or ""),
                    "title": question[:80] or "员工薪酬福利咨询",
                    "description": (
                        f"员工咨询：{question}\n"
                        "系统已完成制度、咨询资料和分流规则校验。"
                    ),
                    "attachment_ids": [],
                },
            )

        if latest_name == "create_ticket_draft":
            return ModelTurn(
                content=(
                    "工单内容已准备完成，正在等待员工确认。"
                )
            )

        if latest_name == "request_human_review":
            return ModelTurn(
                content=(
                    "该问题需要人工专业判断，已进入人工复核队列。"
                    "在服务人员处理前，系统不会自动提交普通工单。"
                )
            )

        if latest_name == "get_ticket_status":
            return ModelTurn(
                content=f"当前工单状态：{data.get('status', '未知')}。"
            )

        return ModelTurn(content="已完成当前检查。请补充更具体的问题。")


@dataclass(frozen=True)
class AgentStreamEvent:
    """Agent 运行过程中产生的 SSE 业务事件。"""

    event: str
    data: dict[str, Any]


RUNTIME_SAFETY_PROMPT = """
【Agent 运行时安全规则】
1. 每轮薪酬福利业务咨询必须先调用 get_employee_profile，身份只来自登录会话。
2. 涉及薪酬、福利、社保、公积金、假期或入职调薪事实时，必须调用
   search_knowledge，并使用本轮 evidence_id 调用 validate_policy。
3. 只有员工明确说“帮我提交”“帮我建单”等执行语句时，才进入工单链路；
   “能不能提交”“流程是什么”等问法只解释制度，不创建工单。
4. 创建工单前必须依次通过 validate_policy、validate_consultation 和
   route_ticket。资料不足时先向员工索取缺失信息，不得生成空泛工单。
5. 高风险、劳动争议、个人例外、制度冲突或用户要求人工时，调用
   request_human_review，不得强行创建普通工单。
6. Harness 拦截 create_ticket_draft：员工确认前工具不得执行。模型不得
   声称工单已创建或已提交，也不得把模糊回复视为确认。
7. 不得伪造、复用或跨会话使用任何 validation_token。
8. 工具失败时如实说明，不得声称未实际成功的动作已经完成。
9. 最终回答只使用本轮制度证据；没有证据时明确说明“制度中未找到”，
   不得编造薪酬、奖金、社保、公积金或其他福利标准。
10. 为满足最多 5 轮限制，第一轮应同时规划 get_employee_profile 和
    search_knowledge；Harness 会按安全顺序执行。
"""


@dataclass
class AgentPendingConfirmation:
    """等待员工确认的建单动作。"""

    confirmation_id: str
    user_id: str
    conversation_id: str
    request_id: str
    question: str
    round_no: int
    arguments: dict[str, Any]
    preview: dict[str, Any]
    expires_at: datetime


class PendingConfirmationStore:
    """
    建草稿前的人工确认中断点。

    当前原型为单进程内存存储，适合最小验证；正式多实例部署时应替换为
    Redis，并使用服务端加密保存 routing_token。浏览器永远拿不到 Token。
    """

    def __init__(self, *, ttl_seconds: int = 600):
        self._ttl_seconds = ttl_seconds
        self._items: dict[str, AgentPendingConfirmation] = {}
        self._lock = asyncio.Lock()

    def _cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            key
            for key, pending in self._items.items()
            if pending.expires_at <= now
        ]
        for key in expired:
            self._items.pop(key, None)

    async def put(
        self,
        *,
        user_id: str,
        conversation_id: str,
        request_id: str,
        question: str,
        round_no: int,
        arguments: dict[str, Any],
        preview: dict[str, Any],
    ) -> AgentPendingConfirmation:
        async with self._lock:
            self._cleanup()
            confirmation_id = str(uuid.uuid4())
            pending = AgentPendingConfirmation(
                confirmation_id=confirmation_id,
                user_id=user_id,
                conversation_id=conversation_id,
                request_id=request_id,
                question=question,
                round_no=round_no,
                arguments=dict(arguments),
                preview=dict(preview),
                expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=self._ttl_seconds),
            )
            self._items[confirmation_id] = pending
            return pending

    async def take(
        self,
        *,
        confirmation_id: str,
        user_id: str,
    ) -> AgentPendingConfirmation | None:
        """
        一次性取出待确认动作。

        先移除再执行，防止双击确认或并发请求重复消费同一个 routing Token。
        """

        async with self._lock:
            self._cleanup()
            pending = self._items.get(confirmation_id)
            if pending is None or pending.user_id != user_id:
                return None
            return self._items.pop(confirmation_id)


class AgentHarness:
    """
    轻量、供应商无关的 Agent Harness。

    选择 Harness 而非把状态机拆进通用 Agent SDK，是因为本系统有强顺序、
    一次性凭证、SSE 中断确认和中文审计需求。Harness 负责四步循环：
    模型决策 → 安全排序并执行工具 → 结果回填 → 再决策。
    """

    def __init__(
        self,
        *,
        database: SQLiteDatabase,
        runtime: RagRuntime,
        token_service: AgentTokenService,
        user: dict[str, Any],
        conversation_id: str,
        system_prompt: str,
        question: str,
        pending_store: PendingConfirmationStore,
    ):
        self.database = database
        self.runtime = runtime
        self.executor = AgentToolExecutor(
            database=database,
            runtime=runtime,
            token_service=token_service,
            user=user,
            conversation_id=conversation_id,
            request_question=question,
        )
        self.system_prompt = system_prompt
        self.question = question
        self.pending_store = pending_store
        # 业务硬限制最多 5 轮；环境变量只能调小，不能突破。
        self.max_rounds = min(
            max(int(os.getenv("AGENT_MAX_ROUNDS", "5")), 1),
            5,
        )

    def _model_client(self) -> AgentModelClient:
        """按配置创建模型适配器；离线模式使用可重复验证的 Mock。"""

        settings = self.runtime.settings
        if settings.llm_provider == "mock":
            return MockToolCallingClient()
        if settings.llm_provider == "openai_compatible":
            if not settings.llm_api_url:
                raise RuntimeError("缺少 LLM_API_URL。")
            if not settings.llm_api_key:
                raise RuntimeError("缺少 LLM_API_KEY。")
            if not settings.llm_model:
                raise RuntimeError("缺少 LLM_MODEL。")
            return OpenAICompatibleToolClient(
                api_url=settings.llm_api_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout=settings.request_timeout,
            )
        raise RuntimeError(f"不支持的 LLM_PROVIDER: {settings.llm_provider}")

    async def run(
        self,
        *,
        question: str,
        history: Sequence[dict[str, str]],
        request_id: str,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        运行工具调用循环并产出 SSE 业务事件。

        每一轮都会：
        1. 请求模型产生文本或工具调用；
        2. 按服务端安全优先级执行工具；
        3. 将脱敏结果写入 agent_decision_logs；
        4. 把工具结果回填给模型继续决策。

        遇到 create_ticket_draft 时立即中断循环，只流式发送预览。员工确认
        后由独立确认接口执行工具，模型无法替员工越过该确认点。
        """

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt + "\n\n" + RUNTIME_SAFETY_PROMPT,
            },
            *history,
            {
                "role": "user",
                "content": f"【当前问题】\n{question}",
            },
        ]
        client = self._model_client()
        final_answer = ""

        tool_order = {
            "get_employee_profile": 10,
            "search_knowledge": 20,
            "validate_policy": 30,
            "validate_consultation": 40,
            "route_ticket": 50,
            "request_human_review": 60,
            "get_ticket_status": 60,
            "create_ticket_draft": 70,
        }

        for round_no in range(1, self.max_rounds + 1):
            turn = await client.complete(messages, TOOL_DEFINITIONS)
            if turn.tool_calls:
                raw_assistant = turn.raw_assistant_message or {
                    "role": "assistant",
                    "content": turn.content or None,
                    "tool_calls": [
                        {
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(
                                    tool_call.arguments,
                                    ensure_ascii=False,
                                ),
                            },
                        }
                        for tool_call in turn.tool_calls
                    ],
                }
                messages.append(raw_assistant)

                # 模型可以在一轮返回多个工具。Harness 按依赖顺序重排后
                # 顺序执行，绝不能把“先 identity、后搜索”交给并发调度。
                ordered_calls = sorted(
                    enumerate(turn.tool_calls),
                    key=lambda item: (
                        tool_order.get(item[1].name, 999),
                        item[0],
                    ),
                )
                for _, tool_call in ordered_calls:
                    yield AgentStreamEvent(
                        "status",
                        {
                            "stage": "tool",
                            "message": f"正在执行 {tool_call.name}…",
                            "tool": tool_call.name,
                        },
                    )

                    if tool_call.name == "create_ticket_draft":
                        # 强制人工中断点：只做零副作用预检查，不执行工具。
                        started_at = time.perf_counter()
                        try:
                            preview = self.executor.prepare_ticket_draft(
                                call_id=tool_call.call_id,
                                arguments=tool_call.arguments,
                            )
                        except AgentToolError as exc:
                            result = self.executor._failure(
                                call_id=tool_call.call_id,
                                name=tool_call.name,
                                code=exc.code,
                                message=exc.message,
                                details=exc.details,
                            )
                            duration_ms = int(
                                (time.perf_counter() - started_at) * 1000
                            )
                            await self.database.write_agent_decision_log(
                                conversation_id=self.executor.conversation_id,
                                user_id=str(self.executor.user["id"]),
                                request_id=request_id,
                                round_no=round_no,
                                decision_type="confirmation_precheck_failed",
                                tool_name=tool_call.name,
                                arguments=public_tool_payload(
                                    tool_call.arguments
                                ),
                                duration_ms=duration_ms,
                                result_summary=str(exc.message),
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.call_id,
                                    "content": json.dumps(
                                        {
                                            "tool_name": tool_call.name,
                                            "result": result.model_payload,
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            yield AgentStreamEvent(
                                "tool",
                                {
                                    "name": tool_call.name,
                                    "ok": False,
                                    "code": result.public_payload.get("code"),
                                    "message": str(
                                        result.public_payload.get("message") or ""
                                    ),
                                },
                            )
                            continue
                        else:
                            pending = await self.pending_store.put(
                                user_id=str(self.executor.user["id"]),
                                conversation_id=self.executor.conversation_id,
                                request_id=request_id,
                                question=question,
                                round_no=round_no,
                                arguments=tool_call.arguments,
                                preview=preview,
                            )
                            duration_ms = int(
                                (time.perf_counter() - started_at) * 1000
                            )
                            await self.database.write_agent_decision_log(
                                conversation_id=self.executor.conversation_id,
                                user_id=str(self.executor.user["id"]),
                                request_id=request_id,
                                round_no=round_no,
                                decision_type="confirmation_required",
                                tool_name=tool_call.name,
                                arguments=public_tool_payload(
                                    tool_call.arguments
                                ),
                                duration_ms=duration_ms,
                                result_summary=(
                                    "已暂停工具执行，等待员工确认："
                                    f"{preview['title']}"
                                ),
                            )
                            expires_at = pending.expires_at.replace(
                                microsecond=0
                            ).isoformat()
                            preview_text = (
                                "拟创建薪酬福利咨询工单\n"
                                f"标题：{preview['title']}\n"
                                f"业务方向：{preview.get('topic') or preview['category']}\n"
                                f"处理队列：{preview.get('queue') or '员工服务'}\n"
                                f"优先级：{preview.get('priority') or 'normal'}\n"
                                f"咨询说明：\n{preview['description']}\n"
                                "请核对以上内容。确认后系统才会真正执行建单并提交。"
                            )
                            for index in range(0, len(preview_text), 20):
                                yield AgentStreamEvent(
                                    "delta",
                                    {"text": preview_text[index : index + 20]},
                                )
                                await asyncio.sleep(0.01)
                            yield AgentStreamEvent(
                                "confirmation_required",
                                {
                                    "confirmation_id": pending.confirmation_id,
                                    "title": preview["title"],
                                    "description": preview["description"],
                                    "category": preview["category"],
                                    "topic": preview.get("topic"),
                                    "queue": preview.get("queue"),
                                    "priority": preview.get("priority"),
                                    "expires_at": expires_at,
                                    "requires_user_confirmation": True,
                                },
                            )
                            if self.executor.state.citations:
                                yield AgentStreamEvent(
                                    "sources",
                                    {
                                        "sources": [
                                            citation.to_dict()
                                            for citation in self.executor.state.citations
                                        ]
                                    },
                                )
                            return

                    started_at = time.perf_counter()
                    result = await self.executor.execute(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                    duration_ms = int(
                        (time.perf_counter() - started_at) * 1000
                    )
                    await self.database.write_agent_decision_log(
                        conversation_id=self.executor.conversation_id,
                        user_id=str(self.executor.user["id"]),
                        request_id=request_id,
                        round_no=round_no,
                        decision_type="tool_call",
                        tool_name=tool_call.name,
                        arguments=public_tool_payload(tool_call.arguments),
                        duration_ms=duration_ms,
                        result_summary=json.dumps(
                            public_tool_payload(result.public_payload),
                            ensure_ascii=False,
                        )[:900],
                    )
                    await self.database.write_audit_log(
                        actor_id=str(self.executor.user["id"]),
                        actor_role=str(self.executor.user["role"]),
                        action="agent_tool_call",
                        resource_type="conversation",
                        resource_id=self.executor.conversation_id,
                        request_id=request_id,
                        after_data={
                            "tool": tool_call.name,
                            # 工具签名凭证不进入审计日志，只保留非敏感参数。
                            "arguments": public_tool_payload(tool_call.arguments),
                            "result_code": result.model_payload.get("code"),
                            "ok": result.model_payload.get("ok"),
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.call_id,
                            "content": json.dumps(
                                {
                                    "tool_name": tool_call.name,
                                    "result": result.model_payload,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    yield AgentStreamEvent(
                        "tool",
                        {
                            "name": tool_call.name,
                            "ok": bool(result.public_payload.get("ok")),
                            "code": result.public_payload.get("code"),
                            "message": str(
                                result.public_payload.get("message") or ""
                            ),
                        },
                    )
                    if (
                        tool_call.name == "request_human_review"
                        and result.public_payload.get("ok")
                    ):
                        # 转人工是当前问题的终止动作，不需要再消耗第 6 轮
                        # 让模型重复表述。Harness 直接产出确定性的收尾文本。
                        terminal_text = (
                            "该事项涉及个人权益、劳动争议或制度例外，"
                            "已转给薪酬福利专员人工复核。"
                            "服务人员会在核实后继续处理，系统不会自动承诺结果。"
                        )
                        await self.database.write_agent_decision_log(
                            conversation_id=self.executor.conversation_id,
                            user_id=str(self.executor.user["id"]),
                            request_id=request_id,
                            round_no=round_no,
                            decision_type="final_answer",
                            tool_name=None,
                            arguments=None,
                            duration_ms=0,
                            result_summary=terminal_text,
                        )
                        for index in range(0, len(terminal_text), 18):
                            yield AgentStreamEvent(
                                "delta",
                                {"text": terminal_text[index : index + 18]},
                            )
                        if self.executor.state.citations:
                            yield AgentStreamEvent(
                                "sources",
                                {
                                    "sources": [
                                        citation.to_dict()
                                        for citation in self.executor.state.citations
                                    ]
                                },
                            )
                        return
                continue

            candidate = turn.content.strip()
            if not self.executor.state.profile_loaded:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "当前轮尚未取得员工上下文。请先调用 "
                            "get_employee_profile，不得直接回答业务问题。"
                        ),
                    }
                )
                continue
            if not candidate:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "请基于已获得的工具结果继续，或明确说明需要用户补充什么。"
                        ),
                    }
                )
                continue
            await self.database.write_agent_decision_log(
                conversation_id=self.executor.conversation_id,
                user_id=str(self.executor.user["id"]),
                request_id=request_id,
                round_no=round_no,
                decision_type="final_answer",
                tool_name=None,
                arguments=None,
                duration_ms=0,
                result_summary=candidate[:900],
            )
            final_answer = candidate
            break
        else:
            raise RuntimeError("Agent 决策超过最多 5 轮，已安全停止。")

        for index in range(0, len(final_answer), 18):
            yield AgentStreamEvent(
                "delta",
                {"text": final_answer[index : index + 18]},
            )
        if self.executor.state.citations:
            yield AgentStreamEvent(
                "sources",
                {
                    "sources": [
                        citation.to_dict()
                        for citation in self.executor.state.citations
                    ]
                },
            )


# 兼容既有导入路径；实际实现已升级为 Harness。
AgentRunner = AgentHarness
