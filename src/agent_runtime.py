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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Sequence

import httpx

try:
    from .db import SQLiteDatabase
    from .rag_runtime import Citation, RagRuntime
    from .security import AgentTokenService
except ImportError:
    from db import SQLiteDatabase
    from rag_runtime import Citation, RagRuntime
    from security import AgentTokenService


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
    invoice_token: str | None = None
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
                "检索薪酬福利、制度或报销政策。返回带版本和有效期的证据。"
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
                            "reimbursement",
                            "attendance",
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
            "name": "validate_invoice",
            "description": (
                "报销或费用类工单的强制校验。"
                "需先通过 validate_policy，校验金额、日期、费用类型和附件。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_validation_token": {"type": "string"},
                    "invoice_file_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "amount": {"type": "number", "minimum": 0},
                    "currency": {"type": "string"},
                    "expense_type": {"type": "string"},
                    "occurred_at": {"type": "string"},
                },
                "required": [
                    "policy_validation_token",
                    "invoice_file_ids",
                    "amount",
                    "currency",
                    "expense_type",
                    "occurred_at",
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
                "基于已通过的制度/发票校验确定队列、优先级和人工风险。"
                "报销类工单必须提供通过的 invoice_validation_token。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "case_type": {
                        "type": "string",
                        "enum": ["reimbursement", "salary_benefit", "other"],
                    },
                    "policy_validation_token": {"type": "string"},
                    "invoice_validation_token": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "case_type",
                    "policy_validation_token",
                    "invoice_validation_token",
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
                "创建可编辑但未提交的工单草稿。"
                "必须持有 route_ticket 签发的 routing_token。"
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
            if name == "validate_invoice":
                return await self._validate_invoice(call_id, arguments)
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

    async def _validate_invoice(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """
        报销工单校验。

        当前原型校验金额、日期、费用类型和附件 ID 是否完整；正式系统应把
        费用标准接入结构化政策规则表，而不是从自然语言中猜测规则。
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
                "报销校验前必须先通过 validate_policy。",
            ) from exc

        invoice_ids = arguments.get("invoice_file_ids")
        amount = arguments.get("amount")
        currency = str(arguments.get("currency") or "").upper()
        expense_type = str(arguments.get("expense_type") or "").strip()
        occurred_at = str(arguments.get("occurred_at") or "").strip()
        missing: list[str] = []
        checks = []

        if not isinstance(invoice_ids, list) or not invoice_ids:
            missing.append("invoice_file_ids")
            checks.append({"name": "发票附件", "passed": False})
        else:
            checks.append({"name": "发票附件", "passed": True})

        if not isinstance(amount, (int, float)) or float(amount) < 0:
            missing.append("amount")
            checks.append({"name": "金额", "passed": False})
        else:
            checks.append({"name": "金额", "passed": True})

        if currency != "CNY":
            missing.append("currency")
            checks.append({"name": "币种", "passed": False})
        else:
            checks.append({"name": "币种", "passed": True})

        if not expense_type:
            missing.append("expense_type")
            checks.append({"name": "费用类型", "passed": False})
        else:
            checks.append({"name": "费用类型", "passed": True})

        try:
            datetime.fromisoformat(occurred_at)
            date_valid = True
        except ValueError:
            date_valid = False
        if not date_valid:
            missing.append("occurred_at")
            checks.append({"name": "发生日期", "passed": False})
        else:
            checks.append({"name": "发生日期", "passed": True})

        if missing:
            return self._success(
                call_id=call_id,
                name="validate_invoice",
                data={
                    "passed": False,
                    "validation_id": None,
                    "validation_token": None,
                    "checks": checks,
                    "missing_fields": missing,
                    "reason": "报销资料不完整，不能创建工单草稿。",
                },
                citations=list(self.state.evidence.values()),
            )

        threshold = float(os.getenv("INVOICE_HUMAN_REVIEW_THRESHOLD", "3000"))
        risk_flags = ["amount_over_threshold"] if float(amount) >= threshold else []
        payload = {
            "policy_jti": policy_claims["jti"],
            "invoice_file_ids": invoice_ids,
            "amount": float(amount),
            "currency": currency,
            "expense_type": expense_type,
            "occurred_at": occurred_at,
            "risk_flags": risk_flags,
        }
        token = self.token_service.create(
            token_type="invoice",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            payload=payload,
        )
        decoded = self.token_service.decode(
            token,
            expected_type="invoice",
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
        )
        self.state.invoice_token = token
        return self._success(
            call_id=call_id,
            name="validate_invoice",
            data={
                "passed": True,
                "validation_id": decoded["jti"],
                "validation_token": token,
                "checks": checks,
                "missing_fields": [],
                "risk_flags": risk_flags,
                "reason": "报销基础字段和附件引用完整。",
            },
            citations=list(self.state.evidence.values()),
        )

    async def _route_ticket(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """依据校验结果和风险标志签发 routing Token。"""

        case_type = str(arguments.get("case_type") or "other")
        policy_token = str(arguments.get("policy_validation_token") or "")
        invoice_token_raw = arguments.get("invoice_validation_token")
        invoice_token = str(invoice_token_raw) if invoice_token_raw else None
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

        invoice_claims = None
        if invoice_token:
            try:
                invoice_claims = self.token_service.decode(
                    invoice_token,
                    expected_type="invoice",
                    user_id=str(self.user["id"]),
                    case_id=self.conversation_id,
                )
            except ValueError as exc:
                raise AgentToolError(
                    "INVOICE_VALIDATION_REQUIRED",
                    "报销校验凭证无效或已过期。",
                ) from exc

        if case_type == "reimbursement" and invoice_claims is None:
            raise AgentToolError(
                "INVOICE_VALIDATION_REQUIRED",
                "报销类工单必须先通过 validate_invoice。",
            )
        inferred_reimbursement = any(
            keyword in self.request_question
            for keyword in ("报销", "发票", "票据", "附件 file", "附件file")
        )
        if inferred_reimbursement and case_type != "reimbursement":
            raise AgentToolError(
                "CASE_TYPE_MISMATCH",
                "咨询内容涉及报销或发票，不能改标为其他工单类型来绕过校验。",
            )
        if (
            invoice_claims is not None
            and str(invoice_claims.get("payload", {}).get("policy_jti") or "")
            != str(policy_claims["jti"])
        ):
            raise AgentToolError(
                "INVOICE_POLICY_MISMATCH",
                "发票校验凭证与当前制度校验不属于同一轮处理。",
            )

        invoice_risk = (
            invoice_claims.get("payload", {}).get("risk_flags", [])
            if invoice_claims
            else []
        )
        all_risk = sorted(set(risk_flags + list(invoice_risk)))
        requires_human = bool(all_risk)
        queue = (
            "finance_review"
            if case_type == "reimbursement"
            else "employee_services"
        )
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
        if invoice_claims and not await self.database.consume_tool_token(
            jti=str(invoice_claims["jti"]),
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            token_type="invoice",
            purpose="route_ticket",
        ):
            raise AgentToolError("TOKEN_REPLAYED", "报销校验凭证已被使用。")

        route_payload = {
            "case_type": case_type,
            "policy_jti": policy_claims["jti"],
            "invoice_jti": invoice_claims["jti"] if invoice_claims else None,
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

    async def _create_ticket_draft(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolExecution:
        """
        创建工单草稿。

        报销类草稿要求 routing Token 内包含 invoice_jti；如果模型省略
        validate_invoice 或校验失败，route_ticket 不会签发此 Token。
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
        if payload.get("case_type") == "reimbursement" and not payload.get("invoice_jti"):
            raise AgentToolError(
                "INVOICE_VALIDATION_REQUIRED",
                "报销工单草稿必须在 validate_invoice 通过后创建。",
            )
        if payload.get("requires_human"):
            raise AgentToolError(
                "HUMAN_REVIEW_REQUIRED",
                "当前工单存在高风险，必须转人工处理，不能自动创建草稿。",
            )
        if not await self.database.consume_tool_token(
            jti=str(claims["jti"]),
            user_id=str(self.user["id"]),
            case_id=self.conversation_id,
            token_type="routing",
            purpose="create_ticket_draft",
        ):
            raise AgentToolError("TOKEN_REPLAYED", "路由凭证已被使用。")

        title = str(arguments.get("title") or "").strip()
        description = str(arguments.get("description") or "").strip()
        if not title or not description:
            raise AgentToolError("DRAFT_CONTENT_REQUIRED", "工单标题和描述不能为空。")
        attachment_ids = [
            str(item)
            for item in arguments.get("attachment_ids", [])
            if str(item).strip()
        ]
        validation_summary = {
            "policy_jti": payload.get("policy_jti"),
            "invoice_jti": payload.get("invoice_jti"),
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
            "parallel_tool_calls": False,
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

        if latest_name is None:
            return call(
                "get_employee_profile",
                {"include": ["employment_type", "region", "policy_group"]},
            )

        data = latest_result.get("data", {})
        data = data if isinstance(data, dict) else {}
        if latest_name == "get_employee_profile":
            return call(
                "search_knowledge",
                {
                    "query": question or "薪酬福利制度",
                    "domain": (
                        "reimbursement"
                        if "报销" in question
                        else "salary_benefit"
                    ),
                    "top_k": 3,
                },
            )

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

            wants_ticket = any(
                keyword in question
                for keyword in [
                    "创建工单",
                    "建单",
                    "提交工单",
                    "发起工单",
                    "申请报销",
                    "提交一笔",
                ]
            )
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

            if "报销" in question:
                amount_match = re.search(r"(\d+(?:\.\d+)?)\s*元", question)
                amount = float(amount_match.group(1)) if amount_match else 0
                raw_file_ids = re.findall(
                    r"(?:file|附件)[_-]([A-Za-z0-9_-]+)",
                    question,
                )
                invoice_file_ids = [
                    item if item.startswith("file_") else f"file_{item}"
                    for item in raw_file_ids
                ]
                date_match = re.search(r"20\d{2}-\d{2}-\d{2}", question)
                return call(
                    "validate_invoice",
                    {
                        "policy_validation_token": str(
                            data.get("validation_token") or ""
                        ),
                        "invoice_file_ids": invoice_file_ids,
                        "amount": amount,
                        "currency": "CNY",
                        "expense_type": "travel",
                        "occurred_at": (
                            date_match.group(0) if date_match else ""
                        ),
                    },
                )
            return call(
                "route_ticket",
                {
                    "case_type": "salary_benefit",
                    "policy_validation_token": str(
                        data.get("validation_token") or ""
                    ),
                    "invoice_validation_token": None,
                    "risk_flags": [],
                },
            )

        if latest_name == "validate_invoice":
            if not data.get("passed"):
                missing = "、".join(
                    str(item) for item in data.get("missing_fields", [])
                )
                return ModelTurn(
                    content=(
                        "报销资料尚不完整，暂时不能创建工单草稿。"
                        f"需要补充：{missing or '发票附件、金额或发生日期'}。"
                    )
                )
            policy_result = _find_tool_result(messages, "validate_policy")
            policy_data = policy_result.get("data", {})
            return call(
                "route_ticket",
                {
                    "case_type": "reimbursement",
                    "policy_validation_token": str(
                        policy_data.get("validation_token") or ""
                    ),
                    "invoice_validation_token": str(
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
                        "reason": "工单金额或风险规则达到人工复核阈值。",
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
                        "系统已完成制度、报销字段和路由校验。"
                    ),
                    "attachment_ids": [],
                },
            )

        if latest_name == "create_ticket_draft":
            return ModelTurn(
                content=(
                    "已生成工单草稿，但尚未正式提交。"
                    "请核对草稿内容并点击“确认并提交”。"
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
1. 每轮业务咨询必须先调用 get_employee_profile，身份只来自登录会话。
2. 涉及政策、福利、薪酬或报销事实时，必须调用 search_knowledge，
   并使用本轮检索返回的 evidence_id 调用 validate_policy。
3. 涉及报销、发票或费用金额时，必须继续调用 validate_invoice。
4. 只有 validate_policy 和必要的 validate_invoice 通过，并取得
   route_ticket 签发的 routing_token 后，才能调用 create_ticket_draft。
5. 高风险、制度冲突、资料不足、权限不足或用户要求人工时，调用
   request_human_review，不得绕过校验强行建单。
6. create_ticket_draft 只生成草稿；正式提交必须由员工在前端明确确认。
   模型不得声称草稿已提交，也不得把模糊回复视为确认。
7. 不得伪造、复用或跨会话使用任何 validation_token。
8. 工具失败时如实说明，不得声称未实际成功的动作已经完成。
9. 最终回答只使用本轮工具返回的制度证据；没有证据时明确说明
   “制度中未找到”，不得编造报销标准。
"""


class AgentRunner:
    """
    执行“模型决策 → 工具调用 → 结果回填 → 最终回答”的自主循环。

    模型拥有选择下一步工具的自由，但工具依赖、用户身份、数据归属和
    一次性凭证均由服务端强制执行。提示词不是唯一的安全边界。
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
        self.max_steps = max(int(os.getenv("AGENT_MAX_STEPS", "12")), 1)

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

        模型每轮只能从已注册工具中选择动作；工具返回结构会作为 tool 消息
        回填。达到最大步数时失败关闭，避免无限自主循环。
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

        for _ in range(self.max_steps):
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

                for tool_call in turn.tool_calls:
                    yield AgentStreamEvent(
                        "status",
                        {
                            "stage": "tool",
                            "message": f"正在执行 {tool_call.name}…",
                            "tool": tool_call.name,
                        },
                    )
                    result = await self.executor.execute(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
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
                    if result.draft is not None:
                        yield AgentStreamEvent("draft", result.draft)
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
            final_answer = candidate
            break
        else:
            raise RuntimeError("Agent 工具调用超过最大步数，已停止执行。")

        if self.executor.state.draft is not None:
            final_answer += (
                "\n\n当前状态：仅为工单草稿，尚未提交。"
                "请核对后由你本人点击“确认并提交”。"
            )

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
