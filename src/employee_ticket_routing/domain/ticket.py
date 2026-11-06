"""工单分流领域模型与状态机。

把分类、优先级、队列和状态转换从 FastAPI 路由中分离，能够带来三个直接收益：

1. 测试不再需要启动 Web 服务；
2. 路由层只负责鉴权和 HTTP，业务规则更容易复用；
3. 模型输出的分类不能直接触发非法状态，服务端有最后一道确定性校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TicketCategory(str, Enum):
    """薪酬福利咨询的主要业务分类。"""

    POLICY = "policy"
    PAYROLL = "payroll"
    BENEFITS = "benefits"
    SOCIAL_INSURANCE = "social_insurance"
    HOUSING_FUND = "housing_fund"
    ATTENDANCE = "attendance"
    LABOR_DISPUTE = "labor_dispute"
    UNKNOWN = "unknown"


class TicketPriority(str, Enum):
    """工单优先级。"""

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class QueueName(str, Enum):
    """接收队列。名称保持稳定，便于与外部工单系统映射。"""

    EMPLOYEE_SERVICES = "employee_services"
    COMPENSATION_BENEFITS = "compensation_benefits"
    SOCIAL_INSURANCE = "social_insurance"
    LABOR_RELATIONS = "labor_relations"
    HUMAN_REVIEW = "human_review"


class TicketStatus(str, Enum):
    """工单草稿与提交后的状态。"""

    DRAFT = "draft"
    WAITING_CONFIRMATION = "waiting_confirmation"
    SUBMITTED = "submitted"
    HUMAN_REVIEW = "human_review"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RoutingSignals:
    """分类器可使用的确定性信号。

    模型可以提供摘要，但关键词、风险标签和员工类型等结构化事实由服务端补充。
    """

    question: str = ""
    details: str = ""
    risk_flags: tuple[str, ...] = ()
    employee_type: str = "employee"
    requested_deadline: datetime | None = None

    @property
    def searchable_text(self) -> str:
        return " ".join(
            part.strip()
            for part in (self.question, self.details, *self.risk_flags)
            if part and part.strip()
        ).lower()


@dataclass(frozen=True)
class RoutingDecision:
    """工单分类、优先级、接收队列及解释。"""

    category: TicketCategory
    priority: TicketPriority
    queue: QueueName
    confidence: float
    sla_hours: int
    reasons: tuple[str, ...] = field(default_factory=tuple)
    unresolved_items: tuple[str, ...] = field(default_factory=tuple)
    requires_human_review: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence 必须在 0 到 1 之间")
        if self.sla_hours <= 0:
            raise ValueError("sla_hours 必须大于 0")

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "priority": self.priority.value,
            "queue": self.queue.value,
            "confidence": round(self.confidence, 4),
            "sla_hours": self.sla_hours,
            "reasons": list(self.reasons),
            "unresolved_items": list(self.unresolved_items),
            "requires_human_review": self.requires_human_review,
        }


class InvalidTicketTransition(ValueError):
    """工单状态流转不合法。"""


class TicketStateMachine:
    """限制工单只能按合法路径流转。

    特别约束：``draft`` 不能直接跳到 ``closed``；必须先经员工确认成为
    ``submitted``，避免系统或模型绕过确认流程。
    """

    _ALLOWED: dict[TicketStatus, frozenset[TicketStatus]] = {
        TicketStatus.DRAFT: frozenset(
            {
                TicketStatus.WAITING_CONFIRMATION,
                TicketStatus.CANCELLED,
            }
        ),
        TicketStatus.WAITING_CONFIRMATION: frozenset(
            {
                TicketStatus.SUBMITTED,
                TicketStatus.HUMAN_REVIEW,
                TicketStatus.CANCELLED,
            }
        ),
        TicketStatus.SUBMITTED: frozenset(
            {
                TicketStatus.HUMAN_REVIEW,
                TicketStatus.CLOSED,
                TicketStatus.CANCELLED,
            }
        ),
        TicketStatus.HUMAN_REVIEW: frozenset(
            {
                TicketStatus.SUBMITTED,
                TicketStatus.CLOSED,
                TicketStatus.CANCELLED,
            }
        ),
        TicketStatus.CLOSED: frozenset(),
        TicketStatus.CANCELLED: frozenset(),
    }

    @classmethod
    def can_transition(
        cls,
        current: TicketStatus,
        target: TicketStatus,
    ) -> bool:
        return target in cls._ALLOWED[current]

    @classmethod
    def validate_transition(
        cls,
        current: TicketStatus,
        target: TicketStatus,
    ) -> None:
        if not cls.can_transition(current, target):
            raise InvalidTicketTransition(
                f"工单状态不能从 {current.value} 直接变为 {target.value}"
            )

    @classmethod
    def next_status(
        cls,
        current: TicketStatus,
        *,
        employee_confirmed: bool,
        requires_human_review: bool,
    ) -> TicketStatus:
        """根据业务事实计算下一状态，而不允许调用方任意指定。"""

        if current == TicketStatus.WAITING_CONFIRMATION:
            if not employee_confirmed:
                return current
            if requires_human_review:
                return TicketStatus.HUMAN_REVIEW
            return TicketStatus.SUBMITTED
        if current == TicketStatus.DRAFT:
            return TicketStatus.WAITING_CONFIRMATION
        return current
