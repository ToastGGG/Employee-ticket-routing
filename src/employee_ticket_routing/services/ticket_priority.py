"""确定性的工单分类、优先级与 SLA 规则。

LLM 擅长理解自然语言，但不适合独自决定“是否必须人工”“多长时间必须处理”
这类审计要求稳定的规则。本模块使用可解释关键词和风险标签计算兜底结论，
即使模型不可用也能保证工单进入合理队列。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..domain.ticket import (
    QueueName,
    RoutingDecision,
    RoutingSignals,
    TicketCategory,
    TicketPriority,
)


CATEGORY_KEYWORDS: dict[TicketCategory, tuple[str, ...]] = {
    TicketCategory.PAYROLL: (
        "工资",
        "薪资",
        "薪酬",
        "发薪",
        "奖金",
        "绩效",
        "个税",
        "加班费",
    ),
    TicketCategory.BENEFITS: (
        "福利",
        "年假",
        "商业保险",
        "补充医疗",
        "体检",
        "餐补",
        "津贴",
        "培训",
    ),
    TicketCategory.SOCIAL_INSURANCE: (
        "社保",
        "养老",
        "医疗",
        "失业",
        "工伤",
        "生育",
    ),
    TicketCategory.HOUSING_FUND: (
        "公积金",
        "住房补贴",
        "租房提取",
    ),
    TicketCategory.ATTENDANCE: (
        "考勤",
        "请假",
        "病假",
        "事假",
        "调休",
        "打卡",
    ),
    TicketCategory.LABOR_DISPUTE: (
        "劳动仲裁",
        "劳动争议",
        "申诉",
        "投诉",
        "克扣",
        "违法解除",
        "仲裁",
    ),
}

HIGH_RISK_KEYWORDS = (
    "仲裁",
    "诉讼",
    "律师",
    "工伤",
    "歧视",
    "骚扰",
    "违法",
    "克扣",
    "欠薪",
    "集体投诉",
    "泄露",
)

URGENT_KEYWORDS = (
    "今天",
    "立即",
    "马上",
    "截止",
    "紧急",
    "停发",
    "无法发薪",
    "影响入职",
)

CATEGORY_QUEUE: dict[TicketCategory, QueueName] = {
    TicketCategory.PAYROLL: QueueName.COMPENSATION_BENEFITS,
    TicketCategory.BENEFITS: QueueName.COMPENSATION_BENEFITS,
    TicketCategory.SOCIAL_INSURANCE: QueueName.SOCIAL_INSURANCE,
    TicketCategory.HOUSING_FUND: QueueName.SOCIAL_INSURANCE,
    TicketCategory.ATTENDANCE: QueueName.EMPLOYEE_SERVICES,
    TicketCategory.LABOR_DISPUTE: QueueName.LABOR_RELATIONS,
    TicketCategory.POLICY: QueueName.EMPLOYEE_SERVICES,
    TicketCategory.UNKNOWN: QueueName.EMPLOYEE_SERVICES,
}

SLA_HOURS: dict[TicketPriority, int] = {
    TicketPriority.URGENT: 4,
    TicketPriority.HIGH: 8,
    TicketPriority.NORMAL: 24,
    TicketPriority.LOW: 72,
}


def _category_matches(text: str) -> list[tuple[TicketCategory, tuple[str, ...]]]:
    matches: list[tuple[TicketCategory, tuple[str, ...]]] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        found = tuple(keyword for keyword in keywords if keyword in text)
        if found:
            matches.append((category, found))
    return matches


def _choose_category(
    matches: list[tuple[TicketCategory, tuple[str, ...]]],
) -> tuple[TicketCategory, tuple[str, ...]]:
    if not matches:
        return TicketCategory.UNKNOWN, ()
    ranked = sorted(
        matches,
        key=lambda item: (len(item[1]), item[0] == TicketCategory.PAYROLL),
        reverse=True,
    )
    return ranked[0]


def _calculate_priority(
    text: str,
    *,
    high_risk: bool,
    has_deadline: bool,
) -> tuple[TicketPriority, list[str]]:
    urgent_matches = [keyword for keyword in URGENT_KEYWORDS if keyword in text]
    if high_risk or urgent_matches or has_deadline:
        reasons = []
        if high_risk:
            reasons.append("命中高风险劳动/合规关键词")
        if urgent_matches:
            reasons.append("命中紧急时限：" + "、".join(urgent_matches))
        if has_deadline:
            reasons.append("员工提供了明确截止时间")
        return TicketPriority.URGENT, reasons
    if "工资" in text or "社保" in text or "公积金" in text:
        return TicketPriority.HIGH, ["涉及薪酬、社保或公积金资金事项"]
    if "咨询" in text or "制度" in text:
        return TicketPriority.NORMAL, ["一般制度咨询"]
    return TicketPriority.LOW, ["信息不足，保持普通排序"]


def _deadline_risk(
    signals: RoutingSignals,
    priority: TicketPriority,
    now: datetime | None,
) -> int:
    if signals.requested_deadline is None:
        return SLA_HOURS[priority]
    reference = now or datetime.now(timezone.utc)
    requested = signals.requested_deadline
    if requested.tzinfo is None:
        requested = requested.replace(tzinfo=timezone.utc)
    remaining_hours = int((requested - reference).total_seconds() // 3600)
    return max(1, min(SLA_HOURS[priority], remaining_hours or 1))


def classify_and_route(
    signals: RoutingSignals,
    *,
    now: datetime | None = None,
) -> RoutingDecision:
    """根据确定性信号返回分类、优先级、队列和 SLA。"""

    text = signals.searchable_text
    category, category_matches = _choose_category(_category_matches(text))
    risk_matches = tuple(
        keyword for keyword in HIGH_RISK_KEYWORDS if keyword in text
    )
    high_risk = bool(risk_matches) or category == TicketCategory.LABOR_DISPUTE
    priority, priority_reasons = _calculate_priority(
        text,
        high_risk=high_risk,
        has_deadline=signals.requested_deadline is not None,
    )
    reasons = []
    if category_matches:
        reasons.append("分类关键词：" + "、".join(category_matches))
    reasons.extend(priority_reasons)

    unresolved: list[str] = []
    if len(text) < 12:
        unresolved.append("咨询事项描述过短")
    if category == TicketCategory.UNKNOWN:
        unresolved.append("未识别到明确的薪酬福利分类")
    if high_risk:
        unresolved.append("高风险事项需要人工合规复核")

    requires_human_review = high_risk or category == TicketCategory.UNKNOWN
    queue = (
        QueueName.HUMAN_REVIEW
        if requires_human_review
        else CATEGORY_QUEUE[category]
    )
    confidence = 0.20
    if category_matches:
        confidence = min(0.95, 0.58 + 0.08 * len(category_matches))
    if high_risk:
        confidence = max(confidence, 0.82)
    if category == TicketCategory.UNKNOWN:
        confidence = min(confidence, 0.45)
    if unresolved:
        confidence = max(0.2, confidence - 0.05 * len(unresolved))

    return RoutingDecision(
        category=category,
        priority=priority,
        queue=queue,
        confidence=confidence,
        sla_hours=_deadline_risk(signals, priority, now),
        reasons=tuple(reasons),
        unresolved_items=tuple(unresolved),
        requires_human_review=requires_human_review,
    )


def compute_due_at(
    decision: RoutingDecision,
    *,
    started_at: datetime | None = None,
) -> datetime:
    """计算 SLA 到期时间，固定返回 UTC 时间。"""

    start = started_at or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return start.astimezone(timezone.utc) + timedelta(hours=decision.sla_hours)
