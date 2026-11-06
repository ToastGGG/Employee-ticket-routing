"""工单领域模型、分类和 SLA 测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.employee_ticket_routing.domain.ticket import (
    InvalidTicketTransition,
    QueueName,
    RoutingSignals,
    TicketCategory,
    TicketPriority,
    TicketStateMachine,
    TicketStatus,
)
from src.employee_ticket_routing.services.ticket_priority import (
    classify_and_route,
    compute_due_at,
)


class TicketDomainTests(unittest.TestCase):
    def test_payroll_question_routes_to_compensation_queue(self) -> None:
        decision = classify_and_route(
            RoutingSignals(
                question="新人入职首月工资和社保基数如何确认？"
            )
        )
        self.assertIn(
            decision.category,
            {TicketCategory.PAYROLL, TicketCategory.SOCIAL_INSURANCE},
        )
        self.assertFalse(decision.requires_human_review)
        self.assertGreaterEqual(decision.confidence, 0.5)

    def test_high_risk_labor_dispute_requires_human_review(self) -> None:
        decision = classify_and_route(
            RoutingSignals(
                details="公司克扣绩效奖金，我准备申请劳动仲裁"
            )
        )
        self.assertEqual(decision.category, TicketCategory.LABOR_DISPUTE)
        self.assertEqual(decision.priority, TicketPriority.URGENT)
        self.assertEqual(decision.queue, QueueName.HUMAN_REVIEW)
        self.assertTrue(decision.requires_human_review)
        self.assertIn("高风险事项需要人工合规复核", decision.unresolved_items)

    def test_explicit_deadline_reduces_sla_hours(self) -> None:
        now = datetime(2026, 11, 6, 8, 0, tzinfo=timezone.utc)
        decision = classify_and_route(
            RoutingSignals(
                question="请确认公积金补缴流程",
                requested_deadline=now + timedelta(hours=2),
            ),
            now=now,
        )
        self.assertEqual(decision.priority, TicketPriority.URGENT)
        self.assertEqual(decision.sla_hours, 2)
        self.assertEqual(
            compute_due_at(decision, started_at=now).hour,
            10,
        )

    def test_unknown_short_request_is_not_overconfident(self) -> None:
        decision = classify_and_route(RoutingSignals(question="帮我处理"))
        self.assertEqual(decision.category, TicketCategory.UNKNOWN)
        self.assertLessEqual(decision.confidence, 0.45)
        self.assertTrue(decision.requires_human_review)

    def test_state_machine_blocks_confirmation_bypass(self) -> None:
        self.assertTrue(
            TicketStateMachine.can_transition(
                TicketStatus.DRAFT,
                TicketStatus.WAITING_CONFIRMATION,
            )
        )
        with self.assertRaises(InvalidTicketTransition):
            TicketStateMachine.validate_transition(
                TicketStatus.DRAFT,
                TicketStatus.CLOSED,
            )
        self.assertEqual(
            TicketStateMachine.next_status(
                TicketStatus.WAITING_CONFIRMATION,
                employee_confirmed=False,
                requires_human_review=False,
            ),
            TicketStatus.WAITING_CONFIRMATION,
        )


if __name__ == "__main__":
    unittest.main()
