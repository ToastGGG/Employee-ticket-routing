"""与框架无关的领域模型。"""

from .ticket import (
    QueueName,
    RoutingDecision,
    RoutingSignals,
    TicketCategory,
    TicketPriority,
    TicketStateMachine,
    TicketStatus,
)

__all__ = [
    "QueueName",
    "RoutingDecision",
    "RoutingSignals",
    "TicketCategory",
    "TicketPriority",
    "TicketStateMachine",
    "TicketStatus",
]
