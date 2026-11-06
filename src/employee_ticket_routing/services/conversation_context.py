"""对话上下文选择与引用合并。

LLM 上下文不是越长越好。对于薪酬福利咨询，正确策略通常是：

* 保留最近若干轮完整对话；
* 从更早历史中优先保留包含问题、诉求、制度引用的消息；
* 去除连续重复内容；
* 限制总字符预算，避免挤占知识条文；
* 保持 user/assistant 交替关系，不能只截断其中一半。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ContextMessage:
    role: str
    content: str
    citations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ContextWindow:
    messages: tuple[ContextMessage, ...]
    dropped_count: int
    total_chars: int
    citation_count: int
    source_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_prompt_pairs(self) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]


def _to_context_message(row: ContextMessage | Mapping[str, Any]) -> ContextMessage:
    if isinstance(row, ContextMessage):
        return row
    citations = row.get("citations") or row.get("citations_json") or ()
    if isinstance(citations, str):
        citations = ()
    if not isinstance(citations, (list, tuple)):
        citations = ()
    normalized_citations = tuple(
        dict(item) for item in citations if isinstance(item, Mapping)
    )
    return ContextMessage(
        role=str(row.get("role") or "user"),
        content=str(row.get("content") or ""),
        citations=normalized_citations,
    )


def _merge_citations(
    messages: Iterable[ContextMessage],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    seen: set[str] = set()
    citations: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for message in messages:
        for citation in message.citations:
            file_name = str(
                citation.get("file_name")
                or citation.get("document")
                or citation.get("source")
                or ""
            )
            snippet = str(citation.get("content") or citation.get("snippet") or "")
            key = f"{file_name}\u0000{snippet}"
            if key in seen:
                continue
            seen.add(key)
            citation_copy = dict(citation)
            citations.append(citation_copy)
            source_id = str(citation.get("id") or citation.get("chunk_id") or "")
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
    return tuple(citations), tuple(source_ids)


def select_recent_context(
    rows: Iterable[ContextMessage | Mapping[str, Any]],
    *,
    max_rounds: int = 6,
    max_chars: int = 12_000,
) -> ContextWindow:
    """按预算选择最近对话，并保证至少保留一条 user 与一条 assistant。"""

    if max_rounds < 1:
        raise ValueError("max_rounds 必须大于 0")
    if max_chars < 100:
        raise ValueError("max_chars 必须至少为 100")

    messages = [
        _to_context_message(row)
        for row in rows
        if str((row.get("content") if isinstance(row, Mapping) else row.content) or "").strip()
    ]
    deduplicated: list[ContextMessage] = []
    previous_text = ""
    for message in messages:
        normalized = " ".join(message.content.split())
        if normalized == previous_text:
            continue
        previous_text = normalized
        deduplicated.append(message)

    max_messages = max_rounds * 2
    candidates = deduplicated[-max_messages:]
    selected_reversed: list[ContextMessage] = []
    total_chars = 0
    for message in reversed(candidates):
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        content = message.content
        if len(content) > remaining:
            if selected_reversed:
                break
            content = content[:remaining]
        selected_reversed.append(
            ContextMessage(
                role=message.role,
                content=content,
                citations=message.citations,
            )
        )
        total_chars += len(content)

    selected = tuple(reversed(selected_reversed))
    if not any(message.role == "user" for message in selected):
        user_candidates = [
            message for message in reversed(candidates) if message.role == "user"
        ]
        if user_candidates:
            selected = (user_candidates[0],)
            total_chars = len(user_candidates[0].content)

    citations, source_ids = _merge_citations(selected)
    return ContextWindow(
        messages=selected,
        dropped_count=max(0, len(deduplicated) - len(selected)),
        total_chars=total_chars,
        citation_count=len(citations),
        source_ids=source_ids,
    )
