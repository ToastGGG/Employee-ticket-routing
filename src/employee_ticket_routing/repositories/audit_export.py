"""对话审计导出与数据库摘要。

管理员页面的筛选适合交互查看，但课程验收、季度抽查和故障复盘通常需要可归档
的 CSV/JSON。本模块通过只读 SQL 生成稳定格式，并复用统一脱敏函数。
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from ..core.db import SQLiteDatabase
from ..core.redaction import redact_text


ExportFormat = Literal["json", "csv"]


@dataclass(frozen=True)
class AuditExportOptions:
    employee_no: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    flagged_only: bool = False
    limit: int = 5000
    include_content: bool = True
    max_content_chars: int = 2000

    def __post_init__(self) -> None:
        if self.limit < 1 or self.limit > 100_000:
            raise ValueError("limit 必须在 1 到 100000 之间")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from 不能晚于 date_to")


@dataclass(frozen=True)
class AuditExportRecord:
    message_id: str
    conversation_id: str
    employee_no: str
    employee_name: str
    role: str
    status: str
    content: str
    request_id: str | None
    model_provider: str | None
    model_name: str | None
    is_flagged: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_date_boundary(value: str | None, *, end: bool) -> str | None:
    """允许 YYYY-MM-DD 或完整 ISO 时间。"""

    if not value:
        return None
    text = value.strip()
    if len(text) == 10:
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"日期格式无效：{value}") from exc
        return f"{text}T23:59:59.999999+00:00" if end else f"{text}T00:00:00+00:00"
    return text


async def collect_audit_records(
    database: SQLiteDatabase,
    options: AuditExportOptions | None = None,
) -> list[AuditExportRecord]:
    """按员工、时间和违规标记筛选消息，返回只读记录。"""

    selected = options or AuditExportOptions()
    clauses = ["1 = 1"]
    params: list[Any] = []
    if selected.employee_no:
        clauses.append("u.employee_no = ?")
        params.append(selected.employee_no)
    date_from = _normalize_date_boundary(selected.date_from, end=False)
    date_to = _normalize_date_boundary(selected.date_to, end=True)
    if date_from:
        clauses.append("m.created_at >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("m.created_at <= ?")
        params.append(date_to)
    if selected.flagged_only:
        clauses.append(
            "EXISTS (SELECT 1 FROM audit_flags af "
            "WHERE af.conversation_id = m.conversation_id "
            "AND af.status = 'active')"
        )
    params.append(selected.limit)
    query = f"""
        SELECT
            m.id AS message_id,
            m.conversation_id,
            u.employee_no,
            u.name AS employee_name,
            m.role,
            m.status,
            m.content,
            m.request_id,
            m.model_provider,
            m.model_name,
            CASE WHEN EXISTS (
                SELECT 1 FROM audit_flags af
                WHERE af.conversation_id = m.conversation_id
                  AND af.status = 'active'
            ) THEN 1 ELSE 0 END AS is_flagged,
            m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        JOIN users u ON u.id = c.user_id
        WHERE {' AND '.join(clauses)}
        ORDER BY m.created_at ASC
        LIMIT ?
    """
    async with database._connect() as connection:
        cursor = await connection.execute(query, tuple(params))
        rows = await cursor.fetchall()

    records: list[AuditExportRecord] = []
    for row in rows:
        content = str(row["content"] or "")
        if not selected.include_content:
            content = ""
        else:
            content = redact_text(
                content,
                max_length=selected.max_content_chars,
            )
        records.append(
            AuditExportRecord(
                message_id=str(row["message_id"]),
                conversation_id=str(row["conversation_id"]),
                employee_no=str(row["employee_no"] or ""),
                employee_name=str(row["employee_name"] or ""),
                role=str(row["role"]),
                status=str(row["status"]),
                content=content,
                request_id=str(row["request_id"]) if row["request_id"] else None,
                model_provider=(
                    str(row["model_provider"]) if row["model_provider"] else None
                ),
                model_name=str(row["model_name"]) if row["model_name"] else None,
                is_flagged=bool(row["is_flagged"]),
                created_at=str(row["created_at"]),
            )
        )
    return records


def render_audit_export(
    records: Iterable[AuditExportRecord],
    *,
    export_format: ExportFormat = "json",
) -> str:
    """将审计记录渲染为 JSON 或 CSV 字符串。"""

    rows = [record.to_dict() for record in records]
    if export_format == "json":
        return json.dumps(
            {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(rows),
                "records": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    if export_format != "csv":
        raise ValueError("export_format 仅支持 json 或 csv")

    output = io.StringIO(newline="")
    fieldnames = list(AuditExportRecord.__dataclass_fields__)
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def write_audit_export(
    path: Path,
    records: Iterable[AuditExportRecord],
    *,
    export_format: ExportFormat = "json",
) -> Path:
    """原子写入导出文件，失败时不会留下半成品。"""

    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = render_audit_export(records, export_format=export_format)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


async def database_summary(database: SQLiteDatabase) -> dict[str, int]:
    """查询核心表数量和当前数据健康计数。"""

    table_names = (
        "users",
        "conversations",
        "messages",
        "knowledge_documents",
        "document_chunks",
        "ticket_drafts",
        "prompt_templates",
        "ingestion_jobs",
        "audit_flags",
        "audit_logs",
        "agent_decision_logs",
    )
    result: dict[str, int] = {}
    async with database._connect() as connection:
        for table_name in table_names:
            cursor = await connection.execute(
                f"SELECT COUNT(*) AS count FROM {table_name}"
            )
            row = await cursor.fetchone()
            result[table_name] = int(row["count"])
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM ticket_drafts
            WHERE status IN ('draft', 'waiting_confirmation')
            """
        )
        result["pending_ticket_drafts"] = int((await cursor.fetchone())["count"])
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM knowledge_documents
            WHERE status = 'ready' AND confirmation_status = 'confirmed'
            """
        )
        result["confirmed_ready_policies"] = int(
            (await cursor.fetchone())["count"]
        )
    return result
