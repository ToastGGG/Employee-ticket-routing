"""审计导出和数据库摘要测试。"""

from __future__ import annotations

import asyncio
import json
import shutil
import unittest
import uuid
from pathlib import Path

from src.employee_ticket_routing.core.db import SQLiteDatabase, utc_now
from src.employee_ticket_routing.repositories.audit_export import (
    AuditExportOptions,
    collect_audit_records,
    database_summary,
    render_audit_export,
    write_audit_export,
)


class AuditExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = (
            Path(__file__).resolve().parents[1]
            / f".audit-export-test-{uuid.uuid4().hex}"
        )
        self.temp_dir.mkdir(parents=True)
        self.database_path = self.temp_dir / "audit.sqlite3"
        self.database = SQLiteDatabase(self.database_path)
        asyncio.run(self.database.initialize())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def seed_message(self) -> None:
        async def seed() -> None:
            async with self.database._connect() as connection:
                now = utc_now()
                await connection.execute(
                    """
                    INSERT INTO users
                    (id, employee_no, name, role, status, created_at, updated_at)
                    VALUES ('u1', 'E1001', '张三', 'employee', 'active', ?, ?)
                    """,
                    (now, now),
                )
                await connection.execute(
                    """
                    INSERT INTO conversations
                    (id, user_id, title, created_at, updated_at)
                    VALUES ('c1', 'u1', '薪酬咨询', ?, ?)
                    """,
                    (now, now),
                )
                await connection.execute(
                    """
                    INSERT INTO messages
                    (id, conversation_id, role, content, status, request_id, created_at)
                    VALUES ('m1', 'c1', 'user', ?, 'completed', 'req-1', ?)
                    """,
                    ("我的手机号是 13800138000，请确认工资金额", now),
                )
                await connection.commit()

        asyncio.run(seed())

    def test_records_are_filtered_and_redacted(self) -> None:
        self.seed_message()
        records = asyncio.run(
            collect_audit_records(
                self.database,
                AuditExportOptions(employee_no="E1001"),
            )
        )
        self.assertEqual(len(records), 1)
        self.assertNotIn("13800138000", records[0].content)
        self.assertEqual(records[0].request_id, "req-1")

    def test_json_csv_and_database_summary(self) -> None:
        self.seed_message()
        records = asyncio.run(collect_audit_records(self.database))
        json_text = render_audit_export(records, export_format="json")
        csv_text = render_audit_export(records, export_format="csv")
        self.assertEqual(json.loads(json_text)["record_count"], 1)
        self.assertIn("message_id", csv_text)
        summary = asyncio.run(database_summary(self.database))
        self.assertEqual(summary["messages"], 1)
        self.assertEqual(summary["users"], 1)

    def test_export_writer_is_atomic_and_content_can_be_disabled(self) -> None:
        self.seed_message()
        records = asyncio.run(
            collect_audit_records(
                self.database,
                AuditExportOptions(include_content=False),
            )
        )
        target = self.temp_dir / "out" / "audit.json"
        write_audit_export(target, records, export_format="json")
        self.assertTrue(target.exists())
        self.assertNotIn("工资金额", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
