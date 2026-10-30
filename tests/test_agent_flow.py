"""Agent 工具闭环的端到端回归测试。"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agent_runtime import AgentToolExecutor  # noqa: E402
from src.app import app  # noqa: E402


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """把 SSE 文本解析成事件名和 JSON 数据。"""

    events: list[tuple[str, dict]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


class AgentFlowTests(unittest.TestCase):
    """覆盖草稿确认、人工复核、角色权限和跳过校验四个关键边界。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 某些 Windows 沙箱会禁止 SQLite 在系统 Temp 下建库，因此把测试
        # 数据放到仓库内的短生命周期目录，测试结束自动清理。
        test_root = Path(os.getenv("AGENT_TEST_ROOT", str(REPO_ROOT)))
        test_root.mkdir(parents=True, exist_ok=True)
        cls.data_dir = test_root / f".agent-test-{uuid.uuid4().hex}"
        cls.data_dir.mkdir(parents=True)
        data_dir = cls.data_dir
        os.environ["DATA_DIR"] = str(data_dir)
        os.environ["SQLITE_PATH"] = str(data_dir / "test.sqlite3")
        os.environ["RAG_CACHE_DIR"] = str(data_dir / "faiss_cache")
        os.environ["UPLOAD_DIR"] = str(data_dir / "uploads")
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def login(self, username: str, password: str) -> str:
        response = self.client.post(
            "/api/auth/token",
            json={"username": username, "password": password},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["data"]["access_token"]

    def chat(self, token: str, question: str) -> list[tuple[str, dict]]:
        with self.client.stream(
            "POST",
            "/chat",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "text/event-stream",
            },
            json={
                "session_id": str(uuid.uuid4()),
                "question": question,
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            raw = "".join(response.iter_text())
        return parse_sse(raw)

    def test_reimbursement_draft_is_submitted_only_after_confirmation(self) -> None:
        token = self.login("employee", "Employee@123456")
        events = self.chat(
            token,
            (
                "我要提交一笔 800 元差旅报销，附件 file_inv001，"
                "发生日期 2026-10-20，请帮我建单"
            ),
        )
        event_names = [name for name, _ in events]
        self.assertIn("tool", event_names)
        self.assertIn("draft", event_names)
        self.assertEqual(event_names[-1], "done")

        draft = next(data for name, data in events if name == "draft")
        self.assertEqual(draft["status"], "draft")
        self.assertGreaterEqual(draft["draft_version"], 1)

        no_consent = self.client.post(
            f"/api/tickets/{draft['draft_id']}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json={"confirmed": False, "draft_version": draft["draft_version"]},
        )
        self.assertEqual(no_consent.status_code, 422)
        self.assertEqual(
            no_consent.json()["code"],
            "EXPLICIT_CONFIRMATION_REQUIRED",
        )

        confirmed = self.client.post(
            f"/api/tickets/{draft['draft_id']}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json={"confirmed": True, "draft_version": draft["draft_version"]},
        )
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        self.assertEqual(confirmed.json()["data"]["status"], "submitted")

        stored = asyncio.run(
            app.state.db.get_ticket_status(
                ticket_id=draft["draft_id"],
                user_id="employee-demo",
            )
        )
        self.assertIsNotNone(stored)
        self.assertEqual(stored["status"], "submitted")

        repeated = self.client.post(
            f"/api/tickets/{draft['draft_id']}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json={"confirmed": True, "draft_version": draft["draft_version"]},
        )
        self.assertEqual(repeated.status_code, 409)

    def test_high_value_reimbursement_routes_to_human_review(self) -> None:
        token = self.login("employee", "Employee@123456")
        events = self.chat(
            token,
            (
                "我要提交一笔 5000 元差旅报销，附件 file_inv999，"
                "发生日期 2026-10-20，请帮我建单"
            ),
        )
        self.assertNotIn("draft", [name for name, _ in events])
        answer = "".join(
            data.get("text", "")
            for name, data in events
            if name == "delta"
        )
        self.assertIn("人工", answer)

        async def find_handoff() -> dict | None:
            async with app.state.db._connect() as connection:
                cursor = await connection.execute(
                    """
                    SELECT *
                    FROM ticket_drafts
                    WHERE category = 'human_review'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                row = await cursor.fetchone()
                return dict(row) if row else None

        handoff = asyncio.run(find_handoff())
        self.assertIsNotNone(handoff)
        self.assertEqual(handoff["priority"], "high")

    def test_admin_cannot_confirm_employee_draft(self) -> None:
        employee_token = self.login("employee", "Employee@123456")
        events = self.chat(
            employee_token,
            (
                "我要提交一笔 600 元差旅报销，附件 file_inv777，"
                "发生日期 2026-10-20，请帮我建单"
            ),
        )
        draft = next(data for name, data in events if name == "draft")

        admin_token = self.login("admin", "Admin@123456")
        response = self.client.post(
            f"/api/tickets/{draft['draft_id']}/confirm",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"confirmed": True, "draft_version": draft["draft_version"]},
        )
        self.assertEqual(response.status_code, 403)

    def test_reimbursement_cannot_bypass_invoice_validation(self) -> None:
        """即使有人伪造出合法的 routing 结构，缺少发票校验也必须被拒绝。"""

        async def run_check() -> tuple[dict, dict, dict]:
            executor = AgentToolExecutor(
                database=app.state.db,
                runtime=app.state.runtime,
                token_service=app.state.agent_tokens,
                user={
                    "id": "employee-demo",
                    "role": "employee",
                },
                conversation_id="security-boundary-case",
            )
            forged_routing = app.state.agent_tokens.create(
                token_type="routing",
                user_id="employee-demo",
                case_id="security-boundary-case",
                payload={
                    "case_type": "reimbursement",
                    "policy_jti": "policy",
                    "invoice_jti": None,
                    "queue": "finance_review",
                    "priority": "normal",
                },
            )
            draft_result = await executor.execute(
                call_id="forged-create",
                name="create_ticket_draft",
                arguments={
                    "routing_token": forged_routing,
                    "title": "越权草稿",
                    "description": "试图跳过发票校验",
                    "attachment_ids": [],
                },
            )
            route_result = await executor.execute(
                call_id="direct-route",
                name="route_ticket",
                arguments={
                    "case_type": "reimbursement",
                    "policy_validation_token": forged_routing,
                    "invoice_validation_token": None,
                    "risk_flags": [],
                },
            )
            mismatch_executor = AgentToolExecutor(
                database=app.state.db,
                runtime=app.state.runtime,
                token_service=app.state.agent_tokens,
                user={
                    "id": "employee-demo",
                    "role": "employee",
                },
                conversation_id="security-boundary-case",
                request_question="我要提交差旅报销发票",
            )
            valid_policy_token = app.state.agent_tokens.create(
                token_type="policy",
                user_id="employee-demo",
                case_id="security-boundary-case",
                payload={"evidence_ids": [1]},
            )
            mismatch_result = await mismatch_executor.execute(
                call_id="mislabel-case",
                name="route_ticket",
                arguments={
                    "case_type": "salary_benefit",
                    "policy_validation_token": valid_policy_token,
                    "invoice_validation_token": None,
                    "risk_flags": [],
                },
            )
            return (
                draft_result.model_payload,
                route_result.model_payload,
                mismatch_result.model_payload,
            )

        draft_payload, route_payload, mismatch_payload = asyncio.run(run_check())
        self.assertFalse(draft_payload["ok"])
        self.assertEqual(
            draft_payload["code"],
            "INVOICE_VALIDATION_REQUIRED",
        )
        self.assertFalse(route_payload["ok"])
        self.assertIn(
            route_payload["code"],
            {"POLICY_VALIDATION_REQUIRED", "INVOICE_VALIDATION_REQUIRED"},
        )
        self.assertFalse(mismatch_payload["ok"])
        self.assertEqual(mismatch_payload["code"], "CASE_TYPE_MISMATCH")


if __name__ == "__main__":
    unittest.main()
