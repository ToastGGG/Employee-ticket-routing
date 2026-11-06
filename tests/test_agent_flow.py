"""薪酬福利咨询 Agent Harness 的端到端回归测试。"""

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

from src.employee_ticket_routing.core.agent_runtime import AgentToolExecutor  # noqa: E402
from src.employee_ticket_routing.main import app  # noqa: E402


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
    """覆盖三种典型对话、确认中断、审计日志和越权边界。"""

    @classmethod
    def setUpClass(cls) -> None:
        test_root = Path(os.getenv("AGENT_TEST_ROOT", str(REPO_ROOT)))
        test_root.mkdir(parents=True, exist_ok=True)
        cls.data_dir = test_root / f".agent-test-{uuid.uuid4().hex}"
        cls.data_dir.mkdir(parents=True)
        os.environ["DATA_DIR"] = str(cls.data_dir)
        os.environ["SQLITE_PATH"] = str(cls.data_dir / "test.sqlite3")
        os.environ["RAG_CACHE_DIR"] = str(cls.data_dir / "faiss_cache")
        os.environ["UPLOAD_DIR"] = str(cls.data_dir / "uploads")
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        os.environ["AGENT_MAX_ROUNDS"] = "5"
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

    @staticmethod
    def done_data(events: list[tuple[str, dict]]) -> dict:
        return next(data for name, data in events if name == "done")

    @staticmethod
    def answer_text(events: list[tuple[str, dict]]) -> str:
        return "".join(
            str(data.get("text") or "")
            for name, data in events
            if name == "delta"
        )

    async def ticket_count(self, conversation_id: str) -> int:
        async with app.state.db._connect() as connection:
            cursor = await connection.execute(
                "SELECT COUNT(*) FROM ticket_drafts WHERE conversation_id = ?",
                (conversation_id,),
            )
            return int((await cursor.fetchone())[0])

    def test_new_hire_policy_question_only_queries_and_answers(self) -> None:
        """典型轨迹 a：制度查询只解释政策，不进入建单链路。"""

        token = self.login("employee", "Employee@123456")
        events = self.chat(token, "新人入职薪酬福利是多少")
        names = [name for name, _ in events]
        self.assertNotIn("confirmation_required", names)
        self.assertIn("15", self.answer_text(events))
        conversation_id = self.done_data(events)["conversation_id"]
        self.assertEqual(
            asyncio.run(self.ticket_count(conversation_id)),
            0,
        )

        logs = asyncio.run(
            app.state.db.list_agent_decision_logs(
                conversation_id=conversation_id
            )
        )
        tools = [row["tool_name"] for row in logs if row["tool_name"]]
        self.assertIn("search_knowledge", tools)
        self.assertIn("validate_policy", tools)
        self.assertNotIn("create_ticket_draft", tools)
        self.assertLessEqual(max(row["round_no"] for row in logs), 5)

    def test_vague_submit_request_asks_for_requirement(self) -> None:
        """典型轨迹 b 的前半段：只说建单但缺少诉求时必须先追问。"""

        token = self.login("employee", "Employee@123456")
        events = self.chat(token, "帮我提交我的咨询工单")
        self.assertNotIn(
            "confirmation_required",
            [name for name, _ in events],
        )
        self.assertIn("请补充", self.answer_text(events))

    def test_detailed_submit_is_interrupted_until_employee_confirms(self) -> None:
        """典型轨迹 b：确认前不执行 create_ticket_draft，确认后创建并提交。"""

        token = self.login("employee", "Employee@123456")
        events = self.chat(
            token,
            (
                "帮我提交我的咨询工单，我是新员工，"
                "想确认入职首月薪酬和社保公积金基数"
            ),
        )
        self.assertIn(
            "confirmation_required",
            [name for name, _ in events],
        )
        proposal = next(
            data for name, data in events if name == "confirmation_required"
        )
        done = self.done_data(events)
        self.assertEqual(done["status"], "waiting_confirmation")
        self.assertEqual(
            asyncio.run(self.ticket_count(done["conversation_id"])),
            0,
        )

        rejected = self.client.post(
            f"/api/agent/confirmations/{proposal['confirmation_id']}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json={"confirmed": False},
        )
        self.assertEqual(rejected.status_code, 422)

        confirmed = self.client.post(
            f"/api/agent/confirmations/{proposal['confirmation_id']}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json={"confirmed": True},
        )
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        self.assertEqual(confirmed.json()["data"]["status"], "submitted")

        stored = asyncio.run(
            app.state.db.get_ticket_status(
                ticket_id=confirmed.json()["data"]["draft_id"],
                user_id="employee-demo",
            )
        )
        self.assertEqual(stored["status"], "submitted")

        logs = asyncio.run(
            app.state.db.list_agent_decision_logs(
                conversation_id=done["conversation_id"]
            )
        )
        decisions = [row["decision_type"] for row in logs]
        self.assertIn("confirmation_required", decisions)
        self.assertIn("confirmation_execution", decisions)
        self.assertLessEqual(max(row["round_no"] for row in logs), 5)
        tool_logs = [row for row in logs if row["tool_name"]]
        self.assertTrue(tool_logs)
        for row in tool_logs:
            self.assertIsNotNone(row["arguments_json"])
            self.assertGreaterEqual(row["duration_ms"], 0)
            self.assertTrue(row["result_summary"])
            self.assertNotIn("_token", row["arguments_json"])

        repeated = self.client.post(
            f"/api/agent/confirmations/{proposal['confirmation_id']}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json={"confirmed": True},
        )
        self.assertEqual(repeated.status_code, 404)

    def test_can_i_submit_question_returns_conclusion_without_ticket(self) -> None:
        """典型轨迹 c：询问“能否提交”只输出结论，不误解为执行指令。"""

        token = self.login("employee", "Employee@123456")
        events = self.chat(token, "怎么能不能提交某方面的工单")
        self.assertNotIn(
            "confirmation_required",
            [name for name, _ in events],
        )
        self.assertTrue(self.answer_text(events).strip())

    def test_high_risk_consultation_routes_to_human_review(self) -> None:
        """劳动争议类事项不能走普通建单确认流程。"""

        token = self.login("employee", "Employee@123456")
        events = self.chat(
            token,
            "帮我提交我的咨询工单，我想就绩效奖金争议提出申诉",
        )
        self.assertNotIn(
            "confirmation_required",
            [name for name, _ in events],
        )
        self.assertIn("人工", self.answer_text(events))

        async def latest_handoff() -> dict | None:
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

        handoff = asyncio.run(latest_handoff())
        self.assertIsNotNone(handoff)
        self.assertEqual(handoff["priority"], "high")

    def test_admin_cannot_confirm_employee_proposal(self) -> None:
        employee_token = self.login("employee", "Employee@123456")
        events = self.chat(
            employee_token,
            "帮我提交我的咨询工单，想咨询补充商业保险覆盖家属范围",
        )
        proposal = next(
            data for name, data in events if name == "confirmation_required"
        )
        admin_token = self.login("admin", "Admin@123456")
        response = self.client.post(
            f"/api/agent/confirmations/{proposal['confirmation_id']}/confirm",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"confirmed": True},
        )
        self.assertEqual(response.status_code, 403)

    def test_create_draft_rejects_forged_routing_without_consultation(self) -> None:
        """即使伪造签名正确的 routing 结构，缺少咨询校验也必须拒绝。"""

        async def run_check() -> tuple[dict, dict]:
            executor = AgentToolExecutor(
                database=app.state.db,
                runtime=app.state.runtime,
                token_service=app.state.agent_tokens,
                user={"id": "employee-demo", "role": "employee"},
                conversation_id="security-boundary-case",
            )
            forged_routing = app.state.agent_tokens.create(
                token_type="routing",
                user_id="employee-demo",
                case_id="security-boundary-case",
                payload={
                    "case_type": "compensation",
                    "policy_jti": "policy",
                    "consultation_jti": None,
                    "queue": "compensation_benefits",
                    "priority": "normal",
                },
            )
            create_result = await executor.execute(
                call_id="forged-create",
                name="create_ticket_draft",
                arguments={
                    "routing_token": forged_routing,
                    "title": "越权工单",
                    "description": "试图跳过咨询资料校验",
                    "attachment_ids": [],
                },
            )
            policy_token = app.state.agent_tokens.create(
                token_type="policy",
                user_id="employee-demo",
                case_id="security-boundary-case",
                payload={"evidence_ids": [1]},
            )
            route_result = await executor.execute(
                call_id="direct-route",
                name="route_ticket",
                arguments={
                    "case_type": "compensation",
                    "policy_validation_token": policy_token,
                    "consultation_validation_token": "",
                    "risk_flags": [],
                },
            )
            return create_result.model_payload, route_result.model_payload

        create_payload, route_payload = asyncio.run(run_check())
        self.assertFalse(create_payload["ok"])
        self.assertEqual(
            create_payload["code"],
            "CONSULTATION_VALIDATION_REQUIRED",
        )
        self.assertFalse(route_payload["ok"])
        self.assertEqual(
            route_payload["code"],
            "CONSULTATION_VALIDATION_REQUIRED",
        )


if __name__ == "__main__":
    unittest.main()
