"""
SQLite 数据访问层。

本阶段使用 aiosqlite 提供异步数据库访问。表结构沿用此前确定的最小模型：
users、conversations、messages、knowledge_documents、document_chunks、
ticket_drafts。

当前数据量很小，因此采用轻量模块化单体结构。后续如果迁移到 PostgreSQL，
页面、路由和流式响应层不需要改变，只需要替换本文件中的仓储实现。
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


def utc_now() -> str:
    """返回统一格式的 UTC ISO 时间，数据库中的所有时间字段都使用该格式。"""

    return datetime.now(timezone.utc).isoformat()


class SQLiteDatabase:
    """
    【核心基础设施｜后续可替换为 PostgreSQL】

    本类集中管理建表和查询，避免业务路由直接拼接 SQL。
    """

    def __init__(self, database_path: Path):
        self.database_path = database_path

    @asynccontextmanager
    async def _connect(self):
        """
        创建数据库连接并设置 SQLite 推荐参数。

        WAL 模式允许读请求与写请求并行；foreign_keys 保证消息不会引用不存在的会话。
        连接由调用方通过 async with 管理，防止请求结束后连接泄漏。
        """

        connection = await aiosqlite.connect(self.database_path)
        connection.row_factory = aiosqlite.Row
        try:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
        finally:
            await connection.close()

    async def initialize(self) -> None:
        """
        创建数据目录和数据表。

        CREATE TABLE IF NOT EXISTS 使服务可以重复启动。PRAGMA journal_mode=WAL
        属于数据库级设置，不会因为重复执行而清空数据。
        """

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    employee_no TEXT UNIQUE,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('employee', 'admin', 'auditor')),
                    status TEXT NOT NULL DEFAULT 'active',
                    password_hash TEXT,
                    token_version INTEGER NOT NULL DEFAULT 1,
                    last_login_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
                    content TEXT NOT NULL DEFAULT '',
                    citations_json TEXT,
                    status TEXT NOT NULL DEFAULT 'completed'
                        CHECK (status IN ('streaming', 'completed', 'failed', 'interrupted')),
                    model_provider TEXT,
                    model_name TEXT,
                    request_id TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT '1.0',
                    status TEXT NOT NULL DEFAULT 'active',
                    original_filename TEXT,
                    storage_path TEXT,
                    file_hash TEXT,
                    file_size INTEGER,
                    mime_type TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    effective_from TEXT,
                    effective_to TEXT,
                    created_by TEXT,
                    deleted_by TEXT,
                    deleted_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    vector_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES knowledge_documents(id),
                    UNIQUE (document_id, vector_id)
                );

                CREATE TABLE IF NOT EXISTS ticket_drafts (
                    id TEXT PRIMARY KEY,
                    draft_no TEXT UNIQUE,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    source_message_id TEXT,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    details TEXT NOT NULL DEFAULT '',
                    category TEXT,
                    priority TEXT,
                    fields_json TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    draft_version INTEGER NOT NULL DEFAULT 1,
                    validation_json TEXT,
                    expires_at TEXT,
                    confirmed_at TEXT,
                    confirmation_source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'archived')),
                    created_by TEXT NOT NULL,
                    change_note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'parsing', 'chunking', 'embedding',
                                   'indexing', 'ready', 'failed')
                    ),
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY (document_id) REFERENCES knowledge_documents(id)
                );

                CREATE TABLE IF NOT EXISTS audit_flags (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    ticket_draft_id TEXT,
                    category TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'revoked')),
                    marked_by TEXT NOT NULL,
                    marked_at TEXT NOT NULL,
                    revoked_by TEXT,
                    revoked_at TEXT,
                    revoke_reason TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY (ticket_draft_id) REFERENCES ticket_drafts(id)
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    actor_id TEXT,
                    actor_role TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    before_json TEXT,
                    after_json TEXT,
                    request_id TEXT,
                    ip_address TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_token_usage (
                    jti TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    consumed_at TEXT NOT NULL,
                    purpose TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                    ON conversations(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_request
                    ON messages(request_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_document_vector
                    ON document_chunks(document_id, vector_id);
                CREATE INDEX IF NOT EXISTS idx_prompt_templates_status
                    ON prompt_templates(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_document
                    ON ingestion_jobs(document_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_flags_conversation
                    ON audit_flags(conversation_id, status, marked_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_logs_created
                    ON audit_logs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ticket_drafts_user
                    ON ticket_drafts(user_id, created_at DESC);
                """
            )
            # 对阶段 1 已存在的 SQLite 表做向前兼容迁移。
            await self._ensure_column(connection, "users", "password_hash", "TEXT")
            await self._ensure_column(
                connection, "users", "token_version", "INTEGER NOT NULL DEFAULT 1"
            )
            await self._ensure_column(connection, "users", "last_login_at", "TEXT")
            await self._ensure_column(
                connection, "knowledge_documents", "original_filename", "TEXT"
            )
            await self._ensure_column(
                connection, "knowledge_documents", "storage_path", "TEXT"
            )
            await self._ensure_column(
                connection, "knowledge_documents", "file_hash", "TEXT"
            )
            await self._ensure_column(
                connection, "knowledge_documents", "file_size", "INTEGER"
            )
            await self._ensure_column(
                connection, "knowledge_documents", "mime_type", "TEXT"
            )
            await self._ensure_column(
                connection,
                "knowledge_documents",
                "chunk_count",
                "INTEGER NOT NULL DEFAULT 0",
            )
            await self._ensure_column(
                connection, "knowledge_documents", "created_by", "TEXT"
            )
            await self._ensure_column(
                connection, "knowledge_documents", "deleted_by", "TEXT"
            )
            await self._ensure_column(
                connection, "knowledge_documents", "deleted_at", "TEXT"
            )
            await self._ensure_column(
                connection, "ticket_drafts", "draft_version", "INTEGER NOT NULL DEFAULT 1"
            )
            await self._ensure_column(
                connection, "ticket_drafts", "validation_json", "TEXT"
            )
            await self._ensure_column(connection, "ticket_drafts", "expires_at", "TEXT")
            await self._ensure_column(
                connection, "ticket_drafts", "confirmed_at", "TEXT"
            )
            await self._ensure_column(
                connection, "ticket_drafts", "confirmation_source", "TEXT"
            )
            await connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash
                ON knowledge_documents(file_hash)
                WHERE file_hash IS NOT NULL
                """
            )
            # 阶段 1 使用 active 表示可检索；新阶段统一为 ready。
            await connection.execute(
                """
                UPDATE knowledge_documents
                SET status = 'ready'
                WHERE status = 'active'
                """
            )
            await connection.execute(
                """
                UPDATE knowledge_documents
                SET chunk_count = (
                    SELECT COUNT(*) FROM document_chunks dc
                    WHERE dc.document_id = knowledge_documents.id
                )
                WHERE chunk_count = 0
                """
            )
            await connection.commit()

    @staticmethod
    async def _ensure_column(
        connection: aiosqlite.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """
        SQLite 的 ADD COLUMN 不具备 IF NOT EXISTS，因此先检查现有列。

        这些迁移保证前一阶段创建的数据库可以直接升级，不会清空已有会话。
        """

        cursor = await connection.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        names = {str(row["name"]) for row in rows}
        if column not in names:
            await connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    async def ensure_demo_employee(
        self,
        *,
        user_id: str,
        employee_no: str,
        name: str,
    ) -> dict[str, Any]:
        """
        初始化本地演示员工。

        生产阶段应改为根据 SSO Subject 查询或创建用户；当前接口从
        X-Demo-User-Id 读取演示身份，仅用于本地开发。
        """

        now = utc_now()
        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute(
                """
                INSERT INTO users
                    (id, employee_no, name, role, status, created_at, updated_at)
                VALUES (?, ?, ?, 'employee', 'active', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    employee_no = excluded.employee_no,
                    name = excluded.name,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (user_id, employee_no, name, now, now),
            )
            await connection.commit()
            row = await self._fetchone(
                connection,
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            )
            if row is None:
                raise RuntimeError("演示员工创建失败")
            return dict(row)

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """按用户 ID 读取员工信息；不存在时返回 None。"""

        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            row = await self._fetchone(
                connection,
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            )
        return dict(row) if row is not None else None

    async def get_user_by_employee_no(
        self,
        employee_no: str,
    ) -> dict[str, Any] | None:
        """管理后台登录以员工工号/登录名查询用户。"""

        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                "SELECT * FROM users WHERE employee_no = ?",
                (employee_no,),
            )
        return dict(row) if row is not None else None

    async def upsert_seed_user(
        self,
        *,
        user_id: str,
        employee_no: str,
        name: str,
        role: str,
        password_hash: str,
    ) -> dict[str, Any]:
        """
        初始化或更新演示账号。

        生产环境不应在每次启动时覆盖密码；当前实现仅服务于本地最小可用环境，
        管理员首次部署后应通过环境变量设置独立密码。
        """

        if role not in {"employee", "admin"}:
            raise ValueError("角色只能是 employee 或 admin")
        now = utc_now()
        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO users
                    (
                        id, employee_no, name, role, status, password_hash,
                        token_version, created_at, updated_at
                    )
                VALUES (?, ?, ?, ?, 'active', ?, 1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    employee_no = excluded.employee_no,
                    name = excluded.name,
                    role = excluded.role,
                    status = 'active',
                    password_hash = excluded.password_hash,
                    updated_at = excluded.updated_at
                """,
                (user_id, employee_no, name, role, password_hash, now, now),
            )
            await connection.commit()
        user = await self.get_user(user_id)
        if user is None:
            raise RuntimeError("演示账号创建失败")
        return user

    async def touch_last_login(self, user_id: str) -> None:
        """记录登录时间。登录失败不更新该字段。"""

        async with self._connect() as connection:
            await connection.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?",
                (utc_now(), utc_now(), user_id),
            )
            await connection.commit()

    async def list_employees(self) -> list[dict[str, Any]]:
        """返回审计筛选所需的员工列表。"""

        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                SELECT id, employee_no, name
                FROM users
                WHERE role = 'employee'
                ORDER BY name, employee_no
                """
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def ensure_conversation(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """
        获取会话；如果前端第一次发送新的 UUID，则自动创建。

        这里必须校验会话归属，防止员工通过修改 session_id 查看其他员工的会话。
        """

        now = utc_now()
        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            existing = await self._fetchone(
                connection,
                "SELECT * FROM conversations WHERE id = ?",
                (session_id,),
            )
            if existing is not None:
                if existing["user_id"] != user_id:
                    raise PermissionError("当前会话属于其他员工")
                return dict(existing)

            await connection.execute(
                """
                INSERT INTO conversations
                    (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, "员工薪酬福利咨询", now, now),
            )
            await connection.commit()
            existing = await self._fetchone(
                connection,
                "SELECT * FROM conversations WHERE id = ?",
                (session_id,),
            )
            if existing is None:
                raise RuntimeError("会话创建失败")
            return dict(existing)

    async def get_recent_history(
        self,
        *,
        conversation_id: str,
        rounds: int = 6,
    ) -> list[dict[str, str]]:
        """
        获取最近 N 轮已完成历史。

        一轮由一条 user 消息和一条 assistant 消息组成。先倒序取 2N 条，
        再反转为正序，保证模型看到的时间顺序正确。
        """

        limit = max(rounds, 0) * 2
        if limit == 0:
            return []

        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            cursor = await connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                  AND role IN ('user', 'assistant')
                  AND status = 'completed'
                  AND content <> ''
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = await cursor.fetchall()

        ordered = list(reversed(rows))
        return [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in ordered
        ]

    async def create_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        status: str,
        request_id: str,
        model_provider: str | None = None,
        model_name: str | None = None,
    ) -> str:
        """
        创建消息记录。

        assistant 占位消息在流开始前写入，因此即使服务进程中途退出，
        也能识别出有一次未完成的回答。
        """

        message_id = str(uuid.uuid4())
        now = utc_now()
        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute(
                """
                INSERT INTO messages
                    (
                        id, conversation_id, role, content, status,
                        model_provider, model_name, request_id, created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    status,
                    model_provider,
                    model_name,
                    request_id,
                    now,
                ),
            )
            await connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            await connection.commit()
        return message_id

    async def complete_message(
        self,
        *,
        message_id: str,
        content: str,
        citations: list[dict[str, Any]],
        status: str = "completed",
        error_code: str | None = None,
    ) -> None:
        """保存最终回答、引用和完成状态；失败或中断时也通过同一方法落库。"""

        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute(
                """
                UPDATE messages
                SET content = ?,
                    citations_json = ?,
                    status = ?,
                    error_code = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (
                    content,
                    json.dumps(citations, ensure_ascii=False),
                    status,
                    error_code,
                    utc_now(),
                    message_id,
                ),
            )
            await connection.commit()

    async def consume_tool_token(
        self,
        *,
        jti: str,
        user_id: str,
        case_id: str,
        token_type: str,
        purpose: str,
    ) -> bool:
        """
        记录一次性工具令牌的使用。

        主键 jti 冲突说明令牌被重复使用；服务端据此拒绝重放。
        """

        async with self._connect() as connection:
            try:
                await connection.execute(
                    """
                    INSERT INTO tool_token_usage
                        (jti, user_id, case_id, token_type, consumed_at, purpose)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (jti, user_id, case_id, token_type, utc_now(), purpose),
                )
                await connection.commit()
                return True
            except aiosqlite.IntegrityError:
                await connection.rollback()
                return False

    async def create_ticket_draft(
        self,
        *,
        user_id: str,
        conversation_id: str,
        title: str,
        summary: str,
        details: str,
        category: str,
        priority: str,
        fields: dict[str, Any],
        validation: dict[str, Any],
        expires_at: str,
    ) -> dict[str, Any]:
        """
        创建可编辑、可确认但尚未提交的工单草稿。

        validation_json 只保存校验凭证摘要和证据编号，不保存完整授权 Token。
        """

        draft_id = str(uuid.uuid4())
        draft_no = f"DR-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
        now = utc_now()
        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO ticket_drafts
                    (
                        id, draft_no, user_id, conversation_id, title,
                        summary, details, category, priority, fields_json,
                        status, draft_version, validation_json, expires_at,
                        created_at, updated_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', 1, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    draft_no,
                    user_id,
                    conversation_id,
                    title,
                    summary,
                    details,
                    category,
                    priority,
                    json.dumps(fields, ensure_ascii=False),
                    json.dumps(validation, ensure_ascii=False),
                    expires_at,
                    now,
                    now,
                ),
            )
            await connection.commit()
        return {
            "draft_id": draft_id,
            "draft_no": draft_no,
            "draft_version": 1,
            "status": "draft",
            "expires_at": expires_at,
        }

    async def get_ticket_draft_for_user(
        self,
        *,
        draft_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """读取当前员工自己的工单草稿。"""

        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT * FROM ticket_drafts
                WHERE id = ? AND user_id = ?
                """,
                (draft_id, user_id),
            )
        return dict(row) if row is not None else None

    async def submit_ticket_draft(
        self,
        *,
        draft_id: str,
        user_id: str,
        expected_version: int,
    ) -> dict[str, Any] | None:
        """
        将草稿标记为已提交。

        只有草稿所有者、同一版本且未过期的普通工单草稿才能提交。
        “显式确认”是独立于模型工具调用的服务端状态转换；模型不能替代员工点击确认。
        """

        now = utc_now()
        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT *
                FROM ticket_drafts
                WHERE id = ? AND user_id = ?
                """,
                (draft_id, user_id),
            )
            if row is None:
                return None
            draft = dict(row)
            if (
                draft.get("status") != "draft"
                or int(draft.get("draft_version") or 0) != expected_version
                or draft.get("category") == "human_review"
            ):
                return None
            try:
                expires_at = datetime.fromisoformat(str(draft.get("expires_at") or ""))
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
            if expires_at <= datetime.now(timezone.utc):
                return None
            try:
                validation = json.loads(str(draft.get("validation_json") or "{}"))
            except json.JSONDecodeError:
                return None
            digest = str(validation.get("validation_digest") or "")
            if len(digest) != 64:
                return None

            cursor = await connection.execute(
                """
                UPDATE ticket_drafts
                SET status = 'submitted',
                    submitted_at = ?,
                    confirmed_at = ?,
                    confirmation_source = 'employee_explicit',
                    updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND status = 'draft'
                  AND draft_version = ?
                  AND expires_at > ?
                """,
                (
                    now,
                    now,
                    now,
                    draft_id,
                    user_id,
                    expected_version,
                    now,
                ),
            )
            if cursor.rowcount == 0:
                await connection.rollback()
                return None
            await connection.commit()
            row = await self._fetchone(
                connection,
                "SELECT * FROM ticket_drafts WHERE id = ?",
                (draft_id,),
            )
        return dict(row) if row is not None else None

    async def get_ticket_status(
        self,
        *,
        ticket_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """返回当前员工的工单状态，禁止跨员工查询。"""

        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT
                    id, draft_no, title, summary, status, priority,
                    category, created_at, updated_at, submitted_at
                FROM ticket_drafts
                WHERE id = ? AND user_id = ?
                """,
                (ticket_id, user_id),
            )
        return dict(row) if row is not None else None

    async def upsert_knowledge_document(
        self,
        *,
        document_id: str,
        file_name: str,
        title: str,
        version: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        同步知识文档元数据和切片到 SQLite。

        FAISS 只负责保存向量和整数 vector_id；SQLite 保存同一个 vector_id
        对应的原文、文件信息和引用展示数据。
        """

        now = utc_now()
        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute(
                    """
                    INSERT INTO knowledge_documents
                        (id, file_name, title, version, status, created_at)
                    VALUES (?, ?, ?, ?, 'active', ?)
                    ON CONFLICT(id) DO UPDATE SET
                        file_name = excluded.file_name,
                        title = excluded.title,
                        version = excluded.version,
                        status = 'active'
                    """,
                    (document_id, file_name, title, version, now),
                )
                await connection.execute(
                    "DELETE FROM document_chunks WHERE document_id = ?",
                    (document_id,),
                )
                await connection.executemany(
                    """
                    INSERT INTO document_chunks
                        (
                            document_id, vector_id, chunk_index, content,
                            char_start, char_end, created_at
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            document_id,
                            int(chunk["vector_id"]),
                            int(chunk["chunk_index"]),
                            str(chunk["content"]),
                            int(chunk["char_start"]),
                            int(chunk["char_end"]),
                            now,
                        )
                        for chunk in chunks
                    ],
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise

    async def get_source_rows(
        self,
        *,
        document_id: str,
        vector_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """
        按 FAISS vector_id 查询来源原文和文件信息。

        返回字典而不是列表，调用方可以保持 FAISS 的排序结果不变。
        """

        if not vector_ids:
            return {}
        placeholders = ",".join("?" for _ in vector_ids)
        params: list[Any] = [document_id, *vector_ids]
        async with self._connect() as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA busy_timeout = 5000")
            cursor = await connection.execute(
                f"""
                SELECT
                    dc.id AS chunk_db_id,
                    dc.vector_id,
                    dc.chunk_index,
                    dc.content,
                    dc.char_start,
                    dc.char_end,
                    kd.id AS document_id,
                    kd.file_name,
                    kd.title,
                    kd.version
                FROM document_chunks dc
                JOIN knowledge_documents kd ON kd.id = dc.document_id
                WHERE dc.document_id = ?
                  AND dc.vector_id IN ({placeholders})
                """,
                tuple(params),
            )
            rows = await cursor.fetchall()
        return {int(row["vector_id"]): dict(row) for row in rows}

    async def create_knowledge_document(
        self,
        *,
        document_id: str,
        file_name: str,
        title: str,
        version: str,
        file_hash: str,
        storage_path: str,
        file_size: int,
        mime_type: str,
        created_by: str,
        status: str = "processing",
    ) -> None:
        """创建待入库文档记录；文件哈希由唯一索引防止重复上传。"""

        now = utc_now()
        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO knowledge_documents
                    (
                        id, file_name, title, version, status, original_filename,
                        storage_path, file_hash, file_size, mime_type,
                        chunk_count, created_by, created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    document_id,
                    file_name,
                    title,
                    version,
                    status,
                    file_name,
                    storage_path,
                    file_hash,
                    file_size,
                    mime_type,
                    created_by,
                    now,
                ),
            )
            await connection.commit()

    async def find_document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """按文件内容哈希查找重复文档。"""

        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                "SELECT * FROM knowledge_documents WHERE file_hash = ?",
                (file_hash,),
            )
        return dict(row) if row is not None else None

    async def replace_document_chunks(
        self,
        *,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[int]:
        """
        保存文档切片并返回 SQLite 生成的主键。

        这些主键随后作为 FAISS vector_id，因此引用可以稳定映射回原文。
        """

        now = utc_now()
        inserted_ids: list[int] = []
        async with self._connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute(
                    "DELETE FROM document_chunks WHERE document_id = ?",
                    (document_id,),
                )
                for chunk in chunks:
                    chunk_index = int(chunk["chunk_index"])
                    cursor = await connection.execute(
                        """
                        INSERT INTO document_chunks
                            (
                                document_id, vector_id, chunk_index, content,
                                char_start, char_end, created_at
                            )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            # 先用同一文档内唯一的负值占位，避免批量插入时
                            # (document_id, vector_id=0) 触发唯一约束。
                            -(chunk_index + 1),
                            chunk_index,
                            str(chunk["content"]),
                            int(chunk["char_start"]),
                            int(chunk["char_end"]),
                            now,
                        ),
                    )
                    inserted_ids.append(int(cursor.lastrowid))
                # SQLite 主键生成后再回填 vector_id，确保全局唯一。
                await connection.executemany(
                    "UPDATE document_chunks SET vector_id = ? WHERE id = ?",
                    [(chunk_id, chunk_id) for chunk_id in inserted_ids],
                )
                await connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET chunk_count = ?
                    WHERE id = ?
                    """,
                    (len(inserted_ids), document_id),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        return inserted_ids

    async def update_document_status(
        self,
        *,
        document_id: str,
        status: str,
    ) -> None:
        """更新知识文档生命周期状态。"""

        async with self._connect() as connection:
            await connection.execute(
                "UPDATE knowledge_documents SET status = ? WHERE id = ?",
                (status, document_id),
            )
            await connection.commit()

    async def soft_delete_document(
        self,
        *,
        document_id: str,
        deleted_by: str,
    ) -> bool:
        """
        软删除制度文件。

        历史消息的 citations_json 已保存快照，因此删除文档不会破坏旧审计记录。
        """

        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                UPDATE knowledge_documents
                SET status = 'deleted', deleted_by = ?, deleted_at = ?
                WHERE id = ? AND status <> 'deleted'
                """,
                (deleted_by, utc_now(), document_id),
            )
            await connection.commit()
            return cursor.rowcount > 0

    async def list_knowledge_documents(self) -> list[dict[str, Any]]:
        """管理页文档列表，包含当前处理任务状态。"""

        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    kd.*,
                    u.name AS created_by_name,
                    (
                        SELECT ij.id
                        FROM ingestion_jobs ij
                        WHERE ij.document_id = kd.id
                        ORDER BY ij.created_at DESC
                        LIMIT 1
                    ) AS job_id,
                    (
                        SELECT ij.status
                        FROM ingestion_jobs ij
                        WHERE ij.document_id = kd.id
                        ORDER BY ij.created_at DESC
                        LIMIT 1
                    ) AS job_status,
                    (
                        SELECT ij.stage
                        FROM ingestion_jobs ij
                        WHERE ij.document_id = kd.id
                        ORDER BY ij.created_at DESC
                        LIMIT 1
                    ) AS job_stage,
                    (
                        SELECT ij.progress
                        FROM ingestion_jobs ij
                        WHERE ij.document_id = kd.id
                        ORDER BY ij.created_at DESC
                        LIMIT 1
                    ) AS job_progress,
                    (
                        SELECT ij.error_message
                        FROM ingestion_jobs ij
                        WHERE ij.document_id = kd.id
                        ORDER BY ij.created_at DESC
                        LIMIT 1
                    ) AS job_error
                FROM knowledge_documents kd
                LEFT JOIN users u ON u.id = kd.created_by
                ORDER BY kd.created_at DESC
                """
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_ready_chunks(self) -> list[dict[str, Any]]:
        """
        读取所有已发布文档切片，供内存 FAISS 索引重建。

        vector_id 使用 document_chunks.id，删除文档后不会被重新查询。
        """

        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    dc.id AS chunk_id,
                    dc.id AS vector_id,
                    dc.chunk_index,
                    dc.content,
                    dc.char_start,
                    dc.char_end,
                    kd.id AS document_id,
                    kd.file_name,
                    kd.title,
                    kd.version
                FROM document_chunks dc
                JOIN knowledge_documents kd ON kd.id = dc.document_id
                WHERE kd.status = 'ready'
                ORDER BY kd.created_at, dc.chunk_index
                """
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_source_rows_by_vector_ids(
        self,
        *,
        vector_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """按全局 vector_id 获取引用来源，不限制单一文档。"""

        if not vector_ids:
            return {}
        placeholders = ",".join("?" for _ in vector_ids)
        async with self._connect() as connection:
            cursor = await connection.execute(
                f"""
                SELECT
                    dc.id AS chunk_db_id,
                    dc.vector_id,
                    dc.chunk_index,
                    dc.content,
                    dc.char_start,
                    dc.char_end,
                    kd.id AS document_id,
                    kd.file_name,
                    kd.title,
                    kd.version
                FROM document_chunks dc
                JOIN knowledge_documents kd ON kd.id = dc.document_id
                WHERE dc.vector_id IN ({placeholders})
                """,
                tuple(vector_ids),
            )
            rows = await cursor.fetchall()
        return {int(row["vector_id"]): dict(row) for row in rows}

    async def create_ingestion_job(
        self,
        *,
        document_id: str,
        created_by: str,
    ) -> str:
        """创建持久化入库任务，页面刷新后仍可查看进度。"""

        job_id = str(uuid.uuid4())
        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO ingestion_jobs
                    (
                        id, document_id, status, stage, progress,
                        created_by, created_at
                    )
                VALUES (?, ?, 'queued', '等待处理', 0, ?, ?)
                """,
                (job_id, document_id, created_by, utc_now()),
            )
            await connection.commit()
        return job_id

    async def update_ingestion_job(
        self,
        *,
        job_id: str,
        status: str,
        stage: str,
        progress: int,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """更新入库任务阶段；开始和结束时间由状态决定。"""

        now = utc_now()
        async with self._connect() as connection:
            await connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?,
                    stage = ?,
                    progress = ?,
                    error_code = ?,
                    error_message = ?,
                    started_at = COALESCE(started_at, ?),
                    finished_at = CASE
                        WHEN ? IN ('ready', 'failed') THEN ?
                        ELSE finished_at
                    END
                WHERE id = ?
                """,
                (
                    status,
                    stage,
                    progress,
                    error_code,
                    error_message,
                    now,
                    status,
                    now,
                    job_id,
                ),
            )
            await connection.commit()

    async def get_ingestion_job(self, job_id: str) -> dict[str, Any] | None:
        """读取单个入库任务状态。"""

        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                "SELECT * FROM ingestion_jobs WHERE id = ?",
                (job_id,),
            )
        return dict(row) if row is not None else None

    async def fail_unfinished_ingestion_jobs(self) -> None:
        """
        应用重启时，把进程内丢失的未完成任务标记为失败。

        管理员可在知识库页面重新上传文件触发新建索引。
        """

        async with self._connect() as connection:
            await connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = 'failed',
                    stage = '服务重启中断',
                    error_code = 'INGESTION_INTERRUPTED',
                    error_message = '服务重启导致任务中断，请重新上传。',
                    finished_at = ?
                WHERE status NOT IN ('ready', 'failed')
                """,
                (utc_now(),),
            )
            await connection.execute(
                """
                UPDATE knowledge_documents
                SET status = 'failed'
                WHERE status IN ('queued', 'processing')
                """
            )
            await connection.commit()

    async def get_active_prompt(self) -> dict[str, Any] | None:
        """读取当前生效的系统提示词。聊天请求每次从这里读取，保证修改立即生效。"""

        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                """
                SELECT * FROM prompt_templates
                WHERE status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (),
            )
        return dict(row) if row is not None else None

    async def seed_default_prompt(
        self,
        *,
        content: str,
        created_by: str = "system",
    ) -> None:
        """
        数据库没有提示词时插入默认版本。

        默认内容只存在于初始化种子；运行时始终从 prompt_templates 读取。
        """

        async with self._connect() as connection:
            row = await self._fetchone(
                connection,
                "SELECT id FROM prompt_templates LIMIT 1",
                (),
            )
            if row is not None:
                return
            await connection.execute(
                """
                INSERT INTO prompt_templates
                    (
                        id, name, content, version, status,
                        created_by, change_note, created_at
                    )
                VALUES (?, '员工问答系统提示词', ?, 1, 'active', ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    content,
                    created_by,
                    "系统默认模板",
                    utc_now(),
                ),
            )
            await connection.commit()

    async def publish_prompt(
        self,
        *,
        content: str,
        change_note: str,
        created_by: str,
    ) -> dict[str, Any]:
        """
        发布新提示词版本。

        新版本和旧版本停用在同一事务中完成，避免出现多个 active 模板。
        """

        async with self._connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    SELECT COALESCE(MAX(version), 0) AS max_version
                    FROM prompt_templates
                    """
                )
                row = await cursor.fetchone()
                next_version = int(row["max_version"]) + 1
                prompt_id = str(uuid.uuid4())
                now = utc_now()
                await connection.execute(
                    "UPDATE prompt_templates SET status = 'archived' WHERE status = 'active'"
                )
                await connection.execute(
                    """
                    INSERT INTO prompt_templates
                        (
                            id, name, content, version, status,
                            created_by, change_note, created_at
                        )
                    VALUES (?, '员工问答系统提示词', ?, ?, 'active', ?, ?, ?)
                    """,
                    (
                        prompt_id,
                        content,
                        next_version,
                        created_by,
                        change_note,
                        now,
                    ),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        return {
            "id": prompt_id,
            "version": next_version,
            "content": content,
            "change_note": change_note,
            "created_by": created_by,
            "created_at": now,
        }

    async def list_prompt_templates(self) -> list[dict[str, Any]]:
        """返回提示词版本历史，最新版本排在最前。"""

        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT * FROM prompt_templates ORDER BY version DESC"
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def write_audit_log(
        self,
        *,
        actor_id: str | None,
        actor_role: str | None,
        action: str,
        resource_type: str,
        resource_id: str | None,
        request_id: str | None,
        before_data: dict[str, Any] | None = None,
        after_data: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> None:
        """追加后台管理动作和敏感身份操作日志。"""

        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO audit_logs
                    (
                        id, actor_id, actor_role, action, resource_type,
                        resource_id, before_json, after_json, request_id,
                        ip_address, created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    actor_id,
                    actor_role,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(before_data, ensure_ascii=False)
                    if before_data is not None
                    else None,
                    json.dumps(after_data, ensure_ascii=False)
                    if after_data is not None
                    else None,
                    request_id,
                    ip_address,
                    utc_now(),
                ),
            )
            await connection.commit()

    async def dashboard_counts(self) -> dict[str, int]:
        """管理首页需要的轻量统计数据。"""

        async with self._connect() as connection:
            queries = {
                "documents": "SELECT COUNT(*) FROM knowledge_documents WHERE status = 'ready'",
                "processing": "SELECT COUNT(*) FROM ingestion_jobs WHERE status NOT IN ('ready', 'failed')",
                "conversations": "SELECT COUNT(*) FROM conversations",
                "flags": "SELECT COUNT(*) FROM audit_flags WHERE status = 'active'",
            }
            result: dict[str, int] = {}
            for key, sql in queries.items():
                cursor = await connection.execute(sql)
                result[key] = int((await cursor.fetchone())[0])
        return result

    async def list_audit_conversations(
        self,
        *,
        employee_id: str | None,
        date_from: str | None,
        date_to: str | None,
        flagged: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        按员工、UTC 时间范围和违规状态查询会话。

        页面日期筛选在路由层转换成 UTC ISO 时间，避免 SQLite 自行解释时区。
        """

        conditions = ["1 = 1"]
        params: list[Any] = []
        if employee_id:
            conditions.append("c.user_id = ?")
            params.append(employee_id)
        if date_from:
            conditions.append("c.created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("c.created_at <= ?")
            params.append(date_to)
        if flagged == "flagged":
            conditions.append(
                "EXISTS (SELECT 1 FROM audit_flags af "
                "WHERE af.conversation_id = c.id AND af.status = 'active')"
            )
        elif flagged == "unflagged":
            conditions.append(
                "NOT EXISTS (SELECT 1 FROM audit_flags af "
                "WHERE af.conversation_id = c.id AND af.status = 'active')"
            )
        where_sql = " AND ".join(conditions)

        async with self._connect() as connection:
            count_cursor = await connection.execute(
                f"SELECT COUNT(*) FROM conversations c WHERE {where_sql}",
                tuple(params),
            )
            total = int((await count_cursor.fetchone())[0])
            cursor = await connection.execute(
                f"""
                SELECT
                    c.*,
                    u.name AS employee_name,
                    u.employee_no,
                    (
                        SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id AND m.status = 'completed'
                    ) AS completed_messages,
                    (
                        SELECT COUNT(*) FROM ticket_drafts td
                        WHERE td.conversation_id = c.id
                    ) AS ticket_count,
                    (
                        SELECT COUNT(*) FROM audit_flags af
                        WHERE af.conversation_id = c.id AND af.status = 'active'
                    ) AS active_flags,
                    (
                        SELECT m.content FROM messages m
                        WHERE m.conversation_id = c.id AND m.role = 'user'
                        ORDER BY m.created_at DESC LIMIT 1
                    ) AS last_question
                FROM conversations c
                JOIN users u ON u.id = c.user_id
                WHERE {where_sql}
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows], total

    async def get_conversation_audit_detail(
        self,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        """读取会话、消息、工单草稿和违规标记的完整审计视图。"""

        async with self._connect() as connection:
            conversation = await self._fetchone(
                connection,
                """
                SELECT c.*, u.name AS employee_name, u.employee_no
                FROM conversations c
                JOIN users u ON u.id = c.user_id
                WHERE c.id = ?
                """,
                (conversation_id,),
            )
            if conversation is None:
                return None

            message_cursor = await connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at
                """,
                (conversation_id,),
            )
            messages = [dict(row) for row in await message_cursor.fetchall()]

            draft_cursor = await connection.execute(
                """
                SELECT * FROM ticket_drafts
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                """,
                (conversation_id,),
            )
            drafts = [dict(row) for row in await draft_cursor.fetchall()]

            flag_cursor = await connection.execute(
                """
                SELECT * FROM audit_flags
                WHERE conversation_id = ?
                ORDER BY marked_at DESC
                """,
                (conversation_id,),
            )
            flags = [dict(row) for row in await flag_cursor.fetchall()]

        for message in messages:
            raw_citations = message.get("citations_json")
            try:
                message["citations"] = json.loads(raw_citations) if raw_citations else []
            except (TypeError, json.JSONDecodeError):
                message["citations"] = []
        return {
            "conversation": dict(conversation),
            "messages": messages,
            "ticket_drafts": drafts,
            "flags": flags,
        }

    async def flag_conversation(
        self,
        *,
        conversation_id: str,
        ticket_draft_id: str | None,
        category: str,
        reason: str,
        marked_by: str,
    ) -> str:
        """新增疑似违规标记，不覆盖之前的审计历史。"""

        flag_id = str(uuid.uuid4())
        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO audit_flags
                    (
                        id, conversation_id, ticket_draft_id, category,
                        reason, status, marked_by, marked_at
                    )
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    flag_id,
                    conversation_id,
                    ticket_draft_id or None,
                    category,
                    reason,
                    marked_by,
                    utc_now(),
                ),
            )
            await connection.commit()
        return flag_id

    async def revoke_active_flags(
        self,
        *,
        conversation_id: str,
        revoked_by: str,
        revoke_reason: str,
    ) -> int:
        """撤销会话当前全部有效标记，并保留撤销人和原因。"""

        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                UPDATE audit_flags
                SET status = 'revoked',
                    revoked_by = ?,
                    revoked_at = ?,
                    revoke_reason = ?
                WHERE conversation_id = ? AND status = 'active'
                """,
                (revoked_by, utc_now(), revoke_reason, conversation_id),
            )
            await connection.commit()
            return cursor.rowcount

    async def healthcheck(self) -> bool:
        """执行最小数据库查询，供 /api/health 使用。"""

        try:
            async with self._connect() as connection:
                await connection.execute("PRAGMA foreign_keys = ON")
                await connection.execute("PRAGMA busy_timeout = 5000")
                await connection.execute("SELECT 1")
            return True
        except Exception:
            return False

    @staticmethod
    async def _fetchone(
        connection: aiosqlite.Connection,
        sql: str,
        params: tuple[Any, ...],
    ) -> aiosqlite.Row | None:
        """内部查询辅助方法，减少重复创建 cursor。"""

        cursor = await connection.execute(sql, params)
        return await cursor.fetchone()
