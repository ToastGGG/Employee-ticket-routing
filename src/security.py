"""
认证与安全工具。

职责：
1. Argon2id 密码哈希；
2. JWT 签发和验证；
3. 从 HttpOnly Cookie 或 Authorization Header 读取 Token；
4. CSRF Token 生成与校验。
"""

from __future__ import annotations

import hmac
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


ACCESS_COOKIE_NAME = "employee_access_token"
CSRF_COOKIE_NAME = "employee_csrf_token"
JWT_ALGORITHM = "HS256"


class PasswordService:
    """封装密码哈希，避免业务代码直接处理 Argon2 参数。"""

    def __init__(self) -> None:
        self.hasher = PasswordHasher()

    def hash(self, password: str) -> str:
        """生成 Argon2id 哈希。数据库永远不保存明文密码。"""

        return self.hasher.hash(password)

    def verify(self, password_hash: str | None, password: str) -> bool:
        """验证密码；哈希损坏或密码错误统一返回 False。"""

        if not password_hash:
            return False
        try:
            return self.hasher.verify(password_hash, password)
        except (VerificationError, VerifyMismatchError, InvalidHashError):
            return False


def load_or_create_jwt_secret(secret_file: Path) -> str:
    """
    读取 JWT 密钥。

    环境变量 JWT_SECRET 优先；未配置时在受控数据目录生成随机密钥。
    这样不会在源代码中硬编码生产密钥，同时保证本地服务可直接运行。
    """

    import os

    configured = os.getenv("JWT_SECRET")
    if configured:
        return configured
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    if secret_file.exists():
        return secret_file.read_text(encoding="utf-8").strip()
    secret = secrets.token_urlsafe(48)
    secret_file.write_text(secret, encoding="utf-8")
    return secret


class JWTService:
    """JWT 签发与解析服务，供页面和 API 共用。"""

    def __init__(self, *, secret: str, expire_hours: int = 8):
        self.secret = secret
        self.expire_hours = expire_hours

    def create(self, *, user: dict[str, Any]) -> str:
        """为用户签发 JWT，角色和 token_version 写入声明。"""

        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user["id"]),
            "role": str(user["role"]),
            "ver": int(user.get("token_version") or 1),
            "jti": secrets.token_urlsafe(18),
            "iat": now,
            "exp": now + timedelta(hours=self.expire_hours),
        }
        return jwt.encode(payload, self.secret, algorithm=JWT_ALGORITHM)

    def decode(self, token: str) -> dict[str, Any]:
        """验证签名、过期时间和声明结构；失败抛出 ValueError。"""

        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[JWT_ALGORITHM],
                options={"require": ["sub", "role", "exp", "iat", "jti"]},
            )
        except jwt.PyJWTError as exc:
            raise ValueError("登录状态无效或已过期。") from exc
        if payload.get("role") not in {"employee", "admin"}:
            raise ValueError("用户角色无效。")
        return payload


class AgentTokenService:
    """
    Agent 工具凭证服务。

    工具凭证与登录 JWT 使用同一个密钥但不同的 audience，且绑定：
    用户 ID、会话/案件 ID、凭证类型、输入摘要、jti 和过期时间。
    """

    def __init__(self, *, secret: str, ttl_seconds: int = 600):
        self.secret = secret
        self.ttl_seconds = ttl_seconds

    def create(
        self,
        *,
        token_type: str,
        user_id: str,
        case_id: str,
        payload: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> str:
        """签发短期工具凭证；模型只能传递，不能伪造或修改其内容。"""

        now = datetime.now(timezone.utc)
        token_payload = {
            "aud": "employee-agent-tool",
            "typ": token_type,
            "sub": user_id,
            "case": case_id,
            "payload": payload,
            "jti": secrets.token_urlsafe(20),
            "iat": now,
            "exp": now + timedelta(seconds=ttl_seconds or self.ttl_seconds),
        }
        return jwt.encode(token_payload, self.secret, algorithm=JWT_ALGORITHM)

    def decode(
        self,
        token: str,
        *,
        expected_type: str,
        user_id: str,
        case_id: str,
    ) -> dict[str, Any]:
        """验证凭证签名、类型、归属和有效期。"""

        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[JWT_ALGORITHM],
                audience="employee-agent-tool",
                options={"require": ["typ", "sub", "case", "payload", "jti", "exp"]},
            )
        except jwt.PyJWTError as exc:
            raise ValueError("工具校验凭证无效或已过期。") from exc
        if payload.get("typ") != expected_type:
            raise ValueError("工具校验凭证类型不匹配。")
        if payload.get("sub") != user_id or payload.get("case") != case_id:
            raise ValueError("工具校验凭证与当前员工或会话不匹配。")
        return payload


def extract_bearer_token(request) -> str | None:
    """从 Authorization 请求头提取 Bearer Token。"""

    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return None


def create_csrf_token() -> str:
    """生成有足够随机性的 CSRF Token。"""

    return secrets.token_urlsafe(32)


def csrf_matches(cookie_token: str | None, submitted_token: str | None) -> bool:
    """使用常量时间比较，降低时序攻击风险。"""

    if not cookie_token or not submitted_token:
        return False
    return hmac.compare_digest(cookie_token, submitted_token)
