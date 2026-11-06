"""运行配置的集中解析、校验与安全展示。

项目早期直接在 ``main.py`` 中读取环境变量，便于快速完成原型。随着部署场景增加，
配置来源、默认值、敏感字段和非法值的影响需要被统一管理。本模块不替换现有启动流程，
而是为运维 CLI、测试和后续服务层重构提供一套可复用、可验证的配置模型。

设计原则：

1. 解析与业务启动解耦：环境变量不合法时，先返回可操作的问题清单；
2. 默认值显式化：每个默认值都能在代码中检索到，避免隐藏行为；
3. 敏感值不进入快照：API Key、JWT Secret 等仅返回“是否配置”；
4. 兼容现有变量：变量名称与当前项目保持一致，不迁移已有 ``.env``；
5. 便于测试：所有解析函数都允许注入 mapping，不依赖进程全局状态。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """配置存在阻断性问题时抛出的异常。"""

    def __init__(self, issues: list["ConfigIssue"]):
        self.issues = issues
        super().__init__("；".join(issue.message for issue in issues))


@dataclass(frozen=True)
class ConfigIssue:
    """一条可定位、可修复的配置问题。"""

    key: str
    message: str
    severity: str = "error"
    remedy: str = ""
    current_value: str | None = None

    def to_dict(self, *, reveal_value: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not reveal_value:
            payload["current_value"] = None
        return payload


@dataclass(frozen=True)
class StorageSettings:
    """持久化与知识库文件相关配置。"""

    data_dir: Path
    sqlite_path: Path
    upload_dir: Path
    rag_cache_dir: Path
    knowledge_file: Path


@dataclass(frozen=True)
class LLMSettings:
    """模型供应商配置。

    ``provider_requested`` 是用户希望使用的供应商，``provider_effective`` 是当前
    实际可用的供应商。未配置 DeepSeek Key 时，现有系统会回退到 Mock；配置对象
    将这种行为显式表达出来，避免健康检查只看到“provider=mock”却不知道原因。
    """

    provider_requested: str
    provider_effective: str
    api_url: str | None
    model: str | None
    api_key_configured: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class SecuritySettings:
    """认证和浏览器安全相关配置。"""

    jwt_secret_configured: bool
    jwt_expire_hours: int
    cookie_secure: bool
    tool_token_ttl_seconds: int
    confirmation_ttl_seconds: int


@dataclass(frozen=True)
class AgentSettings:
    """Agent 循环和检索参数。"""

    max_rounds: int
    request_timeout_seconds: int
    top_k: int
    chunk_size: int
    chunk_overlap: int
    default_region: str


@dataclass(frozen=True)
class RuntimeSettings:
    """完整运行配置。

    该对象是不可变 dataclass，便于在请求作用域中传递。路径已展开为绝对路径，
    但不会自动创建目录；目录创建仍由应用启动流程负责，便于配置检查在只读环境运行。
    """

    project_root: Path
    storage: StorageSettings
    llm: LLMSettings
    security: SecuritySettings
    agent: AgentSettings
    environment: str = "development"
    issues: list[ConfigIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[ConfigIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ConfigIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def require_valid(self) -> "RuntimeSettings":
        """存在阻断问题时抛出异常，供正式启动和 CI 使用。"""

        if self.blocking_issues:
            raise ConfigurationError(self.blocking_issues)
        return self

    def masked_snapshot(self) -> dict[str, Any]:
        """返回可安全写入日志或诊断接口的配置快照。"""

        def path_text(value: Path | None) -> str | None:
            return str(value) if value is not None else None

        return {
            "environment": self.environment,
            "project_root": path_text(self.project_root),
            "storage": {
                "data_dir": path_text(self.storage.data_dir),
                "sqlite_path": path_text(self.storage.sqlite_path),
                "upload_dir": path_text(self.storage.upload_dir),
                "rag_cache_dir": path_text(self.storage.rag_cache_dir),
                "knowledge_file": path_text(self.storage.knowledge_file),
            },
            "llm": {
                "provider_requested": self.llm.provider_requested,
                "provider_effective": self.llm.provider_effective,
                "api_url": self.llm.api_url,
                "model": self.llm.model,
                "api_key_configured": self.llm.api_key_configured,
                "fallback_reason": self.llm.fallback_reason,
            },
            "security": {
                "jwt_secret_configured": self.security.jwt_secret_configured,
                "jwt_expire_hours": self.security.jwt_expire_hours,
                "cookie_secure": self.security.cookie_secure,
                "tool_token_ttl_seconds": self.security.tool_token_ttl_seconds,
                "confirmation_ttl_seconds": (
                    self.security.confirmation_ttl_seconds
                ),
            },
            "agent": {
                "max_rounds": self.agent.max_rounds,
                "request_timeout_seconds": (
                    self.agent.request_timeout_seconds
                ),
                "top_k": self.agent.top_k,
                "chunk_size": self.agent.chunk_size,
                "chunk_overlap": self.agent.chunk_overlap,
                "default_region": self.agent.default_region,
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _parse_int(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    issues: list[ConfigIssue],
) -> int:
    """解析闭区间整数；非法值保留默认值并登记问题。"""

    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        issues.append(
            ConfigIssue(
                key=key,
                message=f"{key} 必须是整数，当前值无法解析",
                remedy=f"将 {key} 设置为 {minimum} 到 {maximum} 之间的整数",
            )
        )
        return default
    if value < minimum or value > maximum:
        issues.append(
            ConfigIssue(
                key=key,
                message=f"{key} 超出允许范围 {minimum}..{maximum}",
                current_value=str(value),
                remedy=f"将 {key} 调整到 {minimum} 到 {maximum} 之间",
            )
        )
        return default
    return value


def _parse_bool(
    env: Mapping[str, str],
    key: str,
    default: bool,
    *,
    issues: list[ConfigIssue],
) -> bool:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    issues.append(
        ConfigIssue(
            key=key,
            message=f"{key} 不是受支持的布尔值",
            current_value=normalized,
            remedy="使用 true/false、yes/no、on/off 或 1/0",
        )
    )
    return default


def _resolve_path(
    project_root: Path,
    raw_value: str | None,
    default: Path,
) -> Path:
    """将相对路径稳定解析到项目根目录。"""

    value = Path(raw_value).expanduser() if raw_value else default
    if not value.is_absolute():
        value = project_root / value
    return value.resolve(strict=False)


def _validate_url(
    value: str | None,
    *,
    key: str,
    issues: list[ConfigIssue],
) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        issues.append(
            ConfigIssue(
                key=key,
                message=f"{key} 必须是包含协议的完整 URL",
                current_value=value,
                remedy="例如：https://api.deepseek.com/chat/completions",
            )
        )
    return value


def load_runtime_settings(
    env: Mapping[str, str] | None = None,
    *,
    project_root: Path | None = None,
) -> RuntimeSettings:
    """从环境变量构建配置，并收集所有可修复问题。

    该函数不会抛出解析异常，调用方可以一次性展示全部问题。若希望正式环境快速失败，
    再调用 ``settings.require_valid()`` 即可。
    """

    source = dict(os.environ if env is None else env)
    root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    issues: list[ConfigIssue] = []
    environment = source.get("APP_ENV", "development").strip().lower()
    if environment not in {"development", "test", "staging", "production"}:
        issues.append(
            ConfigIssue(
                key="APP_ENV",
                message="APP_ENV 不是受支持的环境名称",
                current_value=environment,
                remedy="使用 development/test/staging/production",
            )
        )

    data_dir = _resolve_path(
        root,
        source.get("DATA_DIR"),
        root / "data",
    )
    storage = StorageSettings(
        data_dir=data_dir,
        sqlite_path=_resolve_path(
            root,
            source.get("SQLITE_PATH"),
            data_dir / "employee_chat.sqlite3",
        ),
        upload_dir=_resolve_path(
            root,
            source.get("UPLOAD_DIR"),
            data_dir / "uploads",
        ),
        rag_cache_dir=_resolve_path(
            root,
            source.get("RAG_CACHE_DIR"),
            data_dir / "faiss_cache",
        ),
        knowledge_file=_resolve_path(
            root,
            source.get("KNOWLEDGE_FILE"),
            root / "docs" / "薪酬福利制度示例.md",
        ),
    )

    provider = source.get("LLM_PROVIDER", "deepseek").strip().lower()
    if provider not in {"deepseek", "openai_compatible", "mock"}:
        issues.append(
            ConfigIssue(
                key="LLM_PROVIDER",
                message="LLM_PROVIDER 不在支持列表中",
                current_value=provider,
                remedy="使用 deepseek、openai_compatible 或 mock",
            )
        )
        provider = "mock"

    deepseek_key = source.get("DEEPSEEK_API_KEY", "").strip()
    generic_key = source.get("LLM_API_KEY", "").strip()
    fallback_reason: str | None = None
    effective_provider = provider
    if provider == "deepseek":
        if not deepseek_key:
            effective_provider = "mock"
            fallback_reason = "未配置 DEEPSEEK_API_KEY，已回退到 Mock"
        api_url = source.get(
            "DEEPSEEK_API_URL",
            "https://api.deepseek.com/chat/completions",
        ).strip()
        model = source.get("DEEPSEEK_MODEL", "deepseek-chat").strip()
        api_key_configured = bool(deepseek_key)
    elif provider == "openai_compatible":
        if not generic_key:
            effective_provider = "mock"
            fallback_reason = "未配置 LLM_API_KEY，已回退到 Mock"
        api_url = source.get("LLM_API_URL", "").strip() or None
        model = source.get("LLM_MODEL", "").strip() or None
        api_key_configured = bool(generic_key)
    else:
        api_url = None
        model = "mock"
        api_key_configured = False
        fallback_reason = "显式启用 Mock 模式"

    if effective_provider != "mock":
        api_url = _validate_url(
            api_url,
            key=(
                "DEEPSEEK_API_URL"
                if provider == "deepseek"
                else "LLM_API_URL"
            ),
            issues=issues,
        )
        if not model:
            issues.append(
                ConfigIssue(
                    key=(
                        "DEEPSEEK_MODEL"
                        if provider == "deepseek"
                        else "LLM_MODEL"
                    ),
                    message="真实模型模式缺少模型名称",
                    remedy="配置供应商支持的模型标识",
                )
            )

    jwt_configured = bool(source.get("JWT_SECRET", "").strip())
    cookie_secure = _parse_bool(
        source,
        "COOKIE_SECURE",
        environment == "production",
        issues=issues,
    )
    if environment == "production" and not cookie_secure:
        issues.append(
            ConfigIssue(
                key="COOKIE_SECURE",
                message="生产环境未启用 Secure Cookie",
                severity="warning",
                remedy="在 HTTPS 部署中设置 COOKIE_SECURE=true",
            )
        )
    if environment == "production" and not jwt_configured:
        issues.append(
            ConfigIssue(
                key="JWT_SECRET",
                message="生产环境未显式配置 JWT_SECRET",
                severity="warning",
                remedy="由密钥管理服务注入至少 32 字节的高熵随机值",
            )
        )

    security = SecuritySettings(
        jwt_secret_configured=jwt_configured,
        jwt_expire_hours=_parse_int(
            source,
            "JWT_EXPIRE_HOURS",
            8,
            minimum=1,
            maximum=168,
            issues=issues,
        ),
        cookie_secure=cookie_secure,
        tool_token_ttl_seconds=_parse_int(
            source,
            "AGENT_TOKEN_TTL_SECONDS",
            600,
            minimum=60,
            maximum=3600,
            issues=issues,
        ),
        confirmation_ttl_seconds=_parse_int(
            source,
            "AGENT_CONFIRMATION_TTL_SECONDS",
            600,
            minimum=60,
            maximum=3600,
            issues=issues,
        ),
    )

    agent = AgentSettings(
        max_rounds=_parse_int(
            source,
            "AGENT_MAX_ROUNDS",
            5,
            minimum=1,
            maximum=5,
            issues=issues,
        ),
        request_timeout_seconds=_parse_int(
            source,
            "REQUEST_TIMEOUT",
            60,
            minimum=5,
            maximum=300,
            issues=issues,
        ),
        top_k=_parse_int(
            source,
            "TOP_K",
            3,
            minimum=1,
            maximum=20,
            issues=issues,
        ),
        chunk_size=_parse_int(
            source,
            "CHUNK_SIZE",
            500,
            minimum=100,
            maximum=4000,
            issues=issues,
        ),
        chunk_overlap=_parse_int(
            source,
            "CHUNK_OVERLAP",
            80,
            minimum=0,
            maximum=1000,
            issues=issues,
        ),
        default_region=(
            source.get("DEFAULT_REGION", "CN").strip().upper() or "CN"
        ),
    )
    if agent.chunk_overlap >= agent.chunk_size:
        issues.append(
            ConfigIssue(
                key="CHUNK_OVERLAP",
                message="切片重叠必须小于切片长度",
                current_value=str(agent.chunk_overlap),
                remedy=f"将 CHUNK_OVERLAP 设置为小于 {agent.chunk_size} 的值",
            )
        )

    llm = LLMSettings(
        provider_requested=provider,
        provider_effective=effective_provider,
        api_url=api_url,
        model=model,
        api_key_configured=api_key_configured,
        fallback_reason=fallback_reason,
    )

    return RuntimeSettings(
        project_root=root,
        storage=storage,
        llm=llm,
        security=security,
        agent=agent,
        environment=environment,
        issues=issues,
    )


def summarize_configuration(settings: RuntimeSettings) -> str:
    """生成适合 CLI 输出的中文配置摘要。"""

    lines = [
        f"环境: {settings.environment}",
        f"项目根目录: {settings.project_root}",
        f"数据库: {settings.storage.sqlite_path}",
        (
            "模型: 请求="
            f"{settings.llm.provider_requested}, 实际={settings.llm.provider_effective}, "
            f"模型={settings.llm.model or '-'}"
        ),
        f"Agent 最大轮次: {settings.agent.max_rounds}",
        f"JWT 密钥已配置: {'是' if settings.security.jwt_secret_configured else '否'}",
    ]
    if settings.llm.fallback_reason:
        lines.append(f"模型回退原因: {settings.llm.fallback_reason}")
    if not settings.issues:
        lines.append("配置检查: 通过")
    else:
        lines.append(f"配置检查: 发现 {len(settings.issues)} 个问题")
        for issue in settings.issues:
            lines.append(
                f"[{issue.severity.upper()}] {issue.key}: {issue.message}"
            )
            if issue.remedy:
                lines.append(f"  修复建议: {issue.remedy}")
    return "\n".join(lines)
