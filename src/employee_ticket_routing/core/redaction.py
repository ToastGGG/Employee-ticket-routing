"""日志、审计导出和工具参数使用的敏感信息脱敏工具。

员工薪酬福利咨询天然可能包含手机号、邮箱、证件号码、银行卡号和模型密钥。
“不要记录敏感信息”不能只依赖开发人员自觉，因此本模块提供统一的深度脱敏函数：

* 对已知敏感键进行整值遮罩；
* 对任意文本中的常见敏感格式执行正则替换；
* 对嵌套字典、列表和元组递归处理；
* 对超长字符串进行截断，降低日志被员工原文淹没的风险；
* 对工具参数中的 token 字段做保护，避免签名凭证进入决策日志。

该模块是纯函数实现，不读取全局环境，便于在 CLI、FastAPI、后台任务和测试中复用。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


MASK = "***"
SENSITIVE_KEY_PARTS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "session",
        "credential",
        "private_key",
        "access_key",
    }
)
SAFE_TOKEN_SUMMARY_KEYS = frozenset(
    {
        "token_type",
        "token_version",
        "tool_token_usage",
    }
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
MOBILE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")
CHINESE_ID_PATTERN = re.compile(
    r"(?<!\d)\d{6}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
BANK_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)")
BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._\-]+",
)
API_KEY_PATTERN = re.compile(
    r"\b(?:sk|ak|pk|ds)-[A-Za-z0-9_\-]{12,}\b",
    re.IGNORECASE,
)


def _key_is_sensitive(key: Any) -> bool:
    """判断字段名是否代表敏感值。

    采用“包含匹配”而非完全匹配，是为了兼容 ``deepseek_api_key``、
    ``authorization_header`` 等常见命名；少数需要保留的摘要字段显式白名单。
    """

    normalized = str(key).strip().lower()
    if normalized in SAFE_TOKEN_SUMMARY_KEYS:
        return False
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def mask_value(value: Any, *, visible_suffix: int = 4) -> Any:
    """遮罩单个敏感值，仅保留可选尾部字符用于排障。

    过短的值绝不保留尾部，否则遮罩本身可能泄露完整内容。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value)
    if not text:
        return text
    if visible_suffix <= 0 or len(text) <= visible_suffix + 4:
        return MASK
    return f"{MASK}{text[-visible_suffix:]}"


def redact_text(text: str, *, max_length: int | None = 4000) -> str:
    """替换文本中的常见敏感格式，并按需截断。

    正则规则是“降低泄漏概率”的工程防线，不替代数据分类、加密和访问控制。
    """

    if not text:
        return text
    redacted = BEARER_PATTERN.sub("Bearer ***", text)
    redacted = API_KEY_PATTERN.sub("***", redacted)
    redacted = EMAIL_PATTERN.sub("***@***", redacted)
    redacted = MOBILE_PATTERN.sub("***PHONE***", redacted)
    redacted = CHINESE_ID_PATTERN.sub("***ID***", redacted)
    redacted = BANK_CARD_PATTERN.sub("***CARD***", redacted)
    redacted = re.sub(
        r"(?i)(password|passwd|secret|token|api[_-]?key)"
        r"(\s*[=:]\s*)([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}***",
        redacted,
    )
    if max_length is not None and len(redacted) > max_length:
        omitted = len(redacted) - max_length
        redacted = f"{redacted[:max_length]}...[已截断{omitted}字符]"
    return redacted


def redact_mapping(
    value: Any,
    *,
    max_string_length: int | None = 4000,
    _depth: int = 0,
) -> Any:
    """递归生成可安全记录的副本，不修改调用方对象。"""

    if _depth > 20:
        return "[达到最大嵌套深度]"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _key_is_sensitive(key):
                result[key_text] = mask_value(item)
                continue
            result[key_text] = redact_mapping(
                item,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            redact_mapping(
                item,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
            for item in value
        ]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return redact_text(value, max_length=max_string_length)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(repr(value), max_length=max_string_length)


def redact_tool_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """对 Function Calling 参数进行日志级脱敏。

    Agent 工具参数中可能包含 policy/routing 签名。即便 JWT 本身不是密码，
    也不应写入分析日志，因此任何以 ``_token`` 结尾的键统一替换为存在性摘要。
    """

    result: dict[str, Any] = {}
    for key, value in arguments.items():
        key_text = str(key)
        if key_text.endswith("_token") or key_text.endswith("_secret"):
            result[key_text] = "<已省略>"
        else:
            result[key_text] = redact_mapping(value, max_string_length=1200)
    return result


def safe_exception_message(exc: BaseException, *, max_length: int = 500) -> str:
    """把异常转换为用户可记录的中文安全摘要。

    只保留异常类型与脱敏后的短消息；调用栈由日志框架记录，但不应原样返回给用户。
    """

    message = redact_text(str(exc).strip(), max_length=max_length)
    if not message:
        message = "无详细信息"
    return f"{exc.__class__.__name__}: {message}"


def looks_like_secret(value: str) -> bool:
    """用于配置检查和发布扫描的轻量密钥形态检测。"""

    if not value or len(value) < 16:
        return False
    return bool(
        API_KEY_PATTERN.search(value)
        or BEARER_PATTERN.search(value)
        or re.fullmatch(r"[A-Za-z0-9_\-]{32,}", value)
    )
