"""
员工薪酬福利咨询工单分流系统。

本文件同时提供：
- 员工单页问答入口与 SSE 接口；
- JWT 登录、员工/管理员接口级权限校验；
- 服务端渲染的管理后台；
- 知识库上传、提示词管理和对话审计的页面路由。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    from .agent_runtime import AgentRunner
    from .db import SQLiteDatabase
    from .ingestion import (
        MAX_FILE_SIZE,
        calculate_hash,
        create_document_job,
        run_ingestion_job,
        seed_default_document,
        validate_upload,
    )
    from .rag_runtime import RagRuntime
    from .security import (
        ACCESS_COOKIE_NAME,
        AgentTokenService,
        CSRF_COOKIE_NAME,
        JWTService,
        PasswordService,
        create_csrf_token,
        csrf_matches,
        extract_bearer_token,
        load_or_create_jwt_secret,
    )
except ImportError:
    from agent_runtime import AgentRunner
    from db import SQLiteDatabase
    from ingestion import (
        MAX_FILE_SIZE,
        calculate_hash,
        create_document_job,
        run_ingestion_job,
        seed_default_document,
        validate_upload,
    )
    from rag_runtime import RagRuntime
    from security import (
        ACCESS_COOKIE_NAME,
        AgentTokenService,
        CSRF_COOKIE_NAME,
        JWTService,
        PasswordService,
        create_csrf_token,
        csrf_matches,
        extract_bearer_token,
        load_or_create_jwt_secret,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("employee-service")

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "employee_chat.sqlite3"
DEFAULT_KNOWLEDGE_FILE = REPO_ROOT / "docs" / "薪酬福利制度示例.md"
DEFAULT_INDEX_CACHE = DEFAULT_DATA_DIR / "faiss_cache"
DEFAULT_UPLOAD_DIR = DEFAULT_DATA_DIR / "uploads"
SHANGHAI = timezone(timedelta(hours=8))

# 这个模板只在数据库没有任何提示词版本时写入一次。
# 实际问答始终从 prompt_templates 表读取，不会在运行时使用此常量。
DEFAULT_SYSTEM_PROMPT = (
    "你是公司员工薪酬福利咨询工单分流助手，只根据提供的制度条文回答，"
    "不确定时明确说'制度中未找到'，回答末尾附引用来源，不得编造报销标准。"
)


class APIError(Exception):
    """应用统一错误类型，包含 HTTP 状态、稳定错误码和中文信息。"""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class ChatRequest(BaseModel):
    """员工问答 API 请求。"""

    session_id: str
    question: str


class TicketConfirmRequest(BaseModel):
    """员工对 Agent 生成的工单草稿进行显式确认。"""

    confirmed: bool
    draft_version: int


class LoginRequest(BaseModel):
    """JSON 登录请求，用于 Bearer Token 客户端。"""

    username: str
    password: str


@dataclass(frozen=True)
class CurrentUser:
    """当前登录用户及其 JWT 来源，用于判断是否需要 CSRF 校验。"""

    record: dict
    authenticated_by_cookie: bool

    @property
    def id(self) -> str:
        return str(self.record["id"])

    @property
    def role(self) -> str:
        return str(self.record["role"])

    @property
    def name(self) -> str:
        return str(self.record["name"])


templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
password_service = PasswordService()


def format_datetime_cn(value: str | None) -> str:
    """将数据库 UTC 时间转换为上海时间，供 Jinja2 模板显示。"""

    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


templates.env.filters["datetime_cn"] = format_datetime_cn


def request_id(request: Request) -> str:
    """读取中间件生成的 Trace ID。"""

    return getattr(request.state, "request_id", str(uuid.uuid4()))


def error_payload(
    *,
    request: Request | None,
    code: str,
    message: str,
    details: dict | None = None,
) -> dict:
    """生成 HTTP JSON 和 SSE error 共用的错误结构。"""

    payload = {
        "code": code,
        "message": message,
        "request_id": request_id(request) if request else str(uuid.uuid4()),
    }
    if details:
        payload["details"] = details
    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    初始化数据库、演示账号、默认提示词和知识索引。

    默认密码可通过环境变量覆盖。正式部署必须修改默认值并启用 HTTPS。
    """

    data_dir = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
    data_dir.mkdir(parents=True, exist_ok=True)
    database = SQLiteDatabase(
        Path(os.getenv("SQLITE_PATH", str(DEFAULT_DB_PATH)))
    )
    await database.initialize()
    await database.fail_unfinished_ingestion_jobs()

    admin_password = os.getenv("SEED_ADMIN_PASSWORD", "Admin@123456")
    employee_password = os.getenv("SEED_EMPLOYEE_PASSWORD", "Employee@123456")
    await database.upsert_seed_user(
        user_id="admin-demo",
        employee_no=os.getenv("SEED_ADMIN_USERNAME", "admin"),
        name=os.getenv("SEED_ADMIN_NAME", "系统管理员"),
        role="admin",
        password_hash=password_service.hash(admin_password),
    )
    await database.upsert_seed_user(
        user_id="employee-demo",
        employee_no=os.getenv("SEED_EMPLOYEE_USERNAME", "employee"),
        name=os.getenv("SEED_EMPLOYEE_NAME", "演示员工"),
        role="employee",
        password_hash=password_service.hash(employee_password),
    )
    await database.seed_default_prompt(content=DEFAULT_SYSTEM_PROMPT)

    runtime = RagRuntime(
        cache_dir=Path(os.getenv("RAG_CACHE_DIR", str(DEFAULT_INDEX_CACHE)))
    )
    upload_dir = Path(os.getenv("UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR)))
    knowledge_file = Path(
        os.getenv("KNOWLEDGE_FILE", str(DEFAULT_KNOWLEDGE_FILE))
    )
    try:
        await seed_default_document(
            database=database,
            runtime=runtime,
            source_path=knowledge_file,
            upload_dir=upload_dir,
            created_by="admin-demo",
        )
    except Exception as exc:
        runtime.ready = False
        runtime.error_message = str(exc)
        logger.exception("初始化知识库失败：%s", exc)

    jwt_secret = load_or_create_jwt_secret(data_dir / ".jwt_secret")
    jwt_service = JWTService(
        secret=jwt_secret,
        expire_hours=int(os.getenv("JWT_EXPIRE_HOURS", "8")),
    )
    agent_token_service = AgentTokenService(
        secret=jwt_secret,
        ttl_seconds=int(os.getenv("AGENT_TOKEN_TTL_SECONDS", "600")),
    )

    app.state.db = database
    app.state.runtime = runtime
    app.state.jwt = jwt_service
    app.state.agent_tokens = agent_token_service
    app.state.upload_dir = upload_dir
    logger.info(
        "服务初始化完成：documents=%s chunks=%s",
        runtime.document_count,
        runtime.chunk_count,
    )
    yield


app = FastAPI(
    title="员工薪酬福利咨询工单分流系统",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """为每个请求分配并返回 Trace ID。"""

    request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(APIError)
async def handle_api_error(request: Request, exc: APIError):
    """
    API、SSE 之前和后台页面共用同一错误码。

    JSON/SSE 接口返回 JSON；后台页面渲染中文错误页。
    """

    payload = error_payload(
        request=request,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )
    if (
        request.url.path in {"/chat", "/api/auth/token", "/api/auth/me"}
        or request.url.path.startswith("/api/")
    ):
        return JSONResponse(status_code=exc.status_code, content=payload)
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "title": "请求无法完成",
            "status_code": exc.status_code,
            "code": exc.code,
            "message": exc.message,
            "request_id": payload["request_id"],
            "current_user": None,
            "csrf_token": request.cookies.get(CSRF_COOKIE_NAME, ""),
        },
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """将 FastAPI 默认校验错误转换成中文统一结构。"""

    return JSONResponse(
        status_code=422,
        content=error_payload(
            request=request,
            code="VALIDATION_ERROR",
            message="请求参数不完整或格式不正确。",
            details={
                "errors": [
                    {
                        "field": ".".join(str(part) for part in item.get("loc", [])),
                        "type": str(item.get("type", "")),
                        "message": str(item.get("msg", "")),
                    }
                    for item in exc.errors()
                ]
            },
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException):
    """处理框架级 404/405 错误。"""

    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    message = (
        "请求的接口不存在。"
        if exc.status_code == 404
        else "请求方法不被允许。"
        if exc.status_code == 405
        else "请求无法完成。"
    )
    payload = error_payload(request=request, code=code, message=message)
    if request.url.path.startswith("/api/") or request.url.path == "/chat":
        return JSONResponse(status_code=exc.status_code, content=payload)
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "title": "页面不存在",
            "status_code": exc.status_code,
            "code": code,
            "message": message,
            "request_id": payload["request_id"],
            "current_user": None,
            "csrf_token": request.cookies.get(CSRF_COOKIE_NAME, ""),
        },
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """隐藏内部堆栈，只记录日志并返回中文错误。"""

    logger.exception("未处理异常 request_id=%s", request_id(request))
    payload = error_payload(
        request=request,
        code="INTERNAL_ERROR",
        message="系统内部错误，请稍后重试。",
    )
    if request.url.path.startswith("/api/") or request.url.path == "/chat":
        return JSONResponse(status_code=500, content=payload)
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "title": "系统错误",
            "status_code": 500,
            "code": "INTERNAL_ERROR",
            "message": payload["message"],
            "request_id": payload["request_id"],
            "current_user": None,
            "csrf_token": request.cookies.get(CSRF_COOKIE_NAME, ""),
        },
        status_code=500,
    )


async def resolve_current_user(request: Request) -> CurrentUser | None:
    """
    从 HttpOnly Cookie 或 Bearer Header 解析当前用户。

    Token 过期、用户停用和 token_version 变化都视为未登录。
    """

    bearer = extract_bearer_token(request)
    cookie_token = request.cookies.get(ACCESS_COOKIE_NAME)
    token = bearer or cookie_token
    if not token:
        return None
    try:
        payload = app.state.jwt.decode(token)
        user = await app.state.db.get_user(str(payload["sub"]))
        if user is None or user.get("status") != "active":
            return None
        if int(payload.get("ver", 1)) != int(user.get("token_version") or 1):
            return None
        return CurrentUser(
            record=user,
            authenticated_by_cookie=bool(cookie_token and not bearer),
        )
    except ValueError:
        return None


async def get_current_user(request: Request) -> CurrentUser:
    """必须登录的接口依赖。"""

    user = await resolve_current_user(request)
    if user is None:
        raise APIError(
            status_code=401,
            code="UNAUTHORIZED",
            message="请先登录后再继续操作。",
        )
    return user


async def require_employee(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """员工聊天入口只允许 employee 角色。"""

    if user.role != "employee":
        raise APIError(
            status_code=403,
            code="FORBIDDEN",
            message="当前账号不能使用员工问答入口。",
        )
    return user


async def require_admin(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """管理后台与后台 API 只允许 admin 角色。"""

    if user.role != "admin":
        raise APIError(
            status_code=403,
            code="FORBIDDEN",
            message="当前账号没有访问管理后台的权限。",
        )
    return user


def verify_csrf(request: Request, submitted_token: str | None) -> None:
    """校验 Cookie 与表单/请求头中的 CSRF Token 是否一致。"""

    if not csrf_matches(
        request.cookies.get(CSRF_COOKIE_NAME),
        submitted_token,
    ):
        raise APIError(
            status_code=403,
            code="CSRF_INVALID",
            message="页面安全令牌已失效，请刷新页面后重试。",
        )


def set_auth_cookies(response, *, token: str) -> None:
    """设置 HttpOnly JWT Cookie 和可供表单读取的 CSRF Cookie。"""

    secure_cookie = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    max_age = int(os.getenv("JWT_EXPIRE_HOURS", "8")) * 3600
    response.set_cookie(
        ACCESS_COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE_NAME,
        create_csrf_token(),
        max_age=max_age,
        httponly=False,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )


def clear_auth_cookies(response) -> None:
    """退出登录时删除两个认证 Cookie。"""

    response.delete_cookie(ACCESS_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")


async def authenticate(username: str, password: str) -> dict | None:
    """校验用户名和密码，成功时更新最后登录时间。"""

    normalized = username.strip()
    if not normalized or not password:
        return None
    user = await app.state.db.get_user_by_employee_no(normalized)
    if (
        user is None
        or user.get("status") != "active"
        or user.get("role") not in {"employee", "admin"}
        or not password_service.verify(user.get("password_hash"), password)
    ):
        return None
    await app.state.db.touch_last_login(user["id"])
    return user


def redirect_for_role(role: str) -> str:
    """登录成功后按角色跳转。"""

    return "/admin" if role == "admin" else "/"


def render_login(
    request: Request,
    *,
    error_message: str = "",
    username: str = "",
    status_code: int = 200,
):
    """渲染登录页，供 GET 和表单鉴权失败复用。"""

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "title": "登录",
            "error_message": error_message,
            "username": username,
        },
        status_code=status_code,
    )


@app.get("/login")
async def login_page(
    request: Request,
    next: str | None = None,
):
    """显示登录页；已登录用户直接进入对应首页。"""

    user = await resolve_current_user(request)
    if user is not None:
        return RedirectResponse(redirect_for_role(user.role), status_code=303)
    return render_login(request)


@app.post("/auth/login")
async def login_form(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    """服务端表单登录，设置 JWT Cookie 后跳转。"""

    user = await authenticate(username, password)
    if user is None:
        await app.state.db.write_audit_log(
            actor_id=None,
            actor_role=None,
            action="login_failed",
            resource_type="auth",
            resource_id=username[:80],
            request_id=request_id(request),
            ip_address=request.client.host if request.client else None,
        )
        return render_login(
            request,
            error_message="用户名或密码不正确。",
            username=username,
            status_code=401,
        )

    token = app.state.jwt.create(user=user)
    response = RedirectResponse(redirect_for_role(user["role"]), status_code=303)
    set_auth_cookies(response, token=token)
    await app.state.db.write_audit_log(
        actor_id=user["id"],
        actor_role=user["role"],
        action="login_success",
        resource_type="auth",
        resource_id=user["id"],
        request_id=request_id(request),
        ip_address=request.client.host if request.client else None,
    )
    return response


@app.post("/api/auth/token")
async def issue_token(request: Request, payload: LoginRequest):
    """JSON 登录接口，适合非浏览器客户端使用 Bearer Token。"""

    user = await authenticate(payload.username, payload.password)
    if user is None:
        raise APIError(
            status_code=401,
            code="INVALID_CREDENTIALS",
            message="用户名或密码不正确。",
        )
    token = app.state.jwt.create(user=user)
    return {
        "code": "OK",
        "message": "登录成功。",
        "request_id": request_id(request),
        "data": {
            "access_token": token,
            "token_type": "bearer",
            "role": user["role"],
            "expires_in": int(os.getenv("JWT_EXPIRE_HOURS", "8")) * 3600,
        },
    }


@app.get("/api/auth/me")
async def me(request: Request, user: CurrentUser = Depends(get_current_user)):
    """返回当前 JWT 对应的用户信息。"""

    return {
        "code": "OK",
        "message": "身份有效。",
        "request_id": request_id(request),
        "data": {
            "id": user.id,
            "employee_no": user.record["employee_no"],
            "name": user.name,
            "role": user.role,
        },
    }


@app.post("/auth/logout")
async def logout_form(
    request: Request,
    csrf_token: Annotated[str, Form()],
    user: CurrentUser = Depends(get_current_user),
):
    """注销登录并删除 JWT/CSRF Cookie。"""

    verify_csrf(request, csrf_token)
    await app.state.db.write_audit_log(
        actor_id=user.id,
        actor_role=user.role,
        action="logout",
        resource_type="auth",
        resource_id=user.id,
        request_id=request_id(request),
        ip_address=request.client.host if request.client else None,
    )
    response = RedirectResponse("/login", status_code=303)
    clear_auth_cookies(response)
    return response


@app.get("/", response_class=FileResponse)
async def employee_chat_page(
    user: CurrentUser = Depends(get_current_user),
):
    """员工聊天页；管理员访问时转到管理后台。"""

    if user.role == "admin":
        return RedirectResponse("/admin", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health(request: Request):
    """公开健康检查接口。"""

    database_ok = await app.state.db.healthcheck()
    runtime = app.state.runtime
    return {
        "code": "OK" if database_ok and runtime.ready else "DEGRADED",
        "message": (
            "服务正常。"
            if database_ok and runtime.ready
            else "服务可用，但知识库尚未准备完成。"
        ),
        "request_id": request_id(request),
        "data": {
            "database": database_ok,
            "knowledge_base": runtime.ready,
            "documents": runtime.document_count,
            "chunks": runtime.chunk_count,
            "knowledge_error": runtime.error_message,
        },
    }


def sse_event(event: str, data: dict) -> str:
    """按 SSE 协议格式化一个事件。"""

    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


@app.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
    user: CurrentUser = Depends(require_employee),
):
    """员工问答流。使用 Cookie 认证时同时校验 CSRF Token。"""

    if user.authenticated_by_cookie:
        verify_csrf(request, x_csrf_token)
    if not app.state.runtime.ready:
        raise APIError(
            status_code=503,
            code="KNOWLEDGE_BASE_NOT_READY",
            message="制度知识库尚未准备完成，请稍后再试。",
        )

    try:
        session_id = str(uuid.UUID(payload.session_id.strip()))
    except ValueError as exc:
        raise APIError(
            status_code=400,
            code="INVALID_SESSION_ID",
            message="会话编号格式不正确，应为 UUID。",
        ) from exc

    question = payload.question.strip()
    if not question:
        raise APIError(
            status_code=422,
            code="EMPTY_QUESTION",
            message="问题不能为空。",
        )
    if len(question) > 2000:
        raise APIError(
            status_code=422,
            code="QUESTION_TOO_LONG",
            message="问题不能超过 2,000 个字符。",
        )

    prompt_template = await app.state.db.get_active_prompt()
    if prompt_template is None:
        raise APIError(
            status_code=503,
            code="PROMPT_NOT_CONFIGURED",
            message="系统提示词尚未配置，请联系管理员。",
        )

    try:
        conversation = await app.state.db.ensure_conversation(
            session_id=session_id,
            user_id=user.id,
        )
    except PermissionError as exc:
        raise APIError(
            status_code=403,
            code="SESSION_FORBIDDEN",
            message="当前用户无权访问该会话。",
        ) from exc

    history = await app.state.db.get_recent_history(
        conversation_id=conversation["id"],
        rounds=6,
    )
    trace_id = request_id(request)
    user_message_id = await app.state.db.create_message(
        conversation_id=conversation["id"],
        role="user",
        content=question,
        status="completed",
        request_id=trace_id,
    )
    assistant_message_id = await app.state.db.create_message(
        conversation_id=conversation["id"],
        role="assistant",
        content="",
        status="streaming",
        request_id=trace_id,
        model_provider=app.state.runtime.settings.llm_provider,
        model_name=app.state.runtime.settings.llm_model,
    )

    async def event_generator() -> AsyncIterator[str]:
        """执行 Agent 工具循环，并将业务事件映射为 SSE。"""

        collected: list[str] = []
        citations: list[dict] = []
        runner = AgentRunner(
            database=app.state.db,
            runtime=app.state.runtime,
            token_service=app.state.agent_tokens,
            user=user.record,
            conversation_id=str(conversation["id"]),
            system_prompt=str(prompt_template["content"]),
            question=question,
        )
        try:
            yield sse_event(
                "status",
                {
                    "stage": "planning",
                    "message": "正在读取员工上下文并规划处理步骤…",
                },
            )
            async for event in runner.run(
                question=question,
                history=history,
                request_id=trace_id,
            ):
                if await request.is_disconnected():
                    raise asyncio.CancelledError()
                if event.event == "delta":
                    text = str(event.data.get("text") or "")
                    if not text:
                        continue
                    collected.append(text)
                    yield sse_event("delta", {"text": text})
                elif event.event == "sources":
                    citations = list(event.data.get("sources") or [])
                    yield sse_event("sources", {"sources": citations})
                elif event.event == "draft":
                    yield sse_event("draft", event.data)
                else:
                    yield sse_event(event.event, event.data)

            answer = "".join(collected).strip()
            if not answer:
                raise RuntimeError("模型没有返回有效内容")
            await app.state.db.complete_message(
                message_id=assistant_message_id,
                content=answer,
                citations=citations,
                status="completed",
            )
            yield sse_event(
                "done",
                {
                    "conversation_id": conversation["id"],
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                    "history_rounds": min(len(history) // 2, 6),
                    "prompt_version": int(prompt_template["version"]),
                    "request_id": trace_id,
                },
            )
        except asyncio.CancelledError:
            await app.state.db.complete_message(
                message_id=assistant_message_id,
                content="".join(collected).strip(),
                citations=citations,
                status="interrupted",
                error_code="STREAM_INTERRUPTED",
            )
            raise
        except Exception:
            logger.exception("Agent 问答流生成失败 request_id=%s", trace_id)
            await app.state.db.complete_message(
                message_id=assistant_message_id,
                content="".join(collected).strip(),
                citations=citations,
                status="failed",
                error_code="AGENT_EXECUTION_FAILED",
            )
            yield sse_event(
                "error",
                error_payload(
                    request=request,
                    code="AGENT_EXECUTION_FAILED",
                    message="智能体处理失败，请稍后重试或转为人工工单。",
                    details={
                        "partial_answer": "".join(collected).strip(),
                    },
                ),
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": trace_id,
        },
    )


@app.post("/api/tickets/{draft_id}/confirm")
async def confirm_ticket_draft(
    draft_id: str,
    payload: TicketConfirmRequest,
    request: Request,
    x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
    user: CurrentUser = Depends(require_employee),
):
    """
    员工显式确认并提交 Agent 生成的工单草稿。

    这是模型工具之外的状态转换。即使模型已经完成全部校验，也不能代替员工
    点击确认；提交条件和草稿所有者、版本、有效期均由服务端再次核对。
    """

    if user.authenticated_by_cookie:
        verify_csrf(request, x_csrf_token)
    if not payload.confirmed:
        raise APIError(
            status_code=422,
            code="EXPLICIT_CONFIRMATION_REQUIRED",
            message="请明确确认草稿内容后再提交。",
        )
    if payload.draft_version < 1:
        raise APIError(
            status_code=422,
            code="INVALID_DRAFT_VERSION",
            message="草稿版本号不正确。",
        )

    draft = await app.state.db.get_ticket_draft_for_user(
        draft_id=draft_id,
        user_id=user.id,
    )
    if draft is None:
        raise APIError(
            status_code=404,
            code="TICKET_DRAFT_NOT_FOUND",
            message="未找到该员工名下的工单草稿。",
        )
    if draft.get("status") == "submitted":
        raise APIError(
            status_code=409,
            code="DRAFT_ALREADY_SUBMITTED",
            message="该工单草稿已经提交，请勿重复操作。",
        )
    if int(draft.get("draft_version") or 0) != payload.draft_version:
        raise APIError(
            status_code=409,
            code="DRAFT_VERSION_CONFLICT",
            message="草稿内容已更新，请刷新后重新确认。",
        )
    if draft.get("category") == "human_review":
        raise APIError(
            status_code=409,
            code="HUMAN_REVIEW_IN_PROGRESS",
            message="该事项正在人工复核，无需重复提交普通工单。",
        )
    try:
        expires_at = datetime.fromisoformat(str(draft.get("expires_at") or ""))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise APIError(
            status_code=409,
            code="DRAFT_EXPIRY_INVALID",
            message="草稿有效期信息无效，请重新发起咨询。",
        ) from exc
    if expires_at <= datetime.now(timezone.utc):
        raise APIError(
            status_code=410,
            code="DRAFT_EXPIRED",
            message="工单草稿已过期，请重新发起咨询并完成校验。",
        )

    submitted = await app.state.db.submit_ticket_draft(
        draft_id=draft_id,
        user_id=user.id,
        expected_version=payload.draft_version,
    )
    if submitted is None:
        raise APIError(
            status_code=409,
            code="DRAFT_STATE_CONFLICT",
            message="草稿状态已变化，请刷新后重试。",
        )

    await app.state.db.write_audit_log(
        actor_id=user.id,
        actor_role=user.role,
        action="ticket_draft_submit",
        resource_type="ticket_draft",
        resource_id=draft_id,
        request_id=request_id(request),
        before_data={
            "status": draft.get("status"),
            "draft_version": draft.get("draft_version"),
        },
        after_data={
            "status": submitted.get("status"),
            "draft_version": submitted.get("draft_version"),
            "confirmation_source": submitted.get("confirmation_source"),
        },
        ip_address=request.client.host if request.client else None,
    )
    return {
        "code": "OK",
        "message": "工单已由员工本人确认并提交。",
        "request_id": request_id(request),
        "data": {
            "draft_id": submitted["id"],
            "draft_no": submitted["draft_no"],
            "status": submitted["status"],
            "submitted_at": submitted["submitted_at"],
            "confirmed_at": submitted["confirmed_at"],
        },
    }


def admin_context(
    request: Request,
    user: CurrentUser,
    *,
    active_page: str,
    message: str = "",
    error: str = "",
) -> dict:
    """管理后台模板公共上下文。"""

    return {
        "request": request,
        "current_user": user,
        "csrf_token": request.cookies.get(CSRF_COOKIE_NAME, ""),
        "active_page": active_page,
        "message": message,
        "error": error,
    }


@app.get("/admin")
async def admin_dashboard(
    request: Request,
    user: CurrentUser = Depends(require_admin),
):
    """管理首页。"""

    counts = await app.state.db.dashboard_counts()
    context = admin_context(request, user, active_page="dashboard")
    context.update(
        {
            "title": "管理首页",
            "counts": counts,
            "runtime": app.state.runtime,
        }
    )
    return templates.TemplateResponse(
        request=request,
        name="admin_dashboard.html",
        context=context,
    )


@app.get("/admin/knowledge")
async def admin_knowledge(
    request: Request,
    message: str = "",
    error: str = "",
    user: CurrentUser = Depends(require_admin),
):
    """知识库管理页。"""

    documents = await app.state.db.list_knowledge_documents()
    context = admin_context(
        request,
        user,
        active_page="knowledge",
        message=message,
        error=error,
    )
    context.update(
        {
            "title": "制度知识库",
            "documents": documents,
            "max_file_size": MAX_FILE_SIZE,
        }
    )
    return templates.TemplateResponse(
        request=request,
        name="admin_knowledge.html",
        context=context,
    )


@app.post("/admin/knowledge/upload")
async def admin_upload_knowledge(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    csrf_token: Annotated[str, Form()],
    title: Annotated[str, Form()] = "",
    version: Annotated[str, Form()] = "1.0",
    user: CurrentUser = Depends(require_admin),
):
    """上传制度文件并启动后台解析、切片和索引重建。"""

    verify_csrf(request, csrf_token)
    raw = await file.read(MAX_FILE_SIZE + 1)
    try:
        normalized_name, mime_type = validate_upload(file.filename or "", raw)
    except ValueError as exc:
        raise APIError(
            status_code=415 if "仅支持" in str(exc) else 413,
            code="UNSUPPORTED_FILE" if "仅支持" in str(exc) else "FILE_TOO_LARGE",
            message=str(exc),
        ) from exc

    content_hash = calculate_hash(raw)
    existing = await app.state.db.find_document_by_hash(content_hash)
    if existing is not None:
        raise APIError(
            status_code=409,
            code="DUPLICATE_DOCUMENT",
            message="该制度文件已经存在，无需重复上传。",
        )

    upload_dir = app.state.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    suffix = Path(normalized_name).suffix.lower()
    storage_path = upload_dir / f"{file_id}{suffix}"
    storage_path.write_bytes(raw)

    try:
        document_id, job_id = await create_document_job(
            database=app.state.db,
            file_path=storage_path,
            original_filename=normalized_name,
            mime_type=mime_type,
            created_by=user.id,
            title=title.strip() or None,
            version=version.strip() or "1.0",
        )
    except Exception:
        storage_path.unlink(missing_ok=True)
        raise

    await app.state.db.write_audit_log(
        actor_id=user.id,
        actor_role=user.role,
        action="knowledge_upload",
        resource_type="knowledge_document",
        resource_id=document_id,
        request_id=request_id(request),
        after_data={"file_name": normalized_name, "job_id": job_id},
        ip_address=request.client.host if request.client else None,
    )
    background_tasks.add_task(
        run_ingestion_job,
        database=app.state.db,
        runtime=app.state.runtime,
        job_id=job_id,
        document_id=document_id,
        storage_path=str(storage_path),
    )
    return RedirectResponse(
        "/admin/knowledge?message=文件已上传，正在后台解析和建立索引。",
        status_code=303,
    )


@app.post("/admin/knowledge/{document_id}/delete")
async def admin_delete_knowledge(
    document_id: str,
    request: Request,
    csrf_token: Annotated[str, Form()],
    user: CurrentUser = Depends(require_admin),
):
    """软删除制度文件，并从当前检索索引中移除。"""

    verify_csrf(request, csrf_token)
    deleted = await app.state.db.soft_delete_document(
        document_id=document_id,
        deleted_by=user.id,
    )
    if not deleted:
        raise APIError(
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
            message="未找到可删除的制度文档。",
        )
    await app.state.db.write_audit_log(
        actor_id=user.id,
        actor_role=user.role,
        action="knowledge_delete",
        resource_type="knowledge_document",
        resource_id=document_id,
        request_id=request_id(request),
        ip_address=request.client.host if request.client else None,
    )
    await app.state.runtime.reload_from_database(app.state.db)
    return RedirectResponse(
        "/admin/knowledge?message=制度文档已从问答索引中删除。",
        status_code=303,
    )


@app.get("/api/admin/knowledge/jobs/{job_id}")
async def admin_job_status(
    job_id: str,
    request: Request,
    user: CurrentUser = Depends(require_admin),
):
    """查询制度文件后台入库进度。"""

    job = await app.state.db.get_ingestion_job(job_id)
    if job is None:
        raise APIError(
            status_code=404,
            code="INGESTION_JOB_NOT_FOUND",
            message="未找到指定的入库任务。",
        )
    return {
        "code": "OK",
        "message": "任务状态读取成功。",
        "request_id": request_id(request),
        "data": job,
    }


@app.get("/admin/prompts")
async def admin_prompts(
    request: Request,
    message: str = "",
    user: CurrentUser = Depends(require_admin),
):
    """提示词模板管理和版本历史。"""

    active = await app.state.db.get_active_prompt()
    versions = await app.state.db.list_prompt_templates()
    context = admin_context(
        request,
        user,
        active_page="prompts",
        message=message,
    )
    context.update(
        {
            "title": "提示词模板",
            "active_prompt": active,
            "versions": versions,
        }
    )
    return templates.TemplateResponse(
        request=request,
        name="admin_prompts.html",
        context=context,
    )


@app.post("/admin/prompts")
async def admin_publish_prompt(
    request: Request,
    csrf_token: Annotated[str, Form()],
    content: Annotated[str, Form()],
    change_note: Annotated[str, Form()],
    user: CurrentUser = Depends(require_admin),
):
    """发布新的系统提示词版本，并在下一个问答请求立即生效。"""

    verify_csrf(request, csrf_token)
    normalized = content.strip()
    note = change_note.strip()
    if not normalized:
        raise APIError(
            status_code=422,
            code="PROMPT_EMPTY",
            message="系统提示词不能为空。",
        )
    if len(normalized) > 8000:
        raise APIError(
            status_code=422,
            code="PROMPT_TOO_LONG",
            message="系统提示词不能超过 8,000 个字符。",
        )
    if not note:
        raise APIError(
            status_code=422,
            code="CHANGE_NOTE_REQUIRED",
            message="请填写本次修改原因。",
        )
    old_prompt = await app.state.db.get_active_prompt()
    new_prompt = await app.state.db.publish_prompt(
        content=normalized,
        change_note=note,
        created_by=user.id,
    )
    await app.state.db.write_audit_log(
        actor_id=user.id,
        actor_role=user.role,
        action="prompt_publish",
        resource_type="prompt_template",
        resource_id=new_prompt["id"],
        request_id=request_id(request),
        before_data=old_prompt,
        after_data=new_prompt,
        ip_address=request.client.host if request.client else None,
    )
    return RedirectResponse(
        f"/admin/prompts?message=提示词 v{new_prompt['version']} 已发布并立即生效。",
        status_code=303,
    )


def local_date_bounds(
    date_from: str | None,
    date_to: str | None,
) -> tuple[str | None, str | None]:
    """
    将页面输入的上海本地日期转换为 UTC ISO 时间。

    date_to 采用“次日零点前”的闭区间，避免漏掉最后一天记录。
    """

    start_iso = None
    end_iso = None
    try:
        if date_from:
            start_local = datetime.combine(
                date.fromisoformat(date_from),
                time.min,
                tzinfo=SHANGHAI,
            )
            start_iso = start_local.astimezone(timezone.utc).isoformat()
        if date_to:
            end_local = datetime.combine(
                date.fromisoformat(date_to) + timedelta(days=1),
                time.min,
                tzinfo=SHANGHAI,
            ) - timedelta(microseconds=1)
            end_iso = end_local.astimezone(timezone.utc).isoformat()
    except ValueError as exc:
        raise APIError(
            status_code=422,
            code="INVALID_DATE_FILTER",
            message="日期筛选格式不正确，应使用 YYYY-MM-DD。",
        ) from exc
    return start_iso, end_iso


@app.get("/admin/audit")
async def admin_audit(
    request: Request,
    employee_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    flagged: str = "all",
    page: int = 1,
    message: str = "",
    user: CurrentUser = Depends(require_admin),
):
    """按员工、时间和违规状态筛选对话审计记录。"""

    if flagged not in {"all", "flagged", "unflagged"}:
        raise APIError(
            status_code=422,
            code="INVALID_FLAG_FILTER",
            message="违规状态筛选值不正确。",
        )
    start_iso, end_iso = local_date_bounds(from_date, to_date)
    page = max(page, 1)
    limit = 50
    rows, total = await app.state.db.list_audit_conversations(
        employee_id=employee_id or None,
        date_from=start_iso,
        date_to=end_iso,
        flagged=flagged,
        limit=limit,
        offset=(page - 1) * limit,
    )
    employees = await app.state.db.list_employees()
    context = admin_context(
        request,
        user,
        active_page="audit",
        message=message,
    )
    context.update(
        {
            "title": "对话审计",
            "conversations": rows,
            "employees": employees,
            "filters": {
                "employee_id": employee_id or "",
                "from_date": from_date or "",
                "to_date": to_date or "",
                "flagged": flagged,
            },
            "page": page,
            "total": total,
            "total_pages": max((total + limit - 1) // limit, 1),
        }
    )
    return templates.TemplateResponse(
        request=request,
        name="admin_audit.html",
        context=context,
    )


@app.get("/admin/audit/{conversation_id}")
async def admin_audit_detail(
    conversation_id: str,
    request: Request,
    message: str = "",
    user: CurrentUser = Depends(require_admin),
):
    """显示单个会话的问答、引用、工单草稿和违规标记。"""

    detail = await app.state.db.get_conversation_audit_detail(conversation_id)
    if detail is None:
        raise APIError(
            status_code=404,
            code="CONVERSATION_NOT_FOUND",
            message="未找到指定对话记录。",
        )
    context = admin_context(
        request,
        user,
        active_page="audit",
        message=message,
    )
    context.update(
        {
            "title": "对话审计详情",
            **detail,
        }
    )
    return templates.TemplateResponse(
        request=request,
        name="admin_audit_detail.html",
        context=context,
    )


@app.post("/admin/audit/{conversation_id}/flag")
async def admin_flag_conversation(
    conversation_id: str,
    request: Request,
    csrf_token: Annotated[str, Form()],
    ticket_draft_id: Annotated[str, Form()] = "",
    category: Annotated[str, Form()] = "其他",
    reason: Annotated[str, Form()] = "",
    user: CurrentUser = Depends(require_admin),
):
    """标记会话中的疑似违规工单提交。"""

    verify_csrf(request, csrf_token)
    detail = await app.state.db.get_conversation_audit_detail(conversation_id)
    if detail is None:
        raise APIError(
            status_code=404,
            code="CONVERSATION_NOT_FOUND",
            message="未找到指定对话记录。",
        )
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise APIError(
            status_code=422,
            code="FLAG_REASON_REQUIRED",
            message="请填写疑似违规原因。",
        )
    flag_id = await app.state.db.flag_conversation(
        conversation_id=conversation_id,
        ticket_draft_id=ticket_draft_id or None,
        category=category.strip() or "其他",
        reason=normalized_reason,
        marked_by=user.id,
    )
    await app.state.db.write_audit_log(
        actor_id=user.id,
        actor_role=user.role,
        action="audit_flag",
        resource_type="conversation",
        resource_id=conversation_id,
        request_id=request_id(request),
        after_data={
            "flag_id": flag_id,
            "ticket_draft_id": ticket_draft_id or None,
            "category": category,
            "reason": normalized_reason,
        },
        ip_address=request.client.host if request.client else None,
    )
    return RedirectResponse(
        f"/admin/audit/{conversation_id}?message=已标记为疑似违规工单。",
        status_code=303,
    )


@app.post("/admin/audit/{conversation_id}/unflag")
async def admin_unflag_conversation(
    conversation_id: str,
    request: Request,
    csrf_token: Annotated[str, Form()],
    revoke_reason: Annotated[str, Form()] = "",
    user: CurrentUser = Depends(require_admin),
):
    """撤销当前有效违规标记，并保留撤销原因。"""

    verify_csrf(request, csrf_token)
    normalized_reason = revoke_reason.strip()
    if not normalized_reason:
        raise APIError(
            status_code=422,
            code="REVOKE_REASON_REQUIRED",
            message="请填写撤销标记的原因。",
        )
    count = await app.state.db.revoke_active_flags(
        conversation_id=conversation_id,
        revoked_by=user.id,
        revoke_reason=normalized_reason,
    )
    if count == 0:
        raise APIError(
            status_code=409,
            code="NO_ACTIVE_FLAG",
            message="该会话没有可撤销的违规标记。",
        )
    await app.state.db.write_audit_log(
        actor_id=user.id,
        actor_role=user.role,
        action="audit_unflag",
        resource_type="conversation",
        resource_id=conversation_id,
        request_id=request_id(request),
        after_data={"revoke_reason": normalized_reason, "count": count},
        ip_address=request.client.host if request.client else None,
    )
    return RedirectResponse(
        f"/admin/audit/{conversation_id}?message=违规标记已撤销。",
        status_code=303,
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
