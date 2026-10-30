# Employee Ticket Routing

员工薪酬福利咨询工单分流系统的 FastAPI 全栈原型。

## 功能

- JWT 登录与员工、管理员接口级权限控制
- 员工单页问答入口与 SSE 流式回答
- SQLite 会话、消息、知识文档、工单草稿和审计记录
- Markdown/PDF 上传、自动切片、向量化和 FAISS 索引重建
- 最近 6 轮对话上下文
- BM25 + FAISS 混合检索与制度引用
- 管理员知识库、提示词模板和对话审计页面
- 动态系统提示词，修改后下一次问答立即生效
- 疑似违规工单标记与撤销记录
- Function Calling 自主工具循环：员工上下文、制度检索、政策校验、
  报销校验、队列路由、草稿、人工复核和工单状态
- 服务端强制工具依赖和一次性签名凭证，防止模型跳过校验创建工单
- 工单草稿必须由员工本人显式确认后才能提交，15 分钟后自动失效
- 统一错误码和中文错误信息

## 启动

建议使用 Python 3.11 或 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.app:app --reload
```

打开：

- 员工/管理员登录：<http://127.0.0.1:8000/login>
- 员工问答：<http://127.0.0.1:8000/>
- 管理后台：<http://127.0.0.1:8000/admin>

## 本地演示账号

| 角色 | 默认登录名 | 默认密码 |
|---|---|---|
| 员工 | `employee` | `Employee@123456` |
| 管理员 | `admin` | `Admin@123456` |

正式部署必须通过环境变量修改账号密码，并启用 HTTPS。

```text
SEED_ADMIN_USERNAME=admin
SEED_ADMIN_PASSWORD=请修改
SEED_EMPLOYEE_USERNAME=employee
SEED_EMPLOYEE_PASSWORD=请修改
JWT_SECRET=请使用足够长的随机值
COOKIE_SECURE=true
```

JWT 默认写入 HttpOnly Cookie；JSON 登录接口也可返回 Bearer Token。

## 模型配置

```text
LLM_PROVIDER=openai_compatible
LLM_API_URL=https://供应商地址/v1/chat/completions
LLM_API_KEY=...
LLM_MODEL=...

EMBEDDING_PROVIDER=remote
EMBEDDING_API_URL=https://供应商地址/v1/embeddings
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=...
```

未配置 API Key 时自动使用 Mock LLM 和 hash 向量，便于本地验证。

其他配置：

```text
KNOWLEDGE_FILE=./docs/薪酬福利制度示例.md
SQLITE_PATH=./data/employee_chat.sqlite3
DATA_DIR=./data
UPLOAD_DIR=./data/uploads
RAG_CACHE_DIR=./data/faiss_cache
JWT_EXPIRE_HOURS=8
TOP_K=3
CHUNK_SIZE=500
CHUNK_OVERLAP=80
AGENT_MAX_STEPS=12
AGENT_TOKEN_TTL_SECONDS=600
INVOICE_HUMAN_REVIEW_THRESHOLD=3000
```

## 主要接口

- `POST /auth/login`：SSR 表单登录
- `POST /api/auth/token`：JSON 登录并返回 Bearer Token
- `GET /api/auth/me`：当前身份
- `POST /chat`：员工问答与自主工具循环 SSE
- `POST /api/tickets/{draft_id}/confirm`：员工本人确认并提交草稿
- `POST /admin/knowledge/upload`：上传制度
- `POST /admin/knowledge/{id}/delete`：软删除制度
- `GET /api/admin/knowledge/jobs/{job_id}`：查询入库状态
- `POST /admin/prompts`：发布提示词版本
- `GET /admin/audit`：筛选对话审计
- `POST /admin/audit/{id}/flag`：标记疑似违规
- `POST /admin/audit/{id}/unflag`：撤销标记

SSE 常见事件顺序：

`status` → `tool`（可出现多次）→ `draft`（按需）→ `delta` →
`sources` → `done`

## Agent 工具约束

模型可以自主决定下一步工具，但不能绕过以下服务端规则：

1. `get_employee_profile` 从登录会话读取员工身份，禁止由模型传入员工 ID。
2. `search_knowledge` 返回本轮有效的 `evidence_id` 和制度来源。
3. `validate_policy` 只能使用本轮检索证据，通过后签发短期政策凭证。
4. 报销事项必须继续通过 `validate_invoice`。
5. `route_ticket` 消费并核验前置凭证；报销工单缺少发票凭证时直接拒绝。
6. `create_ticket_draft` 必须持有一次性 `routing_token`；高风险工单只能
   调用 `request_human_review`。
7. 草稿创建成功也不等于提交。员工在前端点击“确认并提交”后，后端还会
   校验所有者、草稿版本、有效期和显式确认字段。

工具凭证绑定用户、会话、用途和 `jti`，服务端记录一次性消费。签名凭证
不会出现在 SSE、浏览器内存或审计日志中。系统提示词负责告知模型规则，
真正的阻断发生在工具执行器与确认接口。

## 测试

```powershell
python -m unittest -v tests.test_agent_flow
```

测试覆盖报销草稿显式确认、重复提交拦截、高风险转人工和管理员越权提交。

## 文档说明

PDF 首期只支持文本型 PDF，不支持扫描件 OCR。知识文件最大 20 MB。

知识文件“删除”采用软删除，历史对话中的引用快照仍然保留，便于审计。
