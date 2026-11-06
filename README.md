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
- 制度有效期、适用地区、员工类型、政策组和确认状态校验
- Top-1/Top-2 路由候选、置信度、规则依据与接收角色
- 结构化交接包：未结项、截止时间、制度版本和员工确认
- 运营看板：工单提交、人工复核、Agent 工具耗时与异常决策
- Function Calling 自主工具循环：员工上下文、制度检索、政策校验、
  咨询资料校验、队列路由、建单预览、人工复核和工单状态
- 服务端强制工具依赖和一次性签名凭证，防止模型跳过校验创建工单
- `create_ticket_draft` 执行前强制中断，员工确认预览后才执行并提交
- 每轮模型决策写入 `agent_decision_logs`，记录工具、脱敏参数、耗时和结果摘要
- 统一错误码和中文错误信息

## 启动

建议使用 Python 3.11 或 3.12。

Windows 下可直接双击项目根目录的 `start.bat`。脚本会自动创建或修复
`.venv`、检查并安装依赖、加载 `.env`，然后启动服务。

也可以手动启动：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.employee_ticket_routing.main:app --reload
```

只检查启动环境而不启动服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 -CheckOnly
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

默认提供 DeepSeek 接入。复制 `.env.example` 为 `.env` 后填写密钥即可启用：

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_MODEL=deepseek-chat
```

`DEEPSEEK_API_KEY` 为空时不会发起外部模型请求，系统自动回退到现有
Mock 对话模式，知识检索、Agent 工具链和确认流程仍可离线验证。

通用 OpenAI-Compatible 供应商仍可使用：

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
AGENT_TOKEN_TTL_SECONDS=600
AGENT_MAX_ROUNDS=5
AGENT_CONFIRMATION_TTL_SECONDS=600
```

## 主要接口

- `POST /auth/login`：SSR 表单登录
- `POST /api/auth/token`：JSON 登录并返回 Bearer Token
- `GET /api/auth/me`：当前身份
- `POST /chat`：员工问答与自主工具循环 SSE
- `POST /api/agent/confirmations/{id}/confirm`：确认建单预览后执行工具
- `POST /api/tickets/{draft_id}/confirm`：员工本人确认并提交草稿
- `POST /admin/knowledge/upload`：上传制度
- `POST /admin/knowledge/{id}/delete`：软删除制度
- `GET /api/admin/knowledge/jobs/{job_id}`：查询入库状态
- `POST /admin/prompts`：发布提示词版本
- `GET /admin/audit`：筛选对话审计
- `POST /admin/audit/{id}/flag`：标记疑似违规
- `POST /admin/audit/{id}/unflag`：撤销标记

SSE 常见事件顺序：

`status` → `tool`（可出现多次）→ `delta` →
`confirmation_required`（按需）→ `sources` → `done`

## Harness 执行循环

采用轻量、供应商无关的 Agent Harness，实现：

`模型决策 → 工具执行 → 结果回填 → 再决策`

最多 5 轮。模型可以在同一轮规划多个工具，Harness 会按服务端依赖顺序
串行执行，避免身份、检索和校验工具被并发乱序调用。

模型可以自主决定下一步工具，但不能绕过以下规则：

1. `get_employee_profile` 从登录会话读取员工身份，禁止由模型传入员工 ID。
2. `search_knowledge` 返回本轮有效的 `evidence_id` 和制度来源。
3. `validate_policy` 只能使用本轮检索证据，通过后签发短期政策凭证。
4. 建单前必须通过 `validate_consultation`，检查咨询主题和具体诉求；
   资料不足时先向员工追问。
5. `route_ticket` 同时核验政策与咨询资料凭证，并签发一次性
   `routing_token`。
6. Harness 拦截 `create_ticket_draft`：只流式展示标题和说明，不写数据库。
   员工确认后，独立接口才执行建单并提交。
7. 高风险、劳动争议、个人例外、社保公积金补缴等事项只能转人工。

知识库检索会同时检查制度有效期、确认状态、地区和员工类型。已过期、
尚未确认、已被替代或不适用于当前员工的制度不会进入回答依据。

工具凭证绑定用户、会话、用途和 `jti`，服务端记录一次性消费。签名凭证
不会出现在 SSE、浏览器内存或审计日志中。系统提示词负责告知模型规则，
真正的阻断发生在工具执行器与确认接口。

### 典型轨迹

1. `新人入职薪酬福利是多少`
   `get_employee_profile + search_knowledge → validate_policy → 最终回答`。
   只查询制度，不进入工单链路。
2. `帮我提交我的咨询工单`
   缺少具体诉求时先追问；补充后可走
   `validate_policy → validate_consultation → route_ticket →
   create_ticket_draft（暂停确认）→ 员工确认后建单并提交`。
3. `怎么能不能提交某方面的工单`
   属于“能否提交”的制度咨询，只输出结论，不误解为创建工单的执行指令。

## 阶段 5 分析数据

`agent_decision_logs` 按会话、请求和轮次记录：

- `decision_type`
- `tool_name`
- 脱敏后的 `arguments_json`
- `duration_ms`
- `result_summary`

其中签名 Token 永远不会进入日志，适合后续分析工具选择准确率、平均耗时、
人工中断率和转化路径。

## 测试

```powershell
python -m unittest -v tests.test_agent_flow
```

测试覆盖三种典型轨迹、最多 5 轮限制、确认前零写入、确认后提交、
逐轮日志、高风险转人工和管理员越权。

## 文档说明

PDF 首期只支持文本型 PDF，不支持扫描件 OCR。知识文件最大 20 MB。

知识文件“删除”采用软删除，历史对话中的引用快照仍然保留，便于审计。

## 目录结构

```text
src/employee_ticket_routing/
├── __init__.py
├── main.py                 # FastAPI 程序入口
├── core/                   # Agent、RAG、数据库、安全和入库逻辑
├── static/                 # 员工端页面资源
└── templates/              # 服务端渲染管理页面
```
