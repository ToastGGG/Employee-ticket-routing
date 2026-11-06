# 贡献指南

本项目是员工薪酬福利咨询工单分流系统的课程实践，也按可维护的企业项目组织代码。
提交代码前请先阅读本指南、`README.md` 和 `docs/员工薪酬福利咨询工单分流系统上线交付运行手册.docx`。

## 开发原则

1. 先定义业务边界，再写路由或模型提示词；
2. 不绕过员工显式确认直接创建或提交工单；
3. 知识库无依据时明确回答“制度中未找到”；
4. 高风险、劳动仲裁、个人信息泄露等事项必须转人工；
5. 不将 API Key、JWT Secret、员工薪酬数据提交到 Git；
6. 新增行为必须有测试，修复缺陷必须补回归用例；
7. 只增加有价值的功能、模型、测试和文档，不用重复代码或空白行制造虚假指标。

## 分支与提交

建议使用短分支：

- `feature/<short-name>`
- `fix/<short-name>`
- `docs/<short-name>`
- `test/<short-name>`

提交信息使用 Conventional Commits 风格：

```text
feat: add ticket SLA calendar calculation
fix: prevent confirmation reuse after expiry
test: cover managed prompt rollback
docs: document audit export workflow
```

一次提交应保持单一意图。不要在同一提交中混入无关格式化、依赖升级和业务修改。

## 本地验证

Windows 一键质量检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\quality.ps1
```

手工执行：

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -p "test_*.py" -v
ruff check src tests
```

运行应用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 -CheckOnly
start.bat
```

## 代码结构约定

- `domain/`：不依赖 FastAPI 的枚举、实体、状态机；
- `services/`：分类、SLA、上下文、回答质量等可复用业务规则；
- `repositories/`：只读查询、导出和报表；
- `core/`：数据库、安全、RAG、Agent、配置、指标和脱敏；
- `main.py`：HTTP 入口、鉴权与页面组合，不继续堆积可独立测试的业务规则；
- `tests/`：按模块和业务场景组织，测试名称说明预期行为；
- `docs/`：需求、设计、测试、报价、运行手册等交付证据。

## 新增工具调用

新增 Agent 工具必须同时提供：

- 工具名称、参数 Schema 与返回 Schema；
- 调用前置条件和依赖顺序；
- 用户权限校验；
- 超时、失败与重试策略；
- 是否属于外部副作用；
- 是否需要员工确认；
- 日志脱敏字段；
- 单元测试和至少一条多轮轨迹测试。

涉及建单的工具必须遵循：

```text
身份读取 → 制度检索 → 制度校验 → 咨询资料校验
→ 队列路由 → 员工确认 → 创建草稿 → 提交
```

模型不能通过提示词改变该顺序，最终约束必须由服务端执行器完成。

## 新增知识库文件

制度文件应包含：

- 制度标题、版本、生效日期、失效日期；
- 适用公司、地区、员工类型和政策组；
- 可核验的条款编号；
- 责任部门和版本审批人；
- 不得包含真实员工个人薪酬明细。

上传后需检查解析、切片、引用来源与过期制度过滤。

## 安全与隐私

- 使用 `.env.example` 描述变量，不提交 `.env`；
- 测试使用虚构姓名、手机号和工单号；
- 日志通过 `core/redaction.py` 脱敏；
- 生产部署启用 HTTPS、`COOKIE_SECURE=true` 和强 JWT Secret；
- 审计导出按最小权限、最小字段和留存制度执行；
- 发现疑似密钥或个人信息泄露，应立即停止共享并通知项目负责人。
