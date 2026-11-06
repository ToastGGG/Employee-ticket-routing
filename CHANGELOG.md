# Changelog

所有重要变更记录在此文件中。

## [Unreleased]

### Planned

- PostgreSQL 仓储适配器；
- Redis 会话与确认缓存；
- 工单 SLA 日历与节假日规则；
- 管理端指标仪表盘；
- 知识库多版本对比与回滚；
- 更完整的对抗评测集。

## [0.6.0] - 2026-11-06

### Added

- 运行配置集中解析、范围校验和敏感字段脱敏快照；
- 敏感信息正则脱敏、结构化参数清洗和安全异常摘要；
- 进程内请求、Agent 工具和 LLM 指标采集；
- 管理员指标快照与 Prometheus 文本接口；
- 工单领域模型、优先级、队列、SLA 和状态机；
- 对话上下文预算、去重与引用合并；
- 回答质量、引用完整性、提示词注入和虚假建单检查；
- 对话审计 JSON/CSV 导出和数据库摘要；
- `config-check`、`db-stats`、`route-preview`、`export-audit` CLI；
- Ruff、PyTest、GitHub Actions、Issue/PR 模板和本地质量脚本。

### Security

- 生产环境对不安全 Cookie 和缺失 JWT Secret 给出配置警告；
- 审计导出默认对手机号、邮箱、证件号和签名参数脱敏；
- Agent 建单状态仍必须经过员工显式确认。

## [0.5.0] - 2026-11-05

### Added

- DeepSeek/OpenAI Compatible 模型适配与无密钥 Mock 回退；
- Agent Harness、工具依赖、一次性凭证和确认中断；
- 知识库、提示词和对话审计管理后台；
- 上线交付运行手册、设计文档、测试方案和报价材料。
