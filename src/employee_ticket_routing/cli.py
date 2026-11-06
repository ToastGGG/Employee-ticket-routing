"""项目管理与运维 CLI。

用法示例::

    python -m src.employee_ticket_routing config-check
    python -m src.employee_ticket_routing db-stats
    python -m src.employee_ticket_routing route-preview "我想咨询公积金补缴"
    python -m src.employee_ticket_routing export-audit --format csv --output audit.csv

CLI 只使用标准库和项目已有依赖，便于在服务器上快速诊断，不要求启动 Web 服务。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .core.db import SQLiteDatabase
from .core.runtime_config import (
    ConfigurationError,
    RuntimeSettings,
    load_runtime_settings,
    summarize_configuration,
)
from .domain.ticket import RoutingSignals
from .repositories.audit_export import (
    AuditExportOptions,
    collect_audit_records,
    database_summary,
    write_audit_export,
)
from .services.ticket_priority import classify_and_route


def _settings_from_args(args: argparse.Namespace) -> RuntimeSettings:
    return load_runtime_settings(project_root=Path(args.project_root))


def command_config_check(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    print(summarize_configuration(settings))
    if args.json:
        print(json.dumps(settings.masked_snapshot(), ensure_ascii=False, indent=2))
    try:
        settings.require_valid()
    except ConfigurationError:
        return 2
    return 0


def command_db_stats(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    database = SQLiteDatabase(settings.storage.sqlite_path)
    if not settings.storage.sqlite_path.exists():
        print(
            f"数据库不存在：{settings.storage.sqlite_path}",
            file=sys.stderr,
        )
        return 2
    summary = asyncio.run(database_summary(database))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


def command_export_audit(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    if not settings.storage.sqlite_path.exists():
        print(
            f"数据库不存在：{settings.storage.sqlite_path}",
            file=sys.stderr,
        )
        return 2
    database = SQLiteDatabase(settings.storage.sqlite_path)
    options = AuditExportOptions(
        employee_no=args.employee,
        date_from=args.date_from,
        date_to=args.date_to,
        flagged_only=args.flagged_only,
        limit=args.limit,
        include_content=not args.no_content,
    )
    try:
        records = asyncio.run(collect_audit_records(database, options))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    target = write_audit_export(
        Path(args.output),
        records,
        export_format=args.format,
    )
    print(f"已导出 {len(records)} 条记录：{target}")
    return 0


def command_route_preview(args: argparse.Namespace) -> int:
    decision = classify_and_route(
        RoutingSignals(
            question=args.question,
            details=args.details or "",
            risk_flags=tuple(args.risk_flag or ()),
        )
    )
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="employee-ticket-routing",
        description="员工薪酬福利咨询工单分流系统管理工具",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="项目根目录，默认为仓库根目录",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config-check", help="检查运行配置")
    config_parser.add_argument("--json", action="store_true", help="同时输出脱敏 JSON")
    config_parser.set_defaults(handler=command_config_check)

    db_parser = subparsers.add_parser("db-stats", help="统计数据库核心表")
    db_parser.add_argument("--json", action="store_true", help="输出 JSON")
    db_parser.set_defaults(handler=command_db_stats)

    export_parser = subparsers.add_parser("export-audit", help="导出对话审计记录")
    export_parser.add_argument("--employee", help="员工工号")
    export_parser.add_argument("--date-from", help="开始日期 YYYY-MM-DD")
    export_parser.add_argument("--date-to", help="结束日期 YYYY-MM-DD")
    export_parser.add_argument("--flagged-only", action="store_true")
    export_parser.add_argument("--limit", type=int, default=5000)
    export_parser.add_argument("--no-content", action="store_true")
    export_parser.add_argument("--format", choices=("json", "csv"), default="json")
    export_parser.add_argument("--output", required=True, help="输出文件路径")
    export_parser.set_defaults(handler=command_export_audit)

    route_parser = subparsers.add_parser("route-preview", help="预览分类和 SLA")
    route_parser.add_argument("question", help="员工问题")
    route_parser.add_argument("--details", help="补充说明")
    route_parser.add_argument(
        "--risk-flag",
        action="append",
        help="风险标签，可重复",
    )
    route_parser.set_defaults(handler=command_route_preview)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
