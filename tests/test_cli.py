"""管理 CLI 参数与纯业务命令测试。"""

from __future__ import annotations

import argparse
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.employee_ticket_routing.cli import build_parser, command_route_preview


class CliTests(unittest.TestCase):
    def test_parser_recognizes_operational_commands(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "export-audit",
                "--format",
                "csv",
                "--output",
                "audit.csv",
            ]
        )
        self.assertEqual(args.command, "export-audit")
        self.assertEqual(args.format, "csv")
        self.assertEqual(args.output, "audit.csv")

    def test_route_preview_outputs_explainable_json(self) -> None:
        args = argparse.Namespace(
            question="我想申请劳动仲裁，公司克扣工资",
            details="",
            risk_flag=None,
        )
        output = StringIO()
        with redirect_stdout(output):
            exit_code = command_route_preview(args)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["queue"], "human_review")
        self.assertTrue(payload["requires_human_review"])
        self.assertTrue(payload["reasons"])


if __name__ == "__main__":
    unittest.main()
