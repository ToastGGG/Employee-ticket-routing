"""运行配置解析、校验和安全快照测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.employee_ticket_routing.core.runtime_config import (
    ConfigurationError,
    load_runtime_settings,
    summarize_configuration,
)


class RuntimeConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("C:/workspace/employee-ticket-routing")

    def test_deepseek_without_key_falls_back_and_keeps_secret_out_of_snapshot(
        self,
    ) -> None:
        settings = load_runtime_settings(
            {
                "APP_ENV": "development",
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "",
                "JWT_SECRET": "super-secret-value-that-must-not-leak",
            },
            project_root=self.root,
        )

        self.assertEqual(settings.llm.provider_requested, "deepseek")
        self.assertEqual(settings.llm.provider_effective, "mock")
        self.assertIn("DEEPSEEK_API_KEY", settings.llm.fallback_reason or "")
        snapshot_text = repr(settings.masked_snapshot())
        self.assertNotIn("super-secret-value", snapshot_text)
        self.assertTrue(settings.security.jwt_secret_configured)

    def test_relative_paths_are_resolved_against_project_root(self) -> None:
        settings = load_runtime_settings(
            {
                "DATA_DIR": "runtime-data",
                "SQLITE_PATH": "./runtime-data/app.sqlite3",
                "UPLOAD_DIR": "runtime-data/uploads",
                "RAG_CACHE_DIR": "runtime-data/faiss",
                "KNOWLEDGE_FILE": "docs/policy.md",
            },
            project_root=self.root,
        )

        self.assertEqual(
            settings.storage.sqlite_path,
            (self.root / "runtime-data/app.sqlite3").resolve(strict=False),
        )
        self.assertEqual(
            settings.storage.upload_dir,
            (self.root / "runtime-data/uploads").resolve(strict=False),
        )
        self.assertEqual(
            settings.storage.knowledge_file,
            (self.root / "docs/policy.md").resolve(strict=False),
        )

    def test_invalid_integer_and_overlap_are_reported_together(self) -> None:
        settings = load_runtime_settings(
            {
                "AGENT_MAX_ROUNDS": "99",
                "CHUNK_SIZE": "100",
                "CHUNK_OVERLAP": "100",
                "JWT_EXPIRE_HOURS": "not-a-number",
            },
            project_root=self.root,
        )
        keys = {issue.key for issue in settings.issues}
        self.assertEqual(
            keys,
            {"AGENT_MAX_ROUNDS", "CHUNK_OVERLAP", "JWT_EXPIRE_HOURS"},
        )
        with self.assertRaises(ConfigurationError):
            settings.require_valid()

    def test_production_warnings_do_not_block_but_are_visible(self) -> None:
        settings = load_runtime_settings(
            {
                "APP_ENV": "production",
                "COOKIE_SECURE": "false",
                "JWT_SECRET": "",
                "LLM_PROVIDER": "mock",
            },
            project_root=self.root,
        )

        self.assertEqual(len(settings.blocking_issues), 0)
        warning_keys = {issue.key for issue in settings.warnings}
        self.assertEqual(warning_keys, {"COOKIE_SECURE", "JWT_SECRET"})
        summary = summarize_configuration(settings)
        self.assertIn("配置检查", summary)

    def test_invalid_provider_url_is_rejected(self) -> None:
        settings = load_runtime_settings(
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "configured",
                "DEEPSEEK_API_URL": "api.deepseek.com/chat",
            },
            project_root=self.root,
        )

        self.assertEqual(len(settings.blocking_issues), 1)
        self.assertEqual(settings.blocking_issues[0].key, "DEEPSEEK_API_URL")


if __name__ == "__main__":
    unittest.main()
