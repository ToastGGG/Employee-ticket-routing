"""DeepSeek 配置与无密钥回退测试。"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.employee_ticket_routing.core.stage1_rag_cli import (
    OpenAICompatibleLLMProvider,
    Settings,
    create_llm_provider,
)


class LLMConfigurationTests(unittest.TestCase):
    def build_settings(self, env: dict[str, str]) -> Settings:
        with patch.dict(os.environ, env, clear=True):
            return Settings.from_env(
                document_path=Path("policy.md"),
                cache_dir=Path(".rag_cache"),
            )

    def test_deepseek_without_key_falls_back_to_mock(self) -> None:
        settings = self.build_settings(
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "",
            }
        )
        self.assertEqual(settings.llm_provider_requested, "deepseek")
        self.assertEqual(settings.llm_provider, "mock")
        self.assertEqual(settings.llm_model, "deepseek-chat")
        self.assertIn("DEEPSEEK_API_KEY", settings.llm_fallback_reason or "")

    def test_deepseek_with_key_uses_official_defaults(self) -> None:
        settings = self.build_settings(
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-only-key",
            }
        )
        self.assertEqual(settings.llm_provider, "deepseek")
        self.assertEqual(
            settings.llm_api_url,
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(settings.llm_model, "deepseek-chat")
        self.assertIsNone(settings.llm_fallback_reason)
        provider = create_llm_provider(settings)
        self.assertIsInstance(provider, OpenAICompatibleLLMProvider)

    def test_generic_openai_compatible_configuration_is_unchanged(self) -> None:
        settings = self.build_settings(
            {
                "LLM_API_KEY": "generic-key",
                "LLM_API_URL": "https://example.com/v1/chat/completions",
                "LLM_MODEL": "example-model",
            }
        )
        self.assertEqual(settings.llm_provider, "openai_compatible")
        self.assertEqual(settings.llm_api_key, "generic-key")
        self.assertEqual(settings.llm_model, "example-model")


if __name__ == "__main__":
    unittest.main()
