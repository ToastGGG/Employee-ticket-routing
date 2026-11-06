"""敏感数据脱敏模块测试。"""

from __future__ import annotations

import unittest

from src.employee_ticket_routing.core.redaction import (
    looks_like_secret,
    mask_value,
    redact_mapping,
    redact_text,
    redact_tool_arguments,
    safe_exception_message,
)


class RedactionTests(unittest.TestCase):
    def test_text_redaction_covers_common_employee_identifiers(self) -> None:
        text = (
            "联系 13800138000 或 employee@example.com，"
            "身份证 110105199001011234，授权 Bearer abc.def.ghi"
        )
        redacted = redact_text(text)
        self.assertNotIn("13800138000", redacted)
        self.assertNotIn("employee@example.com", redacted)
        self.assertNotIn("110105199001011234", redacted)
        self.assertNotIn("abc.def.ghi", redacted)

    def test_nested_mapping_is_redacted_without_mutating_source(self) -> None:
        source = {
            "employee": {
                "name": "张三",
                "phone": "13900139000",
                "email": "zhangsan@example.com",
            },
            "api_key": "abcdefghijklmnopqrstuvwxyz",
            "items": [{"password": "P@ssword123"}],
        }
        result = redact_mapping(source)

        self.assertEqual(source["employee"]["phone"], "13900139000")
        self.assertNotEqual(result["employee"]["phone"], "13900139000")
        self.assertEqual(result["api_key"], "***wxyz")
        self.assertEqual(result["items"][0]["password"], "***")

    def test_tool_arguments_hide_signing_tokens(self) -> None:
        result = redact_tool_arguments(
            {
                "policy_validation_token": "signed-token",
                "routing_token": "signed-routing-token",
                "title": "薪酬咨询",
            }
        )
        self.assertEqual(result["policy_validation_token"], "<已省略>")
        self.assertEqual(result["routing_token"], "<已省略>")
        self.assertEqual(result["title"], "薪酬咨询")

    def test_long_text_is_truncated_with_explicit_marker(self) -> None:
        result = redact_text("a" * 100, max_length=20)
        self.assertIn("已截断", result)
        self.assertLess(len(result), 100)

    def test_secret_detection_and_exception_summary(self) -> None:
        self.assertTrue(looks_like_secret("sk-abcdefghijklmnopqrstuvwx"))
        self.assertFalse(looks_like_secret("short"))
        summary = safe_exception_message(
            RuntimeError("failed with token=secret-value")
        )
        self.assertNotIn("secret-value", summary)
        self.assertIn("RuntimeError", summary)

    def test_short_sensitive_value_never_reveals_suffix(self) -> None:
        self.assertEqual(mask_value("ab12"), "***")
        self.assertEqual(mask_value("abcdefghi", visible_suffix=2), "***hi")


if __name__ == "__main__":
    unittest.main()
