"""进程内指标模块测试。"""

from __future__ import annotations

import unittest

from src.employee_ticket_routing.core.observability import (
    ServiceMetrics,
    normalize_request_route,
    percentile,
    summarize_latencies,
)


class ObservabilityTests(unittest.TestCase):
    def test_percentiles_are_deterministic_for_known_samples(self) -> None:
        values = [10, 20, 30, 40, 50]
        self.assertEqual(percentile(values, 0.50), 30)
        self.assertEqual(percentile(values, 0.95), 50)
        self.assertEqual(percentile([], 0.95), 0)

    def test_latency_summary_returns_all_reported_fields(self) -> None:
        summary = summarize_latencies([1, 2, 3, 4]).to_dict()
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min_ms"], 1)
        self.assertEqual(summary["max_ms"], 4)
        self.assertEqual(summary["mean_ms"], 2.5)

    def test_dynamic_routes_are_collapsed(self) -> None:
        self.assertEqual(
            normalize_request_route(
                "/api/agent/confirmations/123e4567-e89b-12d3-a456-426614174000/confirm"
            ),
            "/api/agent/confirmations/:id/confirm",
        )
        self.assertEqual(
            normalize_request_route("/admin/knowledge/document-123/delete"),
            "/admin/knowledge/:id/delete",
        )

    def test_metrics_record_request_tool_and_llm(self) -> None:
        metrics = ServiceMetrics()
        metrics.start_request()
        metrics.finish_request(
            method="post",
            route="/chat",
            status_code=200,
            duration_ms=85,
        )
        metrics.record_tool(
            tool_name="search_knowledge",
            duration_ms=20,
            success=True,
        )
        metrics.record_llm(
            provider="mock",
            duration_ms=35,
            success=True,
        )
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["active_requests"], 0)
        names = {row["name"] for row in snapshot["counters"]}
        self.assertIn("http_requests_total", names)
        self.assertIn("agent_tool_calls_total", names)
        self.assertIn("llm_requests_total", names)

    def test_prometheus_output_contains_summary_and_escaped_labels(self) -> None:
        metrics = ServiceMetrics()
        metrics.increment("custom_total", route='a"b')
        metrics.observe("custom_duration_ms", 12.5, route="/chat")
        output = metrics.render_prometheus()
        self.assertIn("employee_custom_total", output)
        self.assertIn('route="a\\"b"', output)
        self.assertIn("employee_custom_duration_ms_p95", output)


if __name__ == "__main__":
    unittest.main()
