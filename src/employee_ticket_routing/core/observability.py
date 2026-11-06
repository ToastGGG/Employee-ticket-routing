"""轻量级进程内指标采集。

课程项目通常只需要观察“服务是否活着”，但真实工单系统还需要知道：

* 哪些接口失败；
* SSE 长连接是否被中断；
* LLM 和 Agent 工具耗时是否异常；
* 活跃请求数是否持续升高；
* 某个发布后 5xx 是否立即上升。

本模块提供无第三方依赖的计数器与延迟摘要，并输出 JSON 和 Prometheus 文本。
它适用于单进程原型；生产多实例部署时应替换为 Prometheus/OpenTelemetry 后端。
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class LatencySummary:
    """一组延迟样本的稳定摘要。"""

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


def percentile(values: list[float], quantile: float) -> float:
    """使用最近秩法计算分位数，空样本返回 0。"""

    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize_latencies(values: Iterable[float]) -> LatencySummary:
    samples = [max(0.0, float(value)) for value in values]
    if not samples:
        return LatencySummary(0, 0, 0, 0, 0, 0, 0)
    return LatencySummary(
        count=len(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        mean_ms=sum(samples) / len(samples),
        p50_ms=percentile(samples, 0.50),
        p95_ms=percentile(samples, 0.95),
        p99_ms=percentile(samples, 0.99),
    )


def _labels_key(labels: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """将标签转换为可哈希、可稳定排序的键。"""

    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    parts = []
    for key, value in labels:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{key}="{escaped}"')
    return "{" + ",".join(parts) + "}"


def normalize_request_route(path: str) -> str:
    """把动态路径折叠为低基数 route 标签。

    如果直接记录原始 URL，每个确认 ID、工单 ID、文档 ID 都会变成新的指标维度，
    最终造成内存持续膨胀。固定接口保持不变，动态段使用占位符。
    """

    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return "/"
    normalized: list[str] = []
    for index, segment in enumerate(segments):
        previous = segments[index - 1] if index else ""
        if previous in {
            "confirmations",
            "tickets",
            "jobs",
            "knowledge",
            "audit",
        }:
            normalized.append(":id")
        elif len(segment) >= 24 and "-" in segment:
            normalized.append(":id")
        else:
            normalized.append(segment)
    return "/" + "/".join(normalized)


class ServiceMetrics:
    """线程安全的进程内计数器与延迟指标。"""

    def __init__(self, *, latency_capacity: int = 4096) -> None:
        self._started_monotonic = time.monotonic()
        self._latency_capacity = max(latency_capacity, 128)
        self._lock = threading.RLock()
        self._counters: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            float,
        ] = defaultdict(float)
        self._latencies: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            deque[float],
        ] = defaultdict(lambda: deque(maxlen=self._latency_capacity))
        self._active_requests = 0

    def increment(
        self,
        name: str,
        value: float = 1.0,
        **labels: Any,
    ) -> None:
        """增加计数器；name 使用下划线命名，标签保持低基数。"""

        key = (name, _labels_key(labels))
        with self._lock:
            self._counters[key] += float(value)

    def observe(
        self,
        name: str,
        duration_ms: float,
        **labels: Any,
    ) -> None:
        """记录一个非负延迟样本。"""

        key = (name, _labels_key(labels))
        with self._lock:
            self._latencies[key].append(max(0.0, float(duration_ms)))

    def start_request(self) -> None:
        with self._lock:
            self._active_requests += 1
            self._counters[("http_active_requests", ())] = float(
                self._active_requests
            )

    def finish_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        method = method.upper()
        status_class = f"{int(status_code) // 100}xx"
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._counters[("http_active_requests", ())] = float(
                self._active_requests
            )
        self.increment(
            "http_requests_total",
            method=method,
            route=route,
            status=status_class,
        )
        self.observe(
            "http_request_duration_ms",
            duration_ms,
            method=method,
            route=route,
        )
        if status_code >= 500:
            self.increment(
                "http_server_error_total",
                method=method,
                route=route,
            )

    def record_tool(
        self,
        *,
        tool_name: str,
        duration_ms: float,
        success: bool,
    ) -> None:
        self.increment(
            "agent_tool_calls_total",
            tool=tool_name,
            success=str(success).lower(),
        )
        self.observe(
            "agent_tool_duration_ms",
            duration_ms,
            tool=tool_name,
        )

    def record_llm(
        self,
        *,
        provider: str,
        duration_ms: float,
        success: bool,
        status_class: str = "2xx",
    ) -> None:
        self.increment(
            "llm_requests_total",
            provider=provider,
            success=str(success).lower(),
            status=status_class,
        )
        self.observe(
            "llm_request_duration_ms",
            duration_ms,
            provider=provider,
        )

    def snapshot(self) -> dict[str, Any]:
        """返回适合管理接口与单元测试的 JSON 快照。"""

        with self._lock:
            counters = list(self._counters.items())
            latencies = [
                (key, list(samples))
                for key, samples in self._latencies.items()
            ]
            active_requests = self._active_requests
            uptime_seconds = time.monotonic() - self._started_monotonic

        counter_payload = [
            {
                "name": key[0],
                "labels": dict(key[1]),
                "value": value,
            }
            for key, value in sorted(counters)
        ]
        latency_payload = [
            {
                "name": key[0],
                "labels": dict(key[1]),
                **summarize_latencies(samples).to_dict(),
            }
            for key, samples in sorted(latencies)
        ]
        return {
            "uptime_seconds": round(uptime_seconds, 2),
            "active_requests": active_requests,
            "counters": counter_payload,
            "latencies": latency_payload,
        }

    def render_prometheus(self) -> str:
        """输出 Prometheus 文本格式，便于接入现有监控系统。"""

        with self._lock:
            counters = list(self._counters.items())
            latencies = [
                (key, list(samples))
                for key, samples in self._latencies.items()
            ]

        lines = [
            "# HELP employee_app_uptime_seconds Application uptime.",
            "# TYPE employee_app_uptime_seconds gauge",
            (
                "employee_app_uptime_seconds "
                f"{time.monotonic() - self._started_monotonic:.3f}"
            ),
        ]
        for (name, labels), value in sorted(counters):
            metric_name = f"employee_{name}"
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name}{_format_labels(labels)} {value:.6f}")
        for (name, labels), samples in sorted(latencies):
            summary = summarize_latencies(samples)
            metric_name = f"employee_{name}"
            lines.append(f"# TYPE {metric_name} summary")
            for suffix, value in (
                ("count", summary.count),
                ("sum", summary.mean_ms * summary.count),
                ("p50", summary.p50_ms),
                ("p95", summary.p95_ms),
                ("p99", summary.p99_ms),
            ):
                lines.append(
                    f"{metric_name}_{suffix}"
                    f"{_format_labels(labels)} {value:.3f}"
                )
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """仅用于测试和进程内显式维护。"""

        with self._lock:
            self._counters.clear()
            self._latencies.clear()
            self._active_requests = 0
            self._started_monotonic = time.monotonic()


SERVICE_METRICS = ServiceMetrics()
