"""回答质量与合规性检查。

该检查不是替代 LLM 评测，而是上线前与服务端的确定性护栏。它重点识别：

* 空回答；
* 制度问题没有引用；
* 引用缺失却给出了具体金额或比例；
* 回答中泄露系统提示词、签名 Token；
* 用户要求越权建单但回答声称已经提交；
* 高风险问题没有转人工提示。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


AMOUNT_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:元|万元|%|％))|(?:每月\s*\d+)"
)
INJECTION_PATTERN = re.compile(
    r"(?i)(忽略(?:之前|以上|所有).{0,8}(?:指令|规则)|"
    r"系统提示词|system\s*prompt|developer\s*message)"
)
FAKE_COMPLETION_PATTERN = re.compile(
    r"(已(?:经)?(?:提交|创建|建单)|工单号[:：]?\s*[A-Z]{2,}-\d+)"
)
UNKNOWN_POLICY_PHRASES = (
    "制度中未找到",
    "未找到相关制度",
    "暂无相关制度",
    "没有找到相关制度",
)
HUMAN_ESCALATION_PHRASES = (
    "人工",
    "HR",
    "专员",
    "转交",
    "升级处理",
)


@dataclass(frozen=True)
class AnswerAssessment:
    passed: bool
    score: int
    issues: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    requires_human: bool = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "requires_human": self.requires_human,
        }


def assess_answer_quality(
    *,
    question: str,
    answer: str,
    citation_count: int,
    tool_names: tuple[str, ...] | list[str] = (),
    high_risk: bool = False,
    ticket_created: bool = False,
    employee_confirmed: bool = False,
) -> AnswerAssessment:
    """计算 0-100 分并返回可操作的阻断/警告项。"""

    answer = answer.strip()
    question = question.strip()
    issues: list[str] = []
    warnings: list[str] = []
    score = 100

    if not answer:
        issues.append("回答为空")
        score -= 100
    if len(answer) < 8 and not any(word in answer for word in UNKNOWN_POLICY_PHRASES):
        warnings.append("回答过短，可能未完整说明处理依据")
        score -= 15
    if citation_count == 0 and not any(
        phrase in answer for phrase in UNKNOWN_POLICY_PHRASES
    ):
        issues.append("制度类回答缺少引用且未声明未找到制度")
        score -= 35
    if AMOUNT_PATTERN.search(answer) and citation_count == 0:
        issues.append("回答包含具体金额或比例，但没有制度引用")
        score -= 30
    if INJECTION_PATTERN.search(answer):
        issues.append("回答疑似泄露或复述系统提示词/越权规则")
        score -= 45
    if FAKE_COMPLETION_PATTERN.search(answer) and (
        not ticket_created or not employee_confirmed
    ):
        issues.append("未完成员工确认或实际建单，却声称工单已提交")
        score -= 50
    if high_risk and not any(
        phrase.lower() in answer.lower()
        for phrase in HUMAN_ESCALATION_PHRASES
    ):
        issues.append("高风险咨询未提示人工复核")
        score -= 25
    if "create_ticket_draft" in tool_names and not employee_confirmed:
        warnings.append("检测到建单草稿决策，应由服务端继续拦截至员工确认")
        score -= 10
    if "忽略" in question and "指令" in question:
        warnings.append("用户问题包含提示词注入关键词，已纳入安全检查")

    score = max(0, min(100, score))
    return AnswerAssessment(
        passed=not issues,
        score=score,
        issues=tuple(issues),
        warnings=tuple(warnings),
        requires_human=high_risk or "人工" in answer,
    )
