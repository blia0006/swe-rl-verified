#!/usr/bin/env python3
"""
判分核心：把 pytest 执行结果换算成 reward
=========================================

线 A（沙箱内 Agent）与线 B（GRPO 训练的 reward function）共用这一份逻辑，
保证「采集时看到的分」与「训练时拿到的分」定义完全一致。

课题规定的奖励口径：
    reward = fail→pass 测试数 / 总相关测试数

本模块在此基础上做了两处**必要**的加固，均不改变主体口径：

一、防 reward hacking（继承上一轮，实际拦截过 6 次）
    · PASS_TO_PASS 出现回归 fail → 整体判 0
      否则模型可以删掉/改坏无关测试，让 F2P "看起来"通过
    · 测试收集失败（collect_error）→ 判 0
      否则"把文件改坏导致 pytest 收集不到用例"会被误判成通过

二、reward 分档（针对上一轮定量诊断出的主因）
    上一轮只分「能否 apply」两档，导致 51% 的 collect_error 样本
    （patch 能应用但把代码插坏）与「位置正确」的样本同得 0.2 分，
    等于告诉模型"写歪的和写对的一样值钱"，格式能力学不动。
    本轮四档（见 `compute_reward` 的 stage 参数）：

        0.00  没抽到 patch / 路径不存在 / 无法 apply
        0.05  apply 成功，但测试无法收集（代码被插坏）   ← 新增，与下一档差 4 倍
        0.20  apply 成功且测试可正常收集
        0.20 + 0.80 × F2P通过率                        ← 课题口径，占主体权重

    权重可由环境变量覆盖（REWARD_APPLY_BONUS / REWARD_TEST_WEIGHT /
    REWARD_COLLECT_ERROR），便于做消融对照。

自测：
    python3 pipeline/reward.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum


# ------------------------------------------------------------------ 可调权重

def _envf(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


APPLY_BONUS = _envf("REWARD_APPLY_BONUS", 0.20)       # apply 成功 + 测试可收集
COLLECT_ERROR_SCORE = _envf("REWARD_COLLECT_ERROR", 0.05)  # apply 成功但代码被插坏
TEST_WEIGHT = _envf("REWARD_TEST_WEIGHT", 0.80)       # 课题口径部分的权重


class Stage(str, Enum):
    """样本走到了哪一步 —— 既用于分档，也是失败归因的一手数据。"""

    NO_PATCH = "no_patch"            # 模型没输出可解析的修改
    APPLY_FAILED = "apply_failed"    # 有修改但打不进代码库
    COLLECT_ERROR = "collect_error"  # 打进去了，但 pytest 收集失败
    TESTED = "tested"                # 正常跑完测试


@dataclass
class TestOutcome:
    """一次 pytest 执行的结构化结果。

    字段与 SWE-bench 官方判据一一对应：
      · f2p_*：FAIL_TO_PASS，修复前 fail、修复后应 pass
      · p2p_*：PASS_TO_PASS，修复前后都应 pass（守住不许回归）
    """

    f2p_total: int = 0
    f2p_passed: int = 0
    p2p_total: int = 0
    p2p_passed: int = 0
    f2p_failing: list[str] = field(default_factory=list)
    p2p_failing: list[str] = field(default_factory=list)
    collect_error: bool = False
    raw_tail: str = ""  # pytest 输出尾部，便于排查

    @property
    def f2p_rate(self) -> float:
        return self.f2p_passed / self.f2p_total if self.f2p_total else 0.0

    @property
    def has_regression(self) -> bool:
        """P2P 里出现 fail，即产生回归。"""
        return self.p2p_total > 0 and self.p2p_passed < self.p2p_total

    @property
    def strict_pass(self) -> bool:
        """严格通过：F2P 全绿、无 P2P 回归、无收集错误。等价于官方 resolved。"""
        return (
            not self.collect_error
            and self.f2p_total > 0
            and self.f2p_passed == self.f2p_total
            and not self.has_regression
        )

    def rate_str(self) -> tuple[str, str]:
        return (
            f"{self.f2p_passed}/{self.f2p_total}",
            f"{self.p2p_passed}/{self.p2p_total}",
        )


@dataclass
class RewardBreakdown:
    """reward 的完整拆解 —— 训练日志里记这个，事后才能做定量归因。

    上一轮的教训：只记 reward 标量，导致无法区分"没写出 patch"和"写了但插错位置"，
    分析时只能靠猜。本轮把 stage 与各分项一并落盘。
    """

    reward: float
    stage: Stage
    apply_component: float = 0.0
    test_component: float = 0.0
    strict_pass: bool = False
    regression_zeroed: bool = False  # 是否因 P2P 回归被判 0
    f2p_rate_str: str = "0/0"
    p2p_rate_str: str = "0/0"
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


# ------------------------------------------------------------------ 核心计算

def compute_reward(
    stage: Stage,
    outcome: TestOutcome | None = None,
    *,
    apply_bonus: float = APPLY_BONUS,
    test_weight: float = TEST_WEIGHT,
    collect_error_score: float = COLLECT_ERROR_SCORE,
    strict: bool = False,
) -> RewardBreakdown:
    """按四档计算 reward。

    `strict=True` 时退化为纯 outcome reward（只有严格通过才得 1.0，否则 0）——
    **评测时必须用这个模式**：分档是训练期的 shaping 手段，衡量真实能力
    不能把"格式分"算进去。
    """
    if stage is Stage.NO_PATCH:
        return RewardBreakdown(0.0, stage, note="模型未产出可解析的修改")

    if stage is Stage.APPLY_FAILED:
        return RewardBreakdown(0.0, stage, note="修改无法应用到代码库")

    if stage is Stage.COLLECT_ERROR:
        return RewardBreakdown(
            0.0 if strict else collect_error_score,
            stage,
            apply_component=0.0 if strict else collect_error_score,
            note="修改已应用但 pytest 无法收集用例（代码被改坏）",
        )

    assert outcome is not None, "stage=TESTED 时必须提供 TestOutcome"

    # 收集错误可能在执行阶段才暴露，这里兜一次
    if outcome.collect_error:
        return compute_reward(
            Stage.COLLECT_ERROR,
            outcome,
            apply_bonus=apply_bonus,
            test_weight=test_weight,
            collect_error_score=collect_error_score,
            strict=strict,
        )

    f2p_str, p2p_str = outcome.rate_str()

    if strict:
        return RewardBreakdown(
            1.0 if outcome.strict_pass else 0.0,
            stage,
            test_component=1.0 if outcome.strict_pass else 0.0,
            strict_pass=outcome.strict_pass,
            regression_zeroed=outcome.has_regression,
            f2p_rate_str=f2p_str,
            p2p_rate_str=p2p_str,
            note="strict 模式：仅严格通过得分",
        )

    # 防 reward hacking：P2P 回归 → 整体判 0（连 apply 分也不给）
    if outcome.has_regression:
        return RewardBreakdown(
            0.0,
            stage,
            strict_pass=False,
            regression_zeroed=True,
            f2p_rate_str=f2p_str,
            p2p_rate_str=p2p_str,
            note=f"PASS_TO_PASS 回归 {len(outcome.p2p_failing)} 例 → 判 0（防 reward hacking）",
        )

    test_part = test_weight * outcome.f2p_rate
    return RewardBreakdown(
        round(apply_bonus + test_part, 6),
        stage,
        apply_component=apply_bonus,
        test_component=round(test_part, 6),
        strict_pass=outcome.strict_pass,
        f2p_rate_str=f2p_str,
        p2p_rate_str=p2p_str,
        note="正常判分",
    )


# ------------------------------------------------------------------ pytest 输出解析

# `pytest -rA` 的汇总行形如：  PASSED tests/test_x.py::test_a
_RESULT_RE = re.compile(
    r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+)", re.MULTILINE
)
# 收集阶段失败的特征（任一命中即判 collect_error）
_COLLECT_ERROR_PATTERNS = (
    "ERRORS ",
    "ERROR collecting",
    "INTERNALERROR",
    "ImportError while loading conftest",
    "no tests ran",
)


def parse_pytest_output(
    stdout: str, f2p_ids: list[str], p2p_ids: list[str]
) -> TestOutcome:
    """从 pytest 输出解析出每个目标测试的状态。

    只统计 F2P / P2P 列表里**点名**的测试 —— 不能用"总通过数"，
    因为仓库里还有大量与本题无关的测试，混进来会让分数失去意义。

    未在输出中出现的目标测试按 **fail** 计（保守）：它可能因收集失败、
    参数化 id 不匹配或提前崩溃而没跑到，这些都不该算通过。
    """
    status: dict[str, str] = {}
    for m in _RESULT_RE.finditer(stdout):
        status[m.group(2)] = m.group(1)

    collect_error = any(p in stdout for p in _COLLECT_ERROR_PATTERNS)

    def tally(ids: list[str]) -> tuple[int, list[str]]:
        passed, failing = 0, []
        for tid in ids:
            st = status.get(tid)
            if st is None:
                # 参数化用例的 id 可能带转义差异，退一步做后缀匹配
                st = next(
                    (v for k, v in status.items() if k.endswith(tid) or tid.endswith(k)),
                    None,
                )
            if st in ("PASSED", "XPASS"):
                passed += 1
            else:
                failing.append(tid)
        return passed, failing

    f2p_passed, f2p_failing = tally(f2p_ids)
    p2p_passed, p2p_failing = tally(p2p_ids)

    return TestOutcome(
        f2p_total=len(f2p_ids),
        f2p_passed=f2p_passed,
        p2p_total=len(p2p_ids),
        p2p_passed=p2p_passed,
        f2p_failing=f2p_failing,
        p2p_failing=p2p_failing,
        collect_error=collect_error,
        raw_tail=stdout[-2000:],
    )


def outcome_from_json(payload: str | dict) -> TestOutcome:
    """从沙箱内脚本产出的 result.json 还原 TestOutcome。"""
    d = json.loads(payload) if isinstance(payload, str) else payload
    f2p = d.get("fail_to_pass", {})
    p2p = d.get("pass_to_pass", {})
    return TestOutcome(
        f2p_total=int(f2p.get("total", 0)),
        f2p_passed=int(f2p.get("passed", 0)),
        p2p_total=int(p2p.get("total", 0)),
        p2p_passed=int(p2p.get("passed", 0)),
        f2p_failing=list(f2p.get("failing", [])),
        p2p_failing=list(p2p.get("failing", [])),
        collect_error=bool(d.get("collect_error", False)),
        raw_tail=str(d.get("raw_tail", "")),
    )


# ------------------------------------------------------------------ 自测

def _selftest() -> int:
    cases: list[tuple[str, RewardBreakdown, float, Stage]] = []

    def check(name, got: RewardBreakdown, want_reward: float, want_stage: Stage):
        cases.append((name, got, want_reward, want_stage))

    check("没输出 patch", compute_reward(Stage.NO_PATCH), 0.0, Stage.NO_PATCH)
    check("apply 失败", compute_reward(Stage.APPLY_FAILED), 0.0, Stage.APPLY_FAILED)
    check("代码被插坏", compute_reward(Stage.COLLECT_ERROR), 0.05, Stage.COLLECT_ERROR)

    full = TestOutcome(f2p_total=3, f2p_passed=3, p2p_total=5, p2p_passed=5)
    check("完全修对", compute_reward(Stage.TESTED, full), 1.0, Stage.TESTED)

    half = TestOutcome(f2p_total=4, f2p_passed=2, p2p_total=5, p2p_passed=5)
    check("修对一半", compute_reward(Stage.TESTED, half), 0.2 + 0.8 * 0.5, Stage.TESTED)

    none_ = TestOutcome(f2p_total=3, f2p_passed=0, p2p_total=5, p2p_passed=5)
    check("一个没修对但没破坏", compute_reward(Stage.TESTED, none_), 0.2, Stage.TESTED)

    regress = TestOutcome(
        f2p_total=3, f2p_passed=3, p2p_total=5, p2p_passed=3, p2p_failing=["a", "b"]
    )
    r = compute_reward(Stage.TESTED, regress)
    check("F2P全绿但P2P回归→判0", r, 0.0, Stage.TESTED)

    # strict 模式（评测口径）
    check("strict/完全修对", compute_reward(Stage.TESTED, full, strict=True), 1.0, Stage.TESTED)
    check("strict/修对一半", compute_reward(Stage.TESTED, half, strict=True), 0.0, Stage.TESTED)
    check("strict/代码插坏", compute_reward(Stage.COLLECT_ERROR, strict=True), 0.0, Stage.COLLECT_ERROR)

    ok = True
    print(f"{'case':28s} {'reward':>8s} {'期望':>8s}  stage")
    for name, got, want, want_stage in cases:
        good = abs(got.reward - want) < 1e-9 and got.stage is want_stage
        ok &= good
        mark = "✓" if good else "✗"
        print(f"{mark} {name:26s} {got.reward:8.3f} {want:8.3f}  {got.stage.value}")

    # 解析器测试
    print("\n--- pytest 输出解析 ---")
    sample = """
PASSED tests/test_a.py::test_one
FAILED tests/test_a.py::test_two
PASSED tests/test_b.py::test_three
"""
    out = parse_pytest_output(
        sample, ["tests/test_a.py::test_one", "tests/test_a.py::test_two"], ["tests/test_b.py::test_three"]
    )
    exp = (1, 2, 1, 1, False)
    got = (out.f2p_passed, out.f2p_total, out.p2p_passed, out.p2p_total, out.collect_error)
    good = got == exp
    ok &= good
    print(f"{'✓' if good else '✗'} 基本解析 got={got} exp={exp}")

    missing = parse_pytest_output(
        sample, ["tests/test_a.py::test_one", "tests/test_never_ran.py::test_x"], []
    )
    good = missing.f2p_passed == 1 and "tests/test_never_ran.py::test_x" in missing.f2p_failing
    ok &= good
    print(f"{'✓' if good else '✗'} 未出现的测试按 fail 计")

    ce = parse_pytest_output("ERROR collecting tests/test_a.py\n", ["tests/test_a.py::t"], [])
    good = ce.collect_error and ce.f2p_passed == 0
    ok &= good
    print(f"{'✓' if good else '✗'} 收集错误可识别")

    print("\n" + ("全部通过 ✓" if ok else "存在失败 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
