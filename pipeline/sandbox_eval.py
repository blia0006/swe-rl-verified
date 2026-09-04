#!/usr/bin/env python3
"""沙箱判分执行器 —— 全项目唯一的判分入口
==========================================

判据验证、tracing 采集、训练 reward 三处**必须**走同一段代码，否则
"验证时能过、训练时拿 0 分"这类问题无法定位。上一轮判据散在三个脚本里
各写一遍，是排查成本最高的一处设计债。

## 职责划分（与课题架构一致）

```
本模块（本机 / TKE 训练侧）          沙箱（纯 CPU 执行环境）
─────────────────────────           ──────────────────────
生成官方 eval 脚本          ──送入──▶  写入 /task/eval.sh
                            ──送入──▶  写入 /task/model.diff
                                       git apply（打模型补丁）
                                       bash /task/eval.sh
截取 START/END 段 ◀──回传──            sed 截取测试输出段
官方 parser 解析日志
判定 F2P/P2P → TestOutcome
compute_reward
```

沙箱**不做任何判定**，只执行与回传。好处：改判据规则无需重建 20 个镜像。

## 为什么在沙箱内先 sed 截取

`install` 阶段（`pip install -e .`）的编译日志动辄几 MB，且里面可能出现
形似 `PASSED xxx` 的字样污染解析。在沙箱内用 sed 只截 START/END 之间的段，
既省带宽又避免误解析。install 失败的情况另有 marker 判定（见 EVAL_WRAPPER）。
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.official_spec import (  # noqa: E402
    END_TEST_OUTPUT,
    START_TEST_OUTPUT,
    grade,
    make_eval_script,
)
from pipeline.reward import (  # noqa: E402
    RewardBreakdown,
    Stage,
    TestOutcome,
    compute_reward,
)

# 沙箱内的固定路径
TASK_DIR = "/task"
REPO_DIR = "/testbed"
EVAL_SH = f"{TASK_DIR}/eval.sh"
MODEL_DIFF = f"{TASK_DIR}/model.diff"
FULL_LOG = f"{TASK_DIR}/eval_full.log"

# 包装脚本：跑 eval 并把完整日志留在沙箱里（便于排查），只回传截取段。
# `2>&1` 合并 stderr —— django 的 runtests / sympy 的 bin/test 都往 stderr 写结果。
EVAL_WRAPPER = f"""
cd {REPO_DIR} 2>/dev/null || exit 91
bash {EVAL_SH} > {FULL_LOG} 2>&1
rc=$?
echo "===EVAL_RC=$rc==="
if grep -qF '{START_TEST_OUTPUT}' {FULL_LOG}; then
  sed -n '/{START_TEST_OUTPUT}/,/{END_TEST_OUTPUT}/p' {FULL_LOG}
else
  # 测试段都没开始 —— 说明卡在 conda/install 阶段，回传尾部用于归因
  echo "===NO_TEST_SECTION==="
  tail -c 4000 {FULL_LOG}
fi
"""


@dataclass
class EvalResult:
    """一次完整判分的全部一手信息（落盘进 tracing，供事后归因）。"""

    stage: Stage
    reward: RewardBreakdown
    outcome: TestOutcome | None = None
    apply_method: str = ""       # 哪种策略成功打上补丁
    eval_rc: int | None = None   # eval.sh 的退出码
    grade_detail: dict[str, Any] | None = None
    error: str = ""
    log_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "reward": self.reward.to_dict(),
            "apply_method": self.apply_method,
            "eval_rc": self.eval_rc,
            "grade_detail": self.grade_detail,
            "error": self.error[:2000],
            "log_tail": self.log_tail[:4000],
        }


def sbx_run(sbx: Any, cmd: str, timeout: int = 900, *, dns_wait_s: float = 90) -> tuple[int, str]:
    """执行沙箱命令，把异常收敛成 (exit_code, output)。

    ⚠️ 两个必须在此处理的 SDK 行为：

    **① 退出码非 0 时 SDK 直接抛异常**而不返回 exit_code。若不收敛，所有测试
    失败都会被外层 except 兜成"沙箱故障" —— 上一轮实测导致 78 次 apply 失败
    被误报为基础设施问题，失败归因完全失真。

    **② e2b 的不同 API 走不同端口子域名**（`49983-<id>.…` 走文件、
    `49999-<id>.…` 走命令），每个子域名都要独立等 DNS 传播。因此
    `Sandbox.connect()` 成功**不代表**后续调用能解析域名 —— 实测 sympy 一题
    connect 通过、紧接着的第一条命令就报 `nodename nor servname`。
    所以 DNS 类错误必须在这里单独重试，而不能只在 connect 处理。
    """
    from clients.sandbox import is_dns_pending

    deadline = time.time() + dns_wait_s
    while True:
        try:
            res = sbx.commands.run(cmd, user="root", timeout=timeout)
            return (res.exit_code or 0), (res.stdout or "") + (res.stderr or "")
        except Exception as e:  # noqa: BLE001 —— 需按错误内容分流
            if is_dns_pending(e) and time.time() < deadline:
                time.sleep(4)
                continue
            code = getattr(e, "exit_code", None)
            out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
            return (code if code is not None else 1), (out or str(e))


def sbx_write(sbx: Any, path: str, content: str, *, dns_wait_s: float = 90) -> None:
    """写文件到沙箱，同样吸收 DNS 传播延迟（文件 API 走另一个端口子域名）。

    ⚠️ 必须显式 user="root"：SWE-bench 官方镜像里没有名为 `user` 的账号，
    而 e2b SDK 默认以 `user` 身份写文件，会报
    `AuthenticationException: error looking up user 'user'`。
    """
    from clients.sandbox import is_dns_pending

    deadline = time.time() + dns_wait_s
    while True:
        try:
            sbx.files.write(path, content, user="root")
            return
        except Exception as e:  # noqa: BLE001
            if is_dns_pending(e) and time.time() < deadline:
                time.sleep(4)
                continue
            raise


def reset_repo(sbx: Any, base_commit: str, timeout: int = 300) -> bool:
    """把 /testbed 还原到 base_commit —— 同一沙箱跑多次判分的前提。

    `git checkout -- .` 不够：模型可能新增文件（untracked），需 `git clean -fd`。
    保留 .gitignore 里的构建产物（不加 -x），否则每次都要重新编译 C 扩展，
    单题耗时从几十秒涨到几分钟。
    """
    code, _ = sbx_run(
        sbx,
        f"cd {REPO_DIR} && git checkout -f {base_commit} -- . "
        f"&& git clean -fd -e '*.so' -e '*.pyc' >/dev/null 2>&1; "
        f"git reset -q --hard {base_commit} 2>&1 | tail -2",
        timeout,
    )
    return code == 0


def apply_patch(sbx: Any, patch: str) -> tuple[bool, str, str]:
    """把模型补丁打入 /testbed，按容忍度递增依次尝试。

    为什么要多策略：模型产出的 diff 常有行号偏移或缺少上下文。逐级放宽能把
    "内容对但格式糙"的样本救回来 —— 这类样本若判 0，会给出错误的学习信号
    （模型以为思路错了，其实只是行号差 2 行）。

    顺序即严格度：`git apply` → 允许行号偏移 → 三方合并 → GNU patch 模糊匹配。
    """
    if not patch.strip():
        return False, "", "补丁为空"
    sbx_write(sbx, MODEL_DIFF, patch)
    strategies = [
        ("git-apply", f"git apply -v {MODEL_DIFF}"),
        ("git-apply-unsafe", f"git apply -v --unsafe-paths --directory=. {MODEL_DIFF}"),
        ("git-apply-3way", f"git apply -v --3way {MODEL_DIFF}"),
        ("patch-p1-fuzz", f"patch --batch --fuzz=5 -p1 -i {MODEL_DIFF}"),
    ]
    last = ""
    for name, cmd in strategies:
        code, out = sbx_run(sbx, f"cd {REPO_DIR} && {cmd}", 180)
        if code == 0:
            return True, name, out[-800:]
        last = out[-800:]
    return False, "", last


def run_eval(
    sbx: Any,
    task: dict[str, Any],
    patch: str | None,
    *,
    timeout: int = 1800,
    strict: bool = False,
    reset_first: bool = True,
) -> EvalResult:
    """在沙箱内对给定补丁做一次完整官方判分。

    Args:
        patch: 模型补丁。传 None 或空串表示**空解基线**（用于验证题目有效性：
               此时 F2P 应全 fail、P2P 应全 pass）。
        strict: 评测模式。True 时只有官方 resolved 才得 1.0，不给格式分。
        reset_first: 复用同一沙箱连续判分时必须为 True，否则上次的改动会残留。

    Returns:
        EvalResult —— 无论成败都返回，不抛异常（判分失败也是一条训练样本）。
    """
    base_commit = task["base_commit"]

    if reset_first and not reset_repo(sbx, base_commit):
        # 还原失败意味着后续判分结果不可信，必须显式标记而非当作 0 分样本
        rb = compute_reward(Stage.APPLY_FAILED, strict=strict)
        return EvalResult(Stage.APPLY_FAILED, rb, error="仓库还原失败，判分结果不可信")

    apply_method = "none(empty-baseline)"
    if patch and patch.strip():
        ok, apply_method, detail = apply_patch(sbx, patch)
        if not ok:
            rb = compute_reward(Stage.APPLY_FAILED, strict=strict)
            return EvalResult(
                Stage.APPLY_FAILED, rb, apply_method="", error=detail, log_tail=detail
            )

    # 官方 eval 脚本按 (repo, version) 生成，含 install / test_cmd / test_patch
    sbx_write(sbx, EVAL_SH, make_eval_script(task))
    code, out = sbx_run(sbx, EVAL_WRAPPER, timeout)

    m = re.search(r"===EVAL_RC=(-?\d+)===", out)
    eval_rc = int(m.group(1)) if m else None

    if "===NO_TEST_SECTION===" in out:
        # 测试段未出现：卡在 conda activate / pip install / tox 准备阶段。
        # 这是环境问题而非模型问题，单列出来 —— 若大量出现说明镜像有问题，
        # 不该当成模型能力差。
        rb = compute_reward(Stage.COLLECT_ERROR, strict=strict)
        return EvalResult(
            Stage.COLLECT_ERROR,
            rb,
            apply_method=apply_method,
            eval_rc=eval_rc,
            error="测试段未产生（install/环境准备阶段失败）",
            log_tail=out[-4000:],
        )

    detail = grade(task, out)
    outcome = TestOutcome(
        f2p_total=detail["f2p_total"],
        f2p_passed=detail["f2p_pass"],
        p2p_total=detail["p2p_total"],
        p2p_passed=detail["p2p_pass"],
        f2p_failing=detail["f2p_missing"],
        p2p_failing=detail["p2p_failed"],
        collect_error=detail["parsed_tests"] == 0,
        raw_tail=out[-1500:],
    )

    if outcome.collect_error:
        # 官方 parser 一个测试都没解析出来 = 代码被改坏导致收集失败
        rb = compute_reward(Stage.COLLECT_ERROR, strict=strict)
        return EvalResult(
            Stage.COLLECT_ERROR,
            rb,
            outcome=outcome,
            apply_method=apply_method,
            eval_rc=eval_rc,
            grade_detail=detail,
            log_tail=out[-4000:],
        )

    rb = compute_reward(Stage.TESTED, outcome, strict=strict)
    return EvalResult(
        Stage.TESTED,
        rb,
        outcome=outcome,
        apply_method=apply_method,
        eval_rc=eval_rc,
        grade_detail=detail,
        log_tail=out[-2000:],
    )
