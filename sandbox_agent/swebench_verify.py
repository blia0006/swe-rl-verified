#!/usr/bin/env python3
"""
沙箱内判据脚本：跑 SWE-bench 官方测试并输出结构化判分结果
=========================================================

这是**注入沙箱内执行**的脚本（不在本机、不在 GPU 节点跑），职责单一：
    应用测试补丁 → 跑指定的 F2P/P2P 测试 → 产出 result.json

为什么需要这一层：上一轮用的自建题目自带 `/task/verify.sh`；
SWE-bench 官方镜像**没有** verify.sh，判据是数据集里的两个测试 id 列表，
必须自己实现一个等价物，且输出格式要与 `pipeline/reward.py` 对齐
（这样判分逻辑在线 A / 线 B / 评测三处完全一致）。

设计约束（都是硬约束，改动前先想清楚）：
  1. **只依赖标准库**。沙箱内不能联网 pip install。
  2. **必须先应用 test_patch，再跑测试**。SWE-bench 的官方测试文件在
     `test_patch` 里，题目态镜像的代码库里**没有**这些测试 —— 不打就全是
     "测试不存在"，F2P 恒为 0，reward 恒 0，训练直接空转。
  3. **模型 patch 与 test_patch 必须分开应用**，且 test_patch 后打。
     否则模型可以改测试文件作弊（改完再被 test_patch 覆盖才是安全的）。
  4. 退出码恒为 0。判分结果只看 result.json —— 因为 pytest 判不通过时退出码非 0，
     而 e2b SDK 对非 0 退出码会抛异常（上一轮踩过：78 次 apply 失败被误报成
     "基础设施故障"，失败归因完全失真）。

用法（沙箱内）：
    python3 swebench_verify.py --spec /task/spec.json --out /task/result.json
    python3 swebench_verify.py --spec ... --patch /task/model_patch.diff
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# SWE-bench 官方镜像的固定布局
REPO_DIR = "/testbed"
CONDA_ENV_BIN = "/opt/miniconda3/envs/testbed/bin"
# 题目依赖装在 testbed 环境里（可能老至 py3.6），必须用它跑 pytest；
# 而本脚本自身由系统 python(3.10) 执行 —— 两者职责分离，不可混用
TESTBED_PY = CONDA_ENV_BIN + "/python"

MAX_OUTPUT = 20000  # result.json 里保留的输出上限，防止 tracing 膨胀


def run(cmd, cwd=REPO_DIR, timeout=600, env=None):
    """执行 shell 命令，永不抛异常，返回 (exit_code, 合并输出)。"""
    full_env = os.environ.copy()
    # 让 conda 环境的 python/pytest 优先；PYTEST_ADDOPTS 关掉颜色，
    # 否则 ANSI 转义码会让结果正则失配（上一轮踩过，造成假阴性）
    full_env["PATH"] = f"{CONDA_ENV_BIN}:{full_env.get('PATH', '')}"
    full_env["PYTEST_ADDOPTS"] = "--color=no -p no:cacheprovider"
    if env:
        full_env.update(env)
    try:
        p = subprocess.run(
            cmd,
            shell=True,  # 判据脚本需要 shell 语法（重定向/管道），命令均由本脚本构造，不含外部输入
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
        return 124, f"{out}\n[TIMEOUT] 超过 {timeout}s"
    except OSError as e:
        return 125, f"[OSError] {e}"


# ------------------------------------------------------------------ patch 应用

# 宽松度递增的 apply 级联。上一轮实测：strict 模式下小模型成功率仅 1.8%，
# 而放宽格式容错后可达 34.5% —— 放宽的只是 diff 的**格式容错**，
# 代码改动内容分毫未变，最终仍由真实 pytest 判定对错。
APPLY_STRATEGIES = [
    ("strict", "git apply --whitespace=nowarn -v"),
    ("recount", "git apply --recount --whitespace=nowarn -v"),
    ("recount+C1", "git apply --recount -C1 --unidiff-zero --whitespace=nowarn -v"),
    ("recount+C0", "git apply --recount -C0 --unidiff-zero --whitespace=nowarn -v"),
]


def apply_patch(patch_text, path, strict_only=False):
    """按级联策略应用 patch。返回 (是否成功, 生效策略, 失败时的 stderr)。

    `strict_only=True` 用于**评测**：衡量真实能力时不能放宽标准。
    """
    if not patch_text.strip():
        return False, "", "空 patch"
    Path(path).write_text(patch_text, encoding="utf-8")

    strategies = APPLY_STRATEGIES[:1] if strict_only else APPLY_STRATEGIES
    first_err = ""
    for name, base in strategies:
        code, out = run(f"{base} {path}", timeout=120)
        if code == 0:
            return True, name, ""
        if not first_err:
            # 取第一级（strict）的报错作为失败原因：它最能反映 patch 本身的问题，
            # 最后一级的报错往往是"上下文放宽后仍定位不到"，信息量更低
            first_err = out[-1500:]
    return False, "", first_err


# ------------------------------------------------------------------ 测试执行

_STATUS_PREFIXES = ("PASSED", "FAILED", "ERROR", "XFAIL", "XPASS", "SKIPPED")
_COLLECT_ERROR_MARKERS = (
    "ERROR collecting",
    "INTERNALERROR",
    "ImportError while loading conftest",
    "!!!! Interrupted",
)


def parse_results(stdout):
    """解析 `pytest -rA` 的汇总段，返回 {test_id: status} 与是否收集失败。"""
    status = {}
    for line in stdout.splitlines():
        line = line.strip()
        for prefix in _STATUS_PREFIXES:
            if line.startswith(prefix + " "):
                rest = line[len(prefix) + 1 :].strip()
                test_id = rest.split(" ", 1)[0]
                if test_id:
                    status[test_id] = prefix
                break
    collect_error = any(m in stdout for m in _COLLECT_ERROR_MARKERS)
    # "no tests ran" 且确实点名了测试 → 也是收集异常
    if "no tests ran" in stdout and not status:
        collect_error = True
    return status, collect_error


def tally(status, ids):
    """统计点名测试的通过数。未出现在输出里的按 fail 计（保守）。

    只统计**点名**的测试：仓库里还有大量与本题无关的用例，
    用"总通过数"会让分数完全失去意义。
    """
    passed, failing = 0, []
    for tid in ids:
        st = status.get(tid)
        if st is None:
            # 参数化用例的 id 可能有转义/空格差异，退一步做后缀匹配
            st = next(
                (v for k, v in status.items() if k.endswith(tid) or tid.endswith(k)),
                None,
            )
        if st in ("PASSED", "XPASS"):
            passed += 1
        else:
            failing.append(tid)
    return passed, failing


def build_pytest_cmd(test_ids, extra=""):
    """构造 pytest 命令。测试 id 逐个用单引号包裹，避免参数化 id 里的
    括号/空格/方括号被 shell 解释。"""
    quoted = " ".join("'" + t.replace("'", "'\\''") + "'" for t in test_ids)
    return "{} -m pytest -rA --tb=short --no-header {} {}".format(TESTBED_PY, extra, quoted).strip()


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="题目规格 JSON（含 F2P/P2P/test_patch）")
    ap.add_argument("--patch", default="", help="模型产出的 patch 文件；省略则测空解基线")
    ap.add_argument("--out", default="/task/result.json")
    ap.add_argument("--timeout", type=int, default=900, help="pytest 超时(秒)")
    ap.add_argument("--strict-apply", action="store_true", help="只用 strict 策略（评测用）")
    ap.add_argument("--snapshot", default="/tmp/testbed_pristine.tar.gz")
    ap.add_argument(
        "--restore",
        action="store_true",
        help="执行前先从快照还原代码库（沙箱实例复用时必须）",
    )
    args = ap.parse_args()

    t_start = time.time()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    f2p = spec.get("fail_to_pass", [])
    p2p = spec.get("pass_to_pass", [])

    result = {
        "task_id": spec.get("task_id", ""),
        "stage": "no_patch",
        "apply_ok": False,
        "apply_strategy": "",
        "apply_error": "",
        "fail_to_pass": {"total": len(f2p), "passed": 0, "failing": list(f2p)},
        "pass_to_pass": {"total": len(p2p), "passed": 0, "failing": list(p2p)},
        "collect_error": False,
        "raw_tail": "",
        "timings": {},
    }

    def finish(code=0):
        result["timings"]["total_s"] = round(time.time() - t_start, 2)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[verify] stage={result['stage']} → {args.out}")
        # 恒返回 0：判分结论只看 result.json。pytest 失败时退出码非 0，
        # 而 e2b SDK 遇非 0 退出码会抛异常，导致调用侧把"答错"误判成"沙箱故障"
        return 0

    # ---------- 0) 实例复用：从快照还原干净代码库 ----------
    snap = Path(args.snapshot)
    if args.restore and snap.exists():
        t0 = time.time()
        # 官方镜像的 /testbed 是 git 仓库，但仍用 tar 快照还原：
        # git checkout 无法清除未跟踪文件产生的编译产物/缓存，tar 是确定性的
        run(f"rm -rf {REPO_DIR}", cwd="/", timeout=120)
        code, out = run(f"tar xzf {snap} -C /", cwd="/", timeout=300)
        if code != 0:
            result["stage"] = "restore_failed"
            result["apply_error"] = out[-1000:]
            return finish()
        result["timings"]["restore_s"] = round(time.time() - t0, 2)
    elif not snap.exists():
        t0 = time.time()
        run(f"tar czf {snap} -C / {REPO_DIR.lstrip('/')}", cwd="/", timeout=600)
        result["timings"]["snapshot_s"] = round(time.time() - t0, 2)

    # ---------- 1) 先应用模型 patch（必须早于 test_patch） ----------
    if args.patch and Path(args.patch).exists():
        patch_text = Path(args.patch).read_text(encoding="utf-8", errors="replace")
        t0 = time.time()
        ok, strategy, err = apply_patch(
            patch_text, "/tmp/model.diff", strict_only=args.strict_apply
        )
        result["timings"]["apply_s"] = round(time.time() - t0, 2)
        result["apply_ok"] = ok
        result["apply_strategy"] = strategy
        if not ok:
            result["stage"] = "apply_failed"
            result["apply_error"] = err[:MAX_OUTPUT]
            return finish()
    else:
        # 无 patch = 空解基线。仍继续跑测试，用于确认题目本身"修复前确实 fail"
        result["apply_ok"] = True
        result["apply_strategy"] = "none(baseline)"

    # ---------- 2) 再应用官方 test_patch（模型改不到它，防作弊） ----------
    test_patch = spec.get("test_patch", "")
    if test_patch.strip():
        # 先把测试文件恢复到原始状态，确保 test_patch 能干净应用
        # （模型 patch 可能碰过测试目录）
        files = [
            ln[6:].strip()
            for ln in test_patch.splitlines()
            if ln.startswith("+++ b/")
        ]
        if files:
            run("git checkout -- " + " ".join(f"'{f}'" for f in files), timeout=120)
        ok, strategy, err = apply_patch(test_patch, "/tmp/test.diff")
        if not ok:
            # 官方 test_patch 打不上说明环境与数据集版本不匹配 —— 属基础设施错误，
            # 必须与"模型答错"区分开，否则会把环境问题算成模型的锅
            result["stage"] = "test_patch_failed"
            result["apply_error"] = err[:MAX_OUTPUT]
            return finish()

    # ---------- 3) 跑测试 ----------
    all_ids = list(dict.fromkeys(f2p + p2p))  # 去重且保持顺序
    if not all_ids:
        result["stage"] = "no_tests"
        return finish()

    t0 = time.time()
    code, out = run(build_pytest_cmd(all_ids), timeout=args.timeout)
    result["timings"]["pytest_s"] = round(time.time() - t0, 2)
    result["pytest_exit_code"] = code

    status, collect_error = parse_results(out)
    f2p_passed, f2p_failing = tally(status, f2p)
    p2p_passed, p2p_failing = tally(status, p2p)

    result["fail_to_pass"] = {
        "total": len(f2p),
        "passed": f2p_passed,
        "failing": f2p_failing[:50],
    }
    result["pass_to_pass"] = {
        "total": len(p2p),
        "passed": p2p_passed,
        "failing": p2p_failing[:50],
    }
    result["collect_error"] = collect_error
    result["stage"] = "collect_error" if collect_error else "tested"
    result["raw_tail"] = out[-MAX_OUTPUT:]
    return finish()


if __name__ == "__main__":
    sys.exit(main())
