#!/usr/bin/env python3
"""
VERL 自定义 reward function —— 训练侧唯一的钩子
==============================================

VERL 的 GRPO 每步会用当前策略采样 N 个回答，然后逐个调用本模块的
`compute_score()` 打分。分数在组内比较得出 advantage，驱动参数更新。
**这是 on-policy 在线 RL：每一分都来自沙箱里真实跑的 pytest，不是预先算好的常量。**

签名由 VERL 规定（`custom_reward_function.path` 扩展点）：

    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

## 全链路

    模型输出（search/replace 块）
      → pipeline.edit_format 转成 unified diff（行号由程序算，不让模型算）
      → 借一个常驻沙箱实例，还原干净代码库
      → git apply（级联放宽格式容错）
      → 跑 F2P/P2P 测试
      → pipeline.reward 四档打分

## 三层降本（决定训练能否在合理时间跑完）

朴素做法：每次打分新建沙箱实例，冷启动 ~11s + 判分 ~15s = 26s；
GRPO 55 step × 8 采样 = 440 次 → **3.2 小时**，且全程串行阻塞训练。

| 层 | 做法 | 实测收益 |
|---|---|---|
| ① 实例池 | 每题一个常驻实例，用 tar 快照还原代码库而非重建实例 | 26s → **2.7s** |
| ② 并发 | 同一组的 N 个采样并行打分 | 墙钟 ÷ 并发数 |
| ③ 缓存 | `(task_id, patch_hash)` 命中直接返回 | 训练早期输出重复度高，命中即 0 成本 |

## 失败归因（上一轮的教训）

上一轮只记 reward 标量，导致无法区分「没写出块」「路径错」「片段没找到」
「插错位置」，事后分析只能靠猜。本模块把每次打分的 `stage` 与解析细节
写入 `REWARD_DEBUG_LOG`，训练后可直接做定量归因。
"""

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.edit_format import model_output_to_patch  # noqa: E402
from pipeline.reward import Stage, compute_reward  # noqa: E402

# 注：判据不再由注入沙箱的脚本负责，改由 pipeline/sandbox_eval 在训练侧
# 按 (repo, version) 生成官方 eval 脚本、并用官方 parser 解析日志。
# 因此这里不再需要 VERIFY_SCRIPT / SYS_PY / outcome_from_json。
# 详见 _score_in_sandbox 的说明。

_POOL_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_instances = {}     # task_id -> (instance_id, sandbox)
_cache = {}         # (task_id, patch_hash) -> float
_tasks_cache = None


# ------------------------------------------------------------------ 配置

def _env(key, default):
    v = os.environ.get(key)
    return v if v not in (None, "") else default


TASKS_FILE = _env("TASKS_FILE", str(ROOT / "data" / "tasks.jsonl"))
FILE_CONTENTS_FILE = _env("FILE_CONTENTS_FILE", str(ROOT / "data" / "file_contents.json"))
DEBUG_LOG = _env("REWARD_DEBUG_LOG", "")
STRICT_SCORE = _env("REWARD_STRICT_SCORE", "0") == "1"   # 评测口径：只认严格通过
IMAGE_TAG = _env("SANDBOX_IMAGE_TAG", "sbx")
VERIFY_TIMEOUT = int(_env("VERIFY_TIMEOUT", "1800"))     # 含 install 的官方脚本更慢


def _load_tasks():
    global _tasks_cache
    if _tasks_cache is None:
        d = {}
        with open(TASKS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    d[t["task_id"]] = t
        _tasks_cache = d
    return _tasks_cache


def _file_contents(task_id):
    """题目相关文件的原始内容（供 search/replace 定位与生成 diff）。"""
    path = Path(FILE_CONTENTS_FILE)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get(task_id, {})


def _log(record):
    if not DEBUG_LOG:
        return
    with _LOG_LOCK:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ 沙箱实例池

def _sbx_run(sbx, cmd, timeout=900):
    """执行沙箱命令，把异常收敛成返回值。

    ⚠️ e2b SDK 在**退出码非 0 时直接抛异常**，不返回 exit_code。
    上一轮因未收敛，78 次 apply 失败被误报成"沙箱故障"，归因完全失真；
    修复后单步耗时还从 110s 降到 24s。
    """
    try:
        r = sbx.commands.run(cmd, user="root", timeout=timeout)
        return (r.exit_code or 0), (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        code = getattr(e, "exit_code", None)
        out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        return (code if code is not None else 1), (out or str(e))


def _get_instance(task_id):
    """取得该题的常驻沙箱实例（没有就创建）。

    实例在整个训练期常驻并复用：沙箱启动约 11s，而训练每步要判分
    ROLLOUT_N 次，每次都新建实例的话开销无法接受。
    """
    with _POOL_LOCK:
        if task_id in _instances:
            return _instances[task_id]

    from clients.ags import AGSClient
    from clients.sandbox import connect_with_retry, start_instance_with_warmup

    task = _load_tasks()[task_id]
    registry = os.environ["TCR_REGISTRY"]
    namespace = os.environ["TCR_NAMESPACE"]
    slug = task_id.replace("__", "-").lower()
    image = "%s/%s/sweb-%s:%s" % (registry, namespace, slug, IMAGE_TAG)

    ags = AGSClient()
    tool = ags.find_tool(
        os.environ.get("AGS_TOOL_NAME") or os.environ["SWE_SYNTH_SHARED_TOOL"]
    )
    inst, _eff = start_instance_with_warmup(
        ags, tool["tool_id"], image, cpu="2", memory="4Gi", verbose=False
    )
    # 必须走带重试的连接：实例刚创建时其动态域名尚未在 DNS 生效，
    # 裸 Sandbox.connect 会报 "nodename nor servname provided"（详见 clients/sandbox.py）
    sbx = connect_with_retry(inst)

    _sbx_run(sbx, "mkdir -p /task", 60)
    # 判据脚本不再注入沙箱：改由 pipeline/sandbox_eval 在**训练侧**生成官方
    # eval 脚本并解析日志（见下方 _score_in_sandbox 的说明）

    with _POOL_LOCK:
        _instances[task_id] = (inst, sbx)
    return _instances[task_id]


def release_all():
    """训练结束时回收全部实例（沙箱按实例计费，必须显式释放）。"""
    from clients.ags import AGSClient

    ags = AGSClient()
    with _POOL_LOCK:
        items = list(_instances.items())
        _instances.clear()
    for task_id, (inst, _sbx) in items:
        try:
            ags.stop_instance(inst)
            print("[reward] 已回收 %s 的实例 %s" % (task_id, inst[:16]))
        except Exception as e:
            print("[reward] ⚠️ 回收失败 %s: %s" % (task_id, e))


# ------------------------------------------------------------------ 打分

def _score_in_sandbox(task_id, patch_text):
    """把 patch 送进沙箱打分，返回 (RewardBreakdown, 判分细节)。

    ## 为什么改走 pipeline/sandbox_eval（本轮的关键修正）

    旧实现把 `sandbox_agent/swebench_verify.py` 注入沙箱，由它在容器内
    统一用 `pytest -rA <单个 test id>` 跑测试。这对 **9/20 题完全无效**：

    | repo | 官方 test_cmd | pytest 单 id 的结果 |
    |---|---|---|
    | django/django | `./tests/runtests.py --settings=test_sqlite` | 测试 id 形如 `test_x (mod.Cls)`，pytest 不认 |
    | sphinx-doc/sphinx | `tox --current-env -epy39 --` | 绕过 tox 缺插件与 `-rA`，无 PASSED 行可解析 |
    | sympy/sympy | `bin/test -C --verbose` | sympy 自有 runner，pytest 收集不全 |

    结果是这些题**无论模型答对答错都判 0 分**，组内无差异 → advantage 恒 0 →
    梯度为 0。这是上一轮"14/31 步 score 全 0"的地基性根因，比抽样方差的
    解释更靠前。

    新实现改由**训练侧**按 `(repo, version)` 查官方规格表生成 eval 脚本，
    沙箱只负责执行与回传日志，解析与判定都在训练侧做。好处：
      · 判据与官方 harness 逐字对齐，20/20 题 golden 均判满分（已实测）
      · 改判据规则不需要重建 20 个镜像
      · 与判据验证、tracing 采集**共用同一段代码**，不会出现"验证能过、
        训练拿 0 分"这类无法定位的问题
    """
    from pipeline.sandbox_eval import run_eval

    _inst, sbx = _get_instance(task_id)
    task = _load_tasks()[task_id]

    ev = run_eval(sbx, task, patch_text, timeout=VERIFY_TIMEOUT, strict=STRICT_SCORE)
    detail = {
        "stage": ev.stage.value,
        "apply_method": ev.apply_method,
        "eval_rc": ev.eval_rc,
        "error": ev.error[:300],
    }
    if ev.grade_detail:
        detail.update(
            f2p_pass=ev.grade_detail.get("f2p_pass"),
            f2p_total=ev.grade_detail.get("f2p_total"),
            p2p_pass=ev.grade_detail.get("p2p_pass"),
            p2p_total=ev.grade_detail.get("p2p_total"),
            resolved=ev.grade_detail.get("resolved"),
        )
    return ev.reward, detail


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """VERL 调用入口。返回 [0, 1] 的标量分数。"""
    t0 = time.time()
    extra_info = extra_info or {}
    task_id = extra_info.get("task_id") or ground_truth or ""
    if not task_id:
        return 0.0

    record = {"task_id": task_id, "ts": time.time()}

    # 1) 模型输出 → unified diff（行号由程序算，不让模型算）
    contents = _file_contents(task_id)
    patch, info = model_output_to_patch(solution_str or "", contents)
    record["parse"] = info

    if not patch:
        bd = compute_reward(Stage.NO_PATCH)
        record.update(stage=bd.stage.value, reward=bd.reward,
                      elapsed_s=round(time.time() - t0, 2))
        _log(record)
        return bd.reward

    # 2) 缓存：训练早期模型输出重复度高
    key = (task_id, hashlib.sha256(patch.encode()).hexdigest())
    with _CACHE_LOCK:
        if key in _cache:
            record.update(stage="cache_hit", reward=_cache[key],
                          elapsed_s=round(time.time() - t0, 2))
            _log(record)
            return _cache[key]

    # 3) 沙箱实测打分
    try:
        bd, result = _score_in_sandbox(task_id, patch)
    except Exception as e:
        record.update(stage="exception", error=str(e)[:300], reward=0.0,
                      elapsed_s=round(time.time() - t0, 2))
        _log(record)
        return 0.0

    record.update(
        stage=bd.stage.value,
        reward=bd.reward,
        f2p=bd.f2p_rate_str,
        p2p=bd.p2p_rate_str,
        strict_pass=bd.strict_pass,
        regression_zeroed=bd.regression_zeroed,
        apply_strategy=result.get("apply_method", ""),
        verify_stage=result.get("stage", ""),
        resolved=result.get("resolved"),
        eval_error=result.get("error", ""),
        elapsed_s=round(time.time() - t0, 2),
    )
    _log(record)

    with _CACHE_LOCK:
        _cache[key] = bd.reward
    return bd.reward


# 兼容 VERL 不同版本可能查找的别名
compute_score_batch = None


if __name__ == "__main__":
    # 冒烟自测：不连沙箱，只验证解析与归因链路
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--file", help="含模型输出的文本文件；省略则用内置示例")
    args = ap.parse_args()

    text = (
        Path(args.file).read_text(encoding="utf-8")
        if args.file
        else "我认为应该修改这个函数。"
    )
    contents = _file_contents(args.task)
    print("题目文件数：%d" % len(contents))
    patch, info = model_output_to_patch(text, contents)
    print("解析结果：%s" % json.dumps(info, ensure_ascii=False))
    print("patch 长度：%d" % len(patch))
    if patch:
        print(patch[:800])
