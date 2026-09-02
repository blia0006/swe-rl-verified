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
from pipeline.reward import (  # noqa: E402
    Stage,
    compute_reward,
    outcome_from_json,
)

VERIFY_SCRIPT = ROOT / "sandbox_agent" / "swebench_verify.py"

# 判据脚本必须用**系统** python3.10 跑：题目的 conda 环境可能老至 py3.6。
# 必须写绝对路径 —— 镜像 PATH 里 conda 在前，裸写 python3 会解析错解释器。
SYS_PY = "/usr/bin/python3"

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
STRICT_APPLY = _env("REWARD_STRICT_APPLY", "0") == "1"
STRICT_SCORE = _env("REWARD_STRICT_SCORE", "0") == "1"   # 评测口径：只认严格通过
IMAGE_TAG = _env("SANDBOX_IMAGE_TAG", "sbx")
VERIFY_TIMEOUT = int(_env("VERIFY_TIMEOUT", "900"))


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
    """取得该题的常驻沙箱实例（没有就创建并完成一次性初始化）。"""
    with _POOL_LOCK:
        if task_id in _instances:
            return _instances[task_id]

    from clients.ags import AGSClient
    from clients.sandbox import start_instance_with_warmup
    from e2b_code_interpreter import Sandbox

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
        ags, tool["tool_id"], image, cpu="2", memory="4Gi"
    )
    sbx = Sandbox.connect(inst)

    # 一次性注入：判据脚本 + 题目规格
    _sbx_run(sbx, "mkdir -p /task", 60)
    spec = {
        "task_id": task_id,
        "fail_to_pass": task["fail_to_pass"],
        "pass_to_pass": task["pass_to_pass"],
        "test_patch": task["test_patch"],
    }
    # 必须 user="root"：官方镜像没有名为 `user` 的账号
    sbx.files.write("/task/swebench_verify.py", VERIFY_SCRIPT.read_text(encoding="utf-8"), user="root")
    sbx.files.write("/task/spec.json", json.dumps(spec, ensure_ascii=False), user="root")

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
    """把 patch 送进沙箱打分，返回 (RewardBreakdown, 原始 result)。"""
    _inst, sbx = _get_instance(task_id)

    sbx.files.write("/task/model.diff", patch_text, user="root")
    cmd = (
        "%s /task/swebench_verify.py --spec /task/spec.json "
        "--patch /task/model.diff --out /task/result.json --restore%s"
        % (SYS_PY, " --strict-apply" if STRICT_APPLY else "")
    )
    code, out = _sbx_run(sbx, cmd, VERIFY_TIMEOUT)

    try:
        payload = sbx.files.read("/task/result.json", user="root")
        result = json.loads(payload)
    except Exception as e:
        # 读不到结果 = 沙箱侧真异常（与"答错"区分开，避免像上一轮那样归因失真）
        return (
            compute_reward(Stage.APPLY_FAILED),
            {"stage": "sandbox_error", "error": str(e)[:300], "stdout_tail": out[-500:]},
        )

    stage_name = result.get("stage", "")
    if stage_name == "tested":
        bd = compute_reward(Stage.TESTED, outcome_from_json(result), strict=STRICT_SCORE)
    elif stage_name == "collect_error":
        bd = compute_reward(Stage.COLLECT_ERROR, strict=STRICT_SCORE)
    elif stage_name == "apply_failed":
        bd = compute_reward(Stage.APPLY_FAILED)
    else:
        # test_patch_failed / restore_failed / no_tests：属基础设施问题，
        # 不该记在模型头上，但也无法给分，记 0 并在日志里标明
        bd = compute_reward(Stage.APPLY_FAILED)
    return bd, result


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
        apply_strategy=result.get("apply_strategy", ""),
        verify_stage=result.get("stage", ""),
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
