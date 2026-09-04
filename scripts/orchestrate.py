#!/usr/bin/env python3
"""全流程自动编排：采集 → 训练 → 闭环评测
==========================================

在 GPU 节点上无人值守跑完整条 RL 流水线。每个阶段的结果与失败原因都落盘，
中途任何一步失败都会**记录并继续/降级**，而不是静默卡死。

## 为什么必须串行（GPU 独占约束）

单卡 5090 只有 24GB，而：
  · 采集阶段：vLLM 推理服务常驻，占约 21.5GB（给沙箱里的 ReAct Agent 用）
  · 训练阶段：VERL hybrid engine（FSDP actor + vLLM rollout）要独占显存

两者**不可能同时存在**。因此编排必须：
    采集(vLLM 在) → 停 vLLM → 训练(独占) → 起 vLLM → 评测(vLLM 在)

## 模型选择的实测约束

上一轮已实测：**7B 在 24GB 单卡训不动** ——
FSDP actor 即便开 param_offload 残留仍约 15.8GB，vLLM 再要 15.2GB 权重，
合计 31GB > 23.4GB 可用；三档 gpu_memory_utilization 全部失败。

但本轮采集用的是 7B（它能力足够，实测能解对题）。这产生一个矛盾：
**训练模型必须与采集/评测模型一致，否则前后对比无意义。**

本脚本的处理：先给 7B 一次机会（用更激进的省显存配置做冒烟测试），
装不下就降级到 3B，并**自动用 3B 重采基线**以保证闭环一致。
无论走哪条路，都在 stage 记录里写明实际用的模型与原因。

用法（在节点上后台跑）：
    python3 scripts/orchestrate.py --stages collect,train,eval
    python3 scripts/orchestrate.py --stages train,eval --skip-collect-wait
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "logs" / "orchestrate_state.json"
PY_ORCH = "/data/swe-rl/venv-orch/bin/python"   # 编排/采集用（含 e2b、swebench）
CONTAINER_SH = ROOT / "scripts" / "start_train_container.sh"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_state(stage: str, **kw) -> None:
    """把每阶段结果落盘。断点续跑与事后归因都依赖它。"""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if STATE.is_file():
        try:
            data = json.loads(STATE.read_text())
        except Exception:  # noqa: BLE001
            data = {}
    data.setdefault("stages", {})[stage] = {"ts": time.time(), **kw}
    STATE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sh(cmd: str, timeout: int = 3600, cwd: str | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, shell=True, timeout=timeout, cwd=cwd or str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, f"[超时] 超过 {timeout}s"


# ------------------------------------------------------------------ GPU 服务

def vllm_up() -> bool:
    code, _ = sh("curl -s -m 5 http://127.0.0.1:8000/v1/models > /dev/null", 30)
    return code == 0


def gpu_used_mib() -> int:
    code, out = sh(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", 30
    )
    first = out.strip().split("\n")[0].strip() if out.strip() else ""
    return int(first) if first.isdigit() else 99999


def kill_gpu_hogs() -> None:
    """强制清掉所有占用 GPU 的自建进程/容器。

    ## 为什么必须有这一步（实测事故）

    编排被中途 kill 时，它启动的 `swe-rl-serve`（vLLM）容器**不会**跟着退出 ——
    `ctr run -d` 起的容器由 containerd 托管，与编排进程无父子关系。
    实测留下一个 `VLLM::EngineCore` 孤儿进程独占 21.5GB，导致后续训练报

        Free memory on device (7.76/23.41 GiB) < desired (0.72, 16.86 GiB)
        torch.OutOfMemoryError: ... 23.41 GiB of which 264.12 MiB is free

    这个报错**看起来像"7B 装不下 24G 卡"**，与上一轮的历史结论高度吻合，
    极易据此错误地放弃 7B。真因只是没清干净。

    因此每次进入训练前都要显式清场，并**验证显存真的降下来**。
    """
    log("清场：停止所有占用 GPU 的自建容器/进程…")
    sh(f"bash {CONTAINER_SH} serve-stop", 180)
    sh(f"bash {CONTAINER_SH} stop", 180)
    # ray/vllm 可能留下游离进程（不在容器内），按进程名兜底清理
    sh("pkill -f 'VLLM::' 2>/dev/null; pkill -f 'ray::' 2>/dev/null; ray stop 2>/dev/null", 120)
    for i in range(36):
        used = gpu_used_mib()
        if used < 1500:
            log(f"  ✓ 显存已释放（{used} MiB / 24455 MiB）")
            return
        if i == 10:
            # 等了 50s 还没释放，列出占用者并强杀
            _, apps = sh(
                "nvidia-smi --query-compute-apps=pid,used_memory,process_name "
                "--format=csv,noheader", 30
            )
            log(f"  ⚠️ 仍被占用：{apps.strip()[:300]}")
            for ln in apps.strip().splitlines():
                pid = ln.split(",")[0].strip()
                if pid.isdigit():
                    sh(f"kill -9 {pid} 2>/dev/null", 30)
        time.sleep(5)
    log(f"  ✗ 显存仍未释放（{gpu_used_mib()} MiB），训练很可能 OOM")


def stop_vllm() -> None:
    """停 vLLM 并确认显存真的释放（容器退出与显存回收之间有延迟）。"""
    kill_gpu_hogs()


def start_vllm(model_path: str) -> bool:
    log(f"启动 vLLM 服务：{model_path}")
    sh(f"MODEL_PATH={model_path} bash {CONTAINER_SH} serve", 300)
    for i in range(60):
        if vllm_up():
            log(f"  推理服务就绪（等待 {i * 5}s）")
            return True
        time.sleep(5)
    log("  ✗ 推理服务 300s 内未就绪")
    return False


# ------------------------------------------------------------------ 各阶段

def wait_collect(timeout_s: int = 10800) -> dict:
    """等已在跑的采集任务结束，并解析其汇总结果。"""
    pidf = ROOT / "logs" / "collect.pid"
    if not pidf.is_file():
        log("未发现采集任务，跳过等待")
        return {"skipped": True}
    pid = pidf.read_text().strip()
    log(f"等待采集任务 pid={pid} 结束（上限 {timeout_s // 60} 分钟）…")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        code, _ = sh(f"ps -p {pid} > /dev/null 2>&1", 30)
        if code != 0:
            log(f"  采集已结束（耗时 {(time.time() - t0) / 60:.0f} 分钟）")
            break
        time.sleep(30)
    else:
        log("  ⚠️ 采集超时，强制进入下一阶段")

    # 解析采集结果：有梯度的题数决定训练是否值得跑
    summ = sorted((ROOT / "data" / "tracing").glob("summary_*.json"))
    if not summ:
        return {"ok": False, "reason": "未找到采集汇总"}
    data = json.loads(summ[-1].read_text())
    res = [r for r in data.get("results", []) if r.get("rollouts")]
    with_var = [r for r in res if r.get("has_variance")]
    n_resolved = sum(r.get("pass_count", 0) for r in res)
    n_roll = sum(len(r["rollouts"]) for r in res)
    out = {
        "ok": True,
        "tasks": len(res),
        "rollouts": n_roll,
        "with_variance": len(with_var),
        "resolved": n_resolved,
        "pass_at_1": round(n_resolved / n_roll, 4) if n_roll else 0.0,
        "summary_file": summ[-1].name,
    }
    log(
        f"  采集结果：{len(res)} 题 / {n_roll} rollout，"
        f"有梯度 {len(with_var)} 题，resolved {n_resolved}（pass@1={out['pass_at_1']:.1%}）"
    )
    return out


def build_dataset(model_path: str) -> dict:
    """生成 GRPO 训练数据（parquet）。"""
    log("生成训练数据…")
    code, out = sh(
        f"{PY_ORCH} pipeline/extract_file_contents.py 2>&1 | tail -5", 1800
    )
    log(f"  文件内容抽取 rc={code}: {out.strip()[-200:]}")
    code, out = sh(
        f"{PY_ORCH} pipeline/build_grpo_dataset.py 2>&1 | tail -8", 900
    )
    log(f"  数据集构建 rc={code}: {out.strip()[-300:]}")
    pq = ROOT / "data" / "grpo_train.parquet"
    ok = pq.is_file() and pq.stat().st_size > 1000
    return {"ok": ok, "size": pq.stat().st_size if pq.is_file() else 0, "log": out[-500:]}


def smoke_train(model_path: str, tag: str) -> dict:
    """显存冒烟测试：只跑极少步，看能否装下。

    为什么必须单独做：正式训练要几小时，若因显存不足在第 0 步崩，
    等于白等。冒烟测试几分钟就能给出确定答案。

    ## 判据设计（两次实测教训）

    **第一版**只看"日志里有 step/global_step 且无 OOM 关键字"，结果两次冒烟都在
    2 秒内返回失败 —— 真因是 `ctr` 清理残留容器时报
    `cannot delete a deleted process`，训练进程压根没启动。这类"环境没起来"
    被误判成"显存不够"，会导向错误结论（错误地放弃 7B）。

    **第二版**（修 ctr 后）把 `ray`/`actor_rollout`/`WorkerDict` 等关键字也算作
    "启动成功"，于是 7B 冒烟"通过"了 —— 但正式训练仍在 56 秒后死于
    `Free memory on device (7.76/23.41 GiB) < desired (0.72, 16.86 GiB)`。
    原因是这些关键字只证明 **verl 进程起来了**，而显存瓶颈出现在更晚的
    vLLM engine 初始化阶段。冒烟测试若不跑到那一步，就毫无筛查价值。

    **现在**要求看到**至少一步真正完成**的证据（step 指标或 checkpoint 写出），
    并且把 vLLM 显存报错单独识别出来。宁可冒烟慢几分钟，也不能放过假阳性。
    """
    log(f"显存冒烟测试：{tag}")
    # ⚠️ SMOKE_MAX_STEPS 必须 ≥12，不能图快设 1~3。
    # 实测 `fsdp2 + offload_policy=True` 能正常跑完前 10 步，到 **step 10 之后**
    # 才因 LoRA 梯度写回路径触发
    #   RuntimeError: assign a gradient with device type 'cpu' to ... 'cuda'
    # 跑 1~3 步的冒烟测试会给出"通过"的假阳性，然后正式训练在 20 分钟后崩掉。
    steps = os.environ.get("SMOKE_MAX_STEPS", "12")
    env = (
        f"MODEL_PATH={model_path} TOTAL_EPOCHS=1 ROLLOUT_N=4 "
        f"TRAIN_BATCH=1 MINI_BATCH=1 MAX_RESP_LEN=512 SAVE_FREQ=1000 "
        f"SMOKE_MAX_STEPS={steps} "
    )
    code, out = sh(f"{env} bash {CONTAINER_SH} train 2>&1 | tail -100", 5400)

    oom = any(
        k in out
        for k in ("out of memory", "OutOfMemoryError", "KV cache",
                  "Failed to create unquantized linear weights",
                  "No available memory", "CUDA error: out of memory",
                  "less than desired GPU memory", "Free memory on device")
    )
    # 设备不匹配：fsdp2 + CPU offload + LoRA 的组合会在 step 10 后触发
    device_mismatch = "device type 'cpu' to a tensor with device type 'cuda'" in out
    crashed = "Traceback (most recent call last)" in out
    # 实际完成步数 —— 必须真的跑够步数才算通过（见上方 SMOKE_MAX_STEPS 说明）
    steps_done = 0
    for m in re.finditer(r"training/global_step:(\d+)", out):
        steps_done = max(steps_done, int(m.group(1)))
    want = int(steps)
    infra = any(
        k in out
        for k in ("failed to delete task", "cannot delete a deleted process",
                  "already exists", "模型目录不存在", "训练数据不存在",
                  "executable file not found", "拒绝启动")
    )
    verdict = (
        "oom" if oom
        else "device_mismatch" if device_mismatch
        else "infra" if infra
        else "crashed" if crashed
        else "ok" if steps_done >= want
        else f"only_{steps_done}_steps"
    )
    if verdict != "ok":
        log(f"  ✗ {tag} 冒烟失败：{verdict}（完成 {steps_done}/{want} 步）")
        errs = [
            ln for ln in out.splitlines()
            if re.match(r"^\s*(RuntimeError|ValueError|torch\.\w*Error|AssertionError)", ln.strip())
        ]
        for ln in (errs[-2:] or out.strip().splitlines()[-5:]):
            log(f"     {ln.strip()[:190]}")
    else:
        log(f"  ✓ {tag} 冒烟通过（完成 {steps_done} 步，含 LoRA 梯度写回路径）")
    return {
        "ok": verdict == "ok", "verdict": verdict, "oom": oom,
        "device_mismatch": device_mismatch, "steps_done": steps_done,
        "steps_required": want, "infra": infra, "rc": code, "tail": out[-3000:],
    }


def full_train(model_path: str) -> dict:
    """正式训练。

    ## 成功判据（一次实测教训）

    最初写成"checkpoints 目录非空即成功"，结果 **误报了一次成功**：
    目录里是上一轮遗留的 `global_step_10/20/30/31`（前一天的），而本次训练
    其实在 56 秒内因显存不足崩掉了。编排据此进入"训练后评测"，
    拿没训练过的模型跑出一个"after"数字 —— 若不核对时间戳，这个假结果
    会直接写进结论。

    因此判据改为三条**同时**满足：
      · 退出码为 0
      · 出现**训练开始之后**新建的 checkpoint（比时间戳，不看目录是否非空）
      · 日志里能看到 step 推进
    """
    log(f"正式训练：{model_path}")
    t_start = time.time()
    ck_dir = ROOT / "checkpoints"
    before = {p.name for p in ck_dir.glob("global_step_*")} if ck_dir.is_dir() else set()

    # ⚠️ 不能用 `... | tail -80`：管道的退出码是 tail 的（恒 0），
    # 训练真实失败会被吞掉 —— 实测因此把一次 step 10 崩溃误报成 ok=True。
    # 改为全量写文件、再单独取尾部，退出码由训练脚本直接给出。
    tmp_log = ROOT / "logs" / "_full_train_stdout.log"
    code, _ = sh(
        f"MODEL_PATH={model_path} bash {CONTAINER_SH} train > {tmp_log} 2>&1", 36000
    )
    out = ""
    if tmp_log.is_file():
        out = tmp_log.read_text(encoding="utf-8", errors="replace")

    new_ck = sorted(
        (p for p in ck_dir.glob("global_step_*") if p.stat().st_mtime > t_start),
        key=lambda p: p.stat().st_mtime,
    ) if ck_dir.is_dir() else []

    oom = any(
        k in out
        for k in ("out of memory", "OutOfMemoryError", "less than desired GPU memory",
                  "Free memory on device", "Failed to create unquantized linear weights",
                  "No available memory for the cache blocks")
    )
    # 训练是否真的跑到了尾声：verl 正常结束会输出最后一步的指标且无 Traceback
    crashed = "Traceback (most recent call last)" in out or "训练退出码：1" in out
    # 实际完成的步数 —— 判断"跑完"还是"中途崩"的硬指标
    steps_done = 0
    for m in re.finditer(r"training/global_step:(\d+)", out):
        steps_done = max(steps_done, int(m.group(1)))

    ok = code == 0 and not crashed and bool(new_ck)
    wall = round(time.time() - t_start, 1)
    if not ok:
        log(
            f"  ✗ 训练失败（rc={code} crashed={crashed} oom={oom} "
            f"完成 {steps_done} 步，耗时 {wall / 60:.0f} 分钟）"
        )
        # 打印真正的错误行（Traceback 末尾），而不是无信息的栈帧
        errs = [
            ln for ln in out.splitlines()
            if re.match(r"^\s*(RuntimeError|ValueError|TypeError|AssertionError|"
                        r"torch\.\w*Error|OSError|KeyError)", ln.strip())
        ]
        for ln in (errs[-3:] or out.strip().splitlines()[-5:]):
            log(f"     {ln.strip()[:200]}")
    else:
        log(
            f"  ✓ 训练完成 {steps_done} 步，耗时 {wall / 60:.0f} 分钟，"
            f"新增 checkpoint：{[p.name for p in new_ck][-3:]}"
        )
    return {
        "ok": ok,
        "rc": code,
        "crashed": crashed,
        "oom": oom,
        "steps_done": steps_done,
        "wall_s": wall,
        "new_checkpoints": [p.name for p in new_ck],
        "preexisting_checkpoints": sorted(before),
        "error_lines": [ln.strip()[:300] for ln in (errs[-3:] if not ok else [])],
        "tail": out[-3000:],
    }


def closed_loop_eval(model_path: str, label: str, split: str = "eval") -> dict:
    """闭环评测：用 strict 口径跑评测集，得 pass@1。"""
    log(f"闭环评测（{label}）：{model_path}")
    if not start_vllm(model_path):
        return {"ok": False, "reason": "推理服务未就绪"}
    run_id = f"eval-{label}"
    code, out = sh(
        f"{PY_ORCH} pipeline/collect_tracing.py --split {split} -n 1 "
        f"--max-steps 24 --temperature 0.2 --jobs 5 --strict-eval "
        f"--run-id {run_id} 2>&1 | tail -30",
        10800,
    )
    summ = ROOT / "data" / "tracing" / f"summary_{run_id}.json"
    if not summ.is_file():
        return {"ok": False, "reason": "无评测汇总", "tail": out[-800:]}
    data = json.loads(summ.read_text())
    res = [r for r in data.get("results", []) if r.get("rollouts")]
    n_roll = sum(len(r["rollouts"]) for r in res)
    n_res = sum(r.get("pass_count", 0) for r in res)
    out_d = {
        "ok": True, "label": label, "tasks": len(res), "rollouts": n_roll,
        "resolved": n_res, "pass_at_1": round(n_res / n_roll, 4) if n_roll else 0.0,
    }
    log(f"  {label}: resolved {n_res}/{n_roll}（pass@1={out_d['pass_at_1']:.1%}）")
    return out_d


# ------------------------------------------------------------------ 主流程

def rebaseline_with_model(model_path: str, tag: str) -> dict:
    """降级换模型后，用新模型**重新采集训练数据与基线**。

    ## 为什么不能沿用旧数据

    训练数据里的 prompt 是固定的（题目 + 文件内容），换模型不影响；
    但 **basline pass@1 必须用同一模型测**，否则"训练前 20%（7B）→
    训练后 15%（3B）"这种对比毫无意义 —— 差异全部来自模型大小，
    与训练效果无关。

    这是闭环验证的底线：**前后必须同模型**。
    """
    log(f"用 {tag} 重建基线（换模型后旧基线失效）")
    if not start_vllm(model_path):
        return {"ok": False, "reason": "推理服务未就绪"}
    r = closed_loop_eval(model_path, f"before-{tag}")
    kill_gpu_hogs()
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="collect,train,eval")
    ap.add_argument("--model-7b", default="/data/swe-rl/model/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--model-3b", default="/data/swe-rl/model/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--skip-collect-wait", action="store_true")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    log(f"编排开始，阶段：{stages}")

    # ---------- 阶段 1：等采集 ----------
    if "collect" in stages and not args.skip_collect_wait:
        r = wait_collect()
        save_state("collect", **r)
        if r.get("ok") and r.get("with_variance", 0) == 0:
            log("⚠️ 没有任何题目组内有方差 —— GRPO 无梯度可学，但仍继续训练以留下记录")

    # ---------- 阶段 2：训练 ----------
    trained_model = None
    if "train" in stages:
        stop_vllm()

        ds = build_dataset(args.model_7b)
        save_state("dataset", **ds)
        if not ds["ok"]:
            log("✗ 训练数据生成失败，跳过训练")
        else:
            # 先给 7B 一次机会：与采集/评测模型一致才有闭环意义
            sm = smoke_train(args.model_7b, "7B")
            save_state("smoke_7b", **sm)
            if sm["ok"]:
                trained_model = args.model_7b
                log("✓ 7B 可训练，用 7B 正式训练")
            else:
                log(f"✗ 7B 装不下（oom={sm['oom']}），降级到 3B")
                sm3 = smoke_train(args.model_3b, "3B")
                save_state("smoke_3b", **sm3)
                if sm3["ok"]:
                    trained_model = args.model_3b
                    log("✓ 3B 可训练")
                else:
                    log("✗ 3B 也无法训练，请人工介入")

            if trained_model:
                # 训练前基线：与训练同模型，保证可比
                base = closed_loop_eval(trained_model, "before")
                save_state("eval_before", **base)
                stop_vllm()

                tr = full_train(trained_model)
                save_state("train", model=trained_model, **tr)

    # ---------- 阶段 3：训练后评测 ----------
    if "eval" in stages and trained_model:
        after = closed_loop_eval(trained_model, "after")
        save_state("eval_after", **after)

        st = json.loads(STATE.read_text()).get("stages", {})
        b = st.get("eval_before", {}).get("pass_at_1")
        a = after.get("pass_at_1")
        if b is not None and a is not None:
            log(f"\n{'=' * 60}\n闭环对比：pass@1  {b:.1%} → {a:.1%}  (Δ={a - b:+.1%})")

    log("编排结束，状态见 logs/orchestrate_state.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
