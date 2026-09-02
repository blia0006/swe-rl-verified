#!/usr/bin/env python3
"""
训练实时监控面板（本地终端运行，数据来自 GPU 节点）
==================================================

一屏看清训练全貌，同时满足两个目的：
  · **实时发现问题**：reward 是否恒 0、grad_norm 是否为 0、显存是否吃紧、
    沙箱是否连不上 —— 这些都会在几步内暴露，不必等训练跑完
  · **课题交付**：训练过程与 reward 曲线是验收第 3、4 条的证据，
    面板同时把结构化数据落盘（`docs/train_metrics.csv`）

数据来源（都在 GPU 节点上，经云助手读取）：
  · 训练日志  `logs/train_*.log`      —— verl 每步打印的 step/score/grad_norm
  · reward 归因 `logs/reward_debug.jsonl` —— 每次打分的 stage 与耗时
  · `nvidia-smi`                       —— 显存与利用率

## 为什么要看 stage 分布而不只看 reward 均值

上一轮的教训：55 步里 54 步分数 ≤0.2，说明模型只拿到"格式分"、几乎从未
真正修对题 —— 但只看 reward 均值（0.0728）看不出这一点，误以为"在涨"。
本面板把每次打分的 stage 分档统计出来，一眼看清模型卡在哪一层：

    no_patch      → 连编辑块都没写出来
    apply_failed  → 写了但打不进代码库
    collect_error → 打进去了但把代码改坏
    tested        → 真跑了测试（只有这里才可能拿到测试分）

用法：
    python3 scripts/monitor.py              # 实时刷新
    python3 scripts/monitor.py --once       # 打印一次
    python3 scripts/monitor.py --interval 15
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE = str(ROOT / "scripts" / "node.py")
PY = sys.executable

# verl 每步日志形如：
#   step:12 - critic/score/mean:0.15 - actor/grad_norm:0.108 - ...
_METRIC_RE = re.compile(r"([\w/]+):\s*([-+0-9.eE]+)")


def node_run(cmd, timeout=120):
    """经云助手在 GPU 节点执行命令。"""
    try:
        p = subprocess.run(
            [PY, NODE, "run", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        out = p.stdout or ""
        # 滤掉 SDK 的告警噪音
        return "\n".join(
            ln for ln in out.splitlines()
            if "NotOpenSSL" not in ln and "warnings.warn" not in ln
        )
    except subprocess.TimeoutExpired:
        return ""


def parse_steps(log_text):
    """从训练日志里抽出每步的指标。"""
    steps = []
    for line in log_text.splitlines():
        if "step:" not in line or "critic/score/mean" not in line:
            continue
        d = {}
        for k, v in _METRIC_RE.findall(line):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        if "step" in d:
            steps.append(d)
    return steps


def fmt_sparkline(values, width=48):
    """用文本画一条趋势线 —— 终端里最直观的"曲线"。"""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return blocks[0] * len(vals)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.99))] for v in vals)


def moving_avg(values, w):
    if len(values) < w:
        return []
    return [sum(values[i - w + 1 : i + 1]) / w for i in range(w - 1, len(values))]


def collect():
    """一次性从节点抓齐所有数据（合并成一条命令，减少往返）。"""
    cmd = r"""
# 按「含 step 数最多」挑日志，而非按修改时间 ——
# 训练脚本每次用新时间戳建文件，失败的空日志可能反而更新，
# 曾因此读到旧文件、误判训练已停止
LOG=$(bash /data/swe-rl/scripts/pick_train_log.sh)
echo "###LOGFILE"; echo "$LOG"
echo "###GPU"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo "###ALIVE"
pgrep -f "main_ppo" >/dev/null && echo running || echo stopped
echo "###STEPS"
[ -n "$LOG" ] && grep -a "critic/score/mean" "$LOG" | tail -400
echo "###ERRORS"
# 只匹配**训练框架**的错误。不能裸匹配 Error/Traceback ——
# 题目的 issue 描述里本就含 TypeError/Traceback 等字样（会被 verl 打进日志），
# 那是数据不是故障，曾造成整屏误报。
[ -n "$LOG" ] && grep -aE "CUDA out of memory|OutOfMemoryError|RuntimeError:|torch\.|ray\.exceptions|Worker.*died|verl.*Error" "$LOG" \
  | grep -avE "timesince|USE_TZ|offset-naive|depth must be" | tail -5
echo "###REWARD"
[ -f /data/swe-rl/logs/reward_debug.jsonl ] && tail -400 /data/swe-rl/logs/reward_debug.jsonl
echo "###TAIL"
[ -n "$LOG" ] && tail -3 "$LOG"
"""
    out = node_run(cmd, timeout=180)
    sections = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("###"):
            cur = line[3:]
            sections[cur] = []
        elif cur:
            sections[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def render(data, csv_path):
    steps = parse_steps(data.get("STEPS", ""))
    scores = [s.get("critic/score/mean", 0.0) for s in steps]
    grads = [s.get("actor/grad_norm", 0.0) for s in steps]

    print("\033[2J\033[H", end="")  # 清屏
    print("=" * 78)
    print(" SWE-RL 训练监控    %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 78)

    alive = data.get("ALIVE", "").strip()
    gpu = data.get("GPU", "").strip()
    status = "\033[32m● 训练中\033[0m" if alive == "running" else "\033[31m● 已停止\033[0m"
    print(" 状态: %s    GPU: %s" % (status, gpu))

    if not steps:
        print("\n 尚无 step 数据（模型加载 + vLLM 初始化约需 3~5 分钟）")
        tail = data.get("TAIL", "")
        if tail:
            print("\n 日志尾部:")
            for ln in tail.splitlines()[-3:]:
                print("   " + ln[:100])
    else:
        last = steps[-1]
        n = len(steps)
        print("\n 进度: step %d    最新 reward %.4f    grad_norm %.3e"
              % (last.get("step", n), scores[-1], grads[-1]))

        # 关键健康指标：grad_norm 恒 0 意味着参数完全没更新（上一轮前两轮训练的死循环）
        # 首步 grad_norm 常为 0（warmup / 组内 reward 全同 → advantage 为 0），
        # 只有**持续**为 0 才是真问题（上一轮前两轮训练就死在这里）
        nonzero_grad = sum(1 for g in grads if abs(g) > 1e-8)
        recent = grads[-10:]
        recent_zero = sum(1 for g in recent if abs(g) <= 1e-8)
        if recent_zero == len(recent) and len(recent) >= 5:
            flag = "\033[31m← 最近 %d 步梯度全 0，参数未更新\033[0m" % len(recent)
        elif nonzero_grad == n:
            flag = "✓"
        else:
            flag = "（首步为 0 属正常：组内 reward 相同则 advantage 为 0）"
        print(" grad_norm 非零: %d/%d %s" % (nonzero_grad, n, flag))

        print("\n reward 趋势（每步均值）")
        print("   " + fmt_sparkline(scores))
        print("   最小 %.4f   最大 %.4f   均值 %.4f"
              % (min(scores), max(scores), sum(scores) / len(scores)))

        ma = moving_avg(scores, 5)
        if len(ma) >= 2:
            trend = ma[-1] - ma[0]
            arrow = "↑ 上升" if trend > 0.005 else ("↓ 下降" if trend < -0.005 else "→ 持平")
            print("   5 步滑动平均: %.4f → %.4f  (%s)" % (ma[0], ma[-1], arrow))

        # 落盘供课题交付使用（验收第 4 条要 reward 曲线）
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("step,critic/score/mean,critic/score/max,actor/grad_norm,actor/pg_loss\n")
                for s in steps:
                    f.write("%d,%s,%s,%s,%s\n" % (
                        int(s.get("step", 0)),
                        s.get("critic/score/mean", ""),
                        s.get("critic/score/max", ""),
                        s.get("actor/grad_norm", ""),
                        s.get("actor/pg_loss", ""),
                    ))
        except OSError:
            pass

    # ---- reward 归因分布：看清模型卡在哪一层 ----
    rl = data.get("REWARD", "")
    if rl:
        stages, rewards, elapsed = Counter(), [], []
        for line in rl.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            stages[d.get("stage", "?")] += 1
            if isinstance(d.get("reward"), (int, float)):
                rewards.append(d["reward"])
            if isinstance(d.get("elapsed_s"), (int, float)):
                elapsed.append(d["elapsed_s"])
        total = sum(stages.values())
        if total:
            print("\n 打分归因（最近 %d 次）" % total)
            labels = {
                "no_patch": "没写出编辑块",
                "apply_failed": "写了但打不进代码",
                "collect_error": "打进去但代码被改坏",
                "tested": "真跑了测试",
                "cache_hit": "缓存命中",
                "exception": "沙箱异常",
            }
            for st, c in stages.most_common():
                bar = "█" * max(1, int(c / total * 34))
                print("   %-20s %4d  %5.1f%%  %s"
                      % (labels.get(st, st), c, c / total * 100, bar))
            if rewards:
                hi = sum(1 for r in rewards if r > 0.2001)
                print("   拿到测试分(>0.2): %d/%d = %.1f%%%s"
                      % (hi, len(rewards), hi / len(rewards) * 100,
                         "" if hi else "  \033[33m← 尚未真正修对过题\033[0m"))
            if elapsed:
                print("   单次打分耗时: 中位 %.1fs" % sorted(elapsed)[len(elapsed) // 2])

    err = data.get("ERRORS", "").strip()
    if err:
        print("\n \033[31m最近报错\033[0m")
        for ln in err.splitlines()[-4:]:
            print("   " + ln[:110])

    print("\n" + "=" * 78)
    print(" Ctrl-C 退出  |  完整日志: python3 scripts/node.py tail <log> -n 100")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--csv", default=str(ROOT / "docs" / "train_metrics.csv"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    try:
        while True:
            render(collect(), csv_path)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出监控（训练在节点上继续运行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
