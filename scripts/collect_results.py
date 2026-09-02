#!/usr/bin/env python3
"""
训练结果归档：产出课题验收所需的曲线图与分析报告
================================================

对应验收标准：
  · 第 3 条「训练至少 50 step」          → 报告里的 step 数
  · 第 4 条「reward 曲线呈上升趋势，提供图表」→ docs/reward_curve.png
  · 第 7 条「README 含结果分析」          → docs/train_report.md

除了画曲线，还做**失败归因分解** —— 这是上一轮缺失的关键一环：
上一轮只记 reward 标量，事后无法回答"模型到底卡在哪一层"，
只能靠猜。本脚本按 stage 分解每一次打分，让结论有据可依。

用法：
    python3 scripts/collect_results.py               # 从节点拉数据并生成
    python3 scripts/collect_results.py --no-fetch    # 用本地已有数据
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE = str(ROOT / "scripts" / "node.py")
PY = sys.executable
DOCS = ROOT / "docs"

STAGE_LABEL = {
    "no_patch": "没写出编辑块",
    "apply_failed": "写了但打不进代码库",
    "collect_error": "打进去但代码被改坏",
    "tested": "真跑了测试",
    "cache_hit": "缓存命中",
    "exception": "沙箱异常",
    "sandbox_error": "沙箱异常",
}


def node_run(cmd, timeout=180):
    p = subprocess.run(
        [PY, NODE, "run", cmd], capture_output=True, text=True,
        timeout=timeout, cwd=str(ROOT),
    )
    return "\n".join(
        ln for ln in (p.stdout or "").splitlines()
        if "NotOpenSSL" not in ln and "warnings.warn" not in ln
    )


def fetch():
    """从节点取训练日志与 reward 归因数据。"""
    DOCS.mkdir(parents=True, exist_ok=True)
    print("从节点拉取训练数据…")

    steps_raw = node_run(
        'LOG=$(ls -t /data/swe-rl/logs/train_2*.log 2>/dev/null | head -1); '
        '[ -n "$LOG" ] && grep -a "critic/score/mean" "$LOG"'
    )
    (DOCS / "_steps.raw").write_text(steps_raw, encoding="utf-8")

    # reward_debug 可能很大，分块取
    size = node_run('wc -c < /data/swe-rl/logs/reward_debug.jsonl 2>/dev/null || echo 0').strip()
    try:
        nbytes = int(size.split()[0])
    except (ValueError, IndexError):
        nbytes = 0
    print("  reward 归因日志 %.1f KB" % (nbytes / 1024))
    chunks, off = [], 0
    while off < nbytes:
        part = node_run(
            "tail -c +%d /data/swe-rl/logs/reward_debug.jsonl | head -c 40000" % (off + 1)
        )
        if not part:
            break
        chunks.append(part)
        off += 40000
    (DOCS / "_reward.raw").write_text("\n".join(chunks), encoding="utf-8")
    print("  已拉取 %d 行 step 日志" % len(steps_raw.splitlines()))


def parse_steps(text):
    import re

    rx = re.compile(r"([\w/]+):\s*([-+0-9.eE]+)")
    out = []
    for line in text.splitlines():
        if "critic/score/mean" not in line:
            continue
        d = {}
        for k, v in rx.findall(line):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        if "step" in d:
            out.append(d)
    return out


def plot(steps, out_png):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] 未安装 matplotlib，跳过绘图（pip install matplotlib）")
        return False

    xs = [s.get("step", i) for i, s in enumerate(steps)]
    ys = [s.get("critic/score/mean", 0.0) for s in steps]
    gs = [s.get("actor/grad_norm", 0.0) for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(xs, ys, "o-", ms=3, lw=1, alpha=0.45, color="#4C78A8", label="每步均值")
    w = 5
    if len(ys) >= w:
        ma = [sum(ys[i - w + 1 : i + 1]) / w for i in range(w - 1, len(ys))]
        ax1.plot(xs[w - 1 :], ma, lw=2.4, color="#E45756", label="%d 步滑动平均" % w)
    # 线性趋势：验收第 4 条要看"是否呈上升趋势"
    if len(ys) >= 3:
        n = len(ys)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            k = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den
            b = my - k * mx
            ax1.plot(xs, [k * x + b for x in xs], "--", lw=1.5, color="#54A24B",
                     label="线性趋势 (斜率 %+.2e)" % k)
    ax1.axhline(0.2, ls=":", c="gray", lw=1)
    ax1.text(xs[0], 0.205, "0.2 = 仅格式分（patch 可应用、测试可收集）",
             fontsize=8, color="gray")
    ax1.set_ylabel("reward")
    ax1.set_title("GRPO 训练 reward 曲线（Qwen2.5-Coder-3B / SWE-bench Verified）")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(xs, gs, lw=1, color="#B279A2")
    ax2.set_ylabel("grad_norm")
    ax2.set_xlabel("step")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("✓ 曲线图已保存 %s" % out_png)
    return True


def analyse_reward(text):
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    if not args.no_fetch:
        fetch()

    steps = parse_steps((DOCS / "_steps.raw").read_text(encoding="utf-8")
                        if (DOCS / "_steps.raw").exists() else "")
    recs = analyse_reward((DOCS / "_reward.raw").read_text(encoding="utf-8")
                          if (DOCS / "_reward.raw").exists() else "")
    if not steps:
        print("[x] 没有 step 数据")
        return 1

    ys = [s.get("critic/score/mean", 0.0) for s in steps]
    n = len(ys)

    # CSV（原始数据，便于复核）
    csv = DOCS / "reward_curve.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("step,score_mean,score_max,grad_norm,pg_loss\n")
        for s in steps:
            f.write("%d,%s,%s,%s,%s\n" % (
                int(s.get("step", 0)), s.get("critic/score/mean", ""),
                s.get("critic/score/max", ""), s.get("actor/grad_norm", ""),
                s.get("actor/pg_loss", "")))
    print("✓ 原始数据 %s" % csv)

    plot(steps, DOCS / "reward_curve.png")

    # ---- 报告 ----
    third = max(1, n // 3)
    early = sum(ys[:third]) / third
    late = sum(ys[-third:]) / third
    stages = Counter(r.get("stage", "?") for r in recs)
    total = sum(stages.values())
    rewards = [r["reward"] for r in recs if isinstance(r.get("reward"), (int, float))]
    got_test = sum(1 for r in rewards if r > 0.2001)

    lines = [
        "# 训练结果报告",
        "",
        "> 自动生成于 `scripts/collect_results.py`；原始数据见 `reward_curve.csv`。",
        "",
        "## 训练配置",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 模型 | Qwen2.5-Coder-3B-Instruct（7B 因单卡显存装不下） |",
        "| 算法 | GRPO（组大小 8） |",
        "| 题目 | SWE-bench Verified 官方，训练集 9 题 |",
        "| 任务表示 | search/replace 块（不需模型计算行号） |",
        "| reward | 四档：0 / 0.05 / 0.2 / 0.2+0.8×F2P通过率 |",
        "| 硬件 | RTX 5090 24GB（sm_120），单卡 |",
        "",
        "## 训练指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        "| 完成 step | **%d** |" % n,
        "| reward 均值 | %.4f |" % (sum(ys) / n),
        "| reward 最大 | %.4f |" % max(ys),
        "| 前 1/3 均值 | %.4f |" % early,
        "| 后 1/3 均值 | %.4f |" % late,
        "| 变化 | **%+.4f**（%s） |" % (
            late - early,
            "上升" if late > early * 1.05 else ("下降" if late < early * 0.95 else "基本持平")),
        "",
        "![reward 曲线](reward_curve.png)",
        "",
    ]

    if total:
        lines += [
            "## 打分归因分解（共 %d 次）" % total,
            "",
            '> 上一轮只记录 reward 标量，事后无法回答「模型卡在哪一层」。',
            "> 本轮逐次记录 stage，结论可追溯。",
            "",
            "| 阶段 | 次数 | 占比 |",
            "|---|---|---|",
        ]
        for st, c in stages.most_common():
            lines.append("| %s | %d | %.1f%% |" % (
                STAGE_LABEL.get(st, st), c, c / total * 100))
        lines += [
            "",
            "**拿到测试分（reward > 0.2，即真正修对过测试用例）：%d/%d = %.1f%%**"
            % (got_test, len(rewards), got_test / max(1, len(rewards)) * 100),
            "",
            "> 对照：上一轮 55 个 step 中仅 **1 步** 的分数超过 0.2，"
            '即几乎全程只拿到「patch 能应用」的格式分。',
            "",
        ]

    (DOCS / "train_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("✓ 报告 %s" % (DOCS / "train_report.md"))
    print("\n完成 %d step；前 1/3 %.4f → 后 1/3 %.4f（%+.4f）"
          % (n, early, late, late - early))
    return 0


if __name__ == "__main__":
    sys.exit(main())
