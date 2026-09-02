#!/usr/bin/env python3
"""对比两轮训练的 reward 曲线与关键指标。

上一轮：/Users/jacobliang/学习/题目四：强化学习/docs/reward_curve.csv
        1.5B + unified diff + 自建合成题
本轮  ：docs/reward_curve.csv
        3B + search/replace + SWE-bench Verified

两轮的题目集与任务表示都不同，因此**不能直接比 reward 数值大小** ——
本轮题目（真实开源项目 issue）远难于上一轮的合成题。
有意义的比较是「模型走到了哪一层」：能否写出可应用的 patch、能否真正修对题。
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREV = Path("/Users/jacobliang/学习/题目四：强化学习/docs/reward_curve.csv")
CUR = ROOT / "docs" / "reward_curve.csv"


def load(path, col_mean, col_max, col_grad):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "step": int(float(r["step"])),
                    "mean": float(r[col_mean] or 0),
                    "max": float(r[col_max] or 0),
                    "grad": float(r[col_grad] or 0),
                })
            except (ValueError, KeyError):
                continue
    return rows


def stats(rows, label):
    n = len(rows)
    means = [r["mean"] for r in rows]
    maxes = [r["max"] for r in rows]
    grads = [r["grad"] for r in rows]
    t = max(1, n // 3)
    return {
        "label": label,
        "n": n,
        "avg": sum(means) / n,
        "peak_mean": max(means),
        "early": sum(means[:t]) / t,
        "mid": sum(means[t:2 * t]) / max(1, len(means[t:2 * t])),
        "late": sum(means[-t:]) / t,
        "step_gt02": sum(1 for m in means if m > 0.2001),
        "step_perfect": sum(1 for m in maxes if m >= 0.999),
        "step_zero": sum(1 for m in means if m < 1e-9),
        "grad_ok": sum(1 for g in grads if abs(g) > 1e-6),
    }


def bar(v, lo, hi, width=26):
    if hi - lo < 1e-12:
        return ""
    return "█" * max(0, min(width, int((v - lo) / (hi - lo) * width)))


def spark(vals, width=40):
    blocks = "▁▂▃▄▅▆▇█"
    v = vals[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-12:
        return blocks[0] * len(v)
    return "".join(blocks[min(7, int((x - lo) / (hi - lo) * 7.99))] for x in v)


def main():
    if not PREV.exists():
        print("找不到上一轮数据：%s" % PREV)
        return 1
    prev = load(PREV, "critic/score/mean", "critic/score/max", "actor/grad_norm")
    cur = load(CUR, "score_mean", "score_max", "grad_norm")
    a, b = stats(prev, "上一轮"), stats(cur, "本轮")

    print("=" * 74)
    print(" 两轮训练对比")
    print("=" * 74)
    print()
    print(" %-22s %-24s %-24s" % ("", "上一轮", "本轮"))
    print(" %-22s %-24s %-24s" % ("模型", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-3B"))
    print(" %-22s %-24s %-24s" % ("任务表示", "unified diff", "search/replace"))
    print(" %-22s %-24s %-24s" % ("题目", "自建合成题", "SWE-bench Verified 官方"))
    print(" %-22s %-24s %-24s" % ("reward 档位", "2 档", "4 档"))
    print(" %-22s %-24s %-24s" % ("组大小 rollout.n", "8", "8"))
    print()

    print("─" * 74)
    print(" 【A】能力层面：模型走到了哪一步")
    print("─" * 74)
    rows = [
        ("组内出现满分(max=1.0)的 step", a["step_perfect"], a["n"], b["step_perfect"], b["n"]),
        ("均值 > 0.2 的 step", a["step_gt02"], a["n"], b["step_gt02"], b["n"]),
        ("grad_norm 有效的 step", a["grad_ok"], a["n"], b["grad_ok"], b["n"]),
        ("均值为 0 的 step", a["step_zero"], a["n"], b["step_zero"], b["n"]),
    ]
    for name, av, an, bv, bn in rows:
        ap_, bp = av / an * 100, bv / bn * 100
        print(" %-30s %2d/%-2d (%4.1f%%)   →   %2d/%-2d (%4.1f%%)"
              % (name, av, an, ap_, bv, bn, bp))
    print()
    print("   注：满分 = 组内有采样让 F2P 全过且 P2P 无回归，即**真正修对了题**")
    print()

    print("─" * 74)
    print(" 【B】reward 数值（仅供参考，两轮题目难度不同，不可直接比大小）")
    print("─" * 74)
    print(" %-22s %10s %10s" % ("", "上一轮", "本轮"))
    print(" %-22s %10d %10d" % ("总 step", a["n"], b["n"]))
    print(" %-22s %10.4f %10.4f" % ("全程均值", a["avg"], b["avg"]))
    print(" %-22s %10.4f %10.4f" % ("单步均值峰值", a["peak_mean"], b["peak_mean"]))
    print()

    print("─" * 74)
    print(" 【C】趋势：前 1/3 → 中 1/3 → 后 1/3")
    print("─" * 74)
    for s in (a, b):
        delta = s["late"] - s["early"]
        shape = "U 型" if s["mid"] < s["early"] and s["late"] > s["mid"] else (
            "上升" if delta > 0.005 else ("下降" if delta < -0.005 else "持平"))
        print(" %-6s  %.4f → %.4f → %.4f   变化 %+.4f  (%s)"
              % (s["label"], s["early"], s["mid"], s["late"], delta, shape))
    print()

    print("─" * 74)
    print(" 【D】曲线形状")
    print("─" * 74)
    print(" 上一轮 (%d步)  %s" % (a["n"], spark([r["mean"] for r in prev])))
    print(" 本轮   (%d步)  %s" % (b["n"], spark([r["mean"] for r in cur])))
    print()
    print(" 本轮出现满分的 step：%s"
          % ", ".join(str(r["step"]) for r in cur if r["max"] >= 0.999))
    print(" 上一轮出现满分的 step：%s"
          % (", ".join(str(r["step"]) for r in prev if r["max"] >= 0.999) or "无"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
