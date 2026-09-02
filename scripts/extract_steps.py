#!/usr/bin/env python3
"""在节点侧从 verl 日志提取关键指标，输出精简 CSV。

为什么要在节点侧提取：verl 每个 step 打印一整行数百个指标（31 步约 15 万字符），
直接把原始日志回传会超过云助手单次输出上限而被静默截断 ——
实测曾只拿回 11 步，误判训练提前结束。

实现注意：不能用 `txt.split("step:")` 切分 —— 日志里同时存在
`step:20`、`training/global_step:20`、`timing_s/step:44.6` 等，
split 会把一行切碎。改为**逐行**用正则提取。
"""
import re
import sys

FIELDS = [
    ("step", r"training/global_step:(\d+)"),
    ("score_mean", r"critic/score/mean:([0-9.eE+-]+)"),
    ("score_max", r"critic/score/max:([0-9.eE+-]+)"),
    ("grad_norm", r"actor/grad_norm:([0-9.eE+-]+)"),
    ("entropy", r"actor/entropy:([0-9.eE+-]+)"),
    ("pg_loss", r"actor/pg_loss:([0-9.eE+-]+)"),
    ("resp_len", r"response_length/mean:([0-9.eE+-]+)"),
]

path = sys.argv[1]
rows = {}
with open(path, errors="replace") as f:
    for line in f:
        if "critic/score/mean" not in line:
            continue
        vals = {}
        for name, pat in FIELDS:
            m = re.search(pat, line)
            if m:
                try:
                    vals[name] = float(m.group(1))
                except ValueError:
                    pass
        if "step" in vals and "score_mean" in vals:
            rows[int(vals["step"])] = vals

print(",".join(n for n, _ in FIELDS))
for k in sorted(rows):
    print(",".join(str(rows[k].get(n, "")) for n, _ in FIELDS))
