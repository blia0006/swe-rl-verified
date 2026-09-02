#!/usr/bin/env python3
"""分析训练日志：reward 为何未上升。

关注三个量的联动 —— 单看 reward 无法定位根因：
  · entropy   ：策略的输出多样性。GRPO 靠组内 reward 差异产生 advantage，
                entropy 塌了则 8 个采样趋同、advantage 恒 0、无梯度可学
  · grad_norm ：参数实际更新幅度。持续接近 0 = 白训
  · score_max ：组内最高分。>0.99 说明该步真有采样修对了题
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/data/swe-rl/logs/train_20260902_014005.log"
txt = open(path, errors="replace").read()

pat = (
    r"actor/entropy:([0-9.eE+-]+).*?"
    r"actor/grad_norm:([0-9.eE+-]+).*?"
    r"training/global_step:(\d+).*?"
    r"critic/score/mean:([0-9.eE+-]+).*?"
    r"critic/score/max:([0-9.eE+-]+).*?"
    r"response_length/mean:([0-9.]+)"
)
d = {}
for m in re.finditer(pat, txt):
    d[int(m.group(3))] = (
        float(m.group(1)), float(m.group(2)),
        float(m.group(4)), float(m.group(5)), float(m.group(6)),
    )

ks = sorted(d)
print("step   entropy    grad_norm    score   s_max  resp_len")
for k in ks:
    e, g, s, mx, r = d[k]
    hit = " ★" if mx >= 0.999 else ""
    print("%4d %9.4f %12.2e %8.4f %6.2f %8.1f%s" % (k, e, g, s, mx, r, hit))

if not ks:
    print("未解析到数据")
    sys.exit(1)

t = max(1, len(ks) // 3)
ent = [d[k][0] for k in ks]
grd = [d[k][1] for k in ks]
scr = [d[k][2] for k in ks]

print("\n--- 分段均值（前 1/3 → 后 1/3）---")
print("entropy   : %.4f → %.4f" % (sum(ent[:t]) / t, sum(ent[-t:]) / t))
print("grad_norm : %.3e → %.3e" % (sum(grd[:t]) / t, sum(grd[-t:]) / t))
print("score     : %.4f → %.4f" % (sum(scr[:t]) / t, sum(scr[-t:]) / t))

zero_grad = sum(1 for g in grd if g < 1e-6)
zero_score = sum(1 for s in scr if s < 1e-9)
perfect = sum(1 for k in ks if d[k][3] >= 0.999)
print("\ngrad_norm < 1e-6 的步数 : %d/%d" % (zero_grad, len(ks)))
print("score 为 0 的步数       : %d/%d" % (zero_score, len(ks)))
print("组内出现满分的步数      : %d/%d" % (perfect, len(ks)))
