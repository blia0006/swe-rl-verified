#!/usr/bin/env python3
"""
训练前的 GPU 环境冒烟测试（在容器内跑）
======================================

在真正开训之前把「会让训练零步崩溃」的因素全部验一遍。
这些每一条都对应上一轮踩过的坑，或本轮换模型带来的新风险：

| 检查 | 为什么必须查 |
|---|---|
| GPU 与算力 | 5090 是 sm_120，需 cu128；CUDA 12.1 镜像会报 no kernel image |
| `/dev/shm` | 容器默认仅 64MB，NCCL 每 rank 需约 31.5MB → 训练零步崩溃，**运行期无法补救** |
| verl / vLLM 版本 | 配置项名随版本变动 |
| **7B 权重能否加载** | 换模型的最大风险；1.5B→7B 权重从 3GB 涨到 15GB |
| **vLLM 起得来且能生成** | rollout 引擎，起不来就没有采样 |
| 显存余量 | 7B + LoRA + vLLM 同卡共存，24GB 是否够 |
| 沙箱凭证 | reward function 要连沙箱，缺凭证会让每步都判 0 |

用法（容器内）：
    python3 scripts/preflight_gpu.py
    python3 scripts/preflight_gpu.py --skip-vllm    # 只查环境，不加载模型
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", "/data/swe-rl"))
MODEL = Path(os.environ.get("MODEL_PATH", str(WORKDIR / "model" / "Qwen2.5-Coder-3B-Instruct")))

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print("%s %-34s %s" % ("✓" if ok else "✗", name, detail), flush=True)
    return ok


def gb(n):
    return n / 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vllm", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print(" 训练前环境冒烟测试")
    print("=" * 72)

    # ---------- GPU ----------
    try:
        import torch

        avail = torch.cuda.is_available()
        check("torch.cuda 可用", avail, torch.__version__)
        if avail:
            cap = torch.cuda.get_device_capability()
            props = torch.cuda.get_device_properties(0)
            check(
                "GPU 与算力",
                cap >= (8, 0),
                "%s sm_%d%d %.1fGB" % (props.name, cap[0], cap[1], gb(props.total_memory)),
            )
            # bf16 是本轮训练精度，必须原生支持
            check("bf16 支持", torch.cuda.is_bf16_supported(), "训练与 rollout 均用 bfloat16")
            free, total = torch.cuda.mem_get_info()
            check("显存空闲", gb(free) > 20, "%.1f/%.1fGB 空闲" % (gb(free), gb(total)))
    except Exception as e:
        check("torch 导入", False, str(e)[:150])

    # ---------- /dev/shm（上一轮零步崩溃的元凶）----------
    try:
        st = os.statvfs("/dev/shm")
        size_mb = st.f_blocks * st.f_frsize / 1024 ** 2
        # NCCL 每 rank 约需 31.5MB；训练脚本已设 NCCL_SHM_DISABLE=1 兜底
        check(
            "/dev/shm 容量",
            size_mb >= 256 or os.environ.get("NCCL_SHM_DISABLE") == "1",
            "%.0fMB%s" % (size_mb, "（已设 NCCL_SHM_DISABLE 兜底）"
                          if size_mb < 256 else ""),
        )
    except Exception as e:
        check("/dev/shm", False, str(e)[:100])

    # ---------- 框架版本 ----------
    for mod in ("verl", "vllm", "transformers", "peft", "ray"):
        try:
            m = __import__(mod)
            check("%s 版本" % mod, True, getattr(m, "__version__", "?"))
        except Exception as e:
            check("%s 导入" % mod, False, str(e)[:100])

    # ---------- 数据与凭证 ----------
    for label, path in (
        ("模型目录", MODEL),
        ("训练数据", WORKDIR / "data" / "grpo_train.parquet"),
        ("题目元数据", WORKDIR / "data" / "tasks.jsonl"),
        ("文件内容", WORKDIR / "data" / "file_contents.json"),
        ("沙箱凭证", WORKDIR / ".env"),
    ):
        exists = path.exists()
        extra = ""
        if exists and path.is_dir():
            extra = "%.1fGB" % gb(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))
        elif exists:
            extra = "%.1fMB" % (path.stat().st_size / 1024 ** 2)
        check(label, exists, extra or str(path))

    # 凭证内容自检：缺了会导致每步 reward 都是 0，且不易察觉
    env_path = WORKDIR / ".env"
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        # AGS_TOOL_NAME 曾漏投，导致训练时 79% 的打分抛异常、reward 全 0，
        # 且 grad_norm 恒 0（参数完全没更新）—— 必须纳入检查
        need = ["TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY",
                "E2B_API_KEY", "TCR_REGISTRY", "TCR_NAMESPACE",
                "AGS_TOOL_NAME"]
        miss = [k for k in need if ("%s=" % k) not in text or ("%s=\n" % k) in text]
        check("凭证字段完整", not miss, "缺少 %s" % miss if miss else "沙箱可连")

    # ---------- 训练数据自检 ----------
    parquet = WORKDIR / "data" / "grpo_train.parquet"
    if parquet.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(parquet)
            n = len(df)
            # train_batch_size=1 时行数即 step 数；验收要求 ≥50
            check("训练样本数", n >= 50, "%d 行 → 约 %d step（要求 ≥50）" % (n, n))
            lens = df["prompt"].apply(lambda p: len(p[1]["content"]))
            check("prompt 长度", lens.max() < 24000,
                  "最大 %d 字符（约 %d token）" % (lens.max(), lens.max() // 3.5))
        except Exception as e:
            check("读取训练数据", False, str(e)[:150])

    # ---------- 模型加载 + 生成（最大风险项）----------
    if not args.skip_vllm and MODEL.exists():
        print("\n--- 加载 7B 模型并试生成（约 2~4 分钟）---", flush=True)
        try:
            from vllm import LLM, SamplingParams

            llm = LLM(
                model=str(MODEL),
                dtype="bfloat16",
                gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.75")),
                max_model_len=8192,
                enforce_eager=True,
                disable_log_stats=True,
            )
            out = llm.generate(
                ["### calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n"],
                SamplingParams(temperature=0.9, max_tokens=64),
            )
            text = out[0].outputs[0].text
            check("vLLM 加载 + 生成", len(text) > 0, "生成 %d 字符" % len(text))
            import torch

            free, total = torch.cuda.mem_get_info()
            # 注意：此处是**单独**加载 vLLM 的场景，训练时 FSDP 先占位、
            # vLLM 只拿剩余显存，两者不可直接比较。3B 权重 6.2GB，
            # 训练实测占用约 8.1GB/24GB，因此这里 >3GB 即可视为通过。
            check("加载后显存余量", gb(free) > 3,
                  "%.1fGB 空闲（训练时 FSDP 先占位，实际分配与此不同）" % gb(free))
        except Exception as e:
            check("vLLM 加载", False, str(e)[:250])

    # ---------- 汇总 ----------
    print("\n" + "=" * 72)
    failed = [(n, d) for n, ok, d in results if not ok]
    if failed:
        print("✗ %d 项未通过，训练前必须解决：" % len(failed))
        for n, d in failed:
            print("    · %s  %s" % (n, d))
        return 1
    print("✓ 全部 %d 项通过，可以开始训练" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
