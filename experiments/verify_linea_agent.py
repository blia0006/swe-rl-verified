#!/usr/bin/env python3
"""线 A 端到端实测：沙箱内的 ReAct Agent 跑通一道真题
======================================================

验证课题验收第 2 条的完整链路，一次跑通四件事：

```
① 起题目沙箱（含 /testbed + 官方测试环境）
② 沙箱 → VPC 内网 → GPU 节点 vLLM（http://10.0.0.11:8000）        ← 线 A 通路
③ 沙箱内多轮 ReAct：读文件/搜索/编辑/跑测试，落 tracing.jsonl      ← Agent 主体
④ 取回 patch，用官方判据判分                                       ← reward 闭环
```

## 为什么把这四件事放一起测

单独测通路（curl 一下）说明不了问题：真正的风险在**组合处** ——
Agent 能连上模型但输出格式不对、能编辑但 patch 取不出来、能跑测试但判分
对不上。上一轮的教训是各环节单测都过、拼起来 pass@1 为 0。

## 沙箱内的测试脚本

`run_tests` 工具需要一个能跑该题 F2P 的脚本。这里复用
`pipeline/official_spec.make_eval_script` 生成的官方脚本 —— 与最终判分**同源**，
避免 Agent 看到的"测试通过"与判分结果不一致。

用法：
    python3 experiments/verify_linea_agent.py --task <task_id> [--max-steps 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AGENT_SRC = ROOT / "sandbox_agent" / "react_agent.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="swe-rl-policy")
    ap.add_argument("--keep", action="store_true", help="结束后不回收沙箱（便于人工排查）")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    for n in ("e2b", "e2b.api.client_sync", "httpx", "httpcore"):
        logging.getLogger(n).setLevel(logging.WARNING)

    # 复用采集脚本的地址解析：沙箱无法访问 127.0.0.1（那是它自己的 localhost），
    # 必须用节点内网 IP。两处若各写一份逻辑，迟早分叉。
    from pipeline.collect_tracing import resolve_base_url

    base_url = resolve_base_url(args.base_url)

    from clients.ags import AGSClient
    from clients.sandbox import connect_with_retry, start_instance_with_warmup

    from experiments.verify_criteria import tcr_image
    from pipeline.official_spec import make_eval_script
    from pipeline.sandbox_eval import run_eval, sbx_run, sbx_write

    tasks = {
        json.loads(l)["task_id"]: json.loads(l)
        for l in open(ROOT / "data/tasks.jsonl")
        if l.strip()
    }
    task = tasks[args.task]
    image = tcr_image(args.task, os.environ["TCR_REGISTRY"], os.environ["TCR_NAMESPACE"])

    ags = AGSClient()
    tool = ags.find_tool(os.environ.get("AGS_TOOL_NAME", "swe-rl-vpc-runner"))
    assert tool, "找不到沙箱工具"

    print(f"=== 线 A 端到端实测：{args.task}")
    print(f"    推理服务: {base_url}")
    t0 = time.time()
    iid, _ = start_instance_with_warmup(
        ags, tool["tool_id"], image, cpu="2", memory="4Gi", verbose=False
    )
    print(f"    沙箱 {iid} 启动 {time.time() - t0:.1f}s")

    try:
        sbx = connect_with_retry(iid)
        sbx_run(sbx, "mkdir -p /task", 60)

        # ---- 步骤②：先验通路，失败要能立刻区分「网络不通」与「模型有问题」----
        code, out = sbx_run(
            sbx, f"curl -s -m 10 {base_url}/v1/models | head -c 200", 60
        )
        ok_net = "swe-rl-policy" in out or '"object"' in out
        print(f"    ② 线A通路: {'✓' if ok_net else '✗ ' + out[:200]}")
        if not ok_net:
            return 1

        # ---- 投放 Agent 与题目规格 ----
        # spec 里刻意**不含 golden_patch** —— 沙箱内的 Agent 绝不能看到答案，
        # 否则采集到的 tracing 毫无价值（模型会直接抄）。
        spec = {
            "task_id": args.task,
            "repo": task["repo"],
            "problem_statement": task.get("problem_statement") or task.get("issue", ""),
            "test_script_path": "/task/agent_test.sh",
        }
        sbx_write(sbx, "/task/react_agent.py", AGENT_SRC.read_text(encoding="utf-8"))
        sbx_write(sbx, "/task/spec.json", json.dumps(spec, ensure_ascii=False))
        # Agent 的 run_tests 用与最终判分同源的官方脚本
        sbx_write(sbx, "/task/agent_test.sh", make_eval_script(task))

        # ---- 步骤③：跑 ReAct ----
        print(f"    ③ 启动 ReAct（最多 {args.max_steps} 步）…")
        t1 = time.time()
        code, out = sbx_run(
            sbx,
            f"/usr/bin/python3 /task/react_agent.py --spec /task/spec.json "
            f"--base-url {base_url} --model {args.model} "
            f"--max-steps {args.max_steps} --temperature {args.temperature} "
            f"--rollout-id probe 2>&1 | tail -20",
            timeout=2400,
        )
        print(f"       耗时 {time.time() - t1:.0f}s，输出：")
        for ln in out.strip().splitlines()[-6:]:
            print(f"       {ln[:160]}")

        # ---- 取回 tracing 与 patch ----
        _, tr = sbx_run(sbx, "cat /task/tracing.jsonl 2>/dev/null | head -c 400000", 120)
        out_dir = ROOT / "data" / "linea_probe"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{args.task}.tracing.jsonl").write_text(tr, encoding="utf-8")

        steps = [json.loads(l) for l in tr.splitlines() if l.strip()]
        summary = next((s for s in steps if s.get("type") == "summary"), {})
        tool_seq = [s.get("tool") for s in steps if s.get("type") == "step"]
        print(f"    tracing: {len(steps)} 行；工具序列: {' → '.join(t or '?' for t in tool_seq[:14])}")
        print(
            f"    汇总: 步数={summary.get('steps_used')} 成功编辑={summary.get('edits_applied')} "
            f"格式错误={summary.get('parse_errors')} patch={summary.get('patch_chars')}字符 "
            f"主动完成={summary.get('finished_by_agent')}"
        )

        patch = summary.get("patch") or ""
        if not patch.strip():
            print("    ④ 判分: 跳过（Agent 未产出任何改动）")
            return 0

        # ---- 步骤④：官方判据判分（strict=True，评测口径）----
        print("    ④ 官方判据判分…")
        r = run_eval(sbx, task, patch, timeout=1800, strict=True)
        f2p = f"{r.outcome.f2p_passed}/{r.outcome.f2p_total}" if r.outcome else "-"
        p2p = f"{r.outcome.p2p_passed}/{r.outcome.p2p_total}" if r.outcome else "-"
        print(
            f"       stage={r.stage.value} F2P={f2p} P2P={p2p} "
            f"reward={r.reward.reward:.3f} resolved={r.reward.strict_pass}"
        )
        if r.error:
            print(f"       └ {r.error[:200]}")
        (out_dir / f"{args.task}.eval.json").write_text(
            json.dumps(r.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0
    finally:
        if not args.keep:
            try:
                ags.stop_instance(iid)
            except Exception:  # noqa: BLE001,S110
                pass


if __name__ == "__main__":
    sys.exit(main())
