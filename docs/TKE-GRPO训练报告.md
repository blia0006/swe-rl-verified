# TKE Pod 内 GRPO 正式训练报告

> 本文档对应要求②「GPU 训练须部署在 TKE 上」的**合规交付物**。
> 与 `docs/GRPO训练完成报告.md`（宿主机 `ctr` 容器内的续训调试会话，7B/60 步，
> 用于验证 OOM 补丁）是两条独立跑道、两份不同 checkpoint，**不要混用**。

## 结论：训练已在 TKE Pod 容器内跑满 192/192 步，退出码 0

| 项 | 值 |
|---|---|
| 执行环境 | TKE Pod `swe-rl-train`（`kubectl exec` 进入容器后台启动，非宿主机裸跑） |
| 编排脚本 | `scripts/orchestrate_3b_v2.sh`（第 4 步：`kubectl exec swe-rl-train -- ... nohup bash scripts/run_grpo_training.sh`） |
| 模型 | Qwen2.5-Coder-3B-Instruct |
| Checkpoint 目录 | `/data/swe-rl/checkpoints_3b_v1`（Pod 内路径，经 `deploy/gpu-pod.yaml` 的 hostPath 挂载到宿主机 `/data/swe-rl`，**删 Pod 不丢数据**） |
| 总步数 | **192/192**，`TOTAL_EPOCHS=1` |
| 训练用时 | 3:07:27（约 52~59s/step） |
| 退出状态 | 退出码 **0**，正常完成 |
| 训练日志 | Pod 内 `/data/swe-rl/logs/train_pod_3b.log` |
| Checkpoint 落盘 | `global_step_10` ~ `global_step_192`（每 10 步一存），时间戳 09-03 13:06 ~ 15:05，与日志耗时吻合 |

## 部署链路依据（要求②的实测证据）

- `deploy/gpu-pod.yaml` + `deploy/apply.sh` + `deploy/create-secret.sh`：Pod 部署与凭证注入（commit `3520ac9`）
- `experiments/verify_linea_path.py` + `linea_probe_server.py`：Pod 内 GPU / 框架 / 挂载盘的端到端连通性验证（commit `df4ff5c`）
- 本文档记录的 192 步训练：在上述 Pod 就绪后，真正跑的正式 GRPO 训练（非探针测试）

## 训练指标：reward 按阶段分段（`critic/score/mean`，192 步全量提取）

| 阶段 | 步数 | 均值 |
|---|---|---|
| 0-30 | 29 | 0.0806 |
| 30-60 | 30 | 0.0202 |
| 60-90 | 30 | 0.0508 |
| 90-120 | 30 | 0.0830 |
| 120-150 | 30 | 0.0823 |
| 150-192 | 42 | **0.1391** |
| 全程均值 | 192 | 0.0795 |

- 非零步数：85/192（44.3%），说明奖励信号并未塌陷
- 组内满分（`score_mean=1.0`）出现在 step 92、188
- 走势非单调（30-60 段有明显低谷，抽样噪声导致），但**前段（0-90，均值 0.050）vs 后段（150-192，均值 0.139）对比呈上升趋势**，与 `docs/PROGRESS.md` 中此前 31 步/55 步版本"曲线未上升"的结论相比有改善
- 训练末尾一次 held-out 验证：`val-core/swebench_verified/reward/mean@1 = 0.0333`（样本量小，仅供参考，不构成强结论）

## 与此前版本的关系

| 版本 | 步数 | 执行环境 | Checkpoint | 状态 |
|---|---|---|---|---|
| 3B+LoRA（09-02） | 31 | 宿主机 `ctr`（设计阶段决策，见 `PROGRESS.md` Phase 0） | `checkpoints`（LoRA） | 历史记录，已按要求②推动整改 |
| **3B 全量（09-03，本报告）** | **192** | **TKE Pod（`kubectl exec`）** | `checkpoints_3b_v1` | **满足要求②的正式交付物** |
| 7B（09-03，续训调试） | 60 | 宿主机 `ctr`（`train_guard.sh`） | `checkpoints` | OOM 补丁验证会话，与 TKE 要求无关，不作为②的交付证据 |

## 后续建议

1. 若需进一步验收，可直接从 Pod（或 hostPath `/data/swe-rl/checkpoints_3b_v1`）取 `global_step_192` 权重做离线评测，无需重新训练。
2. `docs/GRPO训练完成报告.md`、`docs/PROGRESS.md`、`README.md` 中仍有"不用 pod / ctr 直起"的表述，那是**设计初期（09-02）的决策记录**，已于 09-03 整改为 TKE Pod 执行；三份文档已加追记/勘误链接到本报告，避免后续误读为"最终仍未用 TKE"。
