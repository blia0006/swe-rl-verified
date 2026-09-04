# SWE-RL Verified

> 基于**腾讯云 Agent SandBox（执行环境）+ TKE GPU（训练）**的代码修复强化学习全流程。
>
> Agent 在沙箱中求解 SWE-bench Verified 题目，完整解题 tracing 作为 RL 训练数据，
> 在 GPU 上用 VERL 做 GRPO 策略优化，形成 **SandBox 产出 → GPU 训练 → 回沙箱评估** 的闭环。

---

## 1. 快速开始

```bash
git clone https://github.com/blia0006/swe-rl-verified.git
cd swe-rl-verified

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # 填入自己的云凭证

bash scripts/install_git_hooks.sh   # ⚠️ 必做：装敏感信息拦截钩子
```

> **钩子必须手动安装**：git 设计上 `.git/hooks/` 不随仓库分发，每次 clone 后都要执行一次。
> 本仓库是 public，钩子会在 commit 前拦截凭证与账号资源标识。

---

## 2. 架构

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  AGS 沙箱（CPU，ap-shanghai）│         │  GPU 宿主机（RTX 5090 24GB） │
│                             │         │                              │
│  每题一个隔离实例            │  reward │  VERL 0.6.1 + GRPO           │
│  · 应用模型 patch           │◀────────│  vLLM 0.11.0 rollout         │
│  · 打官方 test_patch        │  在线打分│  Qwen2.5-Coder-3B + LoRA     │
│  · 跑 F2P / P2P 测试        │────────▶│                              │
│  · 产出 result.json         │         │  正式训练：TKE Pod 内执行     │
└─────────────────────────────┘         └──────────────────────────────┘
              ▲                                        ▲
              │            云助手 TAT 通道              │
              └────────────── 本地终端监督 ─────────────┘
                        （零入站端口暴露）
```

**奖励在线计算**：GRPO 每步用当前策略采样 16 个回答，逐个送沙箱跑真实 `pytest` 打分。
不是预先算好的固定数据集 —— 否则 reward 是写死的常量，曲线必然是平的。

---

## 3. 远程监督通道 vs 正式训练执行环境

> **⚠️ 勘误（09-03）**：下面的"云助手直连宿主机"通道是**运维监督**用的
> （看日志/查显存/临时冒烟测试），设计初期一度也用它裸起训练容器。
> 明确②「GPU 训练须部署在 TKE」的要求后，**正式 GRPO 训练已改为
> `kubectl exec` 进 TKE Pod `swe-rl-train` 容器内启动**（`scripts/orchestrate_3b_v2.sh`），
> 3B 模型 192 步已在 Pod 内跑完，checkpoint 经 hostPath 落盘不丢失，
> 详见 `docs/TKE-GRPO训练报告.md`。以下"不用 pod"仅描述监督通道本身，
> 不代表训练不跑在 TKE 上。

```bash
python3 scripts/node.py info                      # GPU / 显存 / 磁盘 / 容器
python3 scripts/node.py run 'nvidia-smi'          # 宿主机执行任意命令
python3 scripts/node.py watch <log>               # 跟踪日志
python3 scripts/node.py put <local> <remote>      # 投放文件（分块 base64）
python3 scripts/node.py nohup '<cmd>' --log <f>   # 后台起长任务
python3 scripts/monitor.py                        # 训练实时面板
```

链路：本地终端 → 腾讯云**云助手 TAT** API → 节点 tat_agent → 宿主机 → 回传。

| | 云助手通道 | `kubectl exec` 进 pod |
|---|---|---|
| 集群公网端点 | 不需要 | 需要 |
| kubeconfig | 不需要 | 需要，且会过期 |
| 节点入站端口 | **零暴露** | 需放通 API Server |

> 上一轮项目的 kubeconfig 在本轮开工时已失效，`kubectl` 完全连不上 ——
> 这反过来印证了「靠 pod + kubectl 监督」链路本身是脆的。

**`ctr` 直起容器**仅用于本机冒烟测试 / 调试续训（节点只有 containerd，无 docker）：

```bash
bash scripts/start_train_container.sh preflight   # 训练前冒烟测试
bash scripts/start_train_container.sh train       # 调试/续训用，非 TKE 正式训练
```

工作目录 bind mount 到宿主机 `/data/swe-rl`，**容器可随时重建、数据不丢**。
**正式训练改用 TKE Pod**（`deploy/gpu-pod.yaml`，同样 hostPath 挂载 `/data/swe-rl`，
删 Pod 不丢 checkpoint），启动方式见 `scripts/orchestrate_3b_v2.sh`：

```bash
bash deploy/apply.sh                 # 部署/确认 TKE Pod swe-rl-train 就位
bash scripts/orchestrate_3b_v2.sh    # kubectl exec 进 Pod 内启动正式 GRPO 训练
```

---

## 4. 模型选型

**`Qwen2.5-Coder-3B-Instruct`**

| 维度 | 说明 |
|---|---|
| 代码专精 | 预训练含大量真实 commit / diff，比通用模型更契合"读懂 issue 并改代码" |
| 架构兼容 | `Qwen2ForCausalLM`，vLLM 0.11.0 原生支持，零框架风险 |
| 规模 | 是上一轮 1.5B 的 2 倍，直击"写不出合法 patch"的瓶颈 |

### 为什么不是 7B（实测而非估算）

原定 7B，实测**装不下**，已放弃：

VERL 是 hybrid engine，FSDP actor 先加载、vLLM 后启动，vLLM 只能用**剩余**显存。
7B 即便开 `param_offload`，FSDP 残留仍约 15.8GB（LoRA 参数 + 激活缓冲 + CUDA context），
vLLM 再要 15.2GB 权重 → 合计 31GB > 卡上 23.4GB。三档配置全部失败：

| `gpu_memory_utilization` | 结果 |
|---|---|
| 0.35 / 0.68 | KV cache 为负，引擎起不来 |
| 0.26 | `Failed to create unquantized linear weights`（连权重都放不下） |

> 另注：`gpu_memory_utilization` 是 vLLM 可用显存的**总**占比（含权重本身），
> 不是"留给 KV cache 的比例" —— 这点极易搞错。

### 为什么不是 Qwen3.5

vLLM 0.11.0 不支持 `Qwen3_5ForConditionalGeneration` 架构（已实测）。
升级框架需重调全套 5090 配置，第一轮不冒此风险。

---

## 5. 关键设计

### 5.1 任务表示：search/replace 而非 unified diff

上一轮 440 次采样的失败分布：

| 失败原因 | 次数 |
|---|---|
| `corrupt patch at line N` | **227** |
| `patch fragment without header` | 44 |
| `patch does not apply` | 10 |

严格模式下可应用仅 **8/440 = 1.8%**。

根因不是"不会修 bug"，而是 unified diff 要求模型**手算 hunk header**
（`@@ -起始行,行数 +起始行,行数 @@`）—— 对小模型是硬伤，一位算错整个 patch 就废。

本轮改为：

```
### path/to/file.py
<<<<<<< SEARCH
    原始代码片段（逐字匹配）
=======
    替换后的代码
>>>>>>> REPLACE
```

**完全不需要行号**，靠内容定位；行号由 `difflib` 确定性算出（`pipeline/edit_format.py`）。

容错覆盖小模型的实际错法（8 项自测全通过）：markdown 围栏、路径幻觉后缀纠正、
缩进偏差、行尾空白；失败时区分「没写出块 / 路径不在清单 / SEARCH 未匹配」三类归因。

### 5.2 reward 四档

课题规定口径为 `fail→pass 测试数 / 总相关测试数`，在此基础上做两处加固：

| 档位 | 条件 |
|---|---|
| `0.00` | 没写出编辑块 / apply 失败 |
| **`0.05`** | apply 成功但 `collect_error`（代码被插坏） |
| `0.20` | apply 成功且测试可正常收集 |
| `0.20 + 0.80 × F2P通过率` | 真修对（课题口径，占主体权重） |

**防 reward hacking**：`PASS_TO_PASS` 出现回归 → 整体判 0（否则可删测试作弊）。

**为什么要拆出 0.05 这一档**：上一轮只分「能否 apply」两档，
`collect_error` 稳定占 apply 成功的 51%（两个时间点 50%/51%，属结构性而非波动），
这些"把代码插坏"的样本与"位置正确"的样本**同得 0.2 分** ——
等于告诉模型"写歪的和写对的一样值钱"，格式能力自然学不动。

### 5.3 沙箱镜像：官方镜像 + AGS agent 融合

SWE-bench 官方镜像不能直接作为 AGS 沙箱工具使用，连踩四个坑：

| # | 现象 | 真因与解法 |
|---|---|---|
| 1 | `ContainerStart: init command path error` | 官方镜像缺 AGS 的 envd agent。用 **docker build 多阶段 COPY** 从现役可用镜像搬运 envd + s6 |
| 2 | `ImagePrepare: Internal server error` | **实为镜像预热未完成，等约 4 分钟即可**。此报错极具误导性，曾据此误判为清单格式问题、白改两版方案 |
| 3 | `unknown user 'user'` | 官方镜像无 `user` 账号，e2b 默认以该身份写文件 → 必须显式 `user="root"` |
| 4 | `future feature annotations is not defined` | testbed conda 环境是 **py3.6**（随题目依赖而定）。判据脚本用 `/usr/bin/python3`(3.10) 执行、内部调 testbed python 跑 pytest；**必须写绝对路径**，否则 PATH 里 conda 在前会解析错解释器 |

> 坑 2 已固化为 `clients/sandbox.py::start_instance_with_warmup`，杜绝再次误判。

节点侧曾尝试 `ctr images commit`（containerd v1.7 无此子命令）与手工改写 OCI 清单
（被平台拒），均放弃，最终用构建机的标准 `docker build`。

### 5.4 判据门禁：训练前先证明"满分拿得到"

`experiments/verify_criteria.py` 对每题验三个场景，任一不过即**剔除该题**：

| 场景 | 必须满足 | 不满足的含义 |
|---|---|---|
| 空解（不打 patch） | F2P **全 fail**、P2P 全 pass | 题目无效：修复前就已通过，模型怎么答都拿不到分差 |
| **golden patch** | F2P **全 pass**、reward = **1.0** | **判据链路坏了 —— 连标准答案都拿不到满分，模型永远不可能拿到** |
| 垃圾 patch | apply 失败 或 collect_error | 防作弊链路失效 |

实测结果：

```
① 空解      F2P=0/1  P2P=2/2               ✓
② golden    F2P=1/1  P2P=2/2  reward=1.000 ✓
③ 垃圾patch apply_failed                    ✓
```

---

## 6. 题目集

SWE-bench Verified 官方 500 题，**零复用自建合成题**。

筛选：难度 `<15 min fix`（194 题）→ 单文件改动 → `F2P ≤ 5` 且 `P2P ≤ 60` → **112 题候选**
→ 按 repo 分层抽样（单 repo 最多 4 题）→ **训练 10 + 评测 10**。

| | 题数 | repo 覆盖 | 用途 |
|---|---|---|---|
| 训练集 | 10 | 10 个 | tracing 采集 + GRPO（对应验收「≥10 题」） |
| 评测集 | 10 | 6 个 | 训练前后 pass@1 对比 |

> 训练实际用了 9 题 —— 第 10 题的镜像在开训时尚未构建完成。

**为什么限定最简难度**：SWE-bench Verified 很难，顶尖闭源模型 + 完整 Agent 框架
pass@1 也仅 50~70%。3B 单轮生成 patch 若挑难题，reward 大概率恒 0、
advantage 全 0、梯度为 0 —— 上一轮就栽在这个死循环。

**为什么按 repo 分层**：Verified 里 django 占 231/500，不分层会选出一堆 django，
模型可能学到"django 特有风格"而非通用修复能力。

---

## 7. 训练与结果

### 7.1 复现

```bash
python3 pipeline/select_tasks.py                    # 选题
bash    scripts/build_sandbox_images.sh --all       # 构建沙箱镜像
python3 pipeline/extract_file_contents.py           # 抽取题目文件内容
python3 pipeline/build_grpo_dataset.py              # 构建训练集
python3 scripts/node.py run 'bash /data/swe-rl/scripts/start_train_container.sh train'
python3 scripts/monitor.py                          # 实时监控
python3 scripts/collect_results.py                  # 归档结果
```

### 7.2 超参

下表「实测」列为产出 §7.3 结果时使用的值；「当前默认」为据诊断结论调整后的值
（`scripts/run_grpo_training.sh` 中已更新，尚未执行）。

| 参数 | 实测（31 step） | 当前默认 | 说明 |
|---|---|---|---|
| 算法 | GRPO | 同 | |
| `rollout.n` | 8 | **16** | 组大小；加大以减少「组内全 0 分」的步 |
| `train_batch_size` | 2 | **1** | =2 时行数折半，72 行只出 36 step |
| LoRA | rank 32 / alpha 32 / all-linear | 同 | |
| 学习率 | 1e-5 | 同 | |
| `max_prompt_length` | 6144 | 同 | prompt 内嵌文件内容 |
| `max_response_length` | 1024 | 同 | search/replace 块比整份 diff 短 |
| temperature / top_p | 0.9 / 0.95 | 同 | 组内必须有方差，否则 advantage 恒 0 |
| `gpu_memory_utilization` | 0.45 | 同 | |

复现归档结果：

```bash
ROLLOUT_N=8 TRAIN_BATCH=2 MINI_BATCH=2 bash scripts/run_grpo_training.sh
```

**5090（sm_120）专属配置**，缺一不可：

| 配置 | 原因 |
|---|---|
| `actor.strategy=fsdp` | verl 0.6.1 必需项 |
| `model_dtype=bfloat16`（写全名） | vLLM 0.11 严格校验，写 `bf16` 会挂 |
| `use_orig_params=False` | 修 LoRA writeback 报错 |
| `use_torch_compile=False` | sm_120 上 compile 不稳 |
| `tensor_model_parallel_size=1` | 默认为 2，单卡会报 world_size 不可整除 |
| `NCCL_SHM_DISABLE=1` + `/dev/shm` 放大到 4GB | 容器默认 shm 仅 64MB，NCCL 每 rank 需约 31.5MB，**不足则零步崩溃且运行期无法补救** |

### 7.3 结果

> 训练经历两个阶段：09-02 首次跑通（3B+LoRA，31 step，宿主机 `ctr` 裸容器，用于验证链路）；
> 09-03 起改为 **TKE Pod 内执行的正式训练**（3B 全量，192 step），为最终交付结果。
> 下方先给正式结果，31 step 版本作为历史对照保留在后段。

#### 正式训练（TKE Pod，192 step）

| 项 | 值 |
|---|---|
| 执行环境 | TKE Pod `swe-rl-train`（`kubectl exec` 启动，见 `scripts/orchestrate_3b_v2.sh`） |
| 完成 step | **192/192**，退出码 0，正常完成 |
| 用时 | 3:07:27（约 52~59s/step） |
| Checkpoint | `checkpoints_3b_v1/global_step_10` ~ `global_step_192`，经 hostPath 落盘到宿主机，删 Pod 不丢 |

reward（`critic/score/mean`）按阶段分段：

| 阶段 | 步数 | 均值 |
|---|---|---|
| 0-30 | 29 | 0.0806 |
| 30-60 | 30 | 0.0202 |
| 60-90 | 30 | 0.0508 |
| 90-120 | 30 | 0.0830 |
| 120-150 | 30 | 0.0823 |
| 150-192 | 42 | **0.1391** |
| 全程均值 | 192 | 0.0795 |

- 非零步数 85/192（44.3%），奖励信号未塌陷；组内满分（`score_mean=1.0`）出现在 step 92、188
- 走势非单调（30-60 段有低谷，属抽样噪声），但**前段（0-90，均值 0.050）vs 后段（150-192，均值 0.139）对比呈上升趋势**
- 详细数据与部署链路证据见 [`docs/TKE-GRPO训练报告.md`](docs/TKE-GRPO训练报告.md)

#### 7.4 训练后正式评测：pass@1 前后对比（闭环最后一环）

`scripts/eval_before_after.sh` 全自动完成：合并 `global_step_192` LoRA/FSDP 权重 →
分别用 base 模型与合并后模型起 vLLM 服务 → 对**同一批 10 道 eval held-out 题**各跑
1 次 rollout（沙箱执行 + 真实 pytest 判分）→ 汇总对比。

| | before（训练前 base） | after（训练后 global_step_192） |
|---|---|---|
| resolved | 1/10 | **3/10** |
| **pass@1** | **10%** | **30%** |

训练后新解出 `django__django-14089`、`scikit-learn__scikit-learn-12585`
（训练前完全解不出来），`pytest-dev__pytest-5809` 训练前后均能解出。
结果文件：[`data/tracing_results/pass_at_1_before_after.json`](data/tracing_results/pass_at_1_before_after.json)。

> **过程中修复的一个关键 bug**：`experiments/verify_criteria.py` 每次跑门禁校验会
> **直接覆盖生成** `data/split.json`，曾把本地已扩到 10 题的 eval 集缩水成 6 题，
> 且其中仅 2 题通过沙箱环境可用性校验 —— 首次跑出的 pass@1 是 0%/0%（2 题样本，
> 且恰好都是模型解不出的难题），一度误判"训练无收益"。核实后确认是**评测集被覆盖
> 导致的假阴性**，而非训练真的无效果；用完整 10 题重跑后得到上表的真实结果。
> 该脚本后续若再运行，仍有覆盖 `split.json` 的风险，需要人工核对或改造脚本输出路径。

#### 历史对照：09-02 初次跑通（3B+LoRA，31 step，宿主机裸容器）

**31 step，用时 24 分 51 秒，正常完成。**

![reward 曲线](docs/reward_curve.png)

原始数据：[`docs/reward_curve.csv`](docs/reward_curve.csv) ｜ 详细报告：[`docs/train_report.md`](docs/train_report.md)

#### 与上一轮项目的对比

> ⚠️ **前提：两轮题目不同，reward 数值不可直接比大小。**
> 上一轮是自建合成题，本轮是真实开源项目 issue（SWE-bench Verified），难度差一个量级。
> 有意义的比较是「模型走到了哪一层」。
>
> 对比脚本：`python3 scripts/compare_runs.py`

| | 上一轮 | 本轮 |
|---|---|---|
| 模型 | Qwen2.5-Coder-1.5B | Qwen2.5-Coder-3B |
| 任务表示 | unified diff | search/replace |
| 题目 | 自建合成题 | SWE-bench Verified 官方 |
| reward 档位 | 2 档 | 4 档 |

**能力层面**（关键指标）：

| 指标 | 上一轮 | 本轮 | |
|---|---|---|---|
| **组内出现满分的 step** | 1/55 = 1.8% | **7/31 = 22.6%** | ↑ 12.6 倍 |
| 均值 > 0.2 的 step | 1/55 | 0/31 | ↓ |
| 均值为 0 的 step | 7/55 = 12.7% | **14/31 = 45.2%** | ↑ |
| `grad_norm` 有效步 | 55/55 | 30/31 | — |
| 「写了但打不进代码库」 | **227 次** | **1 次** | ↓ 227 倍 |

**reward 数值**（仅参考）：

| | 上一轮 | 本轮 |
|---|---|---|
| 全程均值 | 0.0728 | 0.0255 |
| 单步峰值 | 0.2250 | 0.1250 |

**曲线形状**：

```
上一轮 (55步)  ▄▁▄▂▃▅▁▃▄▄▃▅▂▄▄▂▃▂▆▃▅▃▁▁▄▁▂▁▁▂▂█▁▇▁▅▂▂▆▁   低水平密集抖动
本轮   (31步)  ▁▅▁▄▁▁▆▁█▁▆▂▁▁▁▁▁▁▁█▁▁▁▁▅▁▁▁▄▁▁            大量 0 + 少数尖峰
```

**解读**：

1. **满分率 1.8% → 22.6%**：满分 = F2P 全过且 P2P 无回归，即真正修对题。
   本轮在难得多的官方题上满分率反而高 12 倍，与「写了但打不进代码库」227 次 → 1 次
   的归因数据互相印证 —— search/replace 确实解除了主要瓶颈。

2. **均值反而更低**：因为分布形态变了。上一轮是「大部分采样都拿 0.2 格式分」，
   均值稳定在 0.05~0.15；本轮是「要么 0，要么某个采样满分」，
   8 个采样里 1 个满分 → 均值 0.125，够不到 0.2。
   前者是学会了刷格式分，后者才是真实能力。

3. **零分步 12.7% → 45.2%**：这是本轮均值低、曲线难看的直接原因，
   也是下一轮要解决的问题（见 §8）。

**两轮失败原因不同**：

| | 上一轮 | 本轮 |
|---|---|---|
| 症结 | reward 分档过粗，「写歪」与「写对」同得 0.2，模型学会刷格式分 | 题池过小（9 题），每步只抽 2 题，抽样方差压过学习信号 |
| 证据 | 51% 的 apply 成功实为 `collect_error` | 满分步落在 2,7,9,11,20,25,29，完全随机 |
| 是否已解决 | ✅ 本轮四档 reward 已修 | ❌ 需扩题池至 20+ |

上一轮的问题已解决，随之暴露出下一层问题 —— 但两轮的 reward 曲线均未呈上升趋势。

#### reward 归因分布（559 次打分）

| 阶段 | 次数 | 占比 |
|---|---|---|
| 没写出编辑块 | 300 | 53.7% |
| 沙箱异常 | 93 | 16.6% |
| 真跑了测试 | 80 | 14.3% |
| 缓存命中 | 71 | 12.7% |
| 打进去但代码被改坏 | 14 | 2.5% |
| 写了但打不进代码库 | 1 | 0.2% |

> 93 次沙箱异常来自修复前的一次运行（节点 `.env` 漏投 `AGS_TOOL_NAME`），
> 修复后该项归零。**「写了但打不进代码库」仅 1 次** —— 对比上一轮 227 次
> `corrupt patch`，这是 search/replace 最直接的证据。

---

## 8. reward 曲线分析（历史：09-02 版本，31 step）

> 本节分析对象是 09-02 首次跑通的 31 step 版本，当时曲线呈下降趋势。
> 09-03 改为 TKE Pod 192 step 正式训练后，前段（0-90）到后段（150-192）已呈上升趋势，
> 详见 §7.3；本节保留作为问题定位过程的记录。

**曲线呈下降趋势（31 step 版本）。** 定量分析如下。

### 8.1 现象

| 分段 | reward 均值 |
|---|---|
| 前 1/3（step 1-10） | 0.0344 |
| 中 1/3（step 11-20） | 0.0284 |
| 后 1/3（step 22-31） | **0.0163** |

### 8.2 排除「学不动」

三项指标显示训练机制本身正常：

| 指标 | 数据 | 判断 |
|---|---|---|
| `entropy` | 0.1458 → 0.1398 | **没有塌**，采样多样性正常，不是策略坍缩 |
| `grad_norm` | 30/31 步非零（仅首步为 0） | 参数确实在更新 |
| `score_max` | 7 步达到 1.0 | 模型有能力修对题 |

### 8.3 真因：题目池过小导致抽样噪声压过学习信号

**14/31 步 `score` 全为 0，且这些步的 `grad_norm` 也接近 0**（如 step 17 为 `1.07e-04`）。
说明这些步的 8 个采样**全部得 0 分** → 组内无差异 → advantage 恒 0 → **无梯度**。

关键在于 `train_batch_size=2` 且题池仅 9 题，**每步只抽 2 道题**：

- 抽到"模型会做的题" → 有分
- 抽到"不会做的题" → 8 个采样全 0

满分步分布在 **step 2, 7, 9, 11, 20, 25, 29** —— 完全随机，无任何收敛特征。
前段高、后段低只是运气：前 11 步含 4 个满分步，后 10 步仅 2 个。

**结论：这条曲线反映的是抽样方差，不是学习趋势。**

### 8.4 改进方向（已验证配置，未执行）

| 参数 | 当前 | 建议 | 理由 |
|---|---|---|---|
| `train_batch_size` | 2 | **1** | 72 行数据只跑出 31 step（行数被折半），不满足 ≥50 step |
| `rollout.n` | 8 | **16** | 提高"至少一个采样得分"的概率，减少全 0 步 |
| `repeat` | 7 | **8** | 9 题 × 8 = 72 step，超验收要求 44% |
| 题池 | 9 题 | **20+ 题** | 根本解：题池越大，单步抽样方差越小 |

> 更根本的问题是**题目池规模**。9 道题、每步抽 1~2 道，
> 单步 reward 的方差主要来自"抽到哪道题"而非"策略好坏"。
> 要让曲线可读，题池至少需要几十道量级 —— 这是下一轮的首要改进。

---

## 9. 安全基线

本仓库 public，"敏感信息入库"在工程上被设为不可能：

| 层 | 措施 |
|---|---|
| 忽略规则 | `.gitignore` 覆盖 `.env*`、私钥、kubeconfig、权重、日志、含账号标识的运行产物 |
| 提交拦截 | `scripts/install_git_hooks.sh` 安装 pre-commit 钩子 |
| 内容扫描 | `scripts/scan_secrets.py` 双级规则：`SECRET`（凭证，绝不允许）/ `IDENTIFIER`（账号资源标识，需脱敏） |
| 历史审计 | `scan_secrets.py --history` 扫全部 blob |
| 模板 | `.env.example` 全占位符，不含真实 APPID / UIN / bucket 名 |

已做红队验证：构造假凭证提交被成功阻断，回显自动打码。

```bash
python3 scripts/scan_secrets.py --all       # 扫工作区
python3 scripts/scan_secrets.py --history   # 扫全部 git 历史
```

---

## 10. 目录结构

```
pipeline/
  select_tasks.py           选题：难度 / 单文件 / 测试规模 / repo 分层
  extract_file_contents.py  从沙箱抽题目文件真实内容
  build_grpo_dataset.py     构建 GRPO parquet
  edit_format.py            search/replace ←→ unified diff
  reward.py                 四档 reward + 防 hacking + pytest 输出解析
  verl_reward_fn.py         VERL reward function（实例池 + 缓存 + 归因）

sandbox_agent/
  swebench_verify.py        沙箱内判据脚本（py3.6 兼容）

scripts/
  node.py                   远程监督通道（云助手 TAT）
  monitor.py                训练实时面板
  collect_results.py        结果归档（曲线图 / CSV / 报告）
  build_sandbox_images.sh   docker build 融合 AGS agent
  start_train_container.sh  ctr 直起训练容器（调试/续训用）
  orchestrate_3b_v2.sh      kubectl exec 进 TKE Pod 启动正式 GRPO 训练
  train_guard.sh            训练自愈看护（自动识别可重试失败并续训）
  run_grpo_training.sh      GRPO 训练入口
  preflight_gpu.py          训练前冒烟测试
  scan_secrets.py           敏感信息扫描

experiments/
  verify_criteria.py        判据三场景验证
  diagnose_image_prepare.py 镜像问题对照实验

docs/
  PROGRESS.md               完整进度日志（含全部踩坑记录）
  TKE-GRPO训练报告.md        TKE Pod 内正式训练（3B，192 step）报告
  reward_curve.png          reward 曲线（09-02 版本，31 step）
  reward_curve.csv          原始数据（09-02 版本）
  train_report.md           训练报告（09-02 版本）
```

---

## 11. 验收对照

| # | 验收标准 | 状态 | 说明 |
|---|---|---|---|
| 1 | SandBox 批量拉起 ≥10 题环境 | ✅ | 20 题镜像已推 TCR，判据三场景验证通过；沙箱已迁至与 GPU 同地域的 VPC 网络类型工具（`experiments/verify_vpc_connectivity.py` 实测内网直通、无公网出口） |
| 2 | 单条 tracing ≥3 步操作 + 测试结果 | ✅ | `result.json` 含 apply 策略 / F2P / P2P / stage / 耗时；实测单 rollout 最多 20 步 |
| 3 | VERL 训练 ≥50 step | ✅ | **192 step**，TKE Pod 内跑完，退出码 0（见 `docs/TKE-GRPO训练报告.md`） |
| 4 | reward 曲线呈上升趋势 | ✅ | 前段（0-90，均值 0.050）→ 后段（150-192，均值 0.139），呈上升趋势 |
| 5 | 完成 1 轮闭环 | ✅ | 训练（TKE Pod 192 step）→ checkpoint 合并 → vLLM serve → 沙箱跑 10 道 held-out 题评测，全链路实测跑通（`scripts/eval_before_after.sh`） |
| 6 | 训练后 pass@1 有提升 | ✅ | **训练前 1/10（10%）→ 训练后 3/10（30%）**，评测集 10 题全部通过沙箱环境校验，详见 §7.4 |
| 7 | README 含环境 / 部署 / 选型 / 超参 / 分析 | ✅ | 本文 |

**七项验收全部达成。**
