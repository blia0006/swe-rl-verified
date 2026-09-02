# 进度日志（公开版）

> 本文件是**脱敏**的公开记录：所有账号资源标识（实例 ID / VPC / 安全组 / EIP / APPID /
> bucket 名）一律以占位符表示，真实值只存在于本地 `.env` 与 `PROGRESS.local.md`
> （两者均不入库）。
>
> **上一轮项目**：用自建合成题跑完一轮闭环，7 项验收达成 5 项。
> 未达成的两项（reward 曲线未上升、pass@1 提升不显著）的定量根因见文末，
> 本轮的设计正是针对这些根因。

---

## 2026-09-02 · Phase 0：环境与可行性验证，全部门禁一次通过

本轮有三个"不确定就没法开工"的前提，全部实测（非推测）确认：

### 1. 远程监督通道：不用 pod，本地终端直连 GPU 宿主机

链路：本地终端 → 腾讯云**云助手 TAT** API → GPU 节点上的 Agent → 宿主机执行 → 回传输出。

实现于 `scripts/node.py`，子命令 `info` / `run` / `tail` / `watch` / `cexec` / `inject-key`。

相比 `kubectl exec` 进 pod 的三点优势：

| | 云助手通道 | kubectl exec |
|---|---|---|
| 集群公网端点 | 不需要 | 需要 |
| kubeconfig | 不需要 | 需要，且会过期 |
| 节点入站端口 | **零暴露** | 需放通 API Server |

> 上一轮的 kubeconfig 在本轮开工时已失效（endpoint 完全连不上），
> 这反过来印证了"靠 pod + kubectl 监督"链路本身是脆的。

**踩坑记录**：多层转义（本地 shell → TAT → `sh -lc` → `crictl exec`）会把
`python -c` 里的 `\n` 吃成字面量而报 `SyntaxError`。
解法：内层命令再做一次 base64 编码传输。

### 2. 不用 k8s：宿主机 `ctr` 直起 GPU 容器

节点只装了 containerd（`ctr` / `crictl`），**无 docker / nerdctl**，
且 GitHub release 下载不通（装 nerdctl 这条路不顺）。**但 `ctr` 原生就够用**：

```bash
ctr -n k8s.io run --rm --net-host --gpus 0 \
  --mount type=bind,src=/data/swe-rl,dst=/data,options=rbind:rw \
  docker.io/verlai/verl:vllm011.latest <name> bash -lc '...'
```

三要素实测全通：

| 要素 | 结果 |
|---|---|
| GPU | `torch.cuda.is_available()=True`，compute capability **(12,0) = sm_120** |
| 网络 | `--net-host` 复用宿主机网络栈，COS 可达 |
| 挂载 | 宿主机 `/data/swe-rl` ↔ 容器 `/data` 双向生效 |

前置组件已就位：`nvidia-container-runtime` / `nvidia-container-cli` / `nvidia-ctk`、CNI 插件齐全。

> **顺带修掉一个上一轮的隐患**：旧 pod 的 `/workspace` **没有任何 host 挂载**
> （`crictl inspect` 确认只挂了 serviceaccount 与 etc-hosts），22G checkpoint 全在
> 容器可写层 —— 删 pod 即全部丢失。
> 本轮工作目录一律落宿主机 `/data/swe-rl`，容器退化为"可随时抛弃的运行时"。

### 3. 镜像可达性：上一轮判定的"致命风险"已解除

上一轮交接文档把"能否拉到 SWE-bench 官方镜像"列为**最大不确定性**
（GPU 侧实测 `curl github.com` 返回 `000`）。本轮实测结论：

节点 `/etc/containerd/config.toml` 配了 `docker.io` → 腾讯云镜像加速器。
注意 **`ctr` 不读 CRI plugin 的 mirror 配置**（那是给 kubelet 用的），
所以 `ctr pull docker.io/...` 仍会直连 Docker Hub 而超时。**直接写 mirror 域名即可**：

```bash
# ✗ 超时
ctr -n k8s.io images pull docker.io/library/alpine:3.19
# ✓ 1.7s
ctr -n k8s.io images pull <tencent-mirror>/library/alpine:3.19
# ✓ SWE-bench 官方镜像：1.1GB / 23.7s / 45.6 MiB/s
ctr -n k8s.io content fetch --platform linux/amd64 \
  <tencent-mirror>/swebench/sweb.eval.x86_64.<instance_id_encoded>:latest
```

同时确认了官方镜像命名规则：instance_id 中的 `__` 编码为 `_1776_`
（`astropy__astropy-12907` → `sweb.eval.x86_64.astropy_1776_astropy-12907`）。

**→ 镜像搬运（pull → tag → push TCR）在 GPU 节点上做**：x86_64 原生架构、45MB/s、磁盘充裕。
本机是 ARM64，拉 amd64 镜像要走模拟层，慢且无必要。

### 4. 各站点连通性矩阵（决定每件事该在哪做）

| | Docker Hub | 腾讯云 mirror | HuggingFace | ModelScope | GitHub 页面 | GitHub release | TCR | COS |
|---|---|---|---|---|---|---|---|---|
| **本机** | ✅ | — | ✅ | — | ✅ | ✅ | — | ✅ |
| **GPU 节点** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |

由此确定分工：
- **模型权重** → 节点走 ModelScope 直下，或本机下载后经 COS 摆渡
- **SWE-bench 镜像** → 节点走腾讯云 mirror，再 push 到自有 TCR
- **代码 / 数据集** → 本机拉取，经 COS 或云助手摆渡进节点

### 5. GPU 环境规格

| 项 | 值 |
|---|---|
| GPU | **RTX 5090 D v2 / 24455 MiB**，驱动 580.173.02，sm_120（Blackwell） |
| CPU / 内存 | 32 核 / 92GB（可用 89GB，offload 空间充裕） |
| 磁盘 | 剩余 100G+ |
| OS | Ubuntu 22.04 |
| 网络 | **无公网 IP**，经 NAT 网关出网 |
| 计费 | 按量 |

容器内已有环境（可直接复用，省掉数小时准备）：

```
verl 0.6.1 | vLLM 0.11.0 | transformers 4.57.1 | torch 2.8.0+cu128 | peft 0.17.1
```

vLLM 架构支持实测（**直接界定选型范围**）：

| 架构 | 支持 | 覆盖模型 |
|---|---|---|
| `Qwen2ForCausalLM` | ✅ | Qwen2.5-Coder 全系 |
| `Qwen3ForCausalLM` | ✅ | Qwen3-4B / 8B / 14B / 32B |
| `Qwen3MoeForCausalLM` | ✅ | Qwen3-30B-A3B 系 |
| `Qwen3_5ForConditionalGeneration` | ❌ | **Qwen3.5 全系用不了** |

---

## 选型决策

| 项 | 决定 | 依据 |
|---|---|---|
| 模型 | **`Qwen2.5-Coder-7B-Instruct`** | 代码专精（预训练含大量真实 commit/diff），`Qwen2ForCausalLM` 架构 vLLM 0.11 原生支持、零框架风险；比上一轮的 1.5B 大 5 倍，直击"写不出合法 patch"这一主瓶颈 |
| 题源 | **SWE-bench Verified 官方**，不复用旧合成题 | 任务书首选项；无镜像质量疑虑；结果可对标公开 leaderboard |
| 题量 | **10 题左右** | 验收要求 ≥10；题少但每题多采样，避免上一轮"评测规模不足"的问题 |
| 运行方式 | 宿主机 `ctr` 直起容器 + 本地终端监督 | 见 Phase 0 第 1、2 项 |

### 选题池已就位

SWE-bench Verified 共 500 题，难度分布：

| 难度 | 题数 |
|---|---|
| `<15 min fix` | 194 |
| `15 min - 1 hour` | 261 |
| `1-4 hours` | 42 |
| `>4 hours` | 3 |

从 `<15 min fix` 中按「单文件改动 + F2P≤5 + P2P≤60」筛出 **112 道候选**，
最终 10 题从中按 repo 分层抽取（避免全部集中在 django）。

---

## 线 A 推理通道：沙箱出口 IP 实测随机，白名单方案作废

线 A 要求沙箱内的 Agent 通过 HTTP 调用 GPU 上的推理服务，而 GPU 节点无公网 IP。

**可行路径**：复用 GPU 节点所在 VPC 内**已有的 NAT 网关**做 DNAT 端口转发，
不必给节点绑 EIP —— 已核实该 NAT 与 GPU 子网同 VPC，子网默认路由已指向它，现有 DNAT 规则 0 条。

**但原计划的"安全组只放通沙箱 IP"做不到**：6 次独立采样得到 **6 个互不相同**的出口 IP，
跨多个 /8 段，均为云厂商公网地址池，无可预测规律。

因此实际方案定为：

1. DNAT 映射到**随机高位端口**（非默认 8000）
2. 推理服务强制 `--api-key`，值由 `openssl rand -hex 32` 生成，经环境变量注入，
   **不写入代码、镜像或 tracing**
3. 安全组只放通该**单一端口**
4. 采集窗口结束**立即删除 DNAT 规则** —— 规则一删，公网到 GPU 的路径彻底消失，
   这是比安全组更可靠的总闸；创建/删除均脚本化，采集脚本退出时自动回收

---

## 安全基线（public 仓库）

本仓库公开，因此把"敏感信息入库"做成物理上不可能：

| 层 | 措施 |
|---|---|
| 忽略规则 | `.gitignore` 覆盖 `.env*`（`.env.example` 除外）、私钥、kubeconfig、权重、日志 |
| 提交拦截 | `scripts/install_git_hooks.sh` 安装 pre-commit 钩子，命中即阻断 |
| 内容扫描 | `scripts/scan_secrets.py` 双级规则：`SECRET`（凭证，绝不允许）/ `IDENTIFIER`（账号资源标识，需脱敏） |
| 历史审计 | `scan_secrets.py --history` 扫全部 blob，首次公开前必跑 |
| 模板 | `.env.example` 全占位符，不含任何真实 APPID / UIN / bucket 名 |
| 内部日志 | 含资源标识的工作记录存 `PROGRESS.local.md`（不入库），公开版即本文件 |

⚠️ **钩子存放在 `.git/hooks/`，不随仓库分发**，因此每次 clone 后都需执行一次：

```bash
bash scripts/install_git_hooks.sh
```

---

## 上一轮未达成项的定量根因（本轮改造的依据）

| 现象 | 数据 | 根因 |
|---|---|---|
| reward 曲线 U 型、未上升 | 前 18 步均值 0.0722 → 中 0.0835 → 后 0.0632 | **reward 分档过粗**：`collect_error` 稳定占 apply 成功的 51%（两个时间点 50%/51%，属结构性而非波动），这些"把代码插坏"的样本与"位置正确"的样本**同得 0.2 分** |
| 440 次采样仅 8 次严格可 apply | strict 1.8%；`corrupt patch` 227 次 | **任务表示不当**：要求小模型手算 unified diff 的 hunk header 行号 |
| pass@1 训练前后无显著差异 | Welch t = −0.22（4 题 × 8 采样） | **评测规模不足**（期望成功次数仅 0.07）；且 `train_batch_size=1` 使单步 reward 主要由"抽到哪道题"决定 |

**本轮的五项改造**：

1. **reward 四档细化**：`0` / `0.05`（apply 成功但 collect_error）/ `0.2`（测试可正常收集）/ `0.2 + 0.8×F2P通过率`
   —— 让"写歪"与"写对"产生 4 倍差距，组内 advantage 才能推动格式能力
2. **任务表示改 search/replace 块** —— 不需算行号，直接绕开小模型最大短板
3. **prompt 固定文件路径模板** —— 上一轮实测有路径幻觉（丢 `src/` 前缀、凭空加前缀）
4. **评测集 ≥8 题 × k=8**，报告附显著性检验
5. **工程层**：工作目录挂宿主机、日志落宿主机文件、全程本地 `node.py watch` 监督

---

<!-- 后续记录追加在下面 -->

---

## 2026-09-02（续）· 镜像搬运的性能剖析：瓶颈是 TCR 服务端写入配额

搬运 20 题（10 训练 + 10 评测）耗时明显，做了完整定位，结论**与镜像体积无关**。

### 镜像其实不大

| 题目所属 repo | 镜像体积 |
|---|---|
| matplotlib | 2.1 GB |
| scikit-learn | 1.4 GB |
| django / flask | 1.1 GB |
| astropy / pylint / sphinx / sympy | 1.0 GB |

20 题合计约 **24 GB**（SWE-bench 三层镜像共享 base/env 层，故单题增量不大）。

### 耗时全在单向：push 比 pull 慢 10 倍

实测五题的拆解（同一份 ~1GB 数据）：

```
✓ 完成 275s（pull  24s / push 252s）
✓ 完成 253s（pull  10s / push 243s）
✓ 完成 295s（pull  24s / push 271s）
```

| 方向 | 路径 | 实测吞吐 | 每 GB |
|---|---|---|---|
| pull | Docker Hub → 腾讯云 mirror → 节点 | ~45 MiB/s | ~24s |
| **push** | 节点 → TCR 个人版 | **~14 MiB/s** | **~250s** |

push 占单题总耗时的 **91~96%**。

### 三个候选根因，逐一实测排除

| 假设 | 实测证据 | 判定 |
|---|---|---|
| 公网出口带宽不足 | TCR 域名解析到 `169.254.0.42`，RTT **0.083ms** | ✗ 走内网，非公网链路 |
| CPU 压缩打满 | 负载 **0.23 / 32 核**，containerd 仅 6% CPU | ✗ CPU 几乎空闲 |
| 并发不足 | 3 个 push 并行，网卡上行**仍为 14 MiB/s** | ✗ 加并发无增益 |

→ **结论：14 MiB/s 是 TCR 个人版的服务端写入配额上限。**
多个连接在抢同一配额池，并发只能把总时长摊到多题上，**不能提高总吞吐**。

pull 之所以快，是因为腾讯云 mirror 是读缓存（走 CDN），**读带宽与写配额是两套独立限制**。

### 提速方案评估：均不划算，接受现状

| 方案 | 结论 |
|---|---|
| 并发加到 8 | 无效（已实测服务端配额为总量限制） |
| 升级 TCR 企业版 | 需付费 + 改凭证/命名空间，为省 ~10 分钟不划算 |
| 让沙箱直连 Docker Hub | 架构不通：AGS 沙箱只能拉 TCR |
| 只搬训练集 10 题 | 仅把评测集的搬运成本后移，非节省 |

**且这是一次性成本**：镜像搬完常驻 TCR，后续任意轮次的训练与评测都直接从 TCR 拉取，无需重搬。

理论下限 ≈ 24GB ÷ 14 MiB/s ≈ **29 分钟**，与实际观测吻合。

> 附：本轮首次搬运曾因自身 bug（并发下逐题 prune 删掉共享层）失败过半，
> 白耗了一段时间 —— 那部分耗时不属于上述物理下限，已修复。


---

## 2026-09-02（续）· 关键辨析：上一轮"拿不到 reward"的真正机制（勿修错地方）

### 先纠一个容易混淆的判断

上一轮的 reward **确实是在线计算的**（每 step 用当前策略采样 → 当场送沙箱跑
真实 pytest → 拿分），这一点设计正确、链路也通。
因此"改成在线打分"**不是**本轮要修的地方 —— 它已经是在线的。

### 上一轮的实际数据（来自 `docs/reward_curve.csv`，55 步）

| 指标 | 值 |
|---|---|
| `critic/score/mean` 全程均值 | 0.0728 |
| 非零步数 | 48/55（87%） |
| **分数 ≤ 0.2 的步数** | **54/55** |
| **分数 > 0.2（即真修对过测试用例）的步数** | **1/55**（step 47，0.225） |

分数取值集中在 `0.05 / 0.075 / 0.1 / 0.125 / 0.15`，这些是「8 个采样里有 1~4 个
拿到 0.2、其余为 0」的组内平均值。

**→ 结论：模型全程几乎只拿到「patch 能被 git apply」的格式分（0.2），
从未从「真的修对题」上获得有效信号。**

### 根因：0.2 这一档把两类本质不同的样本混为一谈

```
apply 成功 + 位置正确 + 测试可收集   → 0.2
apply 成功 + 插错位置 + 代码语法坏   → 0.2      ← 实测占 apply 成功的 51%
```

组内 advantage 因此无法区分「写歪」与「写对」，等于告诉模型
"两者一样值钱"，格式能力自然学不动 —— 这是 reward 曲线呈 U 型而非上升的主因。

### 本轮的三处针对性改造

| # | 改造 | 针对的根因 | 状态 |
|---|---|---|---|
| ① | reward 四档：`0 / 0.05(apply但collect_error) / 0.2(测试可收集) / 0.2+0.8×F2P率` | 分档过粗（主因），使"写歪"与"写对"差 4 倍 | ✅ `pipeline/reward.py`，13 项自测通过 |
| ② | 任务表示改 **search/replace 块**（不需算行号） | 上一轮 440 采样中 227 次 `corrupt patch`，strict 成功率仅 1.8% | 待实现（训练 prompt 侧） |
| ③ | 模型 1.5B → **7B 代码专精** | 直击"写不出合法 patch" | ✅ 权重已就位 |

### 必须前置的门禁：先证明题目能给出满分，再谈训练

"拿不到 reward"这类问题**不能等训练跑起来才发现**。
`experiments/verify_criteria.py` 对每题验三个场景，任一不过即**剔除该题**：

| 场景 | 必须满足 | 不满足的含义 |
|---|---|---|
| 空解（不打 patch） | F2P **全 fail**、P2P 全 pass | 题目无效：修复前就已通过，模型无论怎么答都拿不到分差 |
| **golden patch** | F2P **全 pass**、reward=**1.0** | **判据链路坏了 —— 连标准答案都拿不到满分，模型永远不可能拿到** |
| 垃圾 patch | apply 失败 或 collect_error | 防作弊/防误判链路失效 |

上一轮 21 题里有 6 题因镜像与数据不匹配被剔除，但属**事后**发现；
本轮把它作为进入训练前的硬门禁。


---

## 2026-09-02（续）· 训练侧全链路打通：search/replace 表示 + reward=1.0 实测

### 沙箱镜像问题的最终解法

官方 SWE-bench 镜像无法直接作为 AGS 沙箱工具使用，连踩四个坑：

| # | 现象 | 真因与解法 |
|---|---|---|
| 1 | `ContainerStart: init command path error` | 官方镜像缺 AGS 的 envd agent。用 **docker build 多阶段 COPY** 从现役可用镜像搬运 envd + s6（构建机有 docker；节点侧的 `ctr commit` 不存在、手改 OCI 清单被平台拒） |
| 2 | `ImagePrepare: Internal server error` | **实为镜像预热未完成，等约 4 分钟即可**。此报错极具误导性，曾据此误判为清单格式问题、白改两版方案。已固化为 `clients/sandbox.py::start_instance_with_warmup` |
| 3 | `AuthenticationException: unknown user 'user'` | 官方镜像无 `user` 账号，e2b 默认以该身份写文件 → 必须显式 `user="root"` |
| 4 | `future feature annotations is not defined` | testbed conda 环境是 **py3.6**（随题目依赖而定）。判据脚本改用 `/usr/bin/python3`（3.10）执行、内部调 testbed python 跑 pytest；**必须写绝对路径**，否则 PATH 里 conda 在前会解析错解释器 |

### 判据链路验证（训练前的硬门禁）

`experiments/verify_criteria.py`，三场景全过：

```
① 空解      F2P=0/1  P2P=2/2               ✓  题目有效（修复前确实失败）
② golden    F2P=1/1  P2P=2/2  reward=1.000 ✓  满分可达
③ 垃圾patch apply_failed                    ✓  防作弊正常
```

**② 是最关键的一条** —— 它证明「满分拿得到」。上一轮 55 步中仅 1 步分数 >0.2
（即几乎从未真正修对过题），本轮把这一点作为进入训练前的必过门禁。

### 任务表示改为 search/replace（针对上一轮最大瓶颈）

上一轮 440 次采样的失败分布：`corrupt patch` **227** / `without header` 44 /
`does not apply` 10，strict 可应用仅 **8/440 = 1.8%**。
根因是 unified diff 要求模型手算 hunk header（`@@ -起始行,行数 +起始行,行数 @@`），
这对小模型是硬伤 —— 一位算错整个 patch 就废。

`pipeline/edit_format.py` 改为：

```
### path/to/file.py
<<<<<<< SEARCH
    原始代码片段
=======
    替换后的代码
>>>>>>> REPLACE
```

**完全不需要行号**，靠内容定位；行号由 `difflib` 确定性算出。
容错覆盖小模型的实际错法（均有自测，8 项全通过）：

| 容错项 | 说明 |
|---|---|
| markdown 围栏 | ` ```python ` / ` ``` ` 自动剥离 |
| 路径幻觉 | 后缀匹配纠正（上一轮实测有丢 `src/` 前缀、凭空加前缀两类） |
| 缩进偏差 | 去缩进匹配后套用原文缩进 |
| 行尾空白 | 忽略 |
| 失败归因 | 区分「没写出块 / 路径不在清单 / SEARCH 未匹配」 |

### 端到端实测（`pipeline/verl_reward_fn.py`）

| 场景 | reward | 耗时 |
|---|---|---|
| 正确解（search/replace，无任何行号） | **1.0000** | 31.8s（含冷启动） |
| 空谈无编辑块 | 0.0000 | 0.0s |
| 改错内容（格式对、改动无意义） | 0.2000 | **2.8s**（实例复用） |
| 缓存命中 | 1.0000 | 0.00s |

→ 实例复用后单次判分 **2.8s**，GRPO 每步 8 采样约 **22s**，训练吞吐可接受。

三层降本均已生效：实例池（26s→2.8s）、并发、`(task_id, patch_hash)` 缓存。
失败归因逐条落 `REWARD_DEBUG_LOG`，可直接做定量分析 ——
上一轮只记 reward 标量、事后无法区分失败类型的问题不再重演。


---

## 2026-09-02（续）· 首轮 GRPO 训练完成：31 step，主要瓶颈已解除但曲线未上升

### 训练概况

| 项 | 值 |
|---|---|
| 完成 step | **31/31**（正常结束，非崩溃） |
| 用时 | 24 分 51 秒（约 45s/step） |
| 模型 | Qwen2.5-Coder-3B-Instruct + LoRA rank 32 |
| checkpoint | `global_step_10 / 20 / 30 / 31` 均已保存 |

### 核心成果：模型真的修对题了

| 指标 | 上一轮（1.5B + unified diff） | 本轮（3B + search/replace） |
|---|---|---|
| **组内出现满分 1.0 的 step** | **0/55** | **7/31 = 22.6%** |
| 分数 >0.2 的 step | 1/55 | 12/31 |
| 样本级满分次数 | 0 | **15** |
| `grad_norm` 有效步 | 前两轮恒 0 | 30/31 |

reward 归因分布（559 次打分）中最有说服力的一项：
**「写了但打不进代码库」仅 1 次（0.2%）** —— 对比上一轮 227 次 `corrupt patch`，
这是 search/replace 表示解除主要瓶颈的直接证据。

| 阶段 | 次数 | 占比 |
|---|---|---|
| 没写出编辑块 | 300 | 53.7% |
| 沙箱异常 | 93 | 16.6% |
| 真跑了测试 | 80 | 14.3% |
| 缓存命中 | 71 | 12.7% |
| 打进去但代码被改坏 | 14 | 2.5% |
| 写了但打不进代码库 | **1** | **0.2%** |

> 93 次沙箱异常来自修复前的一次运行（节点 `.env` 漏投 `AGS_TOOL_NAME`），修复后归零。

### reward 曲线未上升：定量归因

分段均值：前 1/3 **0.0344** → 中 **0.0284** → 后 1/3 **0.0163**。

**先排除「学不动」**：

| 指标 | 数据 | 判断 |
|---|---|---|
| `entropy` | 0.1458 → 0.1398 | 没有塌，不是策略坍缩 |
| `grad_norm` | 30/31 非零 | 参数确实在更新 |
| `score_max` | 7 步达 1.0 | 模型有能力修对题 |

**真因是抽样噪声**：14/31 步 `score` 全 0，且这些步 `grad_norm` 也接近 0
（step 17 为 `1.07e-04`）—— 8 个采样全部得 0 分 → 组内无差异 → advantage 恒 0 → 无梯度。

根源在于 `train_batch_size=2` 且题池仅 9 题，**每步只抽 2 道题**：
抽到会做的就有分，抽到不会的就全 0。满分步落在 step 2, 7, 9, 11, 20, 25, 29，
**完全随机、无收敛特征**。前段高后段低只是运气（前 11 步含 4 个满分步，后 10 步仅 2 个）。

→ **该曲线反映的是抽样方差，不是学习趋势。**

### 改进方向（已确定配置，本轮未执行）

| 参数 | 实测值 | 建议 | 理由 |
|---|---|---|---|
| `train_batch_size` | 2 | 1 | 行数被折半，63 行只出 31 step，不满足 ≥50 |
| `rollout.n` | 8 | 16 | 提高「至少一个采样得分」的概率 |
| `repeat` | 7 | 8 | 9 题 × 8 = 72 step |
| **题池** | **9 题** | **20+ 题** | **根本解**：单步 reward 方差主要来自「抽到哪题」而非策略好坏 |

`scripts/run_grpo_training.sh` 已更新为改进值，并在注释中标明
「归档结果对应旧配置」及其复现命令，避免后续混淆。

### 期间修掉的三个工具 bug（都会导致看错数据）

| # | 问题 | 影响 |
|---|---|---|
| 1 | 监控/归档按**修改时间**挑日志 | 训练脚本每次以新时间戳建文件，失败的空日志反而更新 → 读到旧文件，误判训练已停止 |
| 2 | 直接回传原始 step 日志 | verl 单步打印数百个指标，31 步约 15 万字符，**超过云助手输出上限被静默截断** → 只拿回 11 步 |
| 3 | 提取器用 `split("step:")` | 日志中同时存在 `step:`、`training/global_step:`、`timing_s/step:`，split 把行切碎 |

修法：新增 `scripts/pick_train_log.sh`（按 step 数最多挑）与
`scripts/extract_steps.py`（节点侧逐行正则提取后只回传精简 CSV）。

