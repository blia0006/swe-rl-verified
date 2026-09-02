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
