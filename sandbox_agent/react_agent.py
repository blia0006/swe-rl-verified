#!/usr/bin/env python3
"""沙箱内的多轮 ReAct Agent —— tracing 采集的执行体
====================================================

课题验收第 2 条要求：「Agent 每步操作（读文件/编辑/执行命令/跑测试）+ 观察结果
+ 最终奖励」的结构化 tracing。本脚本是产出这份 tracing 的执行体。

## 为什么必须在沙箱内跑（而非本机驱动）

课题架构写明「Agent 在沙箱中执行修复操作」。若把主循环放在本机、只把单条命令
丢进沙箱，则 Agent 的"身体"在沙箱、"决策"在本机，与架构不符，且每步都要
一次跨公网往返（实测单次 e2b 调用 0.3~1s，30 步就是半分钟纯网络开销）。

放在沙箱内后：主循环、工具执行、tracing 落盘全在容器内，只有 LLM 推理经
**VPC 内网**打到 GPU 节点（`http://10.0.0.11:8000`，走 `--net-host` 的 vLLM）。

## 零依赖约束

官方 SWE-bench 镜像的 conda 环境五花八门（有的 py3.6），且**不能 pip install**
——装包会改变题目依赖，污染判据。因此本脚本：

- 只用**标准库**（`urllib.request` 发 HTTP，不用 requests/openai）
- 由**系统 `/usr/bin/python3`**（3.10）执行，不碰 conda 环境
- 工具实现全部用 `subprocess` 调系统命令（git/grep/sed），不依赖 Python 包

## 工具设计

| 工具 | 作用 | 为什么需要 |
|---|---|---|
| `list_dir` | 列目录 | 定位代码结构 |
| `read_file` | 读文件片段（带行号） | 编辑前必须看到真实内容，否则 search/replace 必然失配 |
| `search` | grep 全仓搜索 | 从 issue 描述定位到具体文件 |
| `edit` | search/replace 编辑 | **不用 unified diff**：模型写 diff 极易算错行号/上下文（上一轮 227 次 corrupt patch），search/replace 只需精确复制一段原文 |
| `run_tests` | 跑 F2P 测试 | 让 Agent 拿到真实执行反馈 —— 这正是 RL 需要的可验证信号 |
| `finish` | 结束 | 显式收尾，便于区分"主动交卷"与"步数耗尽" |

## tracing 格式

`/task/tracing.jsonl` 每行一步：

```json
{"step": 3, "thought": "...", "tool": "edit", "args": {...},
 "observation": "...", "ok": true, "elapsed_s": 1.2, "tokens": {...}}
```

末行是汇总记录（`"type": "summary"`），含最终 patch 与各步统计。
reward 不在此计算 —— 判分由训练侧的 `pipeline/sandbox_eval` 统一负责，
避免两处判据不一致（上一轮的主要排查成本来源）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "/testbed"
TASK_DIR = "/task"
TRACING = f"{TASK_DIR}/tracing.jsonl"
PATCH_OUT = f"{TASK_DIR}/agent_patch.diff"

MAX_OBS_CHARS = 4000      # 单次观察回传上限，防止长文件把上下文吃光
MAX_READ_LINES = 200      # read_file 单次最多返回行数
MAX_HISTORY_CHARS = 24000 # 历史对话裁剪阈值


# ----------------------------------------------------------------- 工具实现

def sh(cmd: str, cwd: str = REPO, timeout: int = 120) -> tuple[int, str]:
    """跑一条 shell 命令。stderr 合并进 stdout —— 报错信息对 Agent 同样有用。"""
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=cwd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, f"[超时] 命令超过 {timeout}s 未结束"


def norm_path(path: str) -> str:
    """把模型给的路径规范成相对 /testbed 的路径。

    ⚠️ 这是实测炸过的地方，务必看完再改。

    模型会**混用**两种写法，两者都必须支持：
      · 相对路径：`pylint/config.py`
      · 绝对路径：`/testbed/pylint/config.py` 或干脆 `/testbed`

    最初的实现是简单 `path.lstrip("/")`，把 `/testbed` 变成 `testbed`，
    而命令的 cwd 本来就是 `/testbed`，于是变成去找 `/testbed/testbed` ——
    **所有工具静默返回"无匹配"/"文件不存在"，Agent 彻底瞎掉**，
    20 步全部耗在乱猜路径上，edits=0。

    更坏的是它**只在部分 rollout 暴露**：单题实测时模型碰巧写了相对路径，
    一次就解对了；批量采集时多数 rollout 写绝对路径，全军覆没。
    所以"单测通过"完全不能说明这里没问题。

    另外顺手挡掉 `..` 逃逸：Agent 没有理由访问 /testbed 之外的文件。
    """
    p = (path or "").strip()
    if not p:
        return "."
    # 用模块级 REPO 的**当前值**而非闭包捕获：便于测试替换工作目录
    repo = globals().get("REPO", "/testbed")
    if p == repo or p.startswith(repo.rstrip("/") + "/"):
        p = p[len(repo):]
    p = p.lstrip("/")
    p = os.path.normpath(p) if p else "."
    if p.startswith("..") or p == "":
        return "."
    return p


def t_list_dir(path: str = ".", **_) -> tuple[bool, str]:
    safe = norm_path(path)
    code, out = sh(f"ls -la --time-style=+ {safe!r} 2>&1 | head -60")
    return code == 0, out


def t_read_file(path: str = "", start: int = 1, end: int = 0, **_) -> tuple[bool, str]:
    """读文件片段。

    ## 为什么**不带**行号（一次实测教训）

    最初实现返回 `    93\tpivot = datetime.datetime(` 这种带行号的格式，理由是
    "Agent 需要行号来定位"。实测结果是灾难性的：模型把带行号的文本**原样抄进**
    `edit` 的 `search` 参数，于是永远匹配不到原文，陷入

        read_file → edit(失败) → read_file → edit(失败) → …

    的死循环，20 步全部烧完、edits=0。django 一题 4 条 rollout 全军覆没。

    根因是工具契约自相矛盾：`edit` 要求 search 与原文**逐字符一致**，而
    `read_file` 却返回了加过料的文本。修法是让 read_file 返回**纯原文**，
    行号信息改放在头部说明里（`【第 80-120 行】`），模型可以知道位置，
    但复制正文时不会带上任何多余字符。
    """
    if not path:
        return False, "缺少 path 参数"
    safe = norm_path(path)
    start = max(1, int(start or 1))
    end = int(end or 0) or (start + MAX_READ_LINES - 1)
    if end - start + 1 > MAX_READ_LINES:
        end = start + MAX_READ_LINES - 1
    code, out = sh(f"sed -n '{start},{end}p' {safe!r}")
    if code != 0 or not out.strip():
        _, out2 = sh(f"test -f {safe!r} && echo EXISTS || echo MISSING")
        if "MISSING" in out2:
            # 给出可操作的提示，而不是让 Agent 反复猜路径
            base = os.path.basename(safe)
            _, hint = sh(
                f"find . -name {base!r} -not -path './.git/*' 2>/dev/null | head -5"
            )
            extra = f"\n同名文件可能在：\n{hint.strip()}" if hint.strip() else ""
            return False, f"文件不存在：{path}（工作目录为 {REPO}，请用相对路径）{extra}"
        return True, f"（{path} 第 {start}-{end} 行为空）"
    body = out.rstrip("\n")
    header = (
        f"【{safe} 第 {start}-{min(end, start + body.count(chr(10)))} 行，"
        f"以下为原文，edit 时请逐字符复制】\n"
    )
    return True, (header + body)[:MAX_OBS_CHARS]


def t_search(pattern: str = "", path: str = ".", **_) -> tuple[bool, str]:
    if not pattern:
        return False, "缺少 pattern 参数"
    safe = norm_path(path)
    # -F 关掉正则：模型给的多是字面量（含括号/点号），当正则用常常报错或漏匹配
    code, out = sh(
        f"grep -rnF --include='*.py' -- {pattern!r} {safe!r} 2>/dev/null | head -40"
    )
    if not out.strip():
        code, out = sh(
            f"grep -rn --include='*.py' -E -- {pattern!r} {safe!r} 2>/dev/null | head -40"
        )
    if not out.strip():
        # 无匹配时给出可操作反馈：否则 Agent 只会换个词再搜，白烧步数
        return True, (
            f"（在 {safe} 下未找到 {pattern!r}）\n"
            f"提示：工作目录是 {REPO}，可先用 list_dir 看顶层结构，"
            f"或用更短的关键词/函数名重搜。"
        )
    return True, out[:MAX_OBS_CHARS]


def strip_line_numbers(text: str) -> str:
    """剥掉模型可能误带的行号前缀。

    即使 `read_file` 已改为返回纯原文，模型仍可能从别处（如 pytest 报错栈、
    自己上一轮的输出）复制到带行号的文本。只在**所有非空行都带前缀**时才剥离，
    避免误伤正文里本就以数字开头的代码（如 `1024 * 1024`）。
    """
    lines = text.split("\n")
    pats = (r"^\s*\d+\t", r"^\s*\d+\s\|\s?", r"^\s*\d+:\s?")
    for pat in pats:
        non_empty = [l for l in lines if l.strip()]
        if non_empty and all(re.match(pat, l) for l in non_empty):
            return "\n".join(re.sub(pat, "", l) if l.strip() else l for l in lines)
    return text


def find_fuzzy_span(src: str, search: str) -> tuple[int, int] | None:
    """在 src 里按「忽略空白差异」找 search 对应的原文区间。

    ## 为什么需要这一层（实测根因）

    模型常把源码里**跨多行**的结构压成一行写进 `search`，例如真实源码是

        def __init__(
            self,
            name: str,
            import_name: str,
        ):

    它却写成 `def __init__(self, name: str, import_name: str):`。语义完全正确，
    但字符串匹配必然 0 处命中。实测 flask 一题因此连续 6 步 `read_file → edit`
    死循环、edits=0，即使已经给出"请复制原文"的警告仍然无效 —— 7B 模型难以
    严格遵守逐字符复制的要求。

    与其把这类样本判为失败（给模型错误信号：以为思路错了，其实只是格式），
    不如在工具层吸收：把两边的连续空白都归一化成单个空格再匹配，命中后**返回
    原文中的真实区间**，替换时按原文边界切割，从而不破坏文件其余部分。

    返回 (start, end) 为 src 上的字符下标；找不到或有歧义则返回 None。
    """
    norm_re = re.compile(r"\s+")
    needle = norm_re.sub(" ", search).strip()
    if not needle:
        return None

    # 建立「归一化后位置 → 原文位置」映射，才能把匹配结果还原回原文区间
    norm_chars: list[str] = []
    pos_map: list[int] = []
    prev_space = True  # 行首视为已有空白，避免开头产生多余空格
    for i, ch in enumerate(src):
        if ch.isspace():
            if not prev_space:
                norm_chars.append(" ")
                pos_map.append(i)
            prev_space = True
        else:
            norm_chars.append(ch)
            pos_map.append(i)
            prev_space = False
    norm_src = "".join(norm_chars)

    first = norm_src.find(needle)
    if first < 0:
        return None
    if norm_src.find(needle, first + 1) >= 0:
        return None  # 多处命中，交给上层报"请扩大范围"

    start = pos_map[first]
    last_idx = first + len(needle) - 1
    end = pos_map[last_idx] + 1
    return start, end


def t_edit(path: str = "", search: str = "", replace: str = "", **_) -> tuple[bool, str]:
    """search/replace 编辑：把 `search` 段**整段**替换为 `replace`。

    为什么不用 unified diff：模型生成 diff 需要同时算对行号、上下文行数、
    `@@` 头，上一轮实测 227 次「写了但打不进代码库」几乎全是这个原因。
    search/replace 只要求精确复制一段原文，容错率高一个量级。

    ## 三级容错（每级都对应一类实测失败）

    1. **精确匹配** —— 理想情况
    2. **剥离行号前缀** —— 模型从报错栈/旧输出里复制到带行号的文本
    3. **忽略空白差异** —— 模型把多行结构压成一行（见 find_fuzzy_span）

    失败时**必须给出可操作的反馈**（找到几处 + 首行位置 + 原文上下文），
    否则 Agent 只知道失败、不知道怎么改，会在同一处反复试错直到步数耗尽。
    """
    if not path or not search:
        return False, "缺少 path 或 search 参数"
    safe = norm_path(path)
    full = os.path.join(REPO, safe)
    if not os.path.isfile(full):
        base = os.path.basename(safe)
        _, hint = sh(f"find . -name {base!r} -not -path './.git/*' 2>/dev/null | head -5")
        extra = f"\n同名文件可能在：\n{hint.strip()}" if hint.strip() else ""
        return False, f"文件不存在：{path}（工作目录为 {REPO}，请用相对路径）{extra}"
    try:
        src = open(full, encoding="utf-8", errors="replace").read()
    except Exception as e:  # noqa: BLE001
        return False, f"读取失败：{e}"

    def write(new_src: str) -> bool:
        try:
            open(full, "w", encoding="utf-8").write(new_src)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- 第 1、2 级：精确 / 剥离行号 ----
    candidates: list[tuple[str, bool]] = [(search, False)]
    stripped = strip_line_numbers(search)
    if stripped != search:
        candidates.append((stripped, True))

    for cand, was_stripped in candidates:
        n = src.count(cand)
        if n == 1:
            new_replace = strip_line_numbers(replace) if was_stripped else replace
            if not write(src.replace(cand, new_replace, 1)):
                return False, "写入失败"
            note = "（已自动剥离行号前缀）" if was_stripped else ""
            return True, (
                f"已替换 {path} 中 1 处{note}"
                f"（-{len(cand.splitlines())} 行 / +{len(new_replace.splitlines())} 行）"
            )
        if n > 1:
            return False, (
                f"search 内容在 {path} 中找到 {n} 处，无法确定替换目标。"
                f"请扩大 search 范围（多带几行上下文）使其唯一。"
            )

    # ---- 第 3 级：忽略空白差异 ----
    span = find_fuzzy_span(src, stripped)
    if span:
        s, e = span
        if not write(src[:s] + replace + src[e:]):
            return False, "写入失败"
        return True, (
            f"已替换 {path} 中 1 处（**按忽略空白差异匹配** —— 你给的 search 与原文"
            f"换行/缩进不同，已按原文边界替换）"
            f"（-{len(src[s:e].splitlines())} 行 / +{len(replace.splitlines())} 行）"
        )

    # ---- 0 处匹配：给出最有用的定位线索 ----
    first = search.strip().split("\n")[0].strip()[:80]
    src_lines = src.split("\n")
    hits = [i + 1 for i, ln in enumerate(src_lines) if first and first in ln][:3]
    if not hits:
        # 首行也没命中时，退一步用首个"单词"定位（如函数名），仍能给出有效线索
        token = re.split(r"[^\w]+", first)
        token = max(token, key=len) if token else ""
        if len(token) >= 4:
            hits = [i + 1 for i, ln in enumerate(src_lines) if token in ln][:3]
    if hits:
        h = hits[0]
        ctx = "\n".join(
            src_lines[i - 1] for i in range(max(1, h - 2), min(len(src_lines) + 1, h + 8))
        )
        hint = (
            f"\n相关代码在第 {hits} 行附近。该处**原文**如下，请**原样复制**你要"
            f"替换的部分作为 search（注意原文可能跨多行，不要压成一行）：\n{ctx}"
        )
    else:
        hint = "\n未找到相关代码，请先用 search 或 read_file 确认文件真实内容。"
    return False, f"search 内容在 {path} 中找到 0 处。{hint}"


def t_run_tests(spec: dict, **_) -> tuple[bool, str]:
    """跑该题的 F2P 测试，把真实执行反馈交给 Agent。

    这是 RL 数据价值的核心：Agent 能看到自己的修改是否真的让测试变绿。
    只跑 F2P（不跑全量 P2P）—— P2P 常有几十项、耗时数分钟，采集阶段跑不起；
    最终判分由训练侧 `sandbox_eval` 跑完整套。
    """
    script = spec.get("test_script_path") or f"{TASK_DIR}/agent_test.sh"
    if not os.path.isfile(script):
        return False, "题目未提供测试脚本，无法运行测试"
    code, out = sh(f"bash {script} 2>&1 | tail -60", timeout=900)
    verdict = "测试全部通过" if code == 0 else f"测试未全过（退出码 {code}）"
    return True, f"{verdict}\n{out[-MAX_OBS_CHARS:]}"


def t_finish(summary: str = "", **_) -> tuple[bool, str]:
    return True, f"已结束。{summary[:500]}"


TOOLS = {
    "list_dir": t_list_dir,
    "read_file": t_read_file,
    "search": t_search,
    "edit": t_edit,
    "run_tests": t_run_tests,
    "finish": t_finish,
}

TOOL_DOC = """可用工具（每次**只能调用一个**）：

- list_dir(path)                     列目录
- read_file(path, start, end)        读文件片段（返回带行号内容）
- search(pattern, path)              全仓搜索字面量
- edit(path, search, replace)        把 search 整段替换为 replace（search 必须与原文逐字符一致且在文件中唯一）
- run_tests()                        运行本题的测试，拿到真实反馈
- finish(summary)                    完成修复，结束

**路径一律用相对 /testbed 的相对路径**，例如 `pylint/checkers/similar.py`，
不要写 `/testbed/...` 开头的绝对路径。省略 path 时默认为仓库根目录。

输出格式（严格遵守，否则无法解析）：

Thought: <一句话说明你要做什么、为什么>
Action: <工具名>
Args: <单行 JSON>

例：
Thought: 先看看 timesince 的实现
Action: read_file
Args: {"path": "django/utils/timesince.py", "start": 80, "end": 120}
"""

SYSTEM_PROMPT = """你是一名资深 Python 工程师，正在修复一个真实开源项目的 bug。

工作目录是 /testbed（该项目的 git 仓库），所有路径都相对它来写。
你的目标是**用最少的改动**让失败的测试通过，同时不破坏其它已通过的测试。

要点：
1. **先定位再改**：用 search / read_file 找到出问题的代码，看清原文后再 edit。
   若 search 没结果，先 list_dir 看清仓库顶层结构，不要凭空猜文件名。
2. **edit 的 search 必须与文件原文逐字符一致**（含缩进），且在文件中唯一。
   不确定就先 read_file 核对。
3. 改完用 run_tests 验证。若没过，读报错继续修。
4. 只改产品代码，**不要修改测试文件** —— 改测试不算修复。
5. 确认通过后调 finish 结束。
"""


# ----------------------------------------------------------------- LLM 调用

def chat(
    base_url: str, model: str, messages: list, *,
    temperature: float, max_tokens: int, timeout: int = 180, api_key: str = "EMPTY",
) -> tuple[str, dict]:
    """调 OpenAI 兼容接口。只用标准库 —— 沙箱内不能装包（见模块 docstring）。"""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["\nObservation:", "\nThought:"],
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    text = payload["choices"][0]["message"]["content"] or ""
    return text, payload.get("usage") or {}


def parse_action(text: str) -> tuple[str, str, dict, str]:
    """从模型输出里解析 (thought, tool, args, error)。

    容错要点（都是实测踩过的）：
    - `Args` 可能被包在 ```json 代码块里
    - 模型可能漏掉 `Args:` 行（无参数工具如 run_tests/finish）
    - JSON 里可能有尾随逗号或单引号
    """
    thought = ""
    m = re.search(r"Thought:\s*(.+?)(?:\n(?:Action|Args):|\Z)", text, re.S)
    if m:
        thought = m.group(1).strip()[:1000]

    m = re.search(r"Action:\s*([A-Za-z_]+)", text)
    if not m:
        return thought, "", {}, "未找到 Action 行"
    tool = m.group(1).strip()
    if tool not in TOOLS:
        return thought, "", {}, f"未知工具 {tool!r}，可用：{', '.join(TOOLS)}"

    m = re.search(r"Args:\s*(.*)", text, re.S)
    raw = (m.group(1).strip() if m else "") or "{}"
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.S).strip()
    if not raw.startswith("{"):
        raw = "{}" if tool in ("run_tests", "finish") else raw
    # 只取第一个完整 JSON 对象：模型常在 Args 之后继续输出别的内容
    depth, endi, instr, esc = 0, -1, False, False
    for i, ch in enumerate(raw):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                endi = i + 1
                break
    if endi > 0:
        raw = raw[:endi]
    try:
        args = json.loads(raw)
    except Exception:
        try:
            args = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
        except Exception as e:  # noqa: BLE001
            return thought, tool, {}, f"Args 不是合法 JSON：{str(e)[:120]}｜原文片段 {raw[:200]!r}"
    if not isinstance(args, dict):
        return thought, tool, {}, "Args 必须是 JSON 对象"
    return thought, tool, args, ""


def trim_history(messages: list) -> list:
    """裁剪历史，保住 system + 首条任务描述 + 最近若干轮。

    为什么必须裁：ReAct 每步都追加"模型输出 + 观察"，读几个大文件后上下文
    就能冲到几万 token，超过模型窗口会直接报错而非降级。保留首条任务描述是
    因为它含 issue 正文 —— 丢了 Agent 就不知道要修什么。
    """
    if sum(len(m["content"]) for m in messages) <= MAX_HISTORY_CHARS:
        return messages
    head, tail = messages[:2], messages[2:]
    while tail and sum(len(m["content"]) for m in head + tail) > MAX_HISTORY_CHARS:
        tail = tail[2:] if len(tail) > 2 else tail[1:]
    if tail and tail[0]["role"] != "assistant":
        note = {"role": "user", "content": "（为控制上下文，较早的探索记录已省略）"}
        return head + [note] + tail
    return head + tail


# ----------------------------------------------------------------- 主循环

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=f"{TASK_DIR}/spec.json")
    ap.add_argument("--base-url", required=True, help="vLLM 的 http://ip:port")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rollout-id", default="0")
    ap.add_argument("--tracing", default=TRACING)
    args = ap.parse_args()

    spec = json.load(open(args.spec, encoding="utf-8"))
    problem = spec.get("problem_statement") or spec.get("issue") or ""
    task_id = spec.get("task_id", "?")

    trace = open(args.tracing, "w", encoding="utf-8")

    def emit(rec: dict) -> None:
        trace.write(json.dumps(rec, ensure_ascii=False) + "\n")
        trace.flush()

    emit({
        "type": "meta", "task_id": task_id, "rollout_id": args.rollout_id,
        "model": args.model, "max_steps": args.max_steps,
        "temperature": args.temperature, "started_at": time.time(),
    })

    user0 = (
        f"# 待修复的问题（仓库 {spec.get('repo', '?')}）\n\n{problem.strip()[:6000]}\n\n"
        f"# 工作目录\n/testbed\n\n{TOOL_DOC}\n\n现在开始，先输出你的第一步。"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user0},
    ]

    t_start = time.time()
    finished = False
    n_edit_ok = 0
    n_parse_err = 0
    tests_run = 0
    tests_passed = False
    # 死循环检测：实测模型会用**完全相同**的参数反复调同一工具（read_file 同一段
    # → edit 同样失败 → 再 read_file 同一段…），20 步全部烧光。
    # 记录 (tool, args) 指纹，重复出现时在 observation 里强制打断，要求换策略。
    seen_calls: dict[str, int] = {}

    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        try:
            text, usage = chat(
                args.base_url, args.model, trim_history(messages),
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            emit({
                "type": "step", "step": step, "error": f"LLM 调用失败：{str(e)[:300]}",
                "elapsed_s": round(time.time() - t0, 2),
            })
            break

        thought, tool, targs, perr = parse_action(text)
        if perr:
            n_parse_err += 1
            obs = f"[格式错误] {perr}\n请严格按 Thought/Action/Args 三行格式输出。"
            ok = False
        else:
            fn = TOOLS[tool]
            try:
                if tool == "run_tests":
                    ok, obs = fn(spec)
                    tests_run += 1
                    tests_passed = ok and "测试全部通过" in obs
                else:
                    ok, obs = fn(**targs)
                    if tool == "edit" and ok:
                        n_edit_ok += 1
            except TypeError as e:
                ok, obs = False, f"参数不匹配：{str(e)[:200]}"
            except Exception as e:  # noqa: BLE001
                ok, obs = False, f"{type(e).__name__}: {str(e)[:200]}"

        obs = obs[:MAX_OBS_CHARS]

        # ---- 死循环打断 ----
        # 同一 (tool, args) 重复出现说明模型在原地打转。此时**必须改变观察内容**，
        # 否则它下一步会得到完全相同的输入、做出完全相同的决策，直到步数耗尽。
        # 只对失败调用计数：重复读同一文件是正常行为（比如改完再确认）。
        if tool and not ok:
            fp = f"{tool}|{json.dumps(targs, sort_keys=True, ensure_ascii=False)[:400]}"
            seen_calls[fp] = seen_calls.get(fp, 0) + 1
            if seen_calls[fp] >= 2:
                obs += (
                    f"\n\n⚠️ 你已用**完全相同的参数**调用 {tool} {seen_calls[fp]} 次并"
                    f"每次都失败。重复同样的操作不会有不同结果，请改变策略："
                    f"\n· 若是 edit 失败：先 read_file 读取要改的确切区间，"
                    f"把返回的原文**原样**作为 search（不要自己重写或凭记忆构造）"
                    f"\n· 若是路径找不到：用 list_dir 或 search 先确认真实路径"
                )
        emit({
            "type": "step", "step": step, "thought": thought, "tool": tool,
            "args": targs, "ok": ok, "observation": obs,
            "raw_output": text[:2000], "usage": usage,
            "elapsed_s": round(time.time() - t0, 2),
        })

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

        if tool == "finish" and not perr:
            finished = True
            break

    # 最终 patch：只取产品代码改动。`git diff` 天然不含 untracked 文件，
    # 而模型偶尔会新建文件，因此显式 add -N 让新增文件也进 diff。
    sh("git add -N . >/dev/null 2>&1")
    _, patch = sh("git diff", timeout=120)
    open(PATCH_OUT, "w", encoding="utf-8").write(patch)

    emit({
        "type": "summary", "task_id": task_id, "rollout_id": args.rollout_id,
        "steps_used": step, "finished_by_agent": finished,
        "edits_applied": n_edit_ok, "parse_errors": n_parse_err,
        "tests_run": tests_run, "tests_passed_in_agent": tests_passed,
        "patch_chars": len(patch), "patch_empty": not patch.strip(),
        "wall_s": round(time.time() - t_start, 1),
        "patch": patch[:20000],
    })
    trace.close()
    print(f"[agent] task={task_id} rollout={args.rollout_id} steps={step} "
          f"edits={n_edit_ok} patch_chars={len(patch)} finished={finished}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
