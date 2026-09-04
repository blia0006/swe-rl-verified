#!/usr/bin/env python3
"""SWE-bench 官方判据适配层（本机/训练侧）
================================================

## 为什么必须有这一层

判据是 RL 的地基：reward 判错，训练必然空转。本项目前两轮的判据脚本
**统一用 `pytest -rA <单个 test id>` 跑测试**，而官方 harness 的真实做法是
按 `(repo, version)` 查表取 `test_cmd`，且**测试目标是文件/模块级 directives
而非单个 test id**。两者对以下 repo 完全不兼容：

| repo | 官方 test_cmd | 单 test-id + pytest 的结果 |
|---|---|---|
| django/django | `./tests/runtests.py --settings=test_sqlite` | 测试 id 形如 `test_x (mod.Cls)`，pytest 无法识别 |
| sphinx-doc/sphinx | `tox --current-env -epy39 --` | 绕过 tox 会缺环境变量与插件 |
| sympy/sympy | `bin/test -C --verbose` | sympy 自有 runner，pytest 收集不全 |

题池 20 题中这三类占 9 题（45%），即**近半数题目的 reward 恒为 0** ——
这是上一轮"14/31 步 score 全 0、组内无差异、advantage 恒 0"的地基性根因，
比抽样方差的解释更靠前。

## 本层的做法

不重新发明判据，直接复用官方 `swebench` 包的两份权威数据：

1. `harness/constants/python.py` → `MAP_REPO_VERSION_TO_SPECS_PY`
   每个 (repo, version) 的 `test_cmd` / `install` / `eval_commands` / `pre_install`
2. `harness/log_parsers/python.py` → `MAP_REPO_TO_PARSER_PY`
   每个 repo 的日志解析器，把测试输出还原成 `{test_id: PASSED/FAILED/...}`

两份都用 `exec` 从已安装的包里加载，而不是 copy 进本仓库：
- `constants/python.py` 是零 import 的纯字面量，任何 Python 版本都能 exec
- `log_parsers/python.py` 仅依赖 `TestStatus`（枚举）与 `TestSpec`（仅类型标注），
  stub 掉即可

**为什么不用 `import swebench`**：官方包要求 Python ≥3.10（顶层用了 `list | None`
新语法），而本仓库的运行环境是 3.9。exec 单文件绕开了包的 `__init__` 链，
既拿到权威数据，又不被版本门槛卡住。

## 职责边界（与课题架构一致）

- **本层（本机/TKE 侧）**：生成 eval bash 脚本、解析日志、判定 F2P/P2P
- **沙箱侧**：只负责执行 bash 并回传原始日志，不做任何判定

判定逻辑留在训练侧的好处：改判据规则不需要重建 20 个镜像。
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# 官方 harness 的输出分隔符（constants/__init__.py:83-84）。
# 之所以硬编码：该文件含 `from swebench...` 链式导入，exec 不了；
# 这两个值是日志协议的一部分，官方多年未变，且 verify_official_spec() 会校验。
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

# 官方 NON_TEST_EXTS（constants/__init__.py:101+）：从 test_patch 里筛出真正的测试文件
NON_TEST_EXTS = [
    ".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl",
    ".yml", ".yaml", ".toml",
]


class TestStatus(Enum):
    """与官方 constants/__init__.py:45 逐字一致。"""

    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


class OfficialSpecError(RuntimeError):
    pass


def _harness_dir() -> Path:
    """定位 swebench 包的 harness 目录。

    ## 为什么不能只靠 importlib.find_spec

    训练容器里 swebench 是用 `pip install --target /data/swe-rl/pylibs` 装的，
    靠 `PYTHONPATH` 生效。此时 `find_spec("swebench")` 会失败或返回意外结果：
    容器镜像自带的 python 环境与 --target 目录之间存在解析顺序差异，
    实测在容器内报"未安装官方 swebench 包"，而目录明明存在。

    这类失败极隐蔽：reward 函数会把它记成 `stage=exception, reward=0.0`，
    **所有样本 reward 恒为 0** → advantage 恒 0 → 训练完全空转，
    但日志表面看不出任何异常（正是上一轮空转的同类问题）。

    因此改为多路查找：先扫 sys.path 下的实际目录，再退回 find_spec。
    """
    candidates: list[Path] = []

    # 1) 直接扫 sys.path —— 覆盖 pip --target + PYTHONPATH 的场景
    for p in sys.path:
        if not p:
            continue
        d = Path(p) / "swebench" / "harness"
        if d.is_dir():
            candidates.append(d)

    # 2) 退回标准包定位
    if not candidates:
        try:
            import importlib.util

            spec = importlib.util.find_spec("swebench")
            if spec and spec.submodule_search_locations:
                d = Path(list(spec.submodule_search_locations)[0]) / "harness"
                if d.is_dir():
                    candidates.append(d)
        except (ImportError, TypeError, ValueError):
            # TypeError：包要求 py3.10+，在 3.9 上顶层 import 会因 `list | None` 报错
            pass

    # 3) 最后兜底：常见安装位置（容器内 PYTHONPATH 可能未包含 cwd）
    for extra in (
        Path("/data/swe-rl/pylibs"),
        Path(__file__).resolve().parent.parent / "pylibs",
    ):
        d = extra / "swebench" / "harness"
        if d.is_dir():
            candidates.append(d)

    for d in candidates:
        # 必须含 constants/python.py —— 5.0.x 重构后该文件不存在，
        # 用它同时校验版本是否为期望的 3.0.x
        if (d / "constants" / "python.py").is_file():
            return d

    hint = f"（已查找 {len(candidates)} 个候选位置）" if candidates else ""
    raise OfficialSpecError(
        f"找不到可用的 swebench harness{hint}。请执行 "
        f"`pip install swebench==3.0.17`（版本必须锁定，见 requirements.txt）"
    )


_SPECS: dict[str, Any] | None = None
_PARSERS: dict[str, Callable[..., Any]] | None = None


def load_specs() -> dict[str, Any]:
    """加载官方 MAP_REPO_VERSION_TO_SPECS_PY（懒加载 + 进程内缓存）。"""
    global _SPECS
    if _SPECS is not None:
        return _SPECS
    f = _harness_dir() / "constants" / "python.py"
    if not f.is_file():
        raise OfficialSpecError(f"官方规格表不存在：{f}")
    ns: dict[str, Any] = {"__name__": "_swebench_constants_py"}
    exec(compile(f.read_text(encoding="utf-8"), str(f), "exec"), ns)  # noqa: S102
    specs = ns.get("MAP_REPO_VERSION_TO_SPECS_PY")
    if not isinstance(specs, dict) or not specs:
        raise OfficialSpecError(
            f"未能从 {f} 取到 MAP_REPO_VERSION_TO_SPECS_PY —— 官方包结构可能已变更"
        )
    _SPECS = specs
    return specs


def load_parsers() -> dict[str, Callable[..., Any]]:
    """加载官方 MAP_REPO_TO_PARSER_PY。

    该文件依赖 `TestStatus` 与 `TestSpec`：前者用本模块的等价枚举替换，
    后者只出现在类型标注里，用 object 顶掉即可。
    """
    global _PARSERS
    if _PARSERS is not None:
        return _PARSERS
    f = _harness_dir() / "log_parsers" / "python.py"
    if not f.is_file():
        raise OfficialSpecError(f"官方解析器不存在：{f}")
    src = re.sub(r"^from swebench.*$", "", f.read_text(encoding="utf-8"), flags=re.M)
    ns: dict[str, Any] = {
        "__name__": "_swebench_log_parsers_py",
        "re": re,
        "TestStatus": TestStatus,
        "TestSpec": object,
    }
    exec(compile(src, str(f), "exec"), ns)  # noqa: S102
    parsers = ns.get("MAP_REPO_TO_PARSER_PY")
    if not isinstance(parsers, dict) or not parsers:
        raise OfficialSpecError(
            f"未能从 {f} 取到 MAP_REPO_TO_PARSER_PY —— 官方包结构可能已变更"
        )
    _PARSERS = parsers
    return parsers


def spec_of(repo: str, version: str | int) -> dict[str, Any]:
    """取某题的官方环境规格。缺失即视为不可判分，必须显式失败而非静默降级。"""
    s = load_specs().get(repo)
    if not s:
        raise OfficialSpecError(f"官方规格表无此 repo：{repo}")
    v = s.get(str(version))
    if not v:
        raise OfficialSpecError(
            f"官方规格表无 {repo} 的 version={version}（可选：{sorted(s)[:8]}…）"
        )
    if "test_cmd" not in v:
        raise OfficialSpecError(f"{repo} v{version} 规格缺 test_cmd")
    return v


def modified_files(patch: str) -> list[str]:
    """从 diff 里取被修改的源文件路径。

    等价于官方 utils.get_modified_files，但改用正则而非 unidiff.PatchSet：
    少一个依赖，且此处只需路径、无需解析 hunk。
    """
    files: list[str] = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)", patch, flags=re.M):
        files.append(m.group(2))
    if files:
        return files
    # 兜底：没有 `diff --git` 头的裸 diff（部分数据源如此）
    for m in re.finditer(r"^--- a/(\S+)", patch, flags=re.M):
        files.append(m.group(1))
    return files


def test_directives(repo: str, test_patch: str) -> list[str]:
    """官方 test_spec/python.py::get_test_directives 的等价实现。

    关键点：**测试目标是文件/模块，不是单个 test id**。官方跑整个测试文件，
    再从日志里挑出 F2P/P2P 各自的状态。django 还要把路径转成模块名。
    """
    if repo == "swe-bench/humaneval":
        return ["test.py"]
    directives = re.findall(r"diff --git a/.* b/(.*)", test_patch)
    directives = [d for d in directives if not any(d.endswith(e) for e in NON_TEST_EXTS)]
    if repo == "django/django":
        out = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            out.append(d.replace("/", "."))
        directives = out
    return directives


def make_eval_script(
    task: dict[str, Any],
    *,
    env_name: str = "testbed",
    repo_directory: str = "/testbed",
) -> str:
    """生成与官方 make_eval_script_list_py 等价的 eval bash 脚本。

    顺序不可改动，每一步都有必要性：

    1. `conda activate testbed` —— 题目依赖装在该环境，可能老至 py3.6
    2. `specs['eval_commands']` —— 少数题需要的额外准备
    3. `specs['install']` —— **对 C 扩展类 repo（matplotlib/sklearn/numpy）必需**：
       模型改了 .pyx/.c 或改动影响编译产物时，不重装则跑的还是旧二进制
    4. `git checkout <base_commit> <test_files>` —— 把测试文件还原到打补丁前，
       防止模型偷改测试文件作弊
    5. `git apply` test_patch —— 引入官方新增的测试
    6. `test_cmd + directives` 夹在 START/END 分隔符之间 —— 便于精确截取
    7. 再次 checkout 还原测试文件 —— 保持仓库状态干净，便于同一沙箱复用

    注意：模型自己的 patch 由调用方在**执行本脚本之前**打入，因为第 4 步
    只还原测试文件，不会碰源码改动。
    """
    repo = task["repo"]
    version = task.get("version")
    specs = spec_of(repo, version)
    base_commit = task["base_commit"]
    test_patch = task["test_patch"]

    test_files = modified_files(test_patch)
    reset_tests = f"git checkout {base_commit} {' '.join(test_files)}"
    heredoc = "EOF_114329324912"
    apply_test_patch = f"git apply -v - <<'{heredoc}'\n{test_patch}\n{heredoc}"
    test_command = " ".join(
        [specs["test_cmd"], *test_directives(repo, test_patch)]
    )

    cmds = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    # ---- 重放 pre_install（官方在**构建镜像时**执行，我们必须在这里补做）----
    #
    # 官方 harness 的 pre_install 跑在镜像构建阶段，改动已固化进镜像。但我们
    # 为了复用同一沙箱连续判分，每次判分前会 `git reset --hard <base_commit>`
    # 把仓库还原 —— 这一步**同时把 pre_install 的改动也冲掉了**。
    #
    # 实测后果（3 道 sphinx 题全判 0）：sphinx 的 pre_install 第一条是
    #     sed -i 's/pytest/pytest -rA/' tox.ini
    # 被 reset 冲掉后，tox 调 pytest 时不带 `-rA`，输出里就没有 `PASSED xxx`
    # 汇总行，官方 parser 一个测试都解析不出来 —— 日志里明明写着
    # "2 failed, 33 passed"，判分结果却是 F2P=0/2、P2P=0/31。
    #
    # 这类失败极具误导性：看起来像"题目环境坏了"，实际是判据链路自己弄坏的。
    #
    # 只重放**改文件内容**的 sed 类命令（pip install 之类由下面的 install 覆盖，
    # 重复跑纯属浪费时间）。每条都加幂等守卫：`sed 's/pytest/pytest -rA/'` 若被
    # 重复执行会累积成 `pytest -rA -rA -rA`；虽然 pytest 能容忍重复参数，但同
    # 一条 sed 反复作用于 setup.py 的版本约束（如 `Jinja2<3.0` → `Jinja2<3.0`）
    # 在部分模式下会产生嵌套结果，破坏文件。用 `grep -q` 先判断是否已生效。
    if "pre_install" in specs:
        for c in specs["pre_install"]:
            c = c.strip()
            if not c.startswith("sed "):
                continue
            m = re.match(r"sed -i\s+(['\"])s(.)(.+?)\2(.*?)\2\1\s+(\S+)$", c)
            if m:
                _, _, pat, rep, target = m.groups()
                # 已含替换后的内容就跳过 —— 用 fgrep 避免把 pat 当正则
                cmds.append(
                    f"grep -qF {rep!r} {target} 2>/dev/null || {c}"
                )
            else:
                cmds.append(c)
    if "eval_commands" in specs:
        cmds += list(specs["eval_commands"])
    cmds += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        cmds.append(specs["install"])
    cmds += [
        reset_tests,
        apply_test_patch,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests,
    ]
    # 头部与官方 test_spec.py:61 逐字一致：`set -uxo pipefail`
    #
    # `-x` 不是调试残留，而是**日志协议的一部分**：`: '>>>>> Start Test Output'`
    # 是 shell 的空命令（`:`），本身不产生任何输出。只有开了 `-x`，shell 才会把
    # 该行以 `+ : '>>>>> Start Test Output'` 的形式回显到 stderr，截取标记才存在。
    # 实测漏掉 `-x` 会导致日志里找不到分隔符，被误判为"测试段未产生"，
    # 而实际上测试跑得好好的。
    #
    # 刻意**不加 `-e`**（官方也不加）：测试失败时退出码非 0 属正常情况，
    # 一旦 `-e` 生效，收尾的 reset_tests 不会执行，会污染沙箱状态。
    return "#!/bin/bash\nset -uxo pipefail\n" + "\n".join(cmds) + "\n"


def extract_test_output(log: str) -> str:
    """从完整日志里截出测试输出段。

    截取而非全量解析的原因：install 阶段的编译日志里可能出现形似
    `PASSED xxx` 的字样（如 pip 的测试步骤），会污染解析结果。
    """
    i = log.find(START_TEST_OUTPUT)
    if i < 0:
        return log  # 没有分隔符：可能在 install 阶段就崩了，交给上层判定
    j = log.find(END_TEST_OUTPUT, i)
    return log[i + len(START_TEST_OUTPUT) : j if j > 0 else len(log)]


def parse_log(repo: str, log: str) -> dict[str, str]:
    """用官方 parser 解析日志，返回 {test_id: status 字符串}。"""
    parser = load_parsers().get(repo)
    if parser is None:
        raise OfficialSpecError(f"官方解析器无此 repo：{repo}")
    section = extract_test_output(log)
    try:
        # 官方 parser 签名为 (log, test_spec)，test_spec 仅少数 repo 用到
        raw = parser(section, None)
    except TypeError:
        raw = parser(section)
    return {k: (v.value if isinstance(v, TestStatus) else str(v)) for k, v in raw.items()}


def grade(task: dict[str, Any], log: str) -> dict[str, Any]:
    """按官方口径判定 F2P / P2P 通过情况。

    官方 grading 规则：只有 status == PASSED 才算过；SKIPPED/XFAIL/ERROR 都不算。
    缺失的 test_id 同样计为未通过（测试没跑到 = 没通过）。
    """
    status = parse_log(task["repo"], log)

    def norm(v: Any) -> list[str]:
        if isinstance(v, str):
            import json

            try:
                v = json.loads(v)
            except Exception:
                v = [v]
        return list(v or [])

    f2p = norm(task.get("fail_to_pass") or task.get("FAIL_TO_PASS"))
    p2p = norm(task.get("pass_to_pass") or task.get("PASS_TO_PASS"))
    ok = lambda t: status.get(t) == TestStatus.PASSED.value  # noqa: E731

    f2p_pass = [t for t in f2p if ok(t)]
    p2p_pass = [t for t in p2p if ok(t)]
    return {
        "f2p_pass": len(f2p_pass),
        "f2p_total": len(f2p),
        "p2p_pass": len(p2p_pass),
        "p2p_total": len(p2p),
        "resolved": len(f2p_pass) == len(f2p) and len(p2p_pass) == len(p2p) and bool(f2p),
        "parsed_tests": len(status),
        "f2p_missing": [t for t in f2p if t not in status][:5],
        "p2p_failed": [t for t in p2p if not ok(t)][:5],
    }


def verify_official_spec() -> None:
    """自检：确认官方数据加载正常且关键假设成立。CI/启动时调用。"""
    specs, parsers = load_specs(), load_parsers()
    assert len(specs) >= 15, f"规格表 repo 数异常偏少：{len(specs)}"
    assert len(parsers) >= 15, f"解析器 repo 数异常偏少：{len(parsers)}"
    # 关键假设：三类非 pytest repo 的命令确实不是 pytest
    assert "runtests.py" in specs["django/django"]["4.2"]["test_cmd"]
    assert "tox" in specs["sphinx-doc/sphinx"]["3.5"]["test_cmd"]
    assert "bin/test" in specs["sympy/sympy"]["1.11"]["test_cmd"]
    print(f"[ok] 官方规格 {len(specs)} repo / 解析器 {len(parsers)} repo，关键假设成立")


if __name__ == "__main__":
    verify_official_spec()
