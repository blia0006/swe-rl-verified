"""任务表示：search/replace 块 ←→ unified diff

## 为什么不让模型直接写 unified diff（上一轮的最大教训）

上一轮实测 440 次采样：

| 失败原因 | 次数 |
|---|---|
| `corrupt patch at line N` | **227** |
| `patch fragment without header` | 44 |
| `patch does not apply` | 10 |

`strict` 模式下可应用的仅 **8/440 = 1.8%**。

根因不是"不会修 bug"，而是 unified diff 要求模型**手工计算 hunk header**
（`@@ -起始行,行数 +起始行,行数 @@`）——这对小模型是硬伤：它得数清楚上下文行数、
算准偏移量，一位算错整个 patch 就废。

## search/replace 表示法

```
### path/to/file.py
<<<<<<< SEARCH
    原始代码片段（逐字匹配）
=======
    替换后的代码
>>>>>>> REPLACE
```

**完全不需要行号** —— 靠内容定位。模型只要能"认出要改的那段代码"就行，
把"算数"这项与修复能力无关的负担彻底移除。
本模块负责把它转成 `git apply` 能吃的 unified diff。

## 附带解决"文件路径幻觉"

上一轮还实测到模型会写错路径（丢 `src/` 前缀、凭空加 `jd/` 前缀），
这类样本连 `--recount -C0` 都救不回。本模块提供 `resolve_path()`，
在候选文件列表里做后缀匹配纠正 —— 因为 prompt 已经给出了确切的文件清单，
模型写错只是"抄错"，不是"不知道改哪"。

自测：
    python3 pipeline/edit_format.py
"""

import difflib
import re

SEARCH_MARK = "<<<<<<< SEARCH"
DIVIDER = "======="
REPLACE_MARK = ">>>>>>> REPLACE"

# 允许模型把文件名写在 ### 后、或写成 markdown 代码块前的一行
_FILE_HEADER = re.compile(r"^\s*(?:#{1,6}\s*)?([\w./\-]+\.\w+)\s*$")


class EditBlock:
    """一次「把 search 换成 replace」的编辑意图。"""

    def __init__(self, path, search, replace):
        self.path = path
        self.search = search
        self.replace = replace

    def __repr__(self):
        return "EditBlock(path=%r, search=%d chars, replace=%d chars)" % (
            self.path,
            len(self.search),
            len(self.replace),
        )


def parse_edit_blocks(text, default_path=None):
    """从模型输出里抽取所有 search/replace 块。

    容错点（都是小模型实际会犯的错，逐条实测过）：
      · markdown 代码围栏（```python / ```）—— 剥掉
      · 文件名写在块前一行、或写在围栏语言标记后
      · 分隔线长度不一（`=====` 到 `=========`）
      · 只有一个待改文件时省略文件名 —— 用 default_path 兜底
    """
    blocks = []
    lines = text.splitlines()
    cur_path = default_path
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 代码围栏：可能是 ```python 或 ```path/to/file.py
        if stripped.startswith("```"):
            rest = stripped[3:].strip()
            m = _FILE_HEADER.match(rest)
            if m:
                cur_path = m.group(1)
            i += 1
            continue

        # 独立的一行文件名
        m = _FILE_HEADER.match(line)
        if m and not stripped.startswith(("<<<", "===", ">>>")):
            cur_path = m.group(1)
            i += 1
            continue

        if stripped.startswith("<<<<<<<") and "SEARCH" in stripped.upper():
            search_lines = []
            replace_lines = []
            i += 1
            # 收集 SEARCH 段
            while i < len(lines) and not _is_divider(lines[i]):
                search_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                break  # 块不完整，丢弃
            i += 1
            # 收集 REPLACE 段
            while i < len(lines) and not lines[i].strip().startswith(">>>>>>>"):
                replace_lines.append(lines[i])
                i += 1
            i += 1
            if cur_path:
                blocks.append(
                    EditBlock(
                        cur_path,
                        "\n".join(search_lines),
                        "\n".join(replace_lines),
                    )
                )
            continue
        i += 1

    return blocks


def _is_divider(line):
    s = line.strip()
    return len(s) >= 3 and set(s) == {"="}


def resolve_path(path, candidates):
    """把模型写的路径纠正到真实存在的文件。

    上一轮实测的两类错法：
        itsdangerous/encoding.py    ← 丢了 src/ 前缀
        jd/tenacity/retry_after.py  ← 凭空多出 jd/ 前缀
    prompt 里已给出确切文件清单，因此按后缀匹配即可纠正。
    """
    if not candidates:
        return path
    if path in candidates:
        return path
    # 后缀匹配：模型少写或多写了前缀目录
    for c in candidates:
        if c.endswith("/" + path) or path.endswith("/" + c):
            return c
    # 只按文件名匹配（唯一命中才采纳，避免张冠李戴）
    base = path.rsplit("/", 1)[-1]
    hits = [c for c in candidates if c.rsplit("/", 1)[-1] == base]
    if len(hits) == 1:
        return hits[0]
    return path


def apply_blocks_to_source(source, blocks):
    """把编辑块应用到源码文本，返回 (新文本, 失败的块列表)。

    匹配策略逐级放宽 —— 小模型常有缩进/空白偏差，但意图明确，
    不该因为多一个空格就判定失败：
      1. 逐字精确匹配
      2. 忽略行尾空白
      3. 忽略每行的前导缩进（按内容匹配，套用原文缩进）
    """
    text = source
    failed = []
    for b in blocks:
        if not b.search.strip():
            failed.append((b, "SEARCH 段为空"))
            continue
        new_text, ok = _replace_once(text, b.search, b.replace)
        if ok:
            text = new_text
        else:
            failed.append((b, "SEARCH 段在源文件中未找到"))
    return text, failed


def _replace_once(text, search, replace):
    # 1) 精确
    if search in text:
        return text.replace(search, replace, 1), True

    # 2) 忽略行尾空白
    def strip_trailing(s):
        return "\n".join(ln.rstrip() for ln in s.split("\n"))

    t2, s2 = strip_trailing(text), strip_trailing(search)
    if s2 in t2:
        idx = t2.index(s2)
        start = _map_index(text, t2, idx)
        end = _map_index(text, t2, idx + len(s2))
        return text[:start] + replace + text[end:], True

    # 3) 忽略前导缩进：按"去缩进后的内容"定位，再套回原文缩进
    src_lines = text.split("\n")
    pat_lines = [ln.strip() for ln in search.split("\n") if ln.strip()]
    if not pat_lines:
        return text, False
    for i in range(len(src_lines) - len(pat_lines) + 1):
        window = [ln.strip() for ln in src_lines[i : i + len(pat_lines)]]
        if window == pat_lines:
            indent = _leading_ws(src_lines[i])
            new_block = "\n".join(
                (indent + ln.lstrip()) if ln.strip() else ln
                for ln in replace.split("\n")
            )
            merged = src_lines[:i] + new_block.split("\n") + src_lines[i + len(pat_lines) :]
            return "\n".join(merged), True
    return text, False


def _leading_ws(line):
    return line[: len(line) - len(line.lstrip())]


def _map_index(orig, stripped, idx):
    """把「去行尾空白文本」中的下标映射回原文下标。"""
    oi = si = 0
    while si < idx and oi < len(orig):
        if orig[oi] == stripped[si] if si < len(stripped) else False:
            oi += 1
            si += 1
        else:
            oi += 1  # 原文里被剥掉的空白
    return oi


def blocks_to_unified_diff(blocks, file_contents, context=3):
    """把编辑块转成 `git apply` 可用的 unified diff。

    行号由 difflib 计算 —— 这正是把「算数」从模型身上卸下来的关键：
    模型只负责认出代码片段，行号交给确定性的程序算，绝不会算错。
    """
    by_path = {}
    for b in blocks:
        by_path.setdefault(b.path, []).append(b)

    parts = []
    for path, bs in by_path.items():
        original = file_contents.get(path)
        if original is None:
            continue
        modified, _failed = apply_blocks_to_source(original, bs)
        if modified == original:
            continue
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="a/" + path,
            tofile="b/" + path,
            n=context,
        )
        chunk = "".join(diff)
        if chunk and not chunk.endswith("\n"):
            chunk += "\n"
        parts.append(chunk)
    return "".join(parts)


def model_output_to_patch(text, file_contents, default_path=None):
    """端到端：模型原始输出 → unified diff。

    返回 (patch_text, info)。info 记录解析细节，用于失败归因 ——
    上一轮的教训是只记 reward 标量，导致无法区分「没写出块」「路径错」
    「片段没找到」，事后只能靠猜。
    """
    candidates = list(file_contents.keys())
    blocks = parse_edit_blocks(text, default_path=default_path)
    info = {
        "n_blocks": len(blocks),
        "paths": [],
        "path_corrected": 0,
        "unmatched": 0,
        "reason": "",
    }
    if not blocks:
        info["reason"] = "未解析出任何 SEARCH/REPLACE 块"
        return "", info

    fixed = []
    for b in blocks:
        rp = resolve_path(b.path, candidates)
        if rp != b.path:
            info["path_corrected"] += 1
        fixed.append(EditBlock(rp, b.search, b.replace))
    info["paths"] = sorted({b.path for b in fixed})

    known = [b for b in fixed if b.path in file_contents]
    if not known:
        info["reason"] = "块中的文件路径均不在题目文件清单内"
        return "", info

    # 统计匹配失败数，供归因
    for path in {b.path for b in known}:
        _mod, failed = apply_blocks_to_source(
            file_contents[path], [b for b in known if b.path == path]
        )
        info["unmatched"] += len(failed)

    patch = blocks_to_unified_diff(known, file_contents)
    if not patch:
        info["reason"] = "块解析成功但未产生任何改动（SEARCH 未匹配到内容）"
    return patch, info


# ------------------------------------------------------------------ 自测

def _selftest():
    src = '''def add(a, b):
    """Add two numbers."""
    return a - b


def sub(a, b):
    return a - b
'''
    files = {"calc/ops.py": src}
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(("✓ " if cond else "✗ ") + name + ("  " + extra if extra else ""))

    # 1) 标准格式
    out1 = """### calc/ops.py
<<<<<<< SEARCH
    return a - b


def sub
=======
    return a + b


def sub
>>>>>>> REPLACE
"""
    p, info = model_output_to_patch(out1, files)
    check("标准块 → 生成 diff", p.startswith("--- a/calc/ops.py") and "+    return a + b" in p,
          "blocks=%d" % info["n_blocks"])

    # 2) 带 markdown 围栏
    out2 = """```python
calc/ops.py
<<<<<<< SEARCH
    \"\"\"Add two numbers.\"\"\"
    return a - b
=======
    \"\"\"Add two numbers.\"\"\"
    return a + b
>>>>>>> REPLACE
```"""
    p2, i2 = model_output_to_patch(out2, files)
    check("markdown 围栏容错", "+    return a + b" in p2, "blocks=%d" % i2["n_blocks"])

    # 3) 路径幻觉纠正（丢前缀）
    out3 = out1.replace("### calc/ops.py", "### ops.py")
    p3, i3 = model_output_to_patch(out3, files)
    check("路径幻觉可纠正", i3["path_corrected"] == 1 and p3 != "",
          "corrected=%d" % i3["path_corrected"])

    # 4) 缩进偏差容错
    out4 = """### calc/ops.py
<<<<<<< SEARCH
return a - b


def sub
=======
return a + b


def sub
>>>>>>> REPLACE
"""
    p4, _ = model_output_to_patch(out4, files)
    check("缩进偏差容错", "+    return a + b" in p4)

    # 5) 生成的 diff 行号正确（关键：这是替代模型算行号的全部意义）
    import re as _re
    m = _re.search(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", p)
    check("hunk header 由程序算出", m is not None, m.group(0) if m else "")

    # 6) 无块时给出明确归因
    _p6, i6 = model_output_to_patch("我觉得应该把减号改成加号。", files)
    check("空输出有归因", i6["reason"] != "" and _p6 == "", i6["reason"])

    # 7) SEARCH 匹配不到时归因
    out7 = """### calc/ops.py
<<<<<<< SEARCH
this line does not exist at all
=======
whatever
>>>>>>> REPLACE
"""
    p7, i7 = model_output_to_patch(out7, files)
    check("未匹配有归因", p7 == "" and i7["unmatched"] == 1, i7["reason"])

    # 8) 多文件
    files2 = dict(files, **{"calc/util.py": "X = 1\n"})
    out8 = out1 + """
### calc/util.py
<<<<<<< SEARCH
X = 1
=======
X = 2
>>>>>>> REPLACE
"""
    p8, i8 = model_output_to_patch(out8, files2)
    check("多文件", p8.count("--- a/") == 2, "paths=%s" % i8["paths"])

    print("\n" + ("全部通过 ✓" if ok else "存在失败 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
