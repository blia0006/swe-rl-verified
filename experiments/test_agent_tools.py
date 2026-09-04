#!/usr/bin/env python3
"""本机自测 ReAct Agent 的工具层（不消耗沙箱）。

覆盖两处实测炸过的地方：
① read_file 必须返回**纯原文**（带行号会让模型抄进 edit 的 search 而永远失配）
② edit 的三级容错：纯原文命中 / 带行号自动剥离 / 0 处匹配给出原文上下文
"""
import importlib.util
import os
import shutil
import tempfile

SAMPLE = """def f():
    pivot = datetime.datetime(
        d.year, d.month
    )
    return pivot
"""


def load_agent(repo: str):
    spec = importlib.util.spec_from_file_location("ra", "sandbox_agent/react_agent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.REPO = repo
    orig = m.sh
    m.sh = lambda cmd, cwd=repo, timeout=120: orig(cmd, cwd=repo, timeout=timeout)
    return m


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="edittest_")
    try:
        open(os.path.join(tmp, "sample.py"), "w").write(SAMPLE)
        m = load_agent(tmp)
        fails = 0

        print("--- 1) read_file 返回纯原文（不得带行号前缀）---")
        ok, out = m.t_read_file("sample.py", 1, 5)
        body = out.split("\n", 1)[1] if "\n" in out else ""
        has_num = any(
            l.lstrip()[:6].strip().isdigit() and "\t" in l[:10]
            for l in body.split("\n") if l.strip()
        )
        print(f"    ok={ok} 含行号={has_num}")
        print("    " + out.replace("\n", "\n    ")[:200])
        fails += 0 if (ok and not has_num) else 1

        print("\n--- 2) edit 用纯原文 → 应成功 ---")
        ok, out = m.t_edit(
            "sample.py", "    pivot = datetime.datetime(", "    pivot = dt.datetime("
        )
        print(f"    ok={ok} {out[:120]}")
        fails += 0 if ok else 1

        print("\n--- 3) edit 带行号前缀 → 应自动剥离并成功 ---")
        ok, out = m.t_edit("sample.py", "     5\t    return pivot", "     5\t    return None")
        print(f"    ok={ok} {out[:150]}")
        cur = open(os.path.join(tmp, "sample.py")).read()
        print(f"    文件现状: {cur.replace(chr(10), ' | ')}")
        fails += 0 if (ok and "return None" in cur and "\t" not in cur) else 1

        print("\n--- 4) 匹配 0 处 → 应返回原文上下文供模型对照 ---")
        ok, out = m.t_edit("sample.py", "    pivot = datetime.datetime(", "x")
        print(f"    ok={ok}")
        print("    " + out.replace("\n", "\n    ")[:350])
        fails += 0 if (not ok and "原文" in out) else 1

        print("\n--- 5) 路径规范化（用真实 REPO 常量 /testbed）---")
        # 单独加载一份未改 REPO 的模块：norm_path 的语义是"相对 /testbed 规范化"，
        # 用被测试改写过的临时 REPO 会验错对象
        spec2 = importlib.util.spec_from_file_location(
            "ra_pure", "sandbox_agent/react_agent.py"
        )
        pure = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(pure)
        cases = [("/testbed", "."), ("/testbed/a.py", "a.py"), ("a.py", "a.py"),
                 ("../x", "."), ("", "."), ("/testbed/pkg/mod.py", "pkg/mod.py")]
        for inp, want in cases:
            got = pure.norm_path(inp)
            mark = "✓" if got == want else "✗"
            if got != want:
                fails += 1
            print(f"    {mark} {inp!r} → {got!r}" + ("" if got == want else f" 期望 {want!r}"))

        print("\n--- 6) edit 空白无关匹配（模型把多行签名压成一行）---")
        # 复刻实测失败场景：flask 的 __init__ 在源码里跨多行，模型写成一行
        multiline = os.path.join(tmp, "blueprints.py")
        open(multiline, "w").write(
            "class Blueprint:\n"
            "    def __init__(\n"
            "        self,\n"
            "        name: str,\n"
            "        import_name: str,\n"
            "    ):\n"
            "        self.name = name\n"
        )
        ok, out = m.t_edit(
            "blueprints.py",
            "def __init__( self, name: str, import_name: str, ):",
            "def __init__(\n        self,\n        name: str,\n        import_name: str,\n    ):\n        if not name:\n            raise ValueError('name may not be empty')",
        )
        print(f"    ok={ok} {out[:150]}")
        cur = open(multiline).read()
        print(f"    结果含校验: {'raise ValueError' in cur}")
        fails += 0 if (ok and "raise ValueError" in cur and "self.name = name" in cur) else 1

        print("\n--- 7) 模糊匹配有歧义时应拒绝（避免改错地方）---")
        dup = os.path.join(tmp, "dup.py")
        open(dup, "w").write("def f(a,\n  b):\n    pass\n\ndef f(a,   b):\n    pass\n")
        ok, out = m.t_edit("dup.py", "def f(a, b):", "def g(a, b):")
        print(f"    ok={ok} {out[:120]}")
        fails += 0 if not ok else 1

        print(f"\n{'=' * 50}\n{'全部通过' if fails == 0 else f'{fails} 项未通过'}")
        return 1 if fails else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
