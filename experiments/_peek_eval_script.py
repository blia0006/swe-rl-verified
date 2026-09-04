#!/usr/bin/env python3
"""抽查生成的 eval 脚本关键行（临时验证用）。"""
import json
import sys

sys.path.insert(0, ".")
from pipeline.official_spec import make_eval_script, test_directives

ts = {}
for line in open("data/tasks.jsonl"):
    if line.strip():
        t = json.loads(line)
        ts[t["task_id"]] = t

for tid in ("django__django-16429", "sympy__sympy-23534", "sphinx-doc__sphinx-8621"):
    t = ts[tid]
    s = make_eval_script(t)
    print(f"--- {tid} ({len(s)} 字符) ---")
    print("    directives:", test_directives(t["repo"], t["test_patch"]))
    for ln in s.splitlines():
        if any(k in ln for k in ("runtests", "bin/test", "tox", "Start Test", "git checkout", "pip install", "conda activate")):
            print("   ", ln[:140])
    print()
