#!/usr/bin/env python3
"""
敏感信息扫描器（public 仓库的守门人）
====================================

用途有两个，共用同一套规则：
  1. 作为 **pre-commit 钩子**：只扫「本次将要提交的内容」，命中即阻断提交
  2. 手工全量审计：`--all` 扫工作区已追踪文件，`--history` 扫全部 git 历史

设计原则：
  · **默认拒绝**：命中任一规则就退出码非 0，宁可误报也不漏报
  · 分两级：`SECRET`（凭证，绝不允许）/ `IDENTIFIER`（账号资源标识，public 仓库应脱敏）
  · 误报可用行内标记 `# noqa: secret-scan` 豁免（需显式写在同一行，避免无意绕过）
  · 扫描器自身与 `.env.example` 的规则示例不参与匹配（否则自己扫自己必命中）

用法：
    python3 scripts/scan_secrets.py --staged      # 钩子模式（默认）
    python3 scripts/scan_secrets.py --all         # 全量扫工作区
    python3 scripts/scan_secrets.py --history     # 扫 git 历史（首次 public 前必跑）
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXEMPT = "noqa: secret-scan"

# 这些文件本身就要写规则/占位符，跳过（但仍会检查它们是否含真凭证——见 SECRET 规则仍生效）
SELF_FILES = {"scripts/scan_secrets.py"}

# 二进制/大文件后缀，不扫
SKIP_SUFFIX = {
    ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz",
    ".tar", ".whl", ".safetensors", ".bin", ".pt", ".pyc", ".ico",
}

# ---------------------------------------------------------------- 规则

# 级别 1：凭证 —— 任何情况都不允许进仓库
SECRET_RULES: list[tuple[str, str]] = [
    # 腾讯云 SecretId：AKID + 32 位
    ("腾讯云 SecretId", r"AKID[0-9A-Za-z]{28,}"),
    # 腾讯云 SecretKey：赋值语句右侧的 30+ 位串（避免裸匹配造成大量误报）
    (
        "腾讯云 SecretKey 赋值",
        r"(?i)(secret[_-]?key|secretkey)\s*[:=]\s*[\"']?[0-9A-Za-z]{30,}",
    ),
    ("TCR/仓库密码赋值", r"(?i)(tcr_password|docker_password|registry_password)\s*[:=]\s*\S{8,}"),
    ("GitHub 传统 token", r"gh[pousr]_[0-9A-Za-z]{30,}"),
    ("GitHub 细粒度 token", r"github_pat_[0-9A-Za-z_]{50,}"),
    ("E2B / AGS API Key 赋值", r"(?i)(e2b_api_key|ags_api_key)\s*[:=]\s*[\"']?e2b_[0-9A-Za-z]{10,}"),
    ("E2B key 字面量", r"e2b_[0-9a-f]{32,}"),
    ("私钥文件头", r"-----BEGIN (RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"),
    ("推理服务 API Key 赋值", r"(?i)(llm_api_key|vllm_api_key|api[_-]?key)\s*[:=]\s*[\"']?[0-9a-f]{32,}"),
    ("Slack/飞书 webhook", r"https://hooks\.(slack|feishu)\S+"),
    ("JWT", r"eyJ[0-9A-Za-z_-]{15,}\.eyJ[0-9A-Za-z_-]{15,}\."),
]

# 级别 2：账号/资源标识 —— 不是凭证，但 public 仓库应脱敏
#   （旧仓库把 APPID 与账号 UIN 片段推上了 GitHub，本轮不重复该做法）
IDENTIFIER_RULES: list[tuple[str, str]] = [
    ("腾讯云 APPID（10位，常见于 COS bucket 后缀）", r"\b12[0-9]{8}\b"),
    ("账号 UIN 片段（TCR 个人版命名空间）", r"tcb-1000[0-9]{8}-[0-9a-z]{4}"),
    ("CAM 角色 ARN 含 UIN", r"qcs::cam::uin/[0-9]{6,}"),
    ("CVM 实例 ID", r"\bins-[0-9a-z]{8}\b"),
    ("TKE 集群 ID", r"\bcls-[0-9a-z]{8}\b"),
    ("VPC / 子网 / 安全组 / NAT ID", r"\b(vpc|subnet|sg|nat|rtb|np)-[0-9a-z]{8}\b"),
    ("SSH 密钥对 ID", r"\bskey-[0-9a-z]{8}\b"),
]

# 私网/本地地址属正常配置，不算标识；只拦公网 IP 字面量
PUBLIC_IP_RE = re.compile(r"\b(?!0\.|10\.|127\.|169\.254\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|22[4-9]\.|2[3-5]\d\.)((?:[1-9]\d{0,2})\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")
# 这些是公开的、非本账号的地址/版本号，白名单放过
IP_ALLOW = {"0.0.0.0", "255.255.255.255", "8.8.8.8", "1.1.1.1"}

SECRET_COMPILED = [(n, re.compile(p)) for n, p in SECRET_RULES]
IDENT_COMPILED = [(n, re.compile(p)) for n, p in IDENTIFIER_RULES]


# ---------------------------------------------------------------- 扫描

def scan_text(text: str, path: str, check_identifiers: bool) -> list[tuple[str, int, str, str]]:
    """返回 [(级别, 行号, 规则名, 命中片段)]。"""
    hits: list[tuple[str, int, str, str]] = []
    is_self = path in SELF_FILES
    for lineno, line in enumerate(text.splitlines(), 1):
        if EXEMPT in line:
            continue
        for name, rx in SECRET_COMPILED:
            if is_self:
                break  # 扫描器自身写着规则本体，跳过；它不含真凭证
            m = rx.search(line)
            if m:
                hits.append(("SECRET", lineno, name, _mask(m.group(0))))
        if not check_identifiers or is_self:
            continue
        for name, rx in IDENT_COMPILED:
            m = rx.search(line)
            if m:
                hits.append(("IDENTIFIER", lineno, name, m.group(0)))
        for m in PUBLIC_IP_RE.finditer(line):
            ip = m.group(0)
            if ip in IP_ALLOW:
                continue
            # 排除版本号误报（如 1.2.3.4 出现在 "version 1.2.3.4"）
            if re.search(r"(?i)(version|v)\s*[:=]?\s*" + re.escape(ip), line):
                continue
            hits.append(("IDENTIFIER", lineno, "公网 IP 字面量", ip))
    return hits


def _mask(s: str) -> str:
    """命中片段回显时打码，避免把凭证抄进 CI 日志。"""
    if len(s) <= 12:
        return s[:4] + "***"
    return s[:8] + "***" + s[-2:]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout


def staged_files() -> list[str]:
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    return [f for f in out.splitlines() if f.strip()]


def tracked_files() -> list[str]:
    return [f for f in git("ls-files").splitlines() if f.strip()]


def read_staged(path: str) -> str:
    return git("show", f":{path}")


def run(mode: str, check_identifiers: bool) -> int:
    if mode == "history":
        return scan_history(check_identifiers)

    files = staged_files() if mode == "staged" else tracked_files()
    if not files:
        print("[scan] 没有需要扫描的文件")
        return 0

    total: list[tuple[str, str, int, str, str]] = []
    for f in files:
        if Path(f).suffix.lower() in SKIP_SUFFIX:
            continue
        try:
            text = read_staged(f) if mode == "staged" else (ROOT / f).read_text(
                encoding="utf-8", errors="replace"
            )
        except (OSError, UnicodeDecodeError):
            continue
        for level, lineno, name, frag in scan_text(text, f, check_identifiers):
            total.append((level, f, lineno, name, frag))

    secrets = [t for t in total if t[0] == "SECRET"]
    idents = [t for t in total if t[0] == "IDENTIFIER"]

    if secrets:
        print("\n\033[31m✗ 检出凭证类敏感信息 —— 已阻断\033[0m")
        for _, f, ln, name, frag in secrets:
            print(f"   {f}:{ln}  [{name}]  {frag}")
    if idents:
        print("\n\033[33m⚠ 检出账号/资源标识（public 仓库建议脱敏）\033[0m")
        for _, f, ln, name, frag in idents:
            print(f"   {f}:{ln}  [{name}]  {frag}")

    if secrets or idents:
        print(
            "\n处理方式：\n"
            "  · 凭证 → 移到 .env（已被 .gitignore 忽略），代码里改成 os.environ 读取\n"
            "  · 账号标识 → 换成占位符（如 <APPID> / <NAMESPACE>），真实值放 .env\n"
            f"  · 确认是误报 → 在该行末尾加注释 `{EXEMPT}`\n"
        )
        return 1

    scope = "本次提交" if mode == "staged" else "工作区"
    print(f"\033[32m✓ {scope}未检出敏感信息（扫描 {len(files)} 个文件）\033[0m")
    return 0


def scan_history(check_identifiers: bool) -> int:
    """扫全部历史 blob —— 首次转 public 之前必须跑一次。"""
    print("[scan] 扫描 git 全部历史（首次 public 前的必要检查）…")
    revs = git("rev-list", "--all").split()
    if not revs:
        print("[scan] 仓库还没有提交")
        return 0
    seen: set[str] = set()
    findings = 0
    for rev in revs:
        for line in git("ls-tree", "-r", rev).splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            blob, path = parts[2], parts[3]
            if blob in seen or Path(path).suffix.lower() in SKIP_SUFFIX:
                continue
            seen.add(blob)
            text = git("cat-file", "-p", blob)
            for level, lineno, name, frag in scan_text(text, path, check_identifiers):
                print(f"   {rev[:8]} {path}:{lineno} [{level}/{name}] {frag}")
                findings += 1
    if findings:
        print(
            f"\n\033[31m✗ 历史中检出 {findings} 处\033[0m —— 仓库转 public 前必须处理。\n"
            "   若已 push：先轮换（作废并重建）对应凭证，再考虑重写历史。\n"
            "   若未 push：可用 git filter-repo 重写，或直接重建仓库（最省事）。"
        )
        return 1
    print(f"\033[32m✓ 历史干净（检查 {len(seen)} 个 blob）\033[0m")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="敏感信息扫描器")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--staged", action="store_true", help="只扫本次提交（钩子模式，默认）")
    g.add_argument("--all", action="store_true", help="扫工作区全部已追踪文件")
    g.add_argument("--history", action="store_true", help="扫 git 全部历史")
    ap.add_argument(
        "--secrets-only",
        action="store_true",
        help="只查凭证，跳过账号标识检查（private 仓库可用）",
    )
    args = ap.parse_args()
    mode = "all" if args.all else "history" if args.history else "staged"
    return run(mode, check_identifiers=not args.secrets_only)


if __name__ == "__main__":
    sys.exit(main())
