#!/usr/bin/env bash
# 安装 git 钩子：每次 commit 前自动扫描敏感信息，命中即阻断
#
#   bash scripts/install_git_hooks.sh
#
# 说明：钩子存放在 .git/hooks/（不随仓库分发），因此**每个 clone 都要跑一次**。
# 这是 git 的设计，不是遗漏 —— README 的"环境准备"一节会提醒。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_DIR="$ROOT/.git/hooks"

if [ ! -d "$ROOT/.git" ]; then
  echo "[x] 还不是 git 仓库，请先 git init"
  exit 1
fi

mkdir -p "$HOOK_DIR"

cat > "$HOOK_DIR/pre-commit" <<'HOOK'
#!/usr/bin/env bash
# 自动生成于 scripts/install_git_hooks.sh —— 请勿手改，改脚本后重新执行安装
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"

# 1) 硬阻断：任何形态的环境/凭证文件被 staged（.env.example 等模板除外）
#    覆盖三类命名：  .env / .env.<any>（如 .env.local）/ <name>.env（如 app.env）
BAD=$(git diff --cached --name-only --diff-filter=ACMR \
      | grep -E '(^|/)(\.env([.][A-Za-z0-9_-]+)?|[A-Za-z0-9_-]+\.env)$' \
      | grep -vE '\.(example|sample|template|dist)$' || true)
if [ -n "$BAD" ]; then
  echo ""
  echo "✗ 拒绝提交：以下文件疑似环境/凭证文件"
  echo "$BAD" | sed 's/^/     /'
  echo "  → git restore --staged <file>，并确认它已被 .gitignore 忽略"
  echo ""
  exit 1
fi

# 2) 内容扫描
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "⚠ 找不到 python3，跳过内容扫描（请手动检查）"
  exit 0
fi
"$PY" "$ROOT/scripts/scan_secrets.py" --staged || {
  echo "→ 确认为误报时，可在该行末尾加注释 'noqa: secret-scan' 后重试"
  echo "→ 紧急绕过（不推荐）：git commit --no-verify"
  exit 1
}
HOOK

chmod +x "$HOOK_DIR/pre-commit"
echo "[ok] 已安装 pre-commit 钩子：$HOOK_DIR/pre-commit"
echo "     自检：python3 scripts/scan_secrets.py --all"
