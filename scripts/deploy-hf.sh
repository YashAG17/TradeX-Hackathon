#!/usr/bin/env bash
# Deploy current working tree to the HF Space as a refreshed orphan bundle.
# Usage: scripts/deploy-hf.sh "commit message"
set -euo pipefail
MSG="${1:-TradeX Space deploy}"
HF_USER="Casp1an"
HF_REMOTE="hf-space"
HF_BRANCH="main"
LOCAL_BUNDLE="hf-space-bundle"

# 1. Sanity checks
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is dirty. Commit or stash first." >&2
  exit 1
fi
if [[ ! -f .env ]] || ! grep -q '^HF_TOKEN=' .env; then
  echo "HF_TOKEN not found in .env" >&2
  exit 1
fi

# 2. Load token
set -a; source .env; set +a

# 3. Build orphan bundle
TMP="_hf_bundle_$(date +%s)"
git checkout --orphan "$TMP"
git reset
git add -A
git commit -m "$MSG"

# 4. Force-push to HF main (token never appears in argv)
git -c "credential.helper=!f() { echo username=$HF_USER; echo \"password=\$HF_TOKEN\"; }; f" \
    push "$HF_REMOTE" "$TMP:$HF_BRANCH" --force

# 5. Move local pointer & clean up
git branch -f "$LOCAL_BUNDLE" "$TMP"
git checkout main
git branch -D "$TMP"
echo "Deployed to https://huggingface.co/spaces/$HF_USER/TradeX"
