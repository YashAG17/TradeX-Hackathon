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
#    HF Space runs the Gradio app (`sdk: gradio`, `app_file: app.py`),
#    so we strip the React frontend, FastAPI backend, and Dockerfile from
#    the bundle. Those still live on GitHub for local dev, but the Space
#    only needs the Gradio entrypoint + Python packages it imports.
HF_EXCLUDE=(
  "Dockerfile"
  "frontend"
  "backend"
  "node_modules"
  ".cursor"
  ".env"
)

TMP="_hf_bundle_$(date +%s)"
git checkout --orphan "$TMP"
git reset
git add -A
for path in "${HF_EXCLUDE[@]}"; do
  if git ls-files --error-unmatch -- "$path" >/dev/null 2>&1 \
     || [ -e "$path" ]; then
    git rm -rf --cached --ignore-unmatch -- "$path" >/dev/null
  fi
done
git commit -m "$MSG"

# 4. Force-push to HF main (token never appears in argv)
git -c "credential.helper=!f() { echo username=$HF_USER; echo \"password=\$HF_TOKEN\"; }; f" \
    push "$HF_REMOTE" "$TMP:$HF_BRANCH" --force

# 5. Move local pointer & clean up
#    `git rm --cached` left the excluded paths as untracked in the working
#    tree, so a normal `git checkout main` would refuse to overwrite them.
#    They are byte-identical to what's tracked on main (we never touched
#    the working tree, only the index), so a force checkout is safe.
git branch -f "$LOCAL_BUNDLE" "$TMP"
git checkout -f main
git branch -D "$TMP"
echo "Deployed to https://huggingface.co/spaces/$HF_USER/TradeX"
