# Hugging Face Space Deployment Guide

## 1. Quick reference card

| Thing | Value |
|---|---|
| Space URL | https://huggingface.co/spaces/Casp1an/TradeX |
| Space owner | Casp1an |
| Space type | Docker SDK (sdk: docker) |
| Git remote name | hf-space |
| Git remote URL | https://huggingface.co/spaces/Casp1an/TradeX |
| HF auth | Personal access token in .env as HF_TOKEN (gitignored) |
| Deploy branch on HF | main |
| Local deploy branch | hf-space-bundle (orphan, single commit) |
| Update mechanism | Force-push an orphan commit (no shared history) |

## 2. One-time setup on a fresh machine
```bash
# 1. Clone the GitHub repo (the source of truth)
git clone git@github.com:YashAG17/TradeX-Hackathon.git
cd TradeX-Hackathon

# 2. Add the HF Space as a second remote
git remote add hf-space https://huggingface.co/spaces/Casp1an/TradeX

# 3. Put your HF token in .env (gitignored already)
cat >> .env <<'EOF'
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
chmod 600 .env

# 4. (optional) fetch what HF currently has
git fetch hf-space
```

## 3. Why force-push?
The HF Space and main have unrelated histories. The Space is updated by force-pushing a single squashed orphan commit (hf-space-bundle). This keeps the Space repo small (HF's storage is tighter than GitHub's) and avoids leaking branch history.

Confirm any time with:
```bash
git merge-base main hf-space/main      # empty output ⇒ unrelated histories
git rev-list --left-right --count main...hf-space/main
```

## 4. Deploy: the full sequence
Run from repo root with a clean working tree on main.

```bash
# 4a. Make sure your work is committed & pushed to GitHub first
git status                  # must be clean
git push origin main        # GitHub source of truth

# 4b. Refresh the bundle: orphan branch with current main's tree as one commit
git checkout --orphan _hf_bundle_tmp
git reset                   # unstage the inherited index
git add -A                  # respects .gitignore (so node_modules, plots, .pth are excluded)
git commit -m "TradeX Space deploy: <one-line summary of what's new>"

# 4c. Force-push the orphan to HF main (auth via .env HF_TOKEN)
set -a && source .env && set +a
git -c "credential.helper=!f() { echo username=Casp1an; echo \"password=$HF_TOKEN\"; }; f" \
    push hf-space _hf_bundle_tmp:main --force

# 4d. Move the local hf-space-bundle pointer forward and clean up
git branch -f hf-space-bundle _hf_bundle_tmp
git checkout main
git branch -D _hf_bundle_tmp

# 4e. Verify
git fetch hf-space
git log --oneline hf-space/main -3
```

## 5. Reusable one-liner script
A deployment script is provided at `scripts/deploy-hf.sh`.

## 6. Hotfix path (skip the local commit on main)
If you want to ship something to HF without committing to GitHub first:
```bash
git stash --include-untracked
git checkout --orphan _hf_hotfix
git reset
git stash pop                # bring changes back into the orphan branch
git add -A
git commit -m "hotfix on HF only"
# ...same force-push + cleanup as section 4...
```

## 7. Auth — three ways, pick one
A. `.env` + credential helper (what we used) — token never enters argv, never written to git config.
B. URL with embedded token — quickest, but token shows in process list briefly.
C. `huggingface_hub` CLI (no git remote needed)

## 8. What the HF Space actually runs
README.md frontmatter (top of the file) controls Space behavior. Currently configured for Docker SDK:
```yaml
---
title: TradeX
emoji: 📈
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---
```
This builds the React frontend via a multi-stage Dockerfile and serves it with FastAPI.

## 9. Common failure modes
Symptom | Cause | Fix
--- | --- | ---
fatal: could not read Username | No credentials | Use auth methods in §7.
Invalid username or password | Token wrong / expired | Regenerate at huggingface.co/settings/tokens.
non-fast-forward | Pushed without --force | Add --force.
Your push was accepted, but ... LFS | A binary >10MB slipped in | Check .gitignore, git rm --cached offender.
Space stuck "Building" | Bad requirements.txt pin | Check Logs tab on HF Space.
Push uploads gigabytes | node_modules / dist slipped in | Verify frontend/.gitignore is committed.

## 10. Verify a deploy worked
```bash
# Latest commit on the Space
git fetch hf-space && git log --oneline hf-space/main -1

# Files actually on the Space
git ls-tree -r hf-space/main | wc -l
git ls-tree -r --long hf-space/main | sort -k4 -n | tail -10
```
