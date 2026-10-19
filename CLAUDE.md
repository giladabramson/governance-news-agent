# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

`c:\dev\` is a workspace, not a single project. Its git repo is `giladabramson/governance-news-agent` (historical name — it now holds several projects plus the GitHub Actions for two of them).

```
c:\dev\
├─ University\        own repo (giladabramson/University) — HUJI course work + Moodle MCP
├─ Projects\          everything else
├─ tools\             workspace tooling (Gmail MCP server)
├─ .github\workflows\ Actions for Projects/watchdog and Projects/tennis-value-bot
└─ .venv\ .venv311\ .node-portable\ .shot-tool\   local tooling — NEVER move: absolute paths are baked in
```

| Path | What | In this repo? | Own CLAUDE.md? |
|---|---|---|---|
| [University/](University/) | HUJI course work. **Own nested repo** (`giladabramson/University`) holding the read-only Moodle MCP server at `tools/moodle_mcp_server.py` (`moodle-readonly`). Each semester is a **separate repo mounted as a submodule** at `<year>/Semester-<X>` (2026-27: A = 52000/52221/52419, B = 52571/52879 — from Moodle's semester field). All commit as the user's GitHub noreply identity. Commit inside the semester first, then the submodule pointer in University. Local files are the source of truth — don't bulk-mirror Moodle; only 52419 is actively used so far. | No (gitignored) | No — see its [README](University/README.md) |
| [Projects/watchdog/](Projects/watchdog/) | Governance Watchdog — Hebrew-language news classifier and daily email digest (Action, daily 07:00 UTC). | Yes | Yes |
| [Projects/tennis-value-bot/](Projects/tennis-value-bot/) | Paper-trading tennis value bot. Action runs every 30 min and **commits `data/ledger.db` back to this repo** — always `git pull --rebase` before pushing. | Yes | No — README |
| [Projects/outliers/](Projects/outliers/) | Outliers Shorts — name → 45–60 s vertical YouTube Short. Rendered media under `content/` is gitignored (local only). | Yes | Yes |
| [Projects/esports-edge/](Projects/esports-edge/) | Dota 2 probability model for Polymarket tier-2 markets. Research phase, NO trading code until each phase passes human review. | Yes | No — README |
| [Projects/adaptive-music-slicer/](Projects/adaptive-music-slicer/) | StemForge / adaptive music slicer. **Own nested repo** (`giladabramson/adaptive-music-slicer`); landing page deploys to Vercel from its `landing/`. Uses `c:\dev\.venv311` (CUDA torch, demucs). | No (gitignored) | No — README |
| [Projects/soccer-edge/](Projects/soccer-edge/), [plant_ai_timelapse/](Projects/plant_ai_timelapse/), [tagah_physics/](Projects/tagah_physics/), [grow_recipes/](Projects/grow_recipes/) | Smaller / finished experiments, kept local only by the user's choice. | No (gitignored) | tagah_physics only |
| [tools/gmail_mcp_server.py](tools/gmail_mcp_server.py) | Read-only Gmail MCP server (`gmail-readonly` in [.mcp.json](.mcp.json)). | Yes | No — see below |

When the user asks about "the watchdog" they mean `Projects/watchdog/`. Gmail tools / `mcp__gmail-readonly__*` → `tools/gmail_mcp_server.py`. Moodle / courses / `mcp__moodle-readonly__*` → `University/`.

## Commands (from workspace root)

```powershell
.\.venv\Scripts\Activate.ps1

# Watchdog
cd Projects\watchdog
pip install -r requirements.txt
python main.py

# Gmail MCP — re-auth (only needed if token.json is deleted or expired)
pip install -r tools\requirements.txt
python tools\gmail_mcp_server.py auth
```

There is no shared test framework, linter, or build step.

## Gmail MCP server

`tools/gmail_mcp_server.py` exposes four read-only tools to Claude Code via [.mcp.json](.mcp.json). It uses an OAuth client under the user's personal Google Cloud project `gmail-mcp` (account `tiulhilufim@gmail.com`). Secrets live outside the repo at `~/.gmail-mcp-readonly/` (never commit those):

- `~/.gmail-mcp-readonly/credentials.json` — OAuth client (download from Google Cloud Console)
- `~/.gmail-mcp-readonly/token.json` — refresh token (auto-created on first auth, auto-renews)

The OAuth scope is hardcoded to `https://www.googleapis.com/auth/gmail.readonly`. The server cannot send, modify, or delete mail even if asked — Google rejects writes at the API layer.

To switch to a different Gmail account: delete `~/.gmail-mcp-readonly/token.json`, then run `python tools\gmail_mcp_server.py auth` again. To revoke entirely: https://myaccount.google.com/permissions → `gmail-mcp` → Remove access.

## Permissions / auto-approve

[.claude/settings.local.json](.claude/settings.local.json) is set to `defaultMode: "bypassPermissions"` with a deny list for destructive Bash patterns (`rm -rf*`, `sudo *`, `git push --force*`, `git reset --hard*`, `git clean -*f*`, `mkfs*`, `dd if=*`, `shutdown`, `reboot`, `chmod -R 777*`, `curl|sh`, `wget|sh`). Anything not on that list runs without confirming. The file is gitignored — these are personal settings, not committed.

## Requirements files

Each project owns its own `requirements.txt`; they share the root `.venv` (pip is idempotent against it). The slicer uses `.venv311` instead.

- [tools/requirements.txt](tools/requirements.txt) — Gmail MCP only
- [Projects/watchdog/requirements.txt](Projects/watchdog/requirements.txt) — Watchdog only

## GitHub Actions

Workflows must stay in `.github/workflows/` at the repo root (GitHub only discovers them there). They point into `Projects/` via `working-directory`:

- `daily_report.yml` → `Projects/watchdog` (the report artifact is written to `$GITHUB_WORKSPACE/report_artifacts/`, i.e. the repo root, not the project folder).
- `tennis_bot.yml` → `Projects/tennis-value-bot`, commits `Projects/tennis-value-bot/data/ledger.db`.

If a project folder moves, update both its `working-directory` and any repo-relative paths in the workflow.
