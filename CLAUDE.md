# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

`c:\dev\` is a workspace, not a single project. It hosts unrelated Python projects that share a single `.venv` and a single git repo:

| Path | What | Owns its own CLAUDE.md? |
|---|---|---|
| [watchdog/](watchdog/) | Governance Watchdog — Hebrew-language news classifier and daily email digest. The user's primary project. | Yes — [watchdog/CLAUDE.md](watchdog/CLAUDE.md) |
| [tagah_physics/](tagah_physics/) | TAGAH physics — physics problem-set generator. Greenfield, no implementation yet. | Yes — [tagah_physics/CLAUDE.md](tagah_physics/CLAUDE.md) |
| [gmail_mcp_server.py](gmail_mcp_server.py) + [.mcp.json](.mcp.json) | Read-only Gmail MCP server registered with Claude Code via project-scope `.mcp.json`. Pure tooling — gives Claude `search_messages`, `get_message`, `get_profile`, `list_labels` against the user's Gmail. | No — see this file |
| [esports-edge/](esports-edge/) | Dota 2 probability model for Polymarket tier-2 markets. Research phase: pipeline + calibrated baseline only, NO trading code until each phase passes human review (build order in its README). | No — see [esports-edge/README.md](esports-edge/README.md) |

When the user asks about "the watchdog" or "the bot," they mean the project under `watchdog/`. When they ask about Gmail tools or `mcp__gmail-readonly__*`, they mean the MCP server at the root.

## Commands (from workspace root)

```powershell
.\.venv\Scripts\Activate.ps1

# Watchdog
cd watchdog
pip install -r requirements.txt
python main.py
cd ..

# Gmail MCP — re-auth (only needed if token.json is deleted or expired)
pip install -r requirements.txt
python gmail_mcp_server.py auth
```

There is no test framework, linter, or build step in either project.

## Gmail MCP server

`gmail_mcp_server.py` exposes four read-only tools to Claude Code via [.mcp.json](.mcp.json). It uses an OAuth client under the user's personal Google Cloud project `gmail-mcp` (account `tiulhilufim@gmail.com`). Secrets live outside the repo at `~/.gmail-mcp-readonly/` (never commit those):

- `~/.gmail-mcp-readonly/credentials.json` — OAuth client (download from Google Cloud Console)
- `~/.gmail-mcp-readonly/token.json` — refresh token (auto-created on first auth, auto-renews)

The OAuth scope is hardcoded to `https://www.googleapis.com/auth/gmail.readonly`. The server cannot send, modify, or delete mail even if asked — Google rejects writes at the API layer.

To switch to a different Gmail account: delete `~/.gmail-mcp-readonly/token.json`, then run `python gmail_mcp_server.py auth` again. To revoke entirely: https://myaccount.google.com/permissions → `gmail-mcp` → Remove access.

Auto-approve for `mcp__gmail-readonly__*` is set in [.claude/settings.local.json](.claude/settings.local.json), so tool calls run without permission prompts.

## Permissions / auto-approve

[.claude/settings.local.json](.claude/settings.local.json) is set to `defaultMode: "bypassPermissions"` with a deny list for destructive Bash patterns (`rm -rf*`, `sudo *`, `git push --force*`, `git reset --hard*`, `git clean -*f*`, `mkfs*`, `dd if=*`, `shutdown`, `reboot`, `chmod -R 777*`, `curl|sh`, `wget|sh`). Anything not on that list runs without confirming. The file is gitignored — these are personal settings, not committed.

## Requirements files

Each project owns its own `requirements.txt`. They share several `google-*` packages, which is fine because pip is idempotent against the shared `.venv`.

- [requirements.txt](requirements.txt) — Gmail MCP only (`mcp`, `google-auth`, `google-auth-oauthlib`, `google-api-python-client`)
- [watchdog/requirements.txt](watchdog/requirements.txt) — Watchdog only (`feedparser`, `google-generativeai`, `requests`, `beautifulsoup4`, `google-auth*`)

## GitHub Actions

`.github/workflows/daily_report.yml` runs the watchdog from inside the `watchdog/` directory (via `working-directory: watchdog`) and installs deps from `watchdog/requirements.txt`. It must remain at the workspace root because GitHub Actions only discovers workflows under `.github/workflows/` at the repo root.
