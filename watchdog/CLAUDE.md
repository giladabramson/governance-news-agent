# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Governance Watchdog: a Python automation agent that ingests Hebrew-language Israeli news RSS feeds (N12/Mako, Ynet, Walla), classifies items as governance/law-enforcement-failure stories, and emails a Hebrew HTML+plain digest. Runs daily on GitHub Actions cron (07:00 UTC) and locally via `python main.py`.

This project lives in `c:\dev\watchdog\`. The parent `c:\dev\` is a workspace that also hosts an unrelated read-only Gmail MCP server for Claude Code tooling — see the root [CLAUDE.md](../CLAUDE.md). The `.venv` and `.git` are shared at the workspace level.

## Commands

Run from the `watchdog/` directory. The `.venv` is at the workspace root (`c:\dev\.venv\`).

```powershell
cd c:\dev\watchdog
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Real run — sends email; needs all required env vars
python main.py

# Dry run — fetches + classifies, no email; cap analyzed articles via DRY_RUN_MAX_ARTICLES (default 10)
$env:DRY_RUN="1"; python main.py

# Microcosm — runs three synthetic articles through heuristic + (if GEMINI_API_KEY set) AI
$env:MICROCOSM_TEST="1"; python main.py
```

There is no test framework, linter, or build step. `MICROCOSM_TEST` and `DRY_RUN` are the regression harness.

## Required environment variables

`GEMINI_API_KEY`, `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECEIVER` are required. `EMAIL_RECEIVER` may be a comma-separated list or a JSON array — see `normalize_receivers`. Optional: `SMTP_HOST` (default `smtp.gmail.com`), `SMTP_PORT` (default 587), `GEMINI_MODEL`, `MAX_AI_RETRIES`, `AI_RETRY_BASE_SECONDS`.

A single `GEMINI_AND_EMAIL_SECRETS` JSON object can supply all of the above; explicit env vars override fields in it. The GitHub Actions workflow passes both the bundle and individual secrets.

## Architecture

The whole pipeline lives in `main.py` as a flat module — no packages, no classes beyond three dataclasses (`Config`, `Article`, `AnalysisResult`). Flow inside `main()`:

1. **Ingest** (`collect_articles` → `fetch_feed_articles`): feedparser over `FEEDS`, 24h lookback (`LOOKBACK_HOURS`), dedup on `(link, title)`.
2. **Stage 1 headline prefilter** (`headline_maybe_relevant`): cheap Hebrew keyword/link-marker check on title+URL. Items that fail this stage are dropped before any AI cost.
3. **Stage 2 skim + classify** (`build_analysis_article` → `analyze_article`): fetches the article body with BeautifulSoup (`SKIM_MAX_CHARS=2500`), appends to summary, sends to Gemini with a Hebrew system prompt that returns strict JSON (`relevant`, `category`, `confidence`, `reason`).
4. **Heuristic safety net** (`heuristic_governance_classification`): keyword/link-marker scan that overrides the AI when AI says irrelevant but heuristic confidence ≥ 0.9. This is how `KNOWN_RELEVANT_LINK_MARKERS` (the Ynet regression URL) is guaranteed to pass even if Gemini drifts.
5. **Cross-article event dedup** (`deduplicate_relevant_by_event`): clusters near-duplicate articles about the same incident (different outlets reword the same story) using char-trigram Jaccard on title + first 300 chars of summary, keeping the highest-confidence representative per cluster. Threshold `EVENT_DEDUP_JACCARD_THRESHOLD = 0.17` is calibrated against real reviewer-flagged duplicates; word-level Jaccard fails because of Hebrew morphology variants (פריצה / הפריצה / פריצת).
6. **Output**: `build_email_content` produces RTL Hebrew HTML+plain; `write_report_artifacts` writes `report_artifacts/daily_report.md` and appends to `$GITHUB_STEP_SUMMARY`; `send_email` uses SMTP+STARTTLS.

### Gemini model selection

`build_gemini_model` calls `genai.list_models()`, then walks an ordered candidate list (`preferred_model` first, then `DEFAULT_MODEL_CANDIDATES`), filtering out anything matching `BLOCKED_MODEL_MARKERS` (currently `gemini-2.0-flash` — known broken for newer keys). If filtering empties the list, it falls back to the unfiltered list. `API_KEY_INVALID` and "model not found for API version" errors are non-retryable and raised as `RuntimeError`.

### Behaviors that look like bugs but aren't

- The default `GEMINI_MODEL` in `load_config` is literally `"gemini-2.0-flash"`, which is on the blocked list. This is intentional — the block list forces a fallback to a working model. Don't "fix" it by changing the default without also reviewing `BLOCKED_MODEL_MARKERS`.
- `KNOWN_RELEVANT_LINK_MARKERS` and the example URL embedded in `build_ai_prompt` are a regression guard for one specific Ynet article. Removing the marker will cause that test to regress.
- The Hebrew prompt enumerates sectors (settlers, Haredi, Bedouin, Arab society, far-left) explicitly to *prevent* sectoral bias by forcing parallel treatment, not to target them. Edit with care.
- The prompt rejects op-eds, "atmosphere" pieces, and broad statistical/trend articles even when they cover governance themes. This is a deliberate tightening from reviewer feedback (Itay Margalit, Apr 29 / May 4) — *specific* events or *specific* procedural failures only. A structural piece like "police has no procedure for handling confiscated equipment" still qualifies because it documents a concrete failure; "why is youth violence raging" does not.

## GitHub Actions

`../.github/workflows/daily_report.yml` (at the workspace root, since GH Actions requires it there) runs `python main.py` from inside `watchdog/` via `working-directory: watchdog`. Artifacts go to `report_artifacts/daily_report.md` at the workspace root because `main.py` writes to `Path(os.getenv("GITHUB_WORKSPACE", os.getcwd())) / "report_artifacts"` and GH sets `GITHUB_WORKSPACE` to the repo root. Locally, the report goes to `watchdog/report_artifacts/`. The artifact upload uses `if: always()`, so failed runs still produce a viewable summary in the Actions tab.

## Files to ignore as scaffolding

`gmail_oauth_example.py` is a standalone Gmail-API readonly snippet — not wired into `main.py`, and now superseded by the workspace-root `gmail_mcp_server.py` for Claude tooling. Keep it only if you want a minimal Gmail-API reference; otherwise it can be deleted.
