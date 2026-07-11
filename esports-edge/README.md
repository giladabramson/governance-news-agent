# esports-edge — Dota 2 probability model (research phase)

Learning project: can a calibrated probability model find real edge in
Polymarket's tier-2/3 Dota 2 match markets? Success = "a calibrated model
that beats the naive baselines," **not** ROI.

## Build order (do NOT build backwards)

1. **Data pipeline + baseline model — THIS IS WHERE WE ARE.** No trading code.
2. Backtest vs historical Polymarket prices — does any edge exist at all?
3. Live paper trading against real CLOB **ask** prices (bid shows fake profit).
4. Only then — small real stakes ($5–50). Legal status from Israel is gray;
   phases 1–3 sidestep it entirely.

Each phase stops for human review before the next begins.

## Why Dota 2 (decided 2026-07-11)

- OpenDota: free structured API, years of pro-match history, no scraping.
  HLTV (CS2) 403s bots; vlr.gg (Valorant) is HTML-parsing.
- Polymarket lists ~13 upcoming Dota events at a time (median vol ~$900,
  max ~$16k) — thin enough that no sharp anchor has compressed the price,
  liquid enough to eventually matter.
- Roster/stand-in news (the exploitable unstructured info) lives on
  Liquipedia, which has a sanctioned API for later feature work.

## Run

```powershell
pip install -r requirements.txt
python ingest.py --pages 400        # ~40k pro matches back in time, resumable
python evaluate.py                  # chronological Elo baseline + calibration
```

`data/matches.db` is created by ingest and gitignored — re-run to rebuild.

## Model principles (fixed, from the project owner)

- Baseline first: the model must beat "always pick the favorite."
- Chronological validation only — never random shuffles.
- No feature that updates after the match (leakage).
- One feature at a time; drop what doesn't help.
- A simple model we understand > a sophisticated one we don't.
- Output must be a **calibrated probability** (calibration plot + Brier),
  because the strategy harvests the model-vs-price gap: an overconfident
  model manufactures phantom edge out of its own errors.
- No positions above 0.90 / below 0.10 — calibration is weakest there and
  spread/fees bite hardest.

## Status

- [x] Phase 1a: OpenDota ingest (matches + league tiers) → SQLite
- [x] Phase 1b: chronological Elo + baselines + calibration report
- [x] Calibration layer (Platt, a=0.520 — raw Elo was ~2x overconfident)
- [x] Phase 2: backtest vs 1,144 historical Polymarket pre-match prices
- [ ] REVIEW CHECKPOINT — phase-2 verdict (2026-07-11):

**Phase-2 result: the market beats the model.** Market Brier 0.2047 vs
model 0.2165; market accuracy 68.4% vs 65.1%. The thin market is also
decently calibrated (bins within ~5pts, mild favorite-longshot bias — 
longshots overpriced, heavy favorites slightly underpriced). The paper
edge rule (|gap|>=5pts, 2c slippage) made 762 bets for **-0.4% ROI**, with
no monotonic gap->profit relationship — i.e. the model's disagreements
with the market are model error, not market error. Elo-only has NO edge
here. Any continuation must first close a ~1.2-Brier-point gap to the
market (features: form, rust, lineups) before the edge rule is worth
re-testing.
