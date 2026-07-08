# Tennis Value Bot — paper-trading build

Value-betting research bot per `tennis-value-bot-spec.md`: Pinnacle no-vig
probabilities as fair value, buy Polymarket tennis match-winner tokens only
when they trade meaningfully below it. **Paper mode only** — this build
contains no order placement, no wallet code, and refuses to start if
`POLYMARKET_PRIVATE_KEY` is present in the environment. Live mode is a
separate, deliberate step gated on paper results (spec §9).

## Prior evidence (read this first)

The sibling project `c:\dev\soccer-edge` tested this exact strategy class on
2,328 small-league soccer matches: Polymarket tracked the bookmaker consensus
to within ~0.7 points and buying disagreements lost at every threshold. The
null hypothesis for tennis is "same result". The CLV ledger this bot produces
is the test. Expect skips, not signals — a 3-point net edge past Pinnacle is
rare by design.

## Setup

```powershell
cd c:\dev\tennis-value-bot
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your free key in it
```

**You need one key:** sign up free at https://the-odds-api.com (500
credits/month) and set `ODDS_API_KEY` in `.env`. Polymarket data is keyless.

## Run

```powershell
python -m src.main --once    # single cycle (sanity check)
python -m src.main           # polling loop; Ctrl-C or BOT_HALT=1 to stop
python -m src.report         # decisions, fills, CLV, P&L — broken down per arm
pytest tests/                # devig / Kelly / filter math (21 tests)
```

## Experiment arms

`config.yaml` defines `arms:` — parallel paper books evaluated against the
same market data and the same odds pulls (zero extra API credits). Each arm
inherits the top-level config plus its overrides, and keeps its own bankroll,
orders, positions, exposure caps, and daily-loss circuit breaker; ledger rows
are tagged with an `arm` column. Current experiment: `base` (control,
`min_volume_usd: 5000`) vs `loose` (`min_volume_usd: 500`) — measuring
whether the liquidity floor filters out real edges or only stale-quote
mirages. Judge `loose` primarily on CLV, not paper P&L: thin books make
simulated fills optimistic, closing-line value doesn't lie.

## Budget & polling (differs from the spec — here's why)

The spec assumed 60s odds polling; **no verified free tier survives that**
(researched 2026-07-08: The Odds API free = 500 credits/mo, 1 credit per
tournament-key pull; OddsPapi free = 250 req/mo). So:

- Polymarket books poll every 60s (free).
- Pinnacle reference polls every 90 min (`odds_poll_interval_sec`), ≈ 11
  credits/day with 2 active tournament keys.
- One extra "closing" pull fires per match you hold a position in, just
  before start — that's what CLV is computed against.
- The provider self-throttles below `min_credits_reserve` (50).

Tighten `odds_poll_interval_sec` only with a paid tier ($30/mo ≈ 4-min polls).
Note The Odds API covers Slams / ATP+WTA 1000 / 500 — Polymarket lists
qualifiers and challengers too, which simply won't pair (no reference = no bet).

## Cost model (also differs from the spec)

Polymarket introduced sports fees after the spec was written: taker pays
`0.03 · p · (1−p)` per share (≈0.75¢ max at p=0.5); makers pay zero. The edge
filter charges the taker fee + gas so a signal must survive worst-case entry;
the paper executor still prefers resting maker orders.

## Paper-fill honesty

Resting orders fill in simulation when the ask crosses the limit — real queue
priority is invisible, so paper fills are OPTIMISTIC. Positive paper CLV over
150+ signals is necessary, not sufficient. All spec §7 risk limits are
enforced in the executor: per-position/day/total caps, daily-loss halt,
`BOT_HALT=1` kill switch checked every loop.

## Layout

```
src/config.py        yaml+env, refuses non-paper mode
src/markets.py       Gamma tennis match discovery + CLOB books (keyless)
src/odds.py          OddsProvider interface + The Odds API (credit-aware)
src/matching.py      surname pairing, ±6h window, skip-on-ambiguity, aliases
src/fair_value.py    two-way proportional devig + sanity gates
src/edge_engine.py   spec §4 filters + fractional Kelly + fee model
src/ledger.py        SQLite: events/quotes/books/decisions/orders/positions
src/executor.py      paper fills (maker/taker), cancels at T-10min, settlement
src/report.py        CLV & P&L report
src/main.py          the loop
tests/               21 unit tests for the money math
data/ledger.db       created on first run (gitignored)
```

*Reality check from the spec applies: this is a learning instrument with
pocket-money downside, not an income strategy. Verify Polymarket legality in
your jurisdiction before ever considering live mode.*
