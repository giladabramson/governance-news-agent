"""Main polling loop (paper mode).

Two cadences share one loop:
- every market_poll_interval_sec: Polymarket discovery/books (free), resting-
  order fills, near-start cancels, settlement of resolved markets.
- every odds_poll_interval_sec: reference-odds refresh (costs credits), fair
  value, edge evaluation, decisions. Plus a one-off closing pull per match
  just before start for CLV.

    python -m src.main            # runs until Ctrl-C or BOT_HALT=1
    python -m src.main --once     # single cycle (for testing)
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from . import markets
from .config import ALIAS_PATH, DB_PATH, arm_configs, halted, load_config
from .edge_engine import evaluate
from .executor import PaperExecutor
from .fair_value import devig_two_way, quote_ok
from .ledger import Ledger
from .matching import load_aliases, pair
from .odds import make_provider

log = logging.getLogger(__name__)


def run(once: bool = False) -> None:
    cfg = load_config()
    ledger = Ledger(DB_PATH)
    provider = make_provider(cfg)
    arms = arm_configs(cfg)
    executors = {name: PaperExecutor(ledger, acfg, arm=name)
                 for name, acfg in arms.items()}
    aliases = load_aliases(ALIAS_PATH)

    while True:
        if halted():
            log.warning("BOT_HALT=1 — exiting")
            return
        # per-arm circuit breaker: a tripped arm stops betting, others continue
        live_arms = {}
        for name, acfg in arms.items():
            pnl = ledger.daily_realized_pnl(arm=name)
            if pnl <= -acfg["daily_loss_halt"]:
                log.error("[%s] daily loss circuit breaker hit (%.2f) — arm halted", name, pnl)
            else:
                live_arms[name] = acfg
        if not live_arms:
            log.error("all arms halted — exiting")
            return
        cycle_start = time.time()
        # odds timing lives in the ledger so scheduled one-shot runs (GitHub
        # Actions) don't re-pull odds every run and burn the monthly credits
        last_odds_pull = float(ledger.get_meta("last_odds_pull") or 0.0)
        try:
            _cycle(cfg, ledger, provider, executors, live_arms, aliases,
                   odds_due=(time.time() - last_odds_pull >= cfg["odds_poll_interval_sec"]))
        except Exception:
            log.exception("cycle failed; continuing")
        if once:
            return
        time.sleep(max(1.0, cfg["market_poll_interval_sec"] - (time.time() - cycle_start)))


def _cycle(cfg, ledger, provider, executors, live_arms, aliases, odds_due: bool) -> None:
    now = datetime.now(timezone.utc)

    # 1. Polymarket side (free): discovery + books
    all_matches = markets.discover_matches()
    upcoming = [m for m in all_matches
                if 0 < (m.start_time - now).total_seconds() < 48 * 3600]
    log.info("%d tennis match markets, %d upcoming <48h", len(all_matches), len(upcoming))

    # persist book rows only for markets we track (quoted / held / resting) so
    # the ledger stays small enough to commit from a scheduled runner
    tracked = ledger.quoted_slugs() | {o["poly_slug"] for o in ledger.resting_orders()}
    books_by_slug: dict = {}
    for m in upcoming:
        books_by_slug[m.slug] = {"a": markets.book_snapshot(m.token_a),
                                 "b": markets.book_snapshot(m.token_b),
                                 "market": m}
        if m.slug in tracked or ledger.has_position(m.slug):
            for side in ("a", "b"):
                ledger.add_book(m.slug, side, books_by_slug[m.slug][side])

    for ex in executors.values():
        ex.check_resting(books_by_slug)
        ex.cancel_near_start({m.slug: m for m in all_matches})
    next(iter(executors.values())).settle_resolved(markets.refresh_resolution)  # all arms

    # 2. reference odds (spends credits) — regular pull, or forced closing pull
    need_closing = [m for m in upcoming
                    if not ledger.closing_pulled(m.slug)
                    and (m.start_time - now).total_seconds() / 60 <= cfg["closing_pull_minutes"] + 5
                    and ledger.has_position(m.slug)]
    if not odds_due and not need_closing:
        return
    if provider.credits_remaining < cfg["min_credits_reserve"]:
        log.warning("credits below reserve (%s) — skipping odds pull",
                    provider.credits_remaining)
        return

    allow = cfg["tournament_allowlist"]
    keys = allow or provider.active_tournaments()
    quotes = []
    for key in keys:
        quotes.extend(provider.quotes(key))
    ledger.set_meta("last_odds_pull", str(time.time()))

    pairings, skips = pair(quotes, upcoming, aliases)
    for s in skips:
        log.warning("matching: %s", s)
    log.info("paired %d reference quotes to Polymarket markets", len(pairings))

    for p in pairings:
        q, mkt = p.ref_quote, p.market
        odds_a, odds_b = (q.odds_a, q.odds_b) if p.aligned else (q.odds_b, q.odds_a)
        fv = devig_two_way(odds_a, odds_b)
        mins_to_start = (mkt.start_time - now).total_seconds() / 60
        is_closing = mins_to_start <= cfg["closing_pull_minutes"] + 5

        ok, why = quote_ok(fv, time.time() - q.pulled_at, ledger.last_fair_a(mkt.slug), cfg)
        ledger.upsert_event(mkt.slug, q.source_event_id, q.tournament_key,
                            mkt.player_a, mkt.player_b, mkt.start_time.isoformat())
        ledger.add_quote(mkt.slug, "pinnacle", odds_a, odds_b, fv, is_closing)
        if is_closing:
            for side, fair in (("a", fv.fair_a), ("b", fv.fair_b)):
                ledger.set_closing_fair(mkt.slug, side, fair)
        if not ok:
            log.info("quote gate failed for %s: %s", mkt.slug, why)
            continue

        books = books_by_slug.get(mkt.slug)
        if not books:
            continue
        for side, player, fair in (("a", mkt.player_a, fv.fair_a),
                                   ("b", mkt.player_b, fv.fair_b)):
            bk = books[side]
            for arm, acfg in live_arms.items():
                sig = evaluate(
                    side=side, player=player, fair=fair, asks=bk.asks,
                    depth_2c_usd=bk.depth_2c_usd, market_volume_usd=mkt.volume_usd,
                    minutes_to_start=mins_to_start,
                    has_position=ledger.has_position(mkt.slug, arm=arm),
                    daily_new_exposure=ledger.daily_new_exposure(arm=arm),
                    open_exposure=ledger.open_exposure(arm=arm),
                    tournament_positions_today=ledger.tournament_positions_today(
                        q.tournament_key, arm=arm),
                    cfg=acfg)
                ledger.add_decision(mkt.slug, sig, arm=arm)
                if sig.action == "bet":
                    log.info("[%s] SIGNAL %s %s fair %.3f ask %.3f net_edge %+.3f stake $%.2f",
                             arm, mkt.slug, player, fair, sig.ask, sig.net_edge, sig.stake)
                    executors[arm].submit(mkt, sig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(once=args.once)


if __name__ == "__main__":
    main()
