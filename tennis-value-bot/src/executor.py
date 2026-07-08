"""Paper executor: simulates GTC limit orders against real book snapshots.

Fill model (conservative, spec §9):
- On placement: if the current best ask <= our limit, we'd have crossed — fill
  as TAKER at the ask (pay taker fee + gas in the recorded entry price).
- Otherwise the order rests. On each later snapshot, if best ask <= limit, we
  assume a MAKER fill at OUR limit price (no fee). This still overstates fill
  quality slightly (queue priority is invisible), noted in the README.
- All resting orders are cancelled at T-minus min_minutes_to_start.

No orders are ever sent anywhere. There is no live path in this module.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .edge_engine import Signal, taker_fee_per_share

log = logging.getLogger(__name__)


class PaperExecutor:
    def __init__(self, ledger, cfg: dict):
        self.ledger = ledger
        self.cfg = cfg

    def submit(self, mkt, sig: Signal) -> None:
        """Place a simulated GTC limit buy for `sig` on market `mkt`."""
        order_id = self.ledger.add_order(mkt.slug, sig)
        if sig.ask <= sig.limit_price:  # crosses immediately -> taker fill
            fee = taker_fee_per_share(sig.ask, self.cfg)
            entry = sig.ask + fee + self.cfg["gas_cost_usd"] / sig.stake
            self.ledger.fill_order(order_id, entry, "taker")
            self._open(mkt, sig, entry)
            log.info("TAKER fill %s %s @ %.3f (eff %.3f) $%.2f",
                     mkt.slug, sig.player, sig.ask, entry, sig.stake)
        else:
            log.info("resting order %s %s limit %.3f $%.2f",
                     mkt.slug, sig.player, sig.limit_price, sig.stake)

    def check_resting(self, books_by_slug: dict) -> None:
        """Fill or keep resting orders based on fresh book snapshots."""
        for od in self.ledger.resting_orders():
            books = books_by_slug.get(od["poly_slug"])
            if not books:
                continue
            bk = books[od["side"]]
            if bk.best_ask <= od["limit_price"]:
                self.ledger.fill_order(od["id"], od["limit_price"], "maker")
                mkt = books["market"]
                sig = Signal(od["side"], od["player"], 0, 0, 0, 0,
                             od["stake_usd"], od["limit_price"], "bet", "maker fill")
                self._open(mkt, sig, od["limit_price"])
                log.info("MAKER fill %s %s @ %.3f", od["poly_slug"], od["player"],
                         od["limit_price"])

    def cancel_near_start(self, markets_by_slug: dict) -> None:
        now = datetime.now(timezone.utc)
        for od in self.ledger.resting_orders():
            mkt = markets_by_slug.get(od["poly_slug"])
            if mkt is None:
                self.ledger.cancel_order(od["id"])
                continue
            mins = (mkt.start_time - now).total_seconds() / 60
            if mins < self.cfg["min_minutes_to_start"]:
                self.ledger.cancel_order(od["id"])
                log.info("cancelled resting order on %s (T-%.0f min)", od["poly_slug"], mins)

    def _open(self, mkt, sig: Signal, entry_price: float) -> None:
        self.ledger.open_position(mkt.slug, sig.side, sig.player,
                                  getattr(mkt, "tournament_key", mkt.tour),
                                  entry_price, sig.stake)

    def settle_resolved(self, refresh_fn) -> None:
        """Settle open positions whose markets have resolved."""
        for pos in self.ledger.open_positions():
            mkt = refresh_fn(pos["poly_slug"])
            if mkt is None or not mkt.resolved:
                continue
            if mkt.winner is None:
                self.ledger.settle(pos["id"], "void", 0.0)
                continue
            won = (mkt.winner == pos["side"])
            pnl = pos["shares"] * 1.0 - pos["stake_usd"] if won else -pos["stake_usd"]
            self.ledger.settle(pos["id"], "won" if won else "lost", pnl)
            log.info("settled %s %s: %s pnl $%.2f", pos["poly_slug"], pos["player"],
                     "WON" if won else "lost", pnl)
