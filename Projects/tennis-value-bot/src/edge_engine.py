"""Edge computation, trade filters, and fractional-Kelly sizing (spec §4–§5).

Cost model (adapted to Polymarket's CURRENT fee schedule, which postdates the
spec): sports markets charge a TAKER fee of taker_fee_rate * p * (1-p) per
share; maker fills are free. The paper executor posts GTC limit orders (maker
path), but edges are evaluated against the taker cost so that a signal is only
taken when it would survive even crossing the spread.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    side: str            # 'a' or 'b'
    player: str
    fair: float
    ask: float           # volume-weighted fill price for the intended stake
    gross_edge: float
    net_edge: float
    stake: float
    limit_price: float
    action: str          # 'bet' | 'skip'
    reason: str


def taker_fee_per_share(p: float, cfg: dict) -> float:
    return cfg["taker_fee_rate"] * p * (1.0 - p)


def vwap_fill_price(asks: list[tuple[float, float]], stake_usd: float) -> tuple[float, float]:
    """Walk the ask ladder [(price, size_shares), ...] best-first; return
    (vwap_price, fillable_usd) for spending up to stake_usd."""
    remaining = stake_usd
    cost = shares = 0.0
    for price, size in asks:
        usd_here = min(remaining, price * size)
        if usd_here <= 0:
            break
        cost += usd_here
        shares += usd_here / price
        remaining -= usd_here
        if remaining <= 1e-9:
            break
    if shares == 0:
        return 0.0, 0.0
    return cost / shares, cost


def kelly_stake(fair: float, price: float, cfg: dict) -> float:
    """Fractional Kelly on a binary token bought at `price` (spec §5)."""
    b = (1.0 - price) / price
    kelly = (fair * b - (1.0 - fair)) / b
    stake = cfg["bankroll_usd"] * kelly * cfg["kelly_fraction"]
    return max(0.0, min(stake, cfg["max_stake"]))


def evaluate(side: str, player: str, fair: float,
             asks: list[tuple[float, float]], depth_2c_usd: float,
             market_volume_usd: float, minutes_to_start: float,
             has_position: bool, daily_new_exposure: float,
             open_exposure: float, tournament_positions_today: int,
             cfg: dict) -> Signal:
    """Run every filter from spec §4; returns a bet or an explained skip."""
    lo, hi = cfg["price_band"]

    def skip(reason: str, ask: float = 0.0, gross: float = 0.0, net: float = 0.0,
             stake: float = 0.0) -> Signal:
        return Signal(side, player, fair, ask, gross, net, stake, 0.0, "skip", reason)

    if not asks:
        return skip("empty ask ladder")
    best_ask = asks[0][0]
    if not (lo <= best_ask <= hi):
        return skip(f"ask {best_ask:.2f} outside band [{lo},{hi}]", best_ask)
    if market_volume_usd < cfg["min_volume_usd"]:
        return skip(f"volume ${market_volume_usd:,.0f} < ${cfg['min_volume_usd']:,}", best_ask)
    if minutes_to_start < cfg["min_minutes_to_start"]:
        return skip(f"{minutes_to_start:.0f} min to start < {cfg['min_minutes_to_start']}", best_ask)
    if has_position:
        return skip("existing position in this match", best_ask)

    # provisional stake from best ask, then re-derive fill price at that size
    stake = kelly_stake(fair, best_ask, cfg)
    if stake < cfg["min_stake"]:
        return skip(f"kelly stake ${stake:.2f} < min ${cfg['min_stake']}", best_ask)
    ask, fillable = vwap_fill_price(asks, stake)
    if fillable < stake * 0.99:
        return skip(f"book can only fill ${fillable:.2f} of ${stake:.2f}", best_ask)

    gross = fair - ask
    fee = taker_fee_per_share(ask, cfg)
    net = gross - cfg["buffer"] - fee - cfg["gas_cost_usd"] / stake
    if net < cfg["min_edge"]:
        return skip(f"net edge {net:+.3f} < {cfg['min_edge']}", ask, gross, net, stake)
    if depth_2c_usd < cfg["depth_multiple"] * stake:
        return skip(f"depth ${depth_2c_usd:.0f} < {cfg['depth_multiple']}x stake", ask, gross, net, stake)
    if daily_new_exposure + stake > cfg["max_daily_new_exposure"]:
        return skip("daily new-exposure cap", ask, gross, net, stake)
    if open_exposure + stake > cfg["max_total_exposure"]:
        return skip("total exposure cap", ask, gross, net, stake)
    if tournament_positions_today >= cfg["max_positions_per_tournament_per_day"]:
        return skip("tournament daily position cap", ask, gross, net, stake)

    limit_price = round(fair - cfg["min_edge"] - cfg["buffer"], 3)
    return Signal(side, player, fair, ask, gross, net, stake, limit_price, "bet", "all filters passed")
