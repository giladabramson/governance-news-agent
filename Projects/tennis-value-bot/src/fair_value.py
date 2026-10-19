"""Fair value from sharp-book odds: two-way proportional devig + sanity gates.

The math here is deliberately tiny and unit-tested — a sign error in devig
will happily lose (paper) money with perfect uptime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FairValue:
    fair_a: float
    fair_b: float
    overround: float


def devig_two_way(odds_a: float, odds_b: float) -> FairValue:
    """Proportional (multiplicative) devig of a two-outcome market."""
    if odds_a <= 1.0 or odds_b <= 1.0:
        raise ValueError(f"decimal odds must be > 1 (got {odds_a}, {odds_b})")
    ia, ib = 1.0 / odds_a, 1.0 / odds_b
    total = ia + ib
    return FairValue(fair_a=ia / total, fair_b=ib / total, overround=total)


def quote_ok(fv: FairValue, quote_age_sec: float, prev_fair_a: float | None,
             cfg: dict) -> tuple[bool, str]:
    """Sanity gates from spec §3. Returns (ok, reason_if_not)."""
    if fv.overround > cfg["max_overround"]:
        return False, f"overround {fv.overround:.3f} > {cfg['max_overround']} (stale/defensive line)"
    if fv.overround < 1.0:
        return False, f"overround {fv.overround:.3f} < 1 (arb/garbage quote)"
    if quote_age_sec > cfg["max_quote_age_sec"]:
        return False, f"quote age {quote_age_sec:.0f}s > {cfg['max_quote_age_sec']}s"
    if prev_fair_a is not None and abs(fv.fair_a - prev_fair_a) > cfg["max_prob_jump"]:
        return False, (f"fair prob jumped {abs(fv.fair_a - prev_fair_a):.3f} "
                       f"> {cfg['max_prob_jump']} since last pull (news?)")
    return True, ""
