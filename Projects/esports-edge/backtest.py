"""Phase 2b: calibrated model vs historical Polymarket pre-match prices.

For every resolved match-winner market with a pre-match price snapshot:
model probability is computed from OpenDota matches that FINISHED BEFORE
the market's game start (single chronological walk — no leakage), mapped
per-game -> match by best-of. Then both the model and the market are scored
against the actual resolution, and a paper edge-rule is simulated.

    python backtest.py --min-edge 0.05 --slippage 0.02
"""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"
PLATT_A, PLATT_B = 0.520, 0.006
MIN_HISTORY = 10


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def match_prob(p: float, best_of: int) -> float:
    if best_of == 1:
        return p
    if best_of == 2:
        # BO2 winner markets void on a 1-1 draw, so the traded probability is
        # conditional on a sweep: P(2-0 | someone sweeps)
        return p * p / (p * p + (1 - p) ** 2)
    if best_of == 3:
        return p * p * (3 - 2 * p)
    if best_of == 5:
        return p ** 3 * (10 - 15 * p + 6 * p * p)
    raise ValueError(best_of)


def calibrated_game_p(elo_a: float, elo_b: float) -> float:
    raw = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400))
    z = PLATT_A * math.log(raw / (1 - raw)) + PLATT_B
    return 1 / (1 + math.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--slippage", type=float, default=0.02)
    ap.add_argument("--min-volume", type=float, default=0.0)
    args = ap.parse_args()

    mcon = sqlite3.connect(DATA / "matches.db")
    mcon.row_factory = sqlite3.Row
    games = mcon.execute(
        """select start_time ts, radiant_team_id a, dire_team_id b, radiant_win y,
                  radiant_name an, dire_name bn from matches
           where radiant_win is not null and radiant_team_id is not null
             and dire_team_id is not null and radiant_team_id != 0 and dire_team_id != 0
           order by start_time, match_id""").fetchall()

    pcon = sqlite3.connect(DATA / "polymarket.db")
    pcon.row_factory = sqlite3.Row
    mkts = pcon.execute(
        """select * from pm_markets
           where snap_price_a is not null and best_of is not null and volume >= ?
           order by game_start_ts""", (args.min_volume,)).fetchall()

    R = defaultdict(lambda: 1500.0)
    G = defaultdict(int)
    by_name: dict[str, int] = {}

    def team_id(name):
        q = norm(name)
        if q in by_name:
            return by_name[q]
        subs = {tid for n, tid in by_name.items() if q and (q in n or n in q)}
        return subs.pop() if len(subs) == 1 else None

    gi = 0
    rows = []
    skip = defaultdict(int)
    for mk in mkts:
        while gi < len(games) and games[gi]["ts"] < mk["game_start_ts"]:
            g = games[gi]
            p = 1 / (1 + 10 ** (-((R[g["a"]] + 10.9) - R[g["b"]]) / 400))
            d = 32 * (g["y"] - p)
            R[g["a"]] += d; R[g["b"]] -= d; G[g["a"]] += 1; G[g["b"]] += 1
            if g["an"]:
                by_name[norm(g["an"])] = g["a"]
            if g["bn"]:
                by_name[norm(g["bn"])] = g["b"]
            gi += 1
        ta, tb = team_id(mk["team_a"]), team_id(mk["team_b"])
        if ta is None or tb is None:
            skip["team unmatched"] += 1
            continue
        if G[ta] < MIN_HISTORY or G[tb] < MIN_HISTORY:
            skip[f"history<{MIN_HISTORY}"] += 1
            continue
        if not (0.02 <= mk["snap_price_a"] <= 0.98):
            skip["snapshot at extreme"] += 1
            continue
        p_model = match_prob(calibrated_game_p(R[ta], R[tb]), mk["best_of"])
        rows.append(dict(p_model=p_model, p_mkt=mk["snap_price_a"],
                         y=1 if mk["winner"] == "a" else 0,
                         vol=mk["volume"], slug=mk["event_slug"],
                         team_a=mk["team_a"], team_b=mk["team_b"]))

    print(f"markets with snapshot: {len(mkts)}; scored: {len(rows)}; "
          f"skipped: {dict(skip)}")
    if not rows:
        return

    bs_model = sum((r["p_model"] - r["y"]) ** 2 for r in rows) / len(rows)
    bs_mkt = sum((r["p_mkt"] - r["y"]) ** 2 for r in rows) / len(rows)
    acc_model = sum(((r["p_model"] >= .5) == bool(r["y"])) for r in rows) / len(rows)
    acc_mkt = sum(((r["p_mkt"] >= .5) == bool(r["y"])) for r in rows) / len(rows)
    print(f"\n{'':16s}{'model':>8s}{'market':>9s}")
    print(f"{'Brier':16s}{bs_model:8.4f}{bs_mkt:9.4f}   (lower wins)")
    print(f"{'accuracy':16s}{acc_model:8.3f}{acc_mkt:9.3f}")

    # market calibration: is the thin market itself honest?
    bins = defaultdict(list)
    for r in rows:
        bins[min(int(r["p_mkt"] * 10), 9)].append(r)
    print("\nmarket calibration:")
    for b in sorted(bins):
        ch = bins[b]
        mp = sum(r["p_mkt"] for r in ch) / len(ch)
        ay = sum(r["y"] for r in ch) / len(ch)
        print(f"  [{b/10:.1f},{b/10+.1:.1f})  n={len(ch):4d}  mkt_says={mp:.3f}  actual={ay:.3f}  gap={ay-mp:+.3f}")

    # paper edge simulation
    bets, pnl = [], 0.0
    for r in rows:
        gap = r["p_model"] - r["p_mkt"]
        if abs(gap) < args.min_edge:
            continue
        if gap > 0:   # buy side A
            q = min(r["p_mkt"] + args.slippage, 0.99)
            win = r["y"] == 1
            fair = r["p_model"]
        else:         # buy side B
            q = min(1 - r["p_mkt"] + args.slippage, 0.99)
            win = r["y"] == 0
            fair = 1 - r["p_model"]
        if not (0.10 <= q <= 0.90):
            continue
        ret = (1 - q) / q if win else -1.0
        bets.append((abs(gap), q, win, ret))
        pnl += ret
    print(f"\npaper edge rule (|gap|>={args.min_edge}, slippage {args.slippage}, "
          f"price band [0.10,0.90]):")
    if bets:
        wins = sum(1 for *_, w, _r in [(g, q, w, r) for g, q, w, r in bets] if w)
        print(f"  bets: {len(bets)}  wins: {wins}  "
              f"P&L: {pnl:+.2f} units on {len(bets)} staked  ROI: {pnl/len(bets)*100:+.1f}%")
        for lo, hi in ((0.05, 0.08), (0.08, 0.12), (0.12, 1.0)):
            sub = [b for b in bets if lo <= b[0] < hi]
            if sub:
                sp = sum(b[3] for b in sub)
                print(f"    gap [{lo:.2f},{hi:.2f}): n={len(sub):4d} pnl {sp:+.2f} "
                      f"roi {sp/len(sub)*100:+.1f}%")
    else:
        print("  no qualifying bets")


if __name__ == "__main__":
    main()
