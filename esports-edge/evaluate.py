"""Phase 1b: chronological Elo baseline + calibration report. No trading.

Walks all pro matches in time order, maintaining team Elo ratings. Every
prediction is made BEFORE the match updates the ratings, so the walk itself
is leak-free. Metrics are reported only on the chronological test tail, and
only for matches where both teams have enough rating history to be
meaningful.

Baselines the model must beat (spec: "baseline first"):
  B0  always predict Radiant win at the train-period base rate
  B1  always pick the Elo favorite (accuracy comparison)

    python evaluate.py --test-frac 0.2 --min-history 10
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "matches.db"

ELO_START = 1500.0
ELO_K = 32.0


def elo_expected(r_radiant: float, r_dire: float, side_offset: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((r_radiant + side_offset) - r_dire) / 400.0))


def brier(preds: list[tuple[float, int]]) -> float:
    return sum((p - y) ** 2 for p, y in preds) / len(preds)


def log_loss(preds: list[tuple[float, int]]) -> float:
    eps = 1e-12
    return -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                for p, y in preds) / len(preds)


def load_matches(con, tiers: set[str] | None):
    q = """select m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                  m.radiant_win, coalesce(l.tier, 'unknown') tier
           from matches m left join leagues l on l.leagueid = m.leagueid
           where m.radiant_win is not null
             and m.radiant_team_id is not null and m.dire_team_id is not null
             and m.radiant_team_id != 0 and m.dire_team_id != 0
           order by m.start_time, m.match_id"""
    rows = con.execute(q).fetchall()
    if tiers:
        rows = [r for r in rows if r["tier"] in tiers]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="chronological tail used for evaluation")
    ap.add_argument("--min-history", type=int, default=10,
                    help="min prior matches per team for a scored prediction")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="league tiers to include (e.g. premium professional); default all")
    ap.add_argument("--plot", default="calibration.png")
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = load_matches(con, set(args.tiers) if args.tiers else None)
    if len(rows) < 1000:
        raise SystemExit(f"only {len(rows)} usable matches — ingest more first")
    split = int(len(rows) * (1 - args.test_frac))
    train, test_start_ts = rows[:split], rows[split]["start_time"]

    # side advantage estimated on TRAIN only, expressed as an Elo offset
    radiant_rate = sum(r["radiant_win"] for r in train) / len(train)
    side_offset = 400.0 * math.log10(radiant_rate / (1.0 - radiant_rate))

    ratings: dict[int, float] = defaultdict(lambda: ELO_START)
    games: dict[int, int] = defaultdict(int)
    preds: list[tuple[float, int]] = []          # (p_radiant, radiant_win) on test
    fav_hits = base_hits = scored = skipped_history = 0

    for i, m in enumerate(rows):
        ra, rd = ratings[m["radiant_team_id"]], ratings[m["dire_team_id"]]
        p = elo_expected(ra, rd, side_offset)
        y = m["radiant_win"]

        in_test = i >= split
        if in_test:
            if (games[m["radiant_team_id"]] >= args.min_history
                    and games[m["dire_team_id"]] >= args.min_history):
                preds.append((p, y))
                scored += 1
                fav_hits += int((p >= 0.5) == bool(y))
                base_hits += int(y)  # B0 predicts radiant every time
            else:
                skipped_history += 1

        # update AFTER predicting
        delta = ELO_K * (y - p)
        ratings[m["radiant_team_id"]] += delta
        ratings[m["dire_team_id"]] -= delta
        games[m["radiant_team_id"]] += 1
        games[m["dire_team_id"]] += 1

    import time as _t
    print(f"matches: {len(rows)} total, train {split}, test {len(rows)-split} "
          f"(test starts {_t.strftime('%Y-%m-%d', _t.gmtime(test_start_ts))})")
    print(f"scored on test: {scored} (skipped {skipped_history} for <{args.min_history} "
          f"matches of team history)")
    print(f"train radiant win rate: {radiant_rate:.3f} -> side offset {side_offset:+.1f} Elo")
    print()
    print(f"B0 always-radiant accuracy:     {base_hits/scored:.3f}")
    print(f"B1 Elo-favorite accuracy:       {fav_hits/scored:.3f}")
    print(f"Elo probability Brier:          {brier(preds):.4f}")
    const = [(radiant_rate, y) for _, y in preds]
    print(f"  vs constant-rate Brier:       {brier(const):.4f}  (must be lower)")
    print(f"Elo probability log-loss:       {log_loss(preds):.4f}")

    # calibration table + plot
    bins = defaultdict(list)
    for p, y in preds:
        bins[min(int(p * 10), 9)].append((p, y))
    print("\ncalibration (test):")
    print("  bin        n    mean_pred  actual")
    xs, ys, ns = [], [], []
    for b in sorted(bins):
        chunk = bins[b]
        mp = sum(p for p, _ in chunk) / len(chunk)
        ay = sum(y for _, y in chunk) / len(chunk)
        xs.append(mp); ys.append(ay); ns.append(len(chunk))
        print(f"  [{b/10:.1f},{b/10+.1:.1f})  {len(chunk):5d}   {mp:.3f}      {ay:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
        ax.scatter(xs, ys, s=[max(10, n / max(ns) * 200) for n in ns], zorder=3)
        ax.set_xlabel("predicted P(radiant win)")
        ax.set_ylabel("actual radiant win rate")
        ax.set_title(f"Elo calibration — test n={scored}, Brier={brier(preds):.4f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(__file__).parent / args.plot, dpi=120)
        print(f"\ncalibration plot -> {args.plot}")
    except ImportError:
        print("\n(matplotlib not installed — skipped plot)")


if __name__ == "__main__":
    main()
