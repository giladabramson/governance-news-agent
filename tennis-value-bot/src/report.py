"""CLV / P&L report (spec §8): python -m src.report"""
from __future__ import annotations

import sqlite3

from .config import DB_PATH


def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    dec = con.execute("select action, count(*) c from decisions group by action").fetchall()
    print("=== decisions ===")
    for r in dec:
        print(f"  {r['action']}: {r['c']}")
    top_skips = con.execute(
        "select reason, count(*) c from decisions where action='skip' "
        "group by reason order by c desc limit 8").fetchall()
    for r in top_skips:
        print(f"    skip[{r['c']}]: {r['reason']}")

    orders = con.execute(
        "select status, count(*) c from orders group by status").fetchall()
    print("=== orders ===")
    for r in orders:
        print(f"  {r['status']}: {r['c']}")

    pos = con.execute("select * from positions").fetchall()
    settled = [p for p in pos if p["settled_ts"]]
    with_clv = [p for p in pos if p["clv"] is not None]
    print(f"=== positions === {len(pos)} opened, {len(settled)} settled")
    if pos:
        avg_entry = sum(p["entry_price"] for p in pos) / len(pos)
        print(f"  avg entry price: {avg_entry:.3f}")
    if with_clv:
        avg_clv = sum(p["clv"] for p in with_clv) / len(with_clv)
        pos_clv = sum(1 for p in with_clv if p["clv"] > 0)
        print(f"  CLV: avg {avg_clv:+.4f} over {len(with_clv)} positions "
              f"({pos_clv} positive) — the number that matters")
    if settled:
        pnl = sum(p["pnl"] for p in settled)
        wins = sum(1 for p in settled if p["result"] == "won")
        staked = sum(p["stake_usd"] for p in settled)
        print(f"  P&L: ${pnl:+.2f} on ${staked:.2f} staked "
              f"({wins}/{len(settled)} won, ROI {pnl/staked*100:+.1f}%)")
        print("  by odds bucket:")
        for lo, hi in ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)):
            b = [p for p in settled if lo <= p["entry_price"] < hi]
            if b:
                bp = sum(p["pnl"] for p in b)
                print(f"    [{lo},{hi}): n={len(b)} pnl ${bp:+.2f}")

    q = con.execute("select count(*) c, sum(is_closing) cl from quotes").fetchone()
    print(f"=== quotes === {q['c']} stored ({q['cl'] or 0} closing)")


if __name__ == "__main__":
    main()
