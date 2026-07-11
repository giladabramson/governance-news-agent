"""Fade-the-favorite study: ALL resolved esports match-winner markets.

Hypothesis (from calibration_map + verification): in fan-heavy esports
titles, moderate favorites are overpriced — fans overbet their team. Test:
every resolved match-winner market across every esports title, pre-match
price >= 15 min before scheduled start, then per-title calibration and a
fade-the-favorite simulation with realistic costs. Dota 2 rides along as
the control (phase 2 showed it calibrated).

    python fade_study.py             # ingest (resumable) + report
    python fade_study.py --report    # report only
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
DB = DATA / "fade_study.db"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BUFFER = 15 * 60
WINDOW_START = "2025-01-01"

TITLE_RE = re.compile(r"^([A-Za-z0-9 .\-]+?): (.+?) vs (.+?) \(BO(\d)\)")
PROP_WORDS = ("Game ", "O/U", "Handicap", "Total", "First Blood", "Winner -",
              "Map ", "Kills", "Rounds")

_SCHEMA = """
create table if not exists fade (
    market_id text primary key,
    game text, team_a text, team_b text, best_of integer,
    start_ts integer, volume real, winner text, snap_ts integer, snap_a real
);
create table if not exists fade_done (window text primary key);
"""


def month_windows(start: str):
    d = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    while d < now:
        nxt = (d.replace(year=d.year + 1, month=1) if d.month == 12
               else d.replace(month=d.month + 1))
        yield d.strftime("%Y-%m"), d.isoformat(), min(nxt, now).isoformat()
        d = nxt


def parse(event):
    for m in event.get("markets", []):
        q = m.get("question", "")
        t = TITLE_RE.match(q)
        if not t or any(w in q for w in PROP_WORDS):
            continue
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
            tokens = json.loads(m.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            continue
        if len(outcomes) != 2 or len(prices) != 2 or len(tokens) != 2:
            continue
        pa = float(prices[0])
        if round(pa) not in (0, 1) or abs(pa - round(pa)) > 0.02:
            continue
        gs = m.get("gameStartTime") or m.get("startDate") or event.get("startDate")
        try:
            start_ts = int(datetime.fromisoformat(gs.replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError):
            continue
        return dict(market_id=m["id"], game=t.group(1).strip(),
                    team_a=outcomes[0], team_b=outcomes[1],
                    best_of=int(t.group(4)), start_ts=start_ts,
                    volume=float(m.get("volumeNum") or m.get("volume") or 0),
                    winner="a" if round(pa) == 1 else "b", token=tokens[0])
    return None


def ingest():
    con = sqlite3.connect(DB)
    con.executescript(_SCHEMA)
    have = {r[0] for r in con.execute("select market_id from fade")}
    done = {r[0] for r in con.execute("select window from fade_done")}
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (research; esports fade study)"
    for label, lo, hi in month_windows(WINDOW_START):
        if label in done:
            continue
        offset = 0
        while True:
            r = s.get(f"{GAMMA}/events", params={
                "tag_slug": "esports", "closed": "true", "limit": 100,
                "offset": offset, "end_date_min": lo, "end_date_max": hi}, timeout=30)
            evs = r.json()
            if not isinstance(evs, list) or not evs:
                break
            for e in evs:
                row = parse(e)
                if row is None or row["market_id"] in have:
                    continue
                pr = s.get(f"{CLOB}/prices-history", params={
                    "market": row.pop("token"),
                    "startTs": row["start_ts"] - 14 * 86400,
                    "endTs": row["start_ts"] - BUFFER, "fidelity": 10}, timeout=30)
                pts = pr.json().get("history", []) if pr.status_code == 200 else []
                if pts and 0.02 <= float(pts[-1]["p"]) <= 0.98:
                    row["snap_ts"], row["snap_a"] = int(pts[-1]["t"]), float(pts[-1]["p"])
                else:
                    row["snap_ts"], row["snap_a"] = None, None
                con.execute("insert or ignore into fade values "
                            "(:market_id,:game,:team_a,:team_b,:best_of,"
                            ":start_ts,:volume,:winner,:snap_ts,:snap_a)", row)
                have.add(row["market_id"])
                time.sleep(0.09)
            con.commit()
            if len(evs) < 100:
                break
            offset += 100
            time.sleep(0.12)
        con.execute("insert or ignore into fade_done values (?)", (label,))
        con.commit()
        n = con.execute("select count(*), count(snap_a) from fade").fetchone()
        print(f"{label}: {n[0]} markets, {n[1]} with pre-match price", flush=True)


def report(slippage=0.02):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""select game, snap_a p, winner, start_ts, volume from fade
                          where snap_a is not null order by start_ts""").fetchall()
    # freshness guard: snapshot must be a trade from the final 24h before start
    fresh = con.execute("""select game, snap_a p, winner, start_ts, volume from fade
                           where snap_a is not null
                             and (start_ts - 900) - snap_ts <= 86400
                           order by start_ts""").fetchall()
    print(f"{len(rows)} priced markets; {len(fresh)} with fresh (<24h) pre-match price\n")
    rows = fresh

    def fav_gap(sub, lo=0.55, hi=0.80):
        """Mean (actual - price) for the favorite side, favorites in [lo,hi]."""
        pts = []
        for r in sub:
            if lo <= r["p"] <= hi:
                pts.append((r["p"], 1 if r["winner"] == "a" else 0))
            elif lo <= 1 - r["p"] <= hi:
                pts.append((1 - r["p"], 1 if r["winner"] == "b" else 0))
        if not pts:
            return 0, 0.0
        gap = sum(y for _, y in pts) / len(pts) - sum(p for p, _ in pts) / len(pts)
        return len(pts), gap

    def fade_sim(sub, lo=0.55, hi=0.80):
        """Buy the underdog side whenever a favorite is priced in [lo,hi]."""
        pnl = n = wins = 0
        for r in sub:
            if lo <= r["p"] <= hi:
                q, win = 1 - r["p"], r["winner"] == "b"
            elif lo <= 1 - r["p"] <= hi:
                q, win = r["p"], r["winner"] == "a"
            else:
                continue
            q = min(q + slippage, 0.99)
            pnl += (1 - q) / q if win else -1.0
            n += 1
            wins += win
        return n, wins, pnl

    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)

    print(f"{'game':16s} {'n':>5s} {'Brier':>7s} {'favN':>5s} {'fav gap':>8s} "
          f"{'fadeN':>6s} {'fade P&L':>9s} {'ROI':>7s}")
    for game, sub in sorted(by_game.items(), key=lambda kv: -len(kv[1])):
        if len(sub) < 30:
            continue
        bs = sum((r["p"] - (1 if r["winner"] == "a" else 0)) ** 2 for r in sub) / len(sub)
        fn, gap = fav_gap(sub)
        bn, bw, bp = fade_sim(sub)
        roi = bp / bn * 100 if bn else 0.0
        print(f"{game:16s} {len(sub):5d} {bs:7.4f} {fn:5d} {gap:+8.3f} "
              f"{bn:6d} {bp:+9.2f} {roi:+6.1f}%")

    print("\ntime-split (all games pooled, favorites .55-.80):")
    half = len(rows) // 2
    for label, sub in (("1st half", rows[:half]), ("2nd half", rows[half:])):
        fn, gap = fav_gap(sub)
        bn, bw, bp = fade_sim(sub)
        lo_d = datetime.fromtimestamp(sub[0]["start_ts"], tz=timezone.utc)
        hi_d = datetime.fromtimestamp(sub[-1]["start_ts"], tz=timezone.utc)
        print(f"  {label} ({lo_d:%Y-%m} .. {hi_d:%Y-%m}): favN={fn} gap={gap:+.3f}  "
              f"fade: n={bn} P&L={bp:+.2f} ROI={bp/bn*100 if bn else 0:+.1f}%")

    print("\ntime-split per major game:")
    for game, sub in sorted(by_game.items(), key=lambda kv: -len(kv[1])):
        if len(sub) < 200:
            continue
        h = len(sub) // 2
        parts = []
        for piece in (sub[:h], sub[h:]):
            fn, gap = fav_gap(piece)
            parts.append(f"n={fn} gap={gap:+.3f}")
        print(f"  {game:16s} 1st: {parts[0]}   2nd: {parts[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if not args.report:
        ingest()
    report()
