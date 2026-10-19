"""Calibration map of Polymarket: which categories have dishonest prices?

For resolved binary markets across ALL categories: snapshot the last traded
price >= 6h before the event's end, then compare price vs outcome per
category. No model involved — this measures the MARKET's calibration.

Method notes:
- categories come from event tags; a market is bucketed by the first tag we
  recognize. Sample cap per category keeps the CLOB call count sane.
- markets need volume >= MIN_VOL (untradeable dust tells us nothing) and a
  clean 0/1 resolution.
- output: per-category n, Brier of the price, calibration bins, and the
  max cost-adjusted bin deviation ("harvest" column) — deviation minus an
  assumed 3c round-trip cost, floored at 0.

    python calibration_map.py            # ingest + report (resumable)
    python calibration_map.py --report   # report only, no network
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
DB = DATA / "calibration_map.db"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SNAPSHOT_BEFORE_END_SEC = 6 * 3600
MIN_VOL = 1000.0
PER_CATEGORY_CAP = 250
WINDOW_START = "2025-11-01"   # ~8 months of resolved markets
COST = 0.03                   # assumed round-trip cost (spread + fees)

KNOWN_TAGS = [
    "nba", "nfl", "mlb", "nhl", "soccer", "epl", "tennis", "ufc", "boxing",
    "cs2", "lol", "dota-2", "valorant", "esports",
    "politics", "geopolitics", "elections",
    "crypto", "bitcoin", "ethereum",
    "pop-culture", "movies", "music", "business", "science", "ai", "weather",
]

_SCHEMA = """
create table if not exists cmap (
    market_id text primary key,
    category text, question text, end_ts integer, volume real,
    outcome integer,            -- 1 if 'Yes'/first outcome won
    snap_price real
);
create table if not exists cmap_done (window text primary key);
"""


def month_windows(start: str):
    d = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    while d < now:
        nxt = (d.replace(year=d.year + 1, month=1) if d.month == 12
               else d.replace(month=d.month + 1))
        yield d.strftime("%Y-%m"), d.isoformat(), min(nxt, now).isoformat()
        d = nxt


def category_of(event) -> str | None:
    slugs = [t.get("slug", "") for t in (event.get("tags") or [])]
    for k in KNOWN_TAGS:
        if k in slugs:
            return k
    return None


def clean_binary(m):
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
        tokens = json.loads(m.get("clobTokenIds") or "[]")
    except json.JSONDecodeError:
        return None
    if len(outcomes) != 2 or len(prices) != 2 or len(tokens) != 2:
        return None
    pa = float(prices[0])
    if round(pa) not in (0, 1) or abs(pa - round(pa)) > 0.02:
        return None
    vol = float(m.get("volumeNum") or m.get("volume") or 0)
    if vol < MIN_VOL:
        return None
    end = m.get("endDate") or ""
    try:
        end_ts = int(datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None
    return dict(market_id=m["id"], question=m.get("question", "")[:120],
                end_ts=end_ts, volume=vol, outcome=round(pa), token=tokens[0])


def ingest():
    con = sqlite3.connect(DB)
    con.executescript(_SCHEMA)
    have = {r[0] for r in con.execute("select market_id from cmap")}
    counts = defaultdict(int)
    for (cat,) in con.execute("select category from cmap"):
        counts[cat] += 1
    done = {r[0] for r in con.execute("select window from cmap_done")}
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (research; polymarket calibration map)"

    for label, lo, hi in month_windows(WINDOW_START):
        if label in done:
            continue
        offset = 0
        while True:
            r = s.get(f"{GAMMA}/events", params={
                "closed": "true", "limit": 100, "offset": offset,
                "end_date_min": lo, "end_date_max": hi}, timeout=30)
            evs = r.json()
            if not isinstance(evs, list) or not evs:
                break
            for e in evs:
                cat = category_of(e)
                if cat is None or counts[cat] >= PER_CATEGORY_CAP:
                    continue
                for m in e.get("markets", []):
                    row = clean_binary(m)
                    if row is None or row["market_id"] in have:
                        continue
                    pr = s.get(f"{CLOB}/prices-history", params={
                        "market": row.pop("token"),
                        "startTs": row["end_ts"] - 14 * 86400,
                        "endTs": row["end_ts"] - SNAPSHOT_BEFORE_END_SEC,
                        "fidelity": 30}, timeout=30)
                    pts = pr.json().get("history", []) if pr.status_code == 200 else []
                    if not pts:
                        continue
                    p = float(pts[-1]["p"])
                    if not (0.02 <= p <= 0.98):
                        continue
                    con.execute("insert or ignore into cmap values (?,?,?,?,?,?,?)",
                                (row["market_id"], cat, row["question"], row["end_ts"],
                                 row["volume"], row["outcome"], p))
                    have.add(row["market_id"])
                    counts[cat] += 1
                    time.sleep(0.1)
            con.commit()
            if len(evs) < 100:
                break
            offset += 100
            time.sleep(0.15)
        con.execute("insert or ignore into cmap_done values (?)", (label,))
        con.commit()
        total = con.execute("select count(*) from cmap").fetchone()[0]
        print(f"window {label} done — {total} markets so far "
              f"({sum(1 for v in counts.values() if v >= PER_CATEGORY_CAP)} categories full)",
              flush=True)


def report():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cats = con.execute(
        "select category, count(*) n, avg(volume) av from cmap "
        "group by category having n >= 30 order by n desc").fetchall()
    print(f"{'category':14s} {'n':>5s} {'avgVol':>10s} {'Brier':>7s} "
          f"{'worst bin (n>=20)':>22s} {'harvest':>8s}")
    for c in cats:
        rows = con.execute("select snap_price p, outcome y from cmap where category=?",
                           (c["category"],)).fetchall()
        bs = sum((r["p"] - r["y"]) ** 2 for r in rows) / len(rows)
        bins = defaultdict(list)
        for r in rows:
            bins[min(int(r["p"] * 10), 9)].append(r)
        worst, wlabel = 0.0, "—"
        for b, ch in bins.items():
            if len(ch) < 20:
                continue
            gap = sum(r["y"] for r in ch) / len(ch) - sum(r["p"] for r in ch) / len(ch)
            if abs(gap) > abs(worst):
                worst, wlabel = gap, f"[{b/10:.1f},{b/10+.1:.1f}) {gap:+.3f} n={len(ch)}"
        harvest = max(0.0, abs(worst) - COST)
        print(f"{c['category']:14s} {len(rows):5d} {c['av']:>10,.0f} {bs:7.4f} "
              f"{wlabel:>22s} {harvest:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if not args.report:
        ingest()
    report()


if __name__ == "__main__":
    main()
