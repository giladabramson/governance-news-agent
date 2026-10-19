"""Phase 2a: historical Polymarket Dota markets + pre-match prices -> SQLite.

For every resolved dota2 match event: the match-winner market, its resolved
outcome, and the last traded price >= SNAPSHOT_BUFFER before the scheduled
game start (from the CLOB price history). The snapshot price is what the
backtest treats as "the market's opinion we could have traded against."

Gamma caps offset pagination at ~2000, so we page in month windows instead.

    python backtest_ingest.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
DB = DATA / "polymarket.db"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SNAPSHOT_BUFFER_SEC = 15 * 60
WINDOW_START = "2025-01-01"

_SCHEMA = """
create table if not exists pm_markets (
    market_id text primary key,
    event_slug text, title text, team_a text, team_b text,
    best_of integer,             -- parsed from title (BO1/BO3/BO5); null if unknown
    game_start_ts integer,       -- scheduled start (unix)
    volume real,
    winner text,                 -- 'a' | 'b' from resolution
    token_a text,                -- clob token for outcome A
    snap_ts integer, snap_price_a real   -- pre-match snapshot
);
"""


def month_windows(start: str):
    d = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    while d < now:
        if d.month == 12:
            nxt = d.replace(year=d.year + 1, month=1)
        else:
            nxt = d.replace(month=d.month + 1)
        yield d.isoformat(), min(nxt, now).isoformat()
        d = nxt


def fetch_events(session):
    for lo, hi in month_windows(WINDOW_START):
        offset = 0
        while True:
            r = session.get(f"{GAMMA}/events", params={
                "tag_slug": "esports", "closed": "true", "limit": 100,
                "offset": offset, "end_date_min": lo, "end_date_max": hi}, timeout=30)
            evs = r.json()
            if not isinstance(evs, list) or not evs:
                break
            for e in evs:
                slug = e.get("slug", "")
                if slug.startswith("dota2-") and "more-markets" not in slug:
                    yield e
            if len(evs) < 100:
                break
            offset += 100
            time.sleep(0.15)


def parse_match_market(event):
    """Return the match-winner market row, or None."""
    t = re.match(r"Dota 2: (.+?) vs (.+?) \(BO(\d)\)", event.get("title", ""))
    for m in event.get("markets", []):
        q = m.get("question", "")
        if not q.startswith("Dota 2:") or "Game" in q or " vs " not in q:
            continue
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
        tokens = json.loads(m.get("clobTokenIds") or "[]")
        if len(outcomes) != 2 or len(prices) != 2 or len(tokens) != 2:
            continue
        pa, pb = float(prices[0]), float(prices[1])
        if {round(pa), round(pb)} != {0, 1}:
            continue  # not cleanly resolved (voided etc.)
        gs = m.get("gameStartTime") or m.get("startDate") or event.get("startDate")
        try:
            start_ts = int(datetime.fromisoformat(gs.replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError):
            continue
        return dict(
            market_id=m["id"], event_slug=event["slug"], title=q,
            team_a=outcomes[0], team_b=outcomes[1],
            best_of=int(t.group(3)) if t else None,
            game_start_ts=start_ts,
            volume=float(m.get("volumeNum") or m.get("volume") or 0),
            winner="a" if round(pa) == 1 else "b",
            token_a=tokens[0])
    return None


def snapshot(session, token_a: str, start_ts: int):
    """Last traded price of token A at least SNAPSHOT_BUFFER before start."""
    r = session.get(f"{CLOB}/prices-history", params={
        "market": token_a, "startTs": start_ts - 7 * 86400,
        "endTs": start_ts - SNAPSHOT_BUFFER_SEC, "fidelity": 10}, timeout=30)
    if r.status_code != 200:
        return None, None
    pts = r.json().get("history", [])
    if not pts:
        return None, None
    last = pts[-1]
    return int(last["t"]), float(last["p"])


def main():
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(_SCHEMA)
    have = {r[0] for r in con.execute("select market_id from pm_markets")}
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (research; esports-edge phase-2 backtest)"

    n_ev = n_new = n_snap = 0
    for e in fetch_events(s):
        n_ev += 1
        row = parse_match_market(e)
        if row is None or row["market_id"] in have:
            continue
        ts, price = snapshot(s, row["token_a"], row["game_start_ts"])
        row["snap_ts"], row["snap_price_a"] = ts, price
        con.execute(
            "insert or ignore into pm_markets values "
            "(:market_id,:event_slug,:title,:team_a,:team_b,:best_of,"
            ":game_start_ts,:volume,:winner,:token_a,:snap_ts,:snap_price_a)", row)
        con.commit()
        n_new += 1
        n_snap += int(price is not None)
        if n_new % 100 == 0:
            print(f"events seen {n_ev}, markets stored {n_new}, with snapshot {n_snap}", flush=True)
        time.sleep(0.12)
    total = con.execute("select count(*), count(snap_price_a) from pm_markets").fetchone()
    print(f"done: {n_ev} events scanned; DB has {total[0]} markets, {total[1]} with pre-match price")


if __name__ == "__main__":
    main()
