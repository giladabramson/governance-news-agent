"""OpenDota -> SQLite ingest (phase 1a). No trading code lives here.

Pulls /proMatches backwards in time (100 rows/call, paginated by
less_than_match_id) plus /leagues once for tier labels. Resumable: re-running
continues from the oldest match already stored, and newer matches are picked
up by a fresh pass from the top (match_id is the primary key, inserts are
idempotent).

Free-tier limits: 60 calls/min, 2000/day. 400 pages ~= 40k matches ~= 7 min.

    python ingest.py --pages 400
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import requests

API = "https://api.opendota.com/api"
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "matches.db"
CALL_GAP_SEC = 1.1  # stay under 60/min

_SCHEMA = """
create table if not exists matches (
    match_id integer primary key,
    start_time integer, duration integer, leagueid integer,
    league_name text, series_type integer,
    radiant_team_id integer, radiant_name text,
    dire_team_id integer, dire_name text,
    radiant_score integer, dire_score integer,
    radiant_win integer
);
create table if not exists leagues (
    leagueid integer primary key, name text, tier text
);
create index if not exists idx_matches_start on matches (start_time);
"""


def _get(session: requests.Session, path: str, **params) -> list | dict:
    for attempt in range(4):
        r = session.get(f"{API}/{path}", params=params, timeout=30)
        if r.status_code == 429:  # rate limited — back off and retry
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate-limited out on {path}")


def ingest_leagues(con: sqlite3.Connection, session: requests.Session) -> None:
    leagues = _get(session, "leagues")
    con.executemany(
        "insert into leagues (leagueid, name, tier) values (?,?,?) "
        "on conflict(leagueid) do update set name=excluded.name, tier=excluded.tier",
        [(l["leagueid"], l.get("name"), l.get("tier")) for l in leagues])
    con.commit()
    print(f"leagues: {len(leagues)} upserted")


def ingest_matches(con: sqlite3.Connection, session: requests.Session, pages: int,
                   resume: bool = False) -> None:
    # default: walk from the newest match down (idempotent inserts dedup any
    # overlap). --resume: continue deeper into history from the oldest stored.
    cursor = None
    if resume:
        row = con.execute("select min(match_id) from matches").fetchone()
        cursor = row[0]
        print(f"resuming below match_id {cursor}")
    inserted = 0
    for page in range(pages):
        params = {"less_than_match_id": cursor} if cursor else {}
        rows = _get(session, "proMatches", **params)
        if not rows:
            print("API returned no more rows — reached the end of history")
            break
        cur = con.executemany(
            "insert or ignore into matches (match_id, start_time, duration, leagueid, "
            "league_name, series_type, radiant_team_id, radiant_name, dire_team_id, "
            "dire_name, radiant_score, dire_score, radiant_win) "
            "values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(m["match_id"], m.get("start_time"), m.get("duration"), m.get("leagueid"),
              m.get("league_name"), m.get("series_type"),
              m.get("radiant_team_id"), m.get("radiant_name"),
              m.get("dire_team_id"), m.get("dire_name"),
              m.get("radiant_score"), m.get("dire_score"),
              None if m.get("radiant_win") is None else int(m["radiant_win"]))
             for m in rows])
        inserted += cur.rowcount
        con.commit()
        cursor = min(m["match_id"] for m in rows)
        if page % 20 == 0:
            oldest = min(m.get("start_time") or 0 for m in rows)
            print(f"page {page}: cursor={cursor} oldest={time.strftime('%Y-%m-%d', time.gmtime(oldest))} "
                  f"inserted so far={inserted}")
        time.sleep(CALL_GAP_SEC)
    n = con.execute("select count(*), min(start_time), max(start_time) from matches").fetchone()
    print(f"done: +{inserted} new rows; total {n[0]} matches "
          f"({time.strftime('%Y-%m-%d', time.gmtime(n[1]))} .. "
          f"{time.strftime('%Y-%m-%d', time.gmtime(n[2]))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=400,
                    help="max proMatches pages (100 matches each)")
    ap.add_argument("--resume", action="store_true",
                    help="continue deeper into history from the oldest stored match")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(_SCHEMA)
    with requests.Session() as session:
        ingest_leagues(con, session)
        ingest_matches(con, session, args.pages, resume=args.resume)


if __name__ == "__main__":
    main()
