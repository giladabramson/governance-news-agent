"""SQLite ledger: every quote, book snapshot, decision, order, and position.

The ledger IS the product of v1 — CLV analysis needs the full history of what
the bot saw and why it acted or didn't (spec §8).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
create table if not exists events (
    id integer primary key autoincrement,
    poly_slug text unique, ref_event_id text, tournament_key text,
    player_a text, player_b text, start_time text, created_ts real
);
create table if not exists quotes (
    id integer primary key autoincrement,
    ts real, poly_slug text, source text,
    odds_a real, odds_b real, fair_a real, fair_b real, overround real,
    is_closing integer default 0
);
create table if not exists books (
    id integer primary key autoincrement,
    ts real, poly_slug text, side text,
    best_bid real, best_ask real, bid_size real, ask_size real, depth_2c_usd real
);
create table if not exists decisions (
    id integer primary key autoincrement,
    ts real, poly_slug text, side text, player text,
    fair real, ask real, gross_edge real, net_edge real, stake real,
    action text, reason text, arm text default 'base'
);
create table if not exists orders (
    id integer primary key autoincrement,
    ts real, poly_slug text, side text, player text,
    limit_price real, stake_usd real, status text,   -- resting|filled|cancelled
    fill_price real, fill_ts real, fill_kind text,   -- maker|taker
    arm text default 'base'
);
create table if not exists positions (
    id integer primary key autoincrement,
    poly_slug text, side text, player text, tournament_key text,
    entry_price real, shares real, stake_usd real, opened_ts real,
    closing_fair real, clv real,
    result text, pnl real, settled_ts real,           -- result: won|lost|void
    arm text default 'base'
);
create table if not exists meta (key text primary key, value text);
"""

# pre-arms ledgers lack the arm column; ALTER is idempotent via the check
_MIGRATIONS = [
    ("decisions", "arm", "alter table decisions add column arm text default 'base'"),
    ("orders", "arm", "alter table orders add column arm text default 'base'"),
    ("positions", "arm", "alter table positions add column arm text default 'base'"),
]


class Ledger:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA)
        for table, column, ddl in _MIGRATIONS:
            cols = {r[1] for r in self.con.execute(f"pragma table_info({table})")}
            if column not in cols:
                self.con.execute(ddl)
        self.con.commit()

    def upsert_event(self, poly_slug, ref_event_id, tournament_key, pa, pb, start_iso):
        self.con.execute(
            "insert into events (poly_slug, ref_event_id, tournament_key, player_a, "
            "player_b, start_time, created_ts) values (?,?,?,?,?,?,?) "
            "on conflict(poly_slug) do nothing",
            (poly_slug, ref_event_id, tournament_key, pa, pb, start_iso, time.time()))
        self.con.commit()

    def add_quote(self, poly_slug, source, odds_a, odds_b, fv, is_closing=False):
        self.con.execute(
            "insert into quotes (ts, poly_slug, source, odds_a, odds_b, fair_a, fair_b, "
            "overround, is_closing) values (?,?,?,?,?,?,?,?,?)",
            (time.time(), poly_slug, source, odds_a, odds_b,
             fv.fair_a, fv.fair_b, fv.overround, int(is_closing)))
        self.con.commit()

    def last_fair_a(self, poly_slug) -> float | None:
        row = self.con.execute(
            "select fair_a from quotes where poly_slug=? order by ts desc limit 1",
            (poly_slug,)).fetchone()
        return row["fair_a"] if row else None

    def add_book(self, poly_slug, side, bk):
        self.con.execute(
            "insert into books (ts, poly_slug, side, best_bid, best_ask, bid_size, "
            "ask_size, depth_2c_usd) values (?,?,?,?,?,?,?,?)",
            (time.time(), poly_slug, side, bk.best_bid, bk.best_ask,
             bk.bid_size, bk.ask_size, bk.depth_2c_usd))
        self.con.commit()

    def add_decision(self, poly_slug, sig, arm="base"):
        self.con.execute(
            "insert into decisions (ts, poly_slug, side, player, fair, ask, gross_edge, "
            "net_edge, stake, action, reason, arm) values (?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), poly_slug, sig.side, sig.player, sig.fair, sig.ask,
             sig.gross_edge, sig.net_edge, sig.stake, sig.action, sig.reason, arm))
        self.con.commit()

    def add_order(self, poly_slug, sig, arm="base") -> int:
        cur = self.con.execute(
            "insert into orders (ts, poly_slug, side, player, limit_price, stake_usd, "
            "status, arm) values (?,?,?,?,?,?, 'resting', ?)",
            (time.time(), poly_slug, sig.side, sig.player, sig.limit_price, sig.stake, arm))
        self.con.commit()
        return cur.lastrowid

    def resting_orders(self, arm=None):
        if arm is None:
            return self.con.execute("select * from orders where status='resting'").fetchall()
        return self.con.execute(
            "select * from orders where status='resting' and arm=?", (arm,)).fetchall()

    def fill_order(self, order_id, fill_price, fill_kind):
        self.con.execute(
            "update orders set status='filled', fill_price=?, fill_ts=?, fill_kind=? "
            "where id=?", (fill_price, time.time(), fill_kind, order_id))
        self.con.commit()

    def cancel_order(self, order_id):
        self.con.execute("update orders set status='cancelled' where id=?", (order_id,))
        self.con.commit()

    def open_position(self, poly_slug, side, player, tournament_key, entry_price, stake,
                      arm="base"):
        self.con.execute(
            "insert into positions (poly_slug, side, player, tournament_key, entry_price, "
            "shares, stake_usd, opened_ts, arm) values (?,?,?,?,?,?,?,?,?)",
            (poly_slug, side, player, tournament_key, entry_price,
             stake / entry_price, stake, time.time(), arm))
        self.con.commit()

    def open_positions(self):
        return self.con.execute("select * from positions where settled_ts is null").fetchall()

    def has_position(self, poly_slug, arm=None) -> bool:
        if arm is None:
            return self.con.execute(
                "select 1 from positions where poly_slug=? limit 1",
                (poly_slug,)).fetchone() is not None
        return self.con.execute(
            "select 1 from positions where poly_slug=? and arm=? limit 1",
            (poly_slug, arm)).fetchone() is not None

    def set_closing_fair(self, poly_slug, side, closing_fair):
        self.con.execute(
            "update positions set closing_fair=?, clv=? - entry_price "
            "where poly_slug=? and side=? and settled_ts is null",
            (closing_fair, closing_fair, poly_slug, side))
        self.con.commit()

    def settle(self, position_id, result, pnl):
        self.con.execute(
            "update positions set result=?, pnl=?, settled_ts=? where id=?",
            (result, pnl, time.time(), position_id))
        self.con.commit()

    # --- cross-run state (needed for scheduled/stateless runs, e.g. GitHub Actions) ---
    def get_meta(self, key: str) -> str | None:
        row = self.con.execute("select value from meta where key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.con.execute("insert into meta (key, value) values (?,?) "
                         "on conflict(key) do update set value=excluded.value", (key, value))
        self.con.commit()

    def closing_pulled(self, poly_slug: str) -> bool:
        return self.con.execute(
            "select 1 from quotes where poly_slug=? and is_closing=1 limit 1",
            (poly_slug,)).fetchone() is not None

    def quoted_slugs(self) -> set[str]:
        return {r["poly_slug"] for r in
                self.con.execute("select distinct poly_slug from quotes").fetchall()}

    # --- risk queries --- (per arm: each arm runs its own paper book)
    def open_exposure(self, arm="base") -> float:
        r = self.con.execute(
            "select coalesce(sum(stake_usd),0) s from positions "
            "where settled_ts is null and arm=?", (arm,)).fetchone()
        return r["s"]

    def daily_new_exposure(self, arm="base") -> float:
        r = self.con.execute(
            "select coalesce(sum(stake_usd),0) s from positions where opened_ts > ? and arm=?",
            (time.time() - 86400, arm)).fetchone()
        return r["s"]

    def daily_realized_pnl(self, arm="base") -> float:
        r = self.con.execute(
            "select coalesce(sum(pnl),0) s from positions where settled_ts > ? and arm=?",
            (time.time() - 86400, arm)).fetchone()
        return r["s"]

    def tournament_positions_today(self, tournament_key, arm="base") -> int:
        r = self.con.execute(
            "select count(*) c from positions where tournament_key=? and opened_ts > ? and arm=?",
            (tournament_key, time.time() - 86400, arm)).fetchone()
        return r["c"]
