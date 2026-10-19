"""Polymarket side: tennis match-market discovery (Gamma) + order books (CLOB).

A tennis match event has slug `atp-<p1>-<p2>-<YYYY-MM-DD>` / `wta-...`, tag
`tennis`, and one MATCH-WINNER market whose two outcomes are the player names
themselves (companion markets are set winners / handicaps / totals — ignored).
Read-only; no auth, no keys.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_SLUG_RE = re.compile(r"^(atp|wta)-")
_NOT_PLAYERS = {"Yes", "No", "Over", "Under"}


@dataclass
class MatchMarket:
    slug: str
    title: str
    tour: str                 # 'atp' | 'wta'
    player_a: str             # first outcome
    player_b: str
    start_time: datetime
    token_a: str
    token_b: str
    condition_id: str
    volume_usd: float
    neg_risk: bool
    resolved: bool = False
    winner: str | None = None  # 'a' | 'b' once resolved


@dataclass
class BookSide:
    best_bid: float = 0.0
    best_ask: float = 1.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    asks: list = field(default_factory=list)   # [(price, shares)] best-first
    depth_2c_usd: float = 0.0


def _get(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_match_event(ev: dict) -> MatchMarket | None:
    slug = ev.get("slug") or ""
    m = _SLUG_RE.match(slug)
    if not m or not ev.get("startTime"):
        return None
    for mk in ev.get("markets") or []:
        try:
            outcomes = json.loads(mk.get("outcomes") or "[]")
            tokens = json.loads(mk.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            continue
        if len(outcomes) != 2 or len(tokens) != 2:
            continue
        if any(o in _NOT_PLAYERS for o in outcomes):
            continue  # set winner markets also have player-name outcomes, so:
        if mk.get("groupItemTitle"):
            continue  # the match-winner market has no groupItemTitle
        start = datetime.fromisoformat(ev["startTime"].replace("Z", "+00:00"))
        resolved = bool(mk.get("closed"))
        winner = None
        if resolved:
            try:
                prices = [float(x) for x in json.loads(mk.get("outcomePrices") or "[]")]
                winner = "a" if prices and prices[0] > 0.5 else "b" if prices else None
            except (json.JSONDecodeError, ValueError):
                winner = None
        return MatchMarket(
            slug=slug, title=ev.get("title") or "", tour=m.group(1),
            player_a=outcomes[0], player_b=outcomes[1], start_time=start,
            token_a=tokens[0], token_b=tokens[1],
            condition_id=mk.get("conditionId") or "",
            volume_usd=float(ev.get("volume") or 0.0),
            neg_risk=bool(mk.get("negRisk")), resolved=resolved, winner=winner)
    return None


def discover_matches(include_closed: bool = False) -> list[MatchMarket]:
    """All current tennis match-winner markets (paginated Gamma query)."""
    out, offset = [], 0
    closed = "true" if include_closed else "false"
    while True:
        batch = _get(f"{GAMMA}/events", {"tag_slug": "tennis", "closed": closed,
                                         "limit": 100, "offset": offset})
        if not batch:
            break
        for ev in batch:
            mm = _parse_match_event(ev)
            if mm:
                out.append(mm)
        if len(batch) < 100:
            break
        offset += 100
    log.info("discovered %d tennis match markets (closed=%s)", len(out), include_closed)
    return out


def refresh_resolution(slug: str) -> MatchMarket | None:
    evs = _get(f"{GAMMA}/events", {"slug": slug})
    return _parse_match_event(evs[0]) if evs else None


def book_snapshot(token_id: str) -> BookSide:
    """Top-of-book + ask ladder + depth within 2c of best ask, in USD."""
    b = _get(f"{CLOB}/book", {"token_id": token_id})
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids") or []),
                  key=lambda t: -t[0])
    asks = sorted(((float(x["price"]), float(x["size"])) for x in b.get("asks") or []),
                  key=lambda t: t[0])
    side = BookSide()
    if bids:
        side.best_bid, side.bid_size = bids[0]
    if asks:
        side.best_ask, side.ask_size = asks[0]
        side.asks = asks
        side.depth_2c_usd = sum(p * s for p, s in asks if p <= asks[0][0] + 0.02)
    return side
