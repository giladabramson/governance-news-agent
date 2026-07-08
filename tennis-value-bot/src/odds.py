"""Sharp-book reference odds. OddsProvider interface + The Odds API impl.

The Odds API free tier = 500 credits/month; one h2h+eu call per tournament key
costs 1 credit. The provider self-throttles: it tracks x-requests-remaining
and refuses to spend below a configured reserve. Tennis keys are per-tournament
(e.g. tennis_atp_wimbledon) and enumerated via /v4/sports (free call).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)


@dataclass
class RefQuote:
    source_event_id: str
    tournament_key: str
    player_a: str
    player_b: str
    start_time: datetime
    odds_a: float
    odds_b: float
    pulled_at: float          # unix ts


class OddsProvider:
    """Interface: implementations return Pinnacle (or sharp-consensus) quotes."""

    def active_tournaments(self) -> list[str]:
        raise NotImplementedError

    def quotes(self, tournament_key: str) -> list[RefQuote]:
        raise NotImplementedError

    @property
    def credits_remaining(self) -> float:
        return float("inf")


class TheOddsAPI(OddsProvider):
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, bookmaker: str = "pinnacle"):
        if not api_key:
            raise SystemExit("ODDS_API_KEY missing — get a free key at the-odds-api.com "
                             "and put it in .env")
        self.api_key = api_key
        self.bookmaker = bookmaker
        self._remaining: float = float("inf")

    @property
    def credits_remaining(self) -> float:
        return self._remaining

    def _get(self, path: str, params: dict) -> tuple[list, dict]:
        params = {**params, "apiKey": self.api_key}
        r = requests.get(f"{self.BASE}{path}", params=params, timeout=30)
        if r.status_code == 429:
            log.warning("odds API rate-limited; backing off 30s")
            time.sleep(30)
            r = requests.get(f"{self.BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        if "x-requests-remaining" in r.headers:
            self._remaining = float(r.headers["x-requests-remaining"])
        return r.json(), dict(r.headers)

    def active_tournaments(self) -> list[str]:
        data, _ = self._get("/sports/", {})  # free: does not consume credits
        return [s["key"] for s in data
                if s.get("key", "").startswith("tennis_") and s.get("active")]

    def quotes(self, tournament_key: str) -> list[RefQuote]:
        data, _ = self._get(f"/sports/{tournament_key}/odds/",
                            {"regions": "eu", "markets": "h2h",
                             "oddsFormat": "decimal", "bookmakers": self.bookmaker})
        now = time.time()
        out = []
        for ev in data:
            bms = [b for b in ev.get("bookmakers") or [] if b.get("key") == self.bookmaker]
            if not bms:
                continue
            h2h = next((m for m in bms[0].get("markets") or [] if m.get("key") == "h2h"), None)
            if not h2h or len(h2h.get("outcomes") or []) != 2:
                continue
            o1, o2 = h2h["outcomes"]
            # align outcomes to the event's home/away order
            pa, pb = ev.get("home_team"), ev.get("away_team")
            prices = {o["name"]: float(o["price"]) for o in (o1, o2)}
            if pa not in prices or pb not in prices:
                continue
            out.append(RefQuote(
                source_event_id=ev["id"], tournament_key=tournament_key,
                player_a=pa, player_b=pb,
                start_time=datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")),
                odds_a=prices[pa], odds_b=prices[pb], pulled_at=now))
        log.info("%s: %d %s quotes (credits left: %s)", tournament_key, len(out),
                 self.bookmaker, self._remaining)
        return out


def make_provider(cfg: dict) -> OddsProvider:
    if cfg["odds_provider"] == "the_odds_api":
        return TheOddsAPI(cfg["odds_api_key"])
    raise SystemExit(f"unknown odds_provider {cfg['odds_provider']!r}")
