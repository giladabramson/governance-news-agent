"""READ-ONLY preview: calibrated model vs live Polymarket prices, today.

This is NOT the decision engine (phase 3) — it places nothing and sizes
nothing. It exists to answer "would the current model see any edge today?"
Run ad hoc; output is a table, the verdict column applies the phase-1
guardrails (>=10 games history per team, |gap| >= 5 pts, price in
[0.10, 0.90]).

    python scan_preview.py
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import requests

DB = Path(__file__).parent / "data" / "matches.db"
PLATT_A, PLATT_B = 0.520, 0.006   # from evaluate.py train fit
MIN_HISTORY = 10
MIN_EDGE = 0.05


def build_ratings():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """select radiant_team_id a, dire_team_id b, radiant_win y,
                  radiant_name an, dire_name bn from matches
           where radiant_win is not null and radiant_team_id is not null
             and dire_team_id is not null and radiant_team_id != 0 and dire_team_id != 0
           order by start_time, match_id""").fetchall()
    R, G, name = defaultdict(lambda: 1500.0), defaultdict(int), {}
    for m in rows:
        p = 1 / (1 + 10 ** (-((R[m["a"]] + 10.9) - R[m["b"]]) / 400))
        d = 32 * (m["y"] - p)
        R[m["a"]] += d
        R[m["b"]] -= d
        G[m["a"]] += 1
        G[m["b"]] += 1
        if m["an"]:
            name[m["a"]] = m["an"]
        if m["bn"]:
            name[m["b"]] = m["bn"]
    return R, G, name


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def find_team(name_map, query):
    q = norm(query)
    exact = [t for t, n in name_map.items() if norm(n) == q]
    if exact:
        return exact[0]
    sub = [t for t, n in name_map.items() if q and (q in norm(n) or norm(n) in q)]
    return sub[0] if len(sub) == 1 else None


def model_p(R, ta, tb):
    raw = 1 / (1 + 10 ** (-(R[ta] - R[tb]) / 400))
    z = PLATT_A * math.log(raw / (1 - raw)) + PLATT_B
    return 1 / (1 + math.exp(-z))


def market_game_p(event) -> float | None:
    """Implied per-game P(team A) from a BO2 three-outcome event."""
    import json as j
    pa = pb = None
    m_title = event["title"]
    m = re.match(r"Dota 2: (.+?) vs (.+?)[ (]", m_title)
    if not m:
        return None
    a_name = m.group(1)
    for mk in event.get("markets", []):
        q = mk.get("question", "")
        prices = mk.get("outcomePrices")
        if not prices:
            continue
        yes = float(j.loads(prices)[0] if isinstance(prices, str) else prices[0])
        if "to win 2-0" in q:
            if norm(a_name) in norm(q):
                pa = math.sqrt(max(yes, 1e-9))
            else:
                pb = math.sqrt(max(yes, 1e-9))
    if pa is None or pb is None:
        return None
    p = (pa + (1 - pb)) / 2  # average the two implied views, kills overround
    return p


def main():
    R, G, name_map = build_ratings()
    evs = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={"closed": "false", "limit": 100, "tag_slug": "esports"},
        timeout=30).json()
    evs = [e for e in evs if e["slug"].startswith("dota2-")
           and "more-markets" not in e["slug"]]
    print(f"{'match':42s} {'market':>7s} {'model':>6s} {'gap':>6s} {'hist':>9s}  verdict")
    for e in evs:
        m = re.match(r"Dota 2: (.+?) vs (.+?)[ (]", e["title"])
        if not m:
            continue
        a_name, b_name = m.group(1).strip(), m.group(2).strip()
        pm = market_game_p(e)
        ta, tb = find_team(name_map, a_name), find_team(name_map, b_name)
        label = f"{a_name} vs {b_name}"[:42]
        if pm is None:
            print(f"{label:42s} {'—':>7s}  (not a BO2 outcome event)")
            continue
        if ta is None or tb is None:
            missing = a_name if ta is None else b_name
            print(f"{label:42s} {pm:7.3f} {'—':>6s} {'—':>6s} {'—':>9s}  SKIP team not in DB: {missing}")
            continue
        pmod = model_p(R, ta, tb)
        gap = pmod - pm
        hist = f"{G[ta]}/{G[tb]}"
        if G[ta] < MIN_HISTORY or G[tb] < MIN_HISTORY:
            verdict = f"SKIP history<{MIN_HISTORY}"
        elif not (0.10 <= pm <= 0.90):
            verdict = "SKIP price band"
        elif abs(gap) < MIN_EDGE:
            verdict = "no edge"
        else:
            side = a_name if gap > 0 else b_name
            verdict = f"EDGE CANDIDATE -> {side} ({abs(gap):.1%})"
        print(f"{label:42s} {pm:7.3f} {pmod:6.3f} {gap:+6.3f} {hist:>9s}  {verdict}")


if __name__ == "__main__":
    main()
