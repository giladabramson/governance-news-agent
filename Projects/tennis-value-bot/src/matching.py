"""Event pairing: reference-odds events <-> Polymarket match markets.

Spec §2.3: require BOTH surname matches (normalized), start time within ±6h,
skip on any ambiguity with a logged reason. A wrong pairing is worse than a
missed bet. Manual alias table at src/player_aliases.csv for recurring
mismatches (columns: ref_name,poly_name).
"""
from __future__ import annotations

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_PARTICLES = {"de", "del", "della", "van", "von", "der", "den", "da", "dos", "el", "al"}


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z\- ]", "", s).strip()


def surname(name: str) -> str:
    """Last token, pulling in name particles ('de minaur', 'van de zandschulp')."""
    tokens = normalize_name(name).replace("-", " ").split()
    if not tokens:
        return ""
    out = [tokens[-1]]
    i = len(tokens) - 2
    while i > 0 and tokens[i] in _PARTICLES:
        out.insert(0, tokens[i])
        i -= 1
    return " ".join(out)


def load_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["ref_name"].strip(): row["poly_name"].strip()
                for row in csv.DictReader(f)}


@dataclass
class Pairing:
    ref_quote: object       # RefQuote
    market: object          # MatchMarket
    aligned: bool           # ref player_a corresponds to market player_a


def _names_match(ref_name: str, poly_name: str, aliases: dict[str, str]) -> bool:
    if aliases.get(ref_name) == poly_name:
        return True
    return surname(ref_name) == surname(poly_name) and surname(ref_name) != ""


def pair(quotes: list, markets: list, aliases: dict[str, str]) -> tuple[list[Pairing], list[str]]:
    """Match quotes to markets. Returns (pairings, skip_log)."""
    pairings, skips = [], []
    for q in quotes:
        cands = []
        for mkt in markets:
            if abs((q.start_time - mkt.start_time).total_seconds()) > 6 * 3600:
                continue
            direct = (_names_match(q.player_a, mkt.player_a, aliases)
                      and _names_match(q.player_b, mkt.player_b, aliases))
            swapped = (_names_match(q.player_a, mkt.player_b, aliases)
                       and _names_match(q.player_b, mkt.player_a, aliases))
            if direct and swapped:
                skips.append(f"AMBIGUOUS orientation {q.player_a}/{q.player_b} vs {mkt.slug} — skipped")
                continue
            if direct or swapped:
                cands.append(Pairing(q, mkt, aligned=direct))
        if len(cands) == 1:
            pairings.append(cands[0])
        elif len(cands) > 1:
            skips.append(f"AMBIGUOUS {q.player_a} vs {q.player_b}: "
                         f"{[c.market.slug for c in cands]} — skipped")
        # zero candidates: normal (Polymarket doesn't list every match) — not logged as skip
    return pairings, skips
