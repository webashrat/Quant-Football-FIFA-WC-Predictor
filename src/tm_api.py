"""
tm_api.py — Transfermarkt unofficial API client.

Endpoint: https://tmapi.transfermarkt.technology/player/{tm_player_id}/performance-game

Returns full career game log per player with exact minutes, goals, assists,
cards, substitution times. Covers both club and national team games.

Key filters:
  - seasonId == 2025           → current 25/26 season
  - isNationalGame == False    → club games (fitness/form proxy)
  - competitionId == 'FIWC'   → FIFA World Cup 2026 games (exact minutes)

Cache: data/tm_api_cache/{tm_player_id}.json  (refreshed every 6 hours)
Rate limit: 0.5s between calls, max 4 parallel threads.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

_CACHE_DIR = DATA_DIR / "tm_api_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_HOURS = 6

_BASE = "https://tmapi.transfermarkt.technology/player"

_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })
    return _SESSION


def _cache_path(tm_player_id: int) -> Path:
    return _CACHE_DIR / f"{tm_player_id}.json"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
    return age_hours < _CACHE_TTL_HOURS


def _fetch_raw(tm_player_id: int) -> list[dict]:
    """Fetch raw performance list from API. Returns [] on any failure."""
    path = _cache_path(tm_player_id)
    if _cache_valid(path):
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    try:
        r = _session().get(f"{_BASE}/{tm_player_id}/performance-game", timeout=20)
        if r.status_code != 200:
            logger.debug(f"TM API {tm_player_id}: HTTP {r.status_code}")
            return []
        data = r.json()
        if not data.get("success"):
            return []
        perf = data["data"]["performance"]
        path.write_text(json.dumps(perf, ensure_ascii=False))
        return perf
    except Exception as e:
        logger.debug(f"TM API {tm_player_id}: {e}")
        return []


# ── Stat extractors ───────────────────────────────────────────────────────────

def _safe(val, default=0):
    return val if val is not None else default


def _parse_game(g: dict) -> dict:
    """Flatten one game record into a simple dict."""
    info  = g["gameInformation"]
    stats = g["statistics"]
    gs    = stats["goalStatistics"]
    cs    = stats["cardStatistics"]
    pt    = stats["playingTimeStatistics"]
    gen   = stats["generalStatistics"]
    sub_in  = pt.get("substitutedIn",  {}) or {}
    sub_out = pt.get("substitutedOut", {}) or {}
    return {
        "date":             info["date"]["dateTimeUTC"][:10],
        "season_id":        info["seasonId"],
        "competition_id":   info["competitionId"],
        "is_national":      info["isNationalGame"],
        "participation":    gen.get("participationState", ""),
        "is_starting":      _safe(pt.get("isStarting"), False),
        "minutes":          _safe(pt.get("playedMinutes")),
        "sub_in_min":       sub_in.get("minute"),
        "sub_out_min":      sub_out.get("minute"),
        "goals":            _safe(gs.get("goalsScoredTotal")),
        "assists":          _safe(gs.get("assists")),
        "own_goals":        _safe(gs.get("ownGoalsScored")),
        "yellow_cards":     _safe(cs.get("yellowCardNet")),
        "red_cards":        _safe(cs.get("yellowCardGross", 0)) - _safe(cs.get("yellowCardNet", 0)),
    }


def get_current_season_club_stats(tm_player_id: int) -> dict:
    """
    Returns 2025/26 club season aggregates for a player.

    Keys: apps, minutes, goals, assists, yellows, reds,
          goals_per_90, assists_per_90, form_score
    """
    empty = {
        "apps": 0, "minutes": 0, "goals": 0, "assists": 0,
        "yellows": 0, "reds": 0,
        "goals_per_90": 0.0, "assists_per_90": 0.0, "form_score": 0.0,
    }
    perf = _fetch_raw(tm_player_id)
    if not perf:
        return empty

    current_club = [
        _parse_game(g) for g in perf
        if g["gameInformation"]["seasonId"] == 2025
        and not g["gameInformation"]["isNationalGame"]
        and g["statistics"]["generalStatistics"].get("participationState") == "played"
    ]
    if not current_club:
        return empty

    apps    = len(current_club)
    minutes = sum(g["minutes"] for g in current_club)
    goals   = sum(g["goals"]   for g in current_club)
    assists = sum(g["assists"]  for g in current_club)
    yellows = sum(g["yellow_cards"] for g in current_club)
    reds    = sum(g["red_cards"]    for g in current_club)

    per90 = minutes / 90 if minutes > 0 else 1
    return {
        "apps":           apps,
        "minutes":        minutes,
        "goals":          goals,
        "assists":        assists,
        "yellows":        yellows,
        "reds":           reds,
        "goals_per_90":   round(goals   / per90, 3),
        "assists_per_90": round(assists / per90, 3),
        # Simple form score: goals+assists per 90 weighted toward recent games
        "form_score":     round((goals + assists) / per90, 3),
    }


def get_wc_2026_stats(tm_player_id: int) -> dict:
    """
    Returns WC 2026 (FIWC, seasonId=2025) aggregates.

    Keys: matches, minutes, goals, assists, yellows, reds
          Also includes per-match list for substitution tracking.
    """
    empty = {"matches": 0, "minutes": 0, "goals": 0, "assists": 0, "yellows": 0, "reds": 0, "games": []}
    perf = _fetch_raw(tm_player_id)
    if not perf:
        return empty

    wc_games = [
        _parse_game(g) for g in perf
        if g["gameInformation"]["competitionId"] == "FIWC"
        and g["gameInformation"]["seasonId"] == 2025
        and g["statistics"]["generalStatistics"].get("participationState") == "played"
    ]
    if not wc_games:
        return empty

    return {
        "matches":  len(wc_games),
        "minutes":  sum(g["minutes"]      for g in wc_games),
        "goals":    sum(g["goals"]        for g in wc_games),
        "assists":  sum(g["assists"]      for g in wc_games),
        "yellows":  sum(g["yellow_cards"] for g in wc_games),
        "reds":     sum(g["red_cards"]    for g in wc_games),
        "games":    wc_games,
    }


def get_exact_wc_minutes(tm_player_id: int, date_str: str) -> Optional[int]:
    """
    Returns exact minutes played in a WC 2026 match on the given date.
    Returns None if not found or player didn't participate.
    """
    perf = _fetch_raw(tm_player_id)
    for g in perf:
        if (g["gameInformation"]["competitionId"] == "FIWC"
                and g["gameInformation"]["date"]["dateTimeUTC"][:10] == date_str
                and g["statistics"]["generalStatistics"].get("participationState") == "played"):
            return g["statistics"]["playingTimeStatistics"].get("playedMinutes")
    return None


def get_exact_sub_minutes(tm_player_id: int, date_str: str) -> dict:
    """
    Returns exact substitution minutes for a WC match.
    dict: {minutes, sub_in_min, sub_out_min, is_starting}
    """
    perf = _fetch_raw(tm_player_id)
    for g in perf:
        if (g["gameInformation"]["competitionId"] == "FIWC"
                and g["gameInformation"]["date"]["dateTimeUTC"][:10] == date_str):
            pg = _parse_game(g)
            return {
                "minutes":     pg["minutes"],
                "sub_in_min":  pg["sub_in_min"],
                "sub_out_min": pg["sub_out_min"],
                "is_starting": pg["is_starting"],
            }
    return {}


# ── Bulk enrichment ───────────────────────────────────────────────────────────

def enrich_squad_with_club_stats(
    squad: list[dict],
    max_workers: int = 4,
    delay: float = 0.4,
) -> list[dict]:
    """
    For each player in squad (with tm_player_id > 0), fetch current season
    club stats and add to the player dict.

    Returns enriched squad list (modifies in place too).
    """
    players_with_ids = [p for p in squad if p.get("tm_player_id", 0) > 0]
    if not players_with_ids:
        return squad

    def _enrich(player: dict) -> dict:
        pid = player["tm_player_id"]
        stats = get_current_season_club_stats(pid)
        player["club_season"] = stats
        time.sleep(delay)
        return player

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_enrich, p): p for p in players_with_ids}
        done = 0
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as e:
                logger.debug(f"enrich error: {e}")

    return squad


def fix_wc_minutes_from_tm(
    records: list[dict],
    squad: list[dict],
    match_date: str,
) -> list[dict]:
    """
    For each performance record, look up the player's TM ID from squad
    and replace the estimated minutes_est with exact TM API minutes.

    Falls back to the ESPN estimate if TM data is unavailable.
    """
    # Build ESPN player id → TM player id lookup
    espn_to_tm = {
        p["espn_player_id"]: p["tm_player_id"]
        for p in squad
        if p.get("espn_player_id") and p.get("tm_player_id", 0) > 0
    }

    updated = 0
    for rec in records:
        espn_pid = rec.get("espn_player_id")
        tm_pid   = espn_to_tm.get(espn_pid)
        if not tm_pid:
            continue
        exact = get_exact_wc_minutes(tm_pid, match_date)
        if exact is not None:
            rec["minutes_est"] = exact
            rec["minutes_exact"] = True
            updated += 1
        else:
            rec["minutes_exact"] = False

    if updated:
        logger.info(f"Replaced {updated}/{len(records)} estimated minutes with exact TM values")
    return records


def get_team_club_season_summary(squad: list[dict]) -> dict:
    """
    Returns team-level club season aggregates from enriched squad.
    Useful for `get_team_player_features()` to report top performers.
    """
    enriched = [p for p in squad if p.get("club_season", {}).get("apps", 0) > 0]
    if not enriched:
        return {}

    total_mins = sum(p["club_season"]["minutes"] for p in enriched)
    avg_mins   = total_mins / len(enriched) if enriched else 0

    # Top scorers this club season
    top_scorers = sorted(
        enriched,
        key=lambda p: (p["club_season"]["goals"], p["club_season"]["assists"]),
        reverse=True,
    )[:5]

    return {
        "players_with_data": len(enriched),
        "avg_club_minutes":  round(avg_mins),
        "top_scorers": [
            {
                "name":    p["name"],
                "apps":    p["club_season"]["apps"],
                "minutes": p["club_season"]["minutes"],
                "goals":   p["club_season"]["goals"],
                "assists": p["club_season"]["assists"],
                "form":    p["club_season"]["form_score"],
            }
            for p in top_scorers
        ],
    }
