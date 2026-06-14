"""
player_scraper.py — scrapes player data from two sources:

  1. Transfermarkt — squad pages: player names, TM IDs, positions, market values
  2. ESPN unofficial API — WC match events: starters, subs, goals, assists, cards

Both run with polite rate limiting and store results via player_store.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_cls
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.player_features import FDORG_TO_TM, _get_session, _parse_value_m
from src.player_store import (
    save_squad, load_squad, squad_initialized,
    log_performances,
)
from src.tm_api import fix_wc_minutes_from_tm

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"

_ESPN_SESSION = None


def _espn_session() -> requests.Session:
    global _ESPN_SESSION
    if _ESPN_SESSION is None:
        _ESPN_SESSION = requests.Session()
        _ESPN_SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })
    return _ESPN_SESSION


# ── Transfermarkt squad scraping ──────────────────────────────────────────────

def scrape_team_squad(fdorg_team_id: int, force: bool = False) -> list[dict]:
    """
    Scrape TM squad page for one team → list of player dicts.
    Saves to player_store. Returns cached version if already scraped today.

    Player dict keys: tm_player_id, name, slug, position, age, market_value_m,
                      fdorg_team_id, team_name, espn_player_id
    """
    if not force and squad_initialized(fdorg_team_id):
        return load_squad(fdorg_team_id)

    tm_entry = FDORG_TO_TM.get(fdorg_team_id)
    if not tm_entry:
        logger.warning(f"No TM mapping for fdorg_team_id={fdorg_team_id}")
        return []

    tm_id, slug, expected_kw = tm_entry
    session = _get_session()

    for saison in ["2025", "2024"]:
        url = (
            f"https://www.transfermarkt.com/{slug}/kader/verein/{tm_id}"
            f"/saison_id/{saison}/plus/1"
        )
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                time.sleep(1)
                continue
            soup = BeautifulSoup(r.text, "lxml")

            # Validate team
            h1 = soup.find("h1")
            h1_text = h1.get_text(" ", strip=True).lower() if h1 else ""
            if expected_kw.lower() not in h1_text:
                logger.debug(f"TM {tm_id} returned '{h1_text[:25]}', expected '{expected_kw}'")
                time.sleep(1)
                continue

            # Squad header details
            avg_age = 27.0
            team_name = h1.get_text(" ", strip=True) if h1 else str(fdorg_team_id)
            for li in soup.select("div.data-header__details li"):
                txt = li.get_text(" ", strip=True)
                if "Average age" in txt:
                    m = re.search(r"[\d.]+", txt)
                    if m:
                        avg_age = float(m.group())

            # Parse player rows
            players = []
            rows = [row for row in soup.select("table.items > tbody > tr") if row.find("td")]
            for row in rows:
                tds = row.find_all("td")
                if len(tds) < 13:
                    continue
                name_td = tds[3]
                link = name_td.find("a")
                if not link:
                    continue

                name = link.get_text(" ", strip=True)
                href = link.get("href", "")
                pid_m  = re.search(r"/spieler/(\d+)", href)
                slug_m = re.search(r"^/([^/]+)/", href)
                tm_player_id = int(pid_m.group(1)) if pid_m else 0
                player_slug  = slug_m.group(1) if slug_m else ""

                position = tds[4].get_text(" ", strip=True) if len(tds) > 4 else ""
                age_raw  = tds[5].get_text(" ", strip=True) if len(tds) > 5 else ""
                age_m    = re.search(r"\((\d+)\)", age_raw)
                age      = int(age_m.group(1)) if age_m else 0
                mv_raw   = tds[12].get_text(strip=True) if len(tds) > 12 else ""
                mv_m     = _parse_value_m(mv_raw)

                if name and tm_player_id:
                    players.append({
                        "tm_player_id":   tm_player_id,
                        "espn_player_id": None,   # filled in during match update
                        "name":           name,
                        "slug":           player_slug,
                        "position":       position,
                        "age":            age,
                        "market_value_m": mv_m,
                        "fdorg_team_id":  fdorg_team_id,
                        "team_name":      team_name,
                    })

            if players:
                save_squad(fdorg_team_id, players)
                logger.info(f"  Scraped {len(players)} players for {team_name} (saison={saison})")
                return players

        except Exception as e:
            logger.debug(f"TM squad scrape error {slug}/{tm_id} saison={saison}: {e}")
        time.sleep(1.0)

    logger.warning(f"Could not scrape squad for fdorg_team_id={fdorg_team_id}")
    return []


def scrape_all_squads(
    fdorg_team_ids: list[int],
    force: bool = False,
    max_workers: int = 3,
) -> dict[int, list[dict]]:
    """
    Scrape TM squads for all given teams in parallel (max_workers threads).
    Returns {fdorg_team_id: [player_dict, ...]}
    """
    results: dict[int, list[dict]] = {}
    pending = [tid for tid in fdorg_team_ids if force or not squad_initialized(tid)]

    if not pending:
        logger.info("All squads already initialized — loading from cache.")
        return {tid: load_squad(tid) for tid in fdorg_team_ids}

    logger.info(f"Scraping {len(pending)} squad(s) from Transfermarkt ({max_workers} threads)...")

    def _scrape(tid: int):
        time.sleep(0.5)   # stagger starts
        return tid, scrape_team_squad(tid, force=force)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scrape, tid): tid for tid in pending}
        for fut in as_completed(futures):
            try:
                tid, squad = fut.result()
                results[tid] = squad
                logger.info(f"  ✓ team {tid}: {len(squad)} players")
            except Exception as e:
                logger.warning(f"  ✗ team {futures[fut]}: {e}")

    # Also load already-cached teams
    for tid in fdorg_team_ids:
        if tid not in results:
            results[tid] = load_squad(tid)

    return results


# ── ESPN match event scraping ─────────────────────────────────────────────────

def _espn_events_for_date(date_str: str) -> list[dict]:
    """Returns ESPN WC events for a date (YYYY-MM-DD)."""
    yyyymmdd = date_str.replace("-", "")
    try:
        r = _espn_session().get(
            f"{ESPN_BASE}/scoreboard?dates={yyyymmdd}", timeout=15
        )
        if r.status_code != 200:
            return []
        return r.json().get("events", [])
    except Exception as e:
        logger.debug(f"ESPN scoreboard error for {date_str}: {e}")
        return []


def _fuzzy_team_match(espn_name: str, candidates: list[str]) -> bool:
    """True if espn_name loosely matches any candidate (first 5 chars)."""
    en = espn_name.lower().strip()
    for c in candidates:
        cn = c.lower().strip()
        if en[:5] == cn[:5] or en in cn or cn in en:
            return True
    return False


def fetch_espn_event_id(
    date_str: str,
    home_name: str,
    away_name: str,
) -> Optional[int]:
    """
    Find the ESPN event ID for a given WC match by date + team names.
    Returns None if not found.
    """
    events = _espn_events_for_date(date_str)
    for ev in events:
        comps = ev.get("competitions", [{}])[0]
        teams = {t["homeAway"]: t["team"]["displayName"]
                 for t in comps.get("competitors", [])}
        home_espn = teams.get("home", "")
        away_espn = teams.get("away", "")
        if (_fuzzy_team_match(home_espn, [home_name]) and
                _fuzzy_team_match(away_espn, [away_name])):
            return int(ev["id"])
    return None


def _minutes_estimate(starter: bool, subbed_in: bool, subbed_out: bool) -> int:
    """
    Approximate minutes played from starter/sub flags.
    Exact minutes require detailed event parsing; this is a reasonable proxy.
    """
    if starter:
        return 65 if subbed_out else 90   # 65 = typical sub time for starters
    return 25 if subbed_in else 0         # 25 = typical minutes for a sub


def fetch_match_performances(
    espn_event_id: int,
    home_fdorg_id: int,
    away_fdorg_id: int,
    fdorg_fixture_id: int,
    match_date: str,
) -> list[dict]:
    """
    Fetch full lineup + player stats for a WC match from ESPN.
    Links ESPN team to fdorg_team_id by roster position (home=home, away=away).

    Also links ESPN player IDs back to TM squad entries (by fuzzy name matching)
    and saves the espn_player_id into the squad JSON for future lookups.

    Returns list of performance dicts ready for player_store.log_performances().
    """
    try:
        r = _espn_session().get(
            f"{ESPN_BASE}/summary?event={espn_event_id}", timeout=15
        )
        if r.status_code != 200:
            logger.warning(f"ESPN summary {espn_event_id} → {r.status_code}")
            return []
        d = r.json()
    except Exception as e:
        logger.warning(f"ESPN summary fetch error: {e}")
        return []

    rosters = d.get("rosters", [])
    if len(rosters) < 2:
        return []

    # ESPN rosters: home=index 0, away=index 1 (matches the competitor order)
    # Map team index to fdorg_id
    comps = d.get("header", {}).get("competitions", [{}])[0]
    espn_teams = comps.get("competitors", [])
    fdorg_ids = []
    for t in espn_teams:
        espn_tname = t.get("team", {}).get("displayName", "")
        home_squad = load_squad(home_fdorg_id)
        away_squad = load_squad(away_fdorg_id)
        home_names = [p["name"][:6].lower() for p in home_squad]
        # Use home/away order from competitors
        if t.get("homeAway") == "home":
            fdorg_ids.append(("home", home_fdorg_id, espn_tname))
        else:
            fdorg_ids.append(("away", away_fdorg_id, espn_tname))

    if len(fdorg_ids) < 2:
        fdorg_ids = [("home", home_fdorg_id, ""), ("away", away_fdorg_id, "")]

    records = []
    for (side, fdorg_id, team_espn_name), team_roster in zip(fdorg_ids, rosters):
        team_display = team_roster.get("team", {}).get("displayName", team_espn_name)
        squad = load_squad(fdorg_id)
        squad_by_tm = {p["name"].lower(): p for p in squad}

        for p in team_roster.get("roster", []):
            athlete   = p.get("athlete", {})
            espn_pid  = int(athlete.get("id", 0))
            espn_name = athlete.get("displayName", "")
            pos       = p.get("position", {}).get("abbreviation", "")
            starter   = bool(p.get("starter", False))
            sub_in    = bool(p.get("subbedIn", False))
            sub_out   = bool(p.get("subbedOut", False))
            active    = bool(p.get("active", True))

            if not active and not starter:
                continue   # unused sub

            stats = {s["name"]: float(s.get("value", 0)) for s in p.get("stats", [])}

            minutes = _minutes_estimate(starter, sub_in, sub_out)

            records.append({
                "fixture_id":      fdorg_fixture_id,
                "date":            match_date,
                "fdorg_team_id":   fdorg_id,
                "team_name":       team_display,
                "espn_player_id":  espn_pid,
                "player_name":     espn_name,
                "position":        pos,
                "starter":         starter,
                "subbed_in":       sub_in,
                "subbed_out":      sub_out,
                "minutes_est":     minutes,
                "goals":           stats.get("totalGoals", 0),
                "assists":         stats.get("goalAssists", 0),
                "yellow_cards":    stats.get("yellowCards", 0),
                "red_cards":       stats.get("redCards", 0),
                "own_goals":       stats.get("ownGoals", 0),
                "saves":           stats.get("saves", 0),
                "shots_on_target": stats.get("shotsOnTarget", 0),
                "fouls_committed": stats.get("foulsCommitted", 0),
            })

            # Back-link ESPN player ID into TM squad (fuzzy name match)
            _link_espn_to_squad(fdorg_id, squad, espn_pid, espn_name)

    # Replace estimated minutes with exact TM API minutes for both teams
    all_squad = load_squad(home_fdorg_id) + load_squad(away_fdorg_id)
    records = fix_wc_minutes_from_tm(records, all_squad, match_date)

    return records


def _link_espn_to_squad(
    fdorg_id: int,
    squad: list[dict],
    espn_pid: int,
    espn_name: str,
) -> None:
    """
    If ESPN player name matches a TM squad entry (fuzzy, first 6 chars of last name),
    save the espn_player_id into that squad entry for future reference.
    """
    espn_last = espn_name.split()[-1].lower()[:6] if espn_name.split() else ""
    for player in squad:
        tm_last = player["name"].split()[-1].lower()[:6]
        if espn_last and tm_last and espn_last == tm_last:
            if player.get("espn_player_id") != espn_pid:
                player["espn_player_id"] = espn_pid
                save_squad(fdorg_id, squad)   # persist the link
            return
