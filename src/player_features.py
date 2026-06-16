"""
player_features.py — player-level intelligence from Transfermarkt.

For each WC 2026 team, fetches (with 1-day disk cache):
  • Squad market value (€M) — quality + depth proxy
  • Top 5 players by value (star-player signal)
  • Injury list — name, type, return date, value lost
  • Suspension list — name, value lost
  • Squad average age
  • Availability rate — usable squad value / total squad value

Data is cached daily at data/player_cache/{tm_id}_{date}.json.
Falls back to neutral defaults if Transfermarkt is unreachable.

Coverage: post-Qatar-2022 squads (saison_id 2023 → 2025, newest available).
"""

import json
import re
import time
import logging
from pathlib import Path
from datetime import date

import requests
from bs4 import BeautifulSoup

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "player_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SESSION = None

# ── fdorg team_id → (Transfermarkt verein_id, tm_slug, expected_h1_keyword) ──
# All 48 WC 2026 teams. Verified: each TM squad page h1 contains expected_h1_keyword.
FDORG_TO_TM: dict[int, tuple[int, str, str]] = {
    762:  (3437,  "argentinien",                   "argentin"),  # Argentina
    760:  (3375,  "spanien",                        "spain"),     # Spain
    773:  (3377,  "frankreich",                     "france"),    # France
    770:  (3299,  "england",                        "england"),   # England
    765:  (3300,  "portugal",                       "portugal"),  # Portugal
    764:  (3439,  "brasilien",                      "brazil"),    # Brazil
    815:  (3575,  "marokko",                        "morocco"),   # Morocco
    8601: (3379,  "niederlande",                    "nether"),    # Netherlands
    805:  (3382,  "belgien",                        "belgi"),     # Belgium
    759:  (3262,  "deutschland",                    "germany"),   # Germany
    799:  (3556,  "kroatien",                       "croat"),     # Croatia
    769:  (6303,  "mexiko",                         "mexico"),    # Mexico
    818:  (3816,  "kolumbien",                      "colomb"),    # Colombia
    804:  (3499,  "senegal",                        "senegal"),   # Senegal
    758:  (3449,  "uruguay",                        "uruguay"),   # Uruguay
    771:  (3505,  "vereinigte-staaten",             "united"),    # USA
    766:  (3435,  "japan",                          "japan"),     # Japan
    788:  (3384,  "schweiz",                        "switz"),     # Switzerland
    840:  (3582,  "iran",                           "iran"),      # Iran
    772:  (3589,  "sudkorea",                       "korea"),     # South Korea
    803:  (3381,  "turkei",                         "turk"),      # Turkey
    791:  (5750,  "ecuador",                        "ecuador"),   # Ecuador
    816:  (3383,  "osterreich",                     "austria"),   # Austria
    779:  (3433,  "australien",                     "austral"),   # Australia
    778:  (3614,  "algerien",                       "alger"),     # Algeria
    825:  (3672,  "agypten",                        "egypt"),     # Egypt
    828:  (3510,  "kanada",                         "canada"),    # Canada
    1935: (3591,  "elfenbeinkuste",                 "ivory"),     # Ivory Coast
    1836: (3577,  "panama",                         "panama"),    # Panama
    761:  (3581,  "paraguay",                       "paraguay"),  # Paraguay
    798:  (3445,  "tschechien",                     "czech"),     # Czechia
    802:  (3670,  "tunesien",                       "tunis"),     # Tunisia
    1934: (3854,  "demokratische-republik-kongo",   "democrat"),  # DR Congo
    8070: (3563,  "usbekistan",                     "uzbek"),     # Uzbekistan
    8030: (14162, "katar",                          "qatar"),     # Qatar
    8062: (3560,  "irak",                           "iraq"),      # Iraq
    801:  (3807,  "saudi-arabien",                  "saudi"),     # Saudi Arabia
    8049: (15737, "jordanien",                      "jordan"),    # Jordan
    1060: (3446,  "bosnien-herzegowina",            "bosn"),      # Bosnia-Herzegovina
    1930: (4311,  "kap-verde",                      "verde"),     # Cape Verde
    763:  (3441,  "ghana",                          "ghana"),     # Ghana
    9460: (32364, "curacao",                        "cura"),      # Curaçao
    783:  (9171,  "neuseeland",                     "zealand"),   # New Zealand
    774:  (3806,  "sudafrika",                      "africa"),    # South Africa
    836:  (14161, "haiti",                          "haiti"),     # Haiti
    8873: (3380,  "schottland",                      "scotland"),  # Scotland
    8872: (3440,  "norwegen",                        "norway"),    # Norway
}


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.transfermarkt.com/",
        })
    return _SESSION


def _parse_value_m(text: str) -> float:
    """Parse '€40.00m', '€500k', '€1.20bn' → float millions."""
    if not text:
        return 0.0
    text = text.strip().replace(",", ".")
    m = re.search(r"[\d.]+", text)
    if not m:
        return 0.0
    num = float(m.group())
    tl = text.lower()
    if "bn" in tl:
        return num * 1000.0
    if "m" in tl:
        return num
    if "k" in tl:
        return num / 1000.0
    return num


def _cache_path(tm_id: int, today: str) -> Path:
    return CACHE_DIR / f"{tm_id}_{today}.json"


def _load_cache(tm_id: int, today: str) -> dict | None:
    p = _cache_path(tm_id, today)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cache(tm_id: int, today: str, data: dict) -> None:
    try:
        _cache_path(tm_id, today).write_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


def _default_features() -> dict:
    return {
        "squad_value_m":      0.0,
        "avg_player_value_m": 0.0,
        "avg_age":            27.0,
        "squad_size":         26,
        "top_players":        [],      # [{name, value_m, position}]
        "injured":            [],      # [{name, value_m, injury, return_date}]
        "suspended":          [],      # [{name, value_m}]
        "available_value_m":  0.0,
        "availability_pct":   1.0,
        "source":             "default",
    }


def _scrape_squad(tm_id: int, slug: str, today: str, expected_keyword: str = "") -> dict:
    """Scrape squad page → player list with market values."""
    session = _get_session()

    # Try current season first, fall back to previous
    for saison in ["2025", "2024"]:
        url = (
            f"https://www.transfermarkt.com/{slug}/kader/verein/{tm_id}"
            f"/saison_id/{saison}/plus/1"
        )
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            # Validate we landed on the expected team using the English team keyword
            h1 = soup.find("h1")
            if not h1:
                continue
            h1_text = h1.get_text(" ", strip=True).lower()
            canonical = soup.find("title")
            canonical_text = canonical.get_text().lower() if canonical else ""
            page_text = h1_text + " " + canonical_text
            if expected_keyword and expected_keyword.lower() not in page_text:
                logger.debug(
                    f"TM ID {tm_id} returned wrong team '{h1_text[:30]}' "
                    f"(expected '{expected_keyword}') — skipping saison={saison}"
                )
                continue

            rows = [row for row in soup.select("table.items > tbody > tr") if row.find("td")]
            if len(rows) < 5:
                continue

            # Squad details header
            avg_age = 27.0
            squad_size = len(rows)
            for li in soup.select("div.data-header__details li"):
                txt = li.get_text(" ", strip=True)
                if "Average age" in txt:
                    m = re.search(r"[\d.]+", txt)
                    if m:
                        avg_age = float(m.group())
                if "Squad size" in txt:
                    m = re.search(r"\d+", txt)
                    if m:
                        squad_size = int(m.group())

            # Parse player rows
            players = []
            for row in rows:
                tds = row.find_all("td")
                if len(tds) < 13:
                    continue
                # td[3] = player name, td[4] = position, td[12] = market value
                name = tds[3].get_text(" ", strip=True) if len(tds) > 3 else ""
                pos  = tds[4].get_text(" ", strip=True) if len(tds) > 4 else ""
                mv_raw = tds[12].get_text(strip=True) if len(tds) > 12 else ""
                mv_m = _parse_value_m(mv_raw)
                if name:
                    players.append({"name": name, "position": pos, "value_m": mv_m})

            if not players:
                continue

            players.sort(key=lambda p: p["value_m"], reverse=True)
            total_value = sum(p["value_m"] for p in players)

            return {
                "players":   players,
                "avg_age":   avg_age,
                "squad_size": squad_size,
                "total_value_m": total_value,
                "saison":    saison,
            }
        except Exception as e:
            logger.debug(f"TM squad scrape failed for {slug}/{tm_id} saison={saison}: {e}")
            time.sleep(0.5)
            continue

    return {}


def _scrape_injuries(tm_id: int, slug: str) -> list[dict]:
    """Scrape injury page → list of currently injured players."""
    session = _get_session()
    url = f"https://www.transfermarkt.com/{slug}/verletzungen/verein/{tm_id}"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        rows = soup.select("table.items > tbody > tr")
        injured = []
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 6:
                continue
            # TM injury page columns: #, player, injury, since, until, games_missed, value
            name_el = tds[1].find("a")
            name = name_el.get_text(" ", strip=True) if name_el else tds[1].get_text(" ", strip=True)
            injury_type = tds[2].get_text(" ", strip=True) if len(tds) > 2 else "Unknown"
            since       = tds[3].get_text(" ", strip=True) if len(tds) > 3 else ""
            until       = tds[4].get_text(" ", strip=True) if len(tds) > 4 else "Unknown"
            mv_raw      = tds[-1].get_text(strip=True) if tds else ""
            mv_m = _parse_value_m(mv_raw)
            if name:
                injured.append({
                    "name":        name,
                    "injury":      injury_type,
                    "out_since":   since,
                    "return_date": until,
                    "value_m":     mv_m,
                })
        return injured
    except Exception as e:
        logger.debug(f"TM injury scrape failed for {slug}/{tm_id}: {e}")
        return []


def _scrape_suspensions(tm_id: int, slug: str, squad_players: list[dict]) -> list[dict]:
    """
    Scrape national team suspension/red-card data.
    TM doesn't have a dedicated suspension page for national teams, so we check
    the team's 'sperren' (bans) section if available, otherwise return [].
    """
    session = _get_session()
    url = f"https://www.transfermarkt.com/{slug}/sperren/verein/{tm_id}"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        rows = soup.select("table.items > tbody > tr")
        suspended = []
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 4:
                continue
            name_el = tds[1].find("a")
            name = name_el.get_text(" ", strip=True) if name_el else tds[1].get_text(" ", strip=True)
            reason = tds[2].get_text(" ", strip=True) if len(tds) > 2 else ""
            until  = tds[3].get_text(" ", strip=True) if len(tds) > 3 else ""
            # Match name to squad to get market value
            mv_m = 0.0
            for p in squad_players:
                if p["name"].lower() in name.lower() or name.lower() in p["name"].lower():
                    mv_m = p["value_m"]
                    break
            if name:
                suspended.append({"name": name, "reason": reason, "until": until, "value_m": mv_m})
        return suspended
    except Exception as e:
        logger.debug(f"TM suspension scrape failed for {slug}/{tm_id}: {e}")
        return []


def get_team_player_features(fdorg_id: int, today: str | None = None) -> dict:
    """
    Main interface. Returns player intelligence dict for a team.

    Keys:
      squad_value_m      — total squad value (€M)
      avg_player_value_m — per-player average (€M)
      avg_age            — squad average age
      squad_size         — number of players in squad
      top_players        — list of top-5 players [{name, value_m, position}]
      injured            — currently injured [{name, value_m, injury, return_date}]
      suspended          — currently suspended [{name, value_m, reason}]
      available_value_m  — squad_value_m minus injured+suspended values
      availability_pct   — available_value_m / squad_value_m (0-1)
      source             — "transfermarkt" | "cache" | "default"
    """
    today = today or date.today().isoformat()
    tm_entry = FDORG_TO_TM.get(fdorg_id)
    if not tm_entry or tm_entry[0] is None:
        return _default_features()

    tm_id, slug, expected_keyword = tm_entry

    # ── Check daily cache ────────────────────────────────────────────────────
    cached = _load_cache(tm_id, today)
    if cached:
        cached["source"] = "cache"
        # Always inject live club season data (not part of TM scraping cache)
        if not cached.get("club_season"):
            from src.player_store import load_squad as _load_sq
            from src.tm_api import get_team_club_season_summary as _cs_summary
            cached["club_season"] = _cs_summary(_load_sq(fdorg_id))
        return cached

    # ── Scrape Transfermarkt ─────────────────────────────────────────────────
    squad_data = _scrape_squad(tm_id, slug, today, expected_keyword)
    if not squad_data:
        return _default_features()

    time.sleep(0.8)   # polite rate limit between squad and injury requests
    injured   = _scrape_injuries(tm_id, slug)
    time.sleep(0.8)
    suspended = _scrape_suspensions(tm_id, slug, squad_data.get("players", []))

    players     = squad_data.get("players", [])
    total_value = squad_data.get("total_value_m", 0.0)
    avg_age     = squad_data.get("avg_age", 27.0)
    squad_size  = squad_data.get("squad_size", len(players))

    lost_value  = sum(p["value_m"] for p in injured) + sum(p["value_m"] for p in suspended)
    avail_value = max(0.0, total_value - lost_value)
    avail_pct   = (avail_value / total_value) if total_value > 0 else 1.0

    # ── Club season stats from TM API (cached in squad JSON) ───────────────────
    from src.player_store import load_squad
    from src.tm_api import get_team_club_season_summary
    stored_squad = load_squad(fdorg_id)
    club_summary = get_team_club_season_summary(stored_squad) if stored_squad else {}

    result = {
        "squad_value_m":      round(total_value, 2),
        "avg_player_value_m": round(total_value / max(squad_size, 1), 2),
        "avg_age":            avg_age,
        "squad_size":         squad_size,
        "top_players":        players[:5],
        "injured":            injured,
        "suspended":          suspended,
        "available_value_m":  round(avail_value, 2),
        "availability_pct":   round(avail_pct, 4),
        "club_season":        club_summary,
        "source":             "transfermarkt",
    }

    _save_cache(tm_id, today, result)
    return result


def availability_prob_nudge(home_feat: dict, away_feat: dict) -> float:
    """
    Compute a probability adjustment (in percentage points) for home team
    based on relative squad availability values.

    Positive = home team benefits from opponent's absentees.
    Capped at ±8pp to avoid overriding the ML signal.
    """
    h_val = home_feat.get("available_value_m", 0.0)
    a_val = away_feat.get("available_value_m", 0.0)
    if h_val == 0.0 and a_val == 0.0:
        return 0.0
    total = h_val + a_val
    if total == 0:
        return 0.0
    # Naive share: if home has 60% of available value → +4pp nudge
    home_share = h_val / total       # 0-1, 0.5 = neutral
    nudge_pp   = (home_share - 0.5) * 16.0   # ±8pp max
    return round(nudge_pp, 2)


def format_player_block(team_name: str, feat: dict, width: int = 28) -> list[str]:
    """Return list of display lines for one team's player context."""
    lines = []
    sv    = feat["squad_value_m"]
    av    = feat["available_value_m"]
    age   = feat["avg_age"]
    src   = feat["source"]

    lines.append(f"  {team_name}  [source: {src}]")
    if sv > 0:
        lines.append(f"    Squad value : €{sv:.1f}M  |  Avail: €{av:.1f}M  |  Age: {age:.1f}")
    else:
        lines.append(f"    Squad value : n/a (no TM data)  |  Age: {age:.1f}")

    # Top-5 players
    if feat["top_players"]:
        lines.append(f"    Top players :")
        for p in feat["top_players"][:5]:
            status = "FIT"
            inj_names = [i["name"].lower() for i in feat["injured"]]
            sus_names = [s["name"].lower() for s in feat["suspended"]]
            pname_l = p["name"].lower()
            if any(pname_l in n or n in pname_l for n in inj_names):
                status = "INJURED"
            elif any(pname_l in n or n in pname_l for n in sus_names):
                status = "SUSPENDED"
            flag = "✓" if status == "FIT" else "✗"
            lines.append(f"      {flag} {p['name']:<24} €{p['value_m']:.1f}M  {status}")

    # Club season form (from TM API)
    cs = feat.get("club_season", {})
    if cs.get("top_scorers"):
        lines.append(f"    Club season form (25/26) :")
        lines.append(f"      {'Player':<24}  {'Apps':>4}  {'Min':>4}  {'G':>3}  {'A':>3}")
        for p in cs["top_scorers"][:5]:
            lines.append(
                f"      {p['name']:<24}  {p['apps']:>4}  {p['minutes']:>4}  "
                f"{p['goals']:>3}  {p['assists']:>3}"
            )
    elif cs.get("players_with_data", 0) == 0:
        lines.append(f"    Club season form : pending enrichment (run enrich cmd)")

    # Injuries
    if feat["injured"]:
        lines.append(f"    Injuries ({len(feat['injured'])}) :")
        for inj in feat["injured"][:5]:
            ret = inj.get("return_date", "?") or "?"
            lines.append(f"      ✗ {inj['name']:<24} {inj['injury'][:20]:<20}  return: {ret}")
    else:
        lines.append(f"    Injuries    : None reported")

    # Suspensions
    if feat["suspended"]:
        lines.append(f"    Suspensions ({len(feat['suspended'])}) :")
        for sus in feat["suspended"][:3]:
            lines.append(f"      ✗ {sus['name']:<24} until: {sus.get('until','?')}")
    else:
        lines.append(f"    Suspensions : None reported")

    return lines
