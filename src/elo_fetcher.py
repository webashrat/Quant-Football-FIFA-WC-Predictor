"""
elo_fetcher.py — Daily Elo seed refresh from eloratings.net.

Fetches real Elo ratings for 244 national teams and saves to data/elo_seeds.json.
Called at the start of each daily run; no-op if already done today.

features.py prefers these seeds over the static initial_elo.json
(which uses FIFA World Ranking points — a different, less principled scale).

Endpoints used:
  World.tsv   — 244 teams, columns: rank, rank, code, current_elo, ...,
                  highest_elo, ..., avg_elo (col[7]), ..., GP, W, D, L
  en.teams.tsv — code→name mapping with all known alternate names

Additional endpoint (available, not yet parsed):
  graph.tsv   — compact full match history since 1872 in format:
                  YYYYMMDD (full date marker, at year boundaries)
                  or {day_of_month}{home_code}{away_code}{elo_delta}... (packed rows)
                  Useful for reconstructing historical Elo series per team,
                  fixing look-ahead bias in d_avg_elo training feature.
"""

import json
import logging
import requests
from datetime import date
from pathlib import Path

from src.config import DATA_DIR

_ELO_SEEDS_PATH = DATA_DIR / "elo_seeds.json"
_WORLD_TSV_URL  = "https://eloratings.net/World.tsv"
_NAMES_TSV_URL  = "https://eloratings.net/en.teams.tsv"
_HTTP_HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; WCPredictor/1.0)"}

logger = logging.getLogger(__name__)

# Parquet team name → eloratings.net canonical name where they differ
_ALIAS: dict[str, str] = {
    "Bosnia-Herzegovina":  "Bosnia and Herzegovina",
    "Cape Verde Islands":  "Cape Verde",
    "Congo DR":            "DR Congo",
    "United States":       "United States",
}

# FDORG ID (football-data.org) → team name as it appears in our parquet.
# Covers all 48 WC 2026 qualified teams.
_FDORG_NAMES: dict[int, str] = {
    758:  "Uruguay",
    759:  "Germany",
    760:  "Spain",
    761:  "Paraguay",
    762:  "Argentina",
    763:  "Ghana",
    764:  "Brazil",
    765:  "Portugal",
    766:  "Japan",
    769:  "Mexico",
    770:  "England",
    771:  "United States",
    772:  "South Korea",
    773:  "France",
    774:  "South Africa",
    778:  "Algeria",
    779:  "Australia",
    783:  "New Zealand",
    788:  "Switzerland",
    791:  "Ecuador",
    792:  "Sweden",
    798:  "Czechia",
    799:  "Croatia",
    801:  "Saudi Arabia",
    802:  "Tunisia",
    803:  "Turkey",
    804:  "Senegal",
    805:  "Belgium",
    815:  "Morocco",
    816:  "Austria",
    818:  "Colombia",
    825:  "Egypt",
    828:  "Canada",
    836:  "Haiti",
    840:  "Iran",
    1060: "Bosnia-Herzegovina",
    1836: "Panama",
    1930: "Cape Verde Islands",
    1934: "Congo DR",
    1935: "Ivory Coast",
    8030: "Qatar",
    8049: "Jordan",
    8062: "Iraq",
    8070: "Uzbekistan",
    8601: "Netherlands",
    8872: "Norway",
    8873: "Scotland",
    9460: "Curaçao",
}


def _fetch_world_data() -> tuple[dict[str, float], dict[str, float]]:
    """
    Fetches World.tsv and en.teams.tsv.

    Returns two dicts, both keyed by lowercase team name (all aliases included):
      name_to_elo     — current Elo rating (col[3])
      name_to_avg_elo — historical average Elo (col[7])

    World.tsv column layout (31 cols):
      [0]  current rank
      [1]  previous rank
      [2]  country code
      [3]  current Elo
      [4]  rank change (recent)
      [5]  highest ever Elo
      [6]  rank at highest
      [7]  historical average Elo
      [8]  rank by average
      [9]  lowest Elo
      [10..15] short-term rank/change data
      [16] rank change 1-week
      [17] Elo change 1-year
      [19] Elo change 5-year
      [22] total matches played
      [23] wins  [24] draws  [25] losses
    """
    try:
        r_world = requests.get(_WORLD_TSV_URL, headers=_HTTP_HEADERS, timeout=20)
        r_world.raise_for_status()

        code_to_elo:     dict[str, float] = {}
        code_to_avg_elo: dict[str, float] = {}
        for line in r_world.content.decode("utf-8").strip().split("\n"):
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            code = parts[2].strip()
            try:
                code_to_elo[code]     = float(parts[3])
                code_to_avg_elo[code] = float(parts[7])
            except ValueError:
                pass

        r_names = requests.get(_NAMES_TSV_URL, headers=_HTTP_HEADERS, timeout=20)
        r_names.raise_for_status()

        name_to_elo:     dict[str, float] = {}
        name_to_avg_elo: dict[str, float] = {}
        for line in r_names.content.decode("utf-8").strip().split("\n"):
            parts = [p.strip() for p in line.split("\t")]
            if not parts or len(parts) < 2:
                continue
            code = parts[0]
            elo     = code_to_elo.get(code)
            avg_elo = code_to_avg_elo.get(code)
            if elo is None:
                continue
            for name in parts[1:]:
                if name:
                    name_to_elo[name.lower()]     = elo
                    if avg_elo is not None:
                        name_to_avg_elo[name.lower()] = avg_elo

        logger.info(
            "elo_fetcher: fetched %d current + %d avg entries from eloratings.net",
            len(name_to_elo), len(name_to_avg_elo),
        )
        return name_to_elo, name_to_avg_elo

    except Exception as exc:
        logger.warning("elo_fetcher: fetch failed — %s", exc)
        return {}, {}


def _build_fdorg_map(
    name_to_cur: dict[str, float],
    name_to_avg: dict[str, float],
) -> tuple[dict[int, float], dict[int, float]]:
    """Resolves FDORG IDs → current and avg Elo via name matching + alias table."""
    cur: dict[int, float] = {}
    avg: dict[int, float] = {}
    missing: list[str] = []

    for fdorg_id, raw_name in _FDORG_NAMES.items():
        canonical = _ALIAS.get(raw_name, raw_name).lower()
        c = name_to_cur.get(canonical)
        a = name_to_avg.get(canonical)
        if c is not None:
            cur[fdorg_id] = c
            if a is not None:
                avg[fdorg_id] = a
        else:
            missing.append(f"{fdorg_id}={raw_name}")

    if missing:
        logger.warning("elo_fetcher: no eloratings match for: %s", ", ".join(missing))
    return cur, avg


def refresh_elo_seeds(force: bool = False) -> None:
    """
    Refresh data/elo_seeds.json from eloratings.net if not already done today.
    Pass force=True to re-fetch regardless of cache date.

    Output format:
      {
        "fetched_date":         "2026-06-14",
        "source":               "eloratings.net",
        "ratings_by_fdorg":     {"762": 2115.0, ...},  # current Elo, 48 WC teams
        "avg_ratings_by_fdorg": {"762": 1987.0, ...},  # historical avg Elo
        "ratings_by_name":      {"argentina": 2115.0, ...}  # current, 244 teams
      }
    """
    today_str = date.today().isoformat()

    if not force and _ELO_SEEDS_PATH.exists():
        try:
            cached = json.loads(_ELO_SEEDS_PATH.read_text())
            if cached.get("fetched_date") == today_str:
                logger.debug("elo_fetcher: seeds are current (fetched today)")
                return
        except Exception:
            pass

    logger.info("elo_fetcher: refreshing Elo seeds from eloratings.net...")
    name_to_elo, name_to_avg = _fetch_world_data()
    if not name_to_elo:
        logger.warning("elo_fetcher: empty result — keeping existing seeds")
        return

    fdorg_cur, fdorg_avg = _build_fdorg_map(name_to_elo, name_to_avg)

    payload = {
        "fetched_date":         today_str,
        "source":               "eloratings.net",
        "ratings_by_fdorg":     {str(k): v for k, v in sorted(fdorg_cur.items())},
        "avg_ratings_by_fdorg": {str(k): v for k, v in sorted(fdorg_avg.items())},
        "ratings_by_name":      name_to_elo,
    }
    _ELO_SEEDS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "elo_fetcher: saved %d FDORG (cur+avg) + %d name seeds → %s",
        len(fdorg_cur), len(name_to_elo), _ELO_SEEDS_PATH,
    )


def get_fdorg_elo_map() -> dict[int, float]:
    """Load the cached FDORG → Elo map. Returns empty dict if not yet fetched."""
    if not _ELO_SEEDS_PATH.exists():
        return {}
    try:
        raw = json.loads(_ELO_SEEDS_PATH.read_text())
        return {int(k): float(v) for k, v in raw.get("ratings_by_fdorg", {}).items()}
    except Exception:
        return {}


def get_avg_fdorg_elo_map() -> dict[int, float]:
    """Load the cached FDORG → historical average Elo map."""
    if not _ELO_SEEDS_PATH.exists():
        return {}
    try:
        raw = json.loads(_ELO_SEEDS_PATH.read_text())
        return {int(k): float(v) for k, v in raw.get("avg_ratings_by_fdorg", {}).items()}
    except Exception:
        return {}


def get_name_elo_map() -> dict[str, float]:
    """
    Load the name → Elo map for all 244 teams (lowercase keys).
    Useful for looking up Elo of opponents not in the FDORG system.
    """
    if not _ELO_SEEDS_PATH.exists():
        return {}
    try:
        raw = json.loads(_ELO_SEEDS_PATH.read_text())
        return {k: float(v) for k, v in raw.get("ratings_by_name", {}).items()}
    except Exception:
        return {}
