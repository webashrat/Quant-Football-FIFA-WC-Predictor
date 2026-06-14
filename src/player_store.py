"""
player_store.py — persistent storage for all WC 2026 player data.

Directory layout:
  data/players/
    squads/          ← one JSON per team: basic squad + market values (from TM)
    wc_performances.parquet  ← row per player per WC match
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

_SQUADS_DIR = DATA_DIR / "players" / "squads"
_PERF_FILE  = DATA_DIR / "players" / "wc_performances.parquet"

_PERF_COLS = [
    "fixture_id",       # fdorg fixture ID (or ESPN event ID if negative)
    "date",
    "fdorg_team_id",
    "team_name",
    "espn_player_id",
    "player_name",
    "position",
    "starter",          # bool
    "subbed_in",        # bool
    "subbed_out",       # bool
    "minutes_est",      # estimated minutes played
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "saves",            # GK only
    "shots_on_target",
    "fouls_committed",
]

_SQUAD_COLS = [
    "tm_player_id",
    "espn_player_id",   # filled in after first match
    "name",
    "slug",
    "position",
    "age",
    "market_value_m",
    "fdorg_team_id",
    "team_name",
    "nationality",
    "club",
]


def _ensure_dirs() -> None:
    _SQUADS_DIR.mkdir(parents=True, exist_ok=True)
    _PERF_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Squad helpers ─────────────────────────────────────────────────────────────

def save_squad(fdorg_team_id: int, players: list[dict]) -> None:
    _ensure_dirs()
    path = _SQUADS_DIR / f"{fdorg_team_id}.json"
    path.write_text(json.dumps(players, ensure_ascii=False, indent=2))


def load_squad(fdorg_team_id: int) -> list[dict]:
    _ensure_dirs()
    path = _SQUADS_DIR / f"{fdorg_team_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []


def squad_initialized(fdorg_team_id: int) -> bool:
    return (_SQUADS_DIR / f"{fdorg_team_id}.json").exists()


def all_initialized_teams() -> list[int]:
    _ensure_dirs()
    return [int(p.stem) for p in _SQUADS_DIR.glob("*.json")]


# ── Performance helpers ───────────────────────────────────────────────────────

def load_performances() -> pd.DataFrame:
    _ensure_dirs()
    if _PERF_FILE.exists():
        try:
            return pd.read_parquet(_PERF_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=_PERF_COLS)


def log_performances(records: list[dict]) -> None:
    """Append match performance records (deduped on fixture_id + espn_player_id)."""
    if not records:
        return
    _ensure_dirs()
    new_df = pd.DataFrame(records)
    for col in _PERF_COLS:
        if col not in new_df.columns:
            new_df[col] = None

    existing = load_performances()
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["fixture_id", "espn_player_id"], keep="last"
        )
    else:
        combined = new_df

    combined.to_parquet(_PERF_FILE, index=False)
    logger.info(f"Logged {len(records)} player performance rows → {_PERF_FILE}")


def get_player_wc_totals(fdorg_team_id: int | None = None) -> pd.DataFrame:
    """
    Returns cumulative WC stats per player (across all matches).
    Optionally filtered to one team.
    """
    df = load_performances()
    if df.empty:
        return df
    if fdorg_team_id is not None:
        df = df[df["fdorg_team_id"] == fdorg_team_id]
    return (
        df.groupby(["espn_player_id", "player_name", "fdorg_team_id", "team_name"])
        .agg(
            matches     = ("fixture_id",     "count"),
            starts      = ("starter",        "sum"),
            minutes_est = ("minutes_est",    "sum"),
            goals       = ("goals",          "sum"),
            assists     = ("assists",         "sum"),
            yellow_cards= ("yellow_cards",   "sum"),
            red_cards   = ("red_cards",      "sum"),
        )
        .reset_index()
    )


def get_team_wc_summary(fdorg_team_id: int) -> dict:
    """
    Returns feature-ready dict for a team:
      goals_per_match, assists_per_match, yellow_cards_total,
      suspended_next_match (yellow_cards >= 2), avg_minutes_per_player
    """
    totals = get_player_wc_totals(fdorg_team_id)
    if totals.empty:
        return {
            "wc_matches_played":   0,
            "goals_per_match":     0.0,
            "assists_per_match":   0.0,
            "yellow_cards_total":  0,
            "suspension_risk":     [],   # players on 1+ yellow
            "avg_minutes":         0.0,
        }
    n_matches = int(totals["matches"].max()) if not totals.empty else 1
    return {
        "wc_matches_played":  n_matches,
        "goals_per_match":    float(totals["goals"].sum() / max(n_matches, 1)),
        "assists_per_match":  float(totals["assists"].sum() / max(n_matches, 1)),
        "yellow_cards_total": int(totals["yellow_cards"].sum()),
        "suspension_risk":    totals[totals["yellow_cards"] >= 2]["player_name"].tolist(),
        "avg_minutes":        float(totals["minutes_est"].mean()),
    }
