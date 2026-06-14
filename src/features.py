import json
import pandas as pd
import numpy as np
from src.config import ROLLING_WINDOW, DATA_DIR

_INITIAL_ELO_PATH = DATA_DIR / "initial_elo.json"
_ELO_SEEDS_PATH   = DATA_DIR / "elo_seeds.json"

# Aliases for names in our match data that differ from eloratings.net
_OPP_NAME_ALIAS: dict[str, str] = {
    "republic of ireland": "ireland",
    "china pr":            "china",
    "sao tome and principe": "são tomé and príncipe",
}

# Competition importance weights — higher = more signal, less noise
_COMP_WEIGHTS: list[tuple[str, float]] = [
    ("FIFA World Cup",              1.00),
    ("WC",                          1.00),   # live API code
    ("Copa America",                0.90),
    ("African Cup of Nations",      0.90),
    ("UEFA European Championship",  0.90),
    ("AFC Asian Cup",               0.90),
    ("CONCACAF Gold Cup",           0.85),
    ("qualification",               0.75),   # any qualifier
    ("Qualifying",                  0.75),
    ("Nations League",              0.70),
    ("Nations Cup",                 0.70),
    ("CONCACAF Nations",            0.70),
    ("Friendly",                    0.30),
]

def match_weight(comp: str) -> float:
    comp_l = comp.lower()
    for pattern, w in _COMP_WEIGHTS:
        if pattern.lower() in comp_l:
            return w
    return 0.60   # unknown competitive match


def _load_name_elo_maps() -> tuple[dict[str, float], dict[str, float]]:
    """
    Returns (name→current_elo, name→avg_elo) from elo_seeds.json.
    Used to resolve opp_id=0 opponents by name (58% of historical training rows).
    """
    if not _ELO_SEEDS_PATH.exists():
        return {}, {}
    try:
        raw = json.loads(_ELO_SEEDS_PATH.read_text())
        cur = {k: float(v) for k, v in raw.get("ratings_by_name", {}).items()}
        # avg_by_name not stored separately; use cur as fallback
        return cur, cur
    except Exception:
        return {}, {}


def _opp_elo_by_name(opp_name: str, name_elo_map: dict[str, float]) -> float:
    """Look up opponent Elo by team name string for opp_id=0 rows."""
    if not opp_name or not name_elo_map:
        return 1500.0
    normalized = str(opp_name).lower().strip()
    canonical  = _OPP_NAME_ALIAS.get(normalized, normalized)
    return name_elo_map.get(canonical, 1500.0)


def _load_initial_elo() -> dict:
    # Prefer live eloratings.net seeds (updated daily by elo_fetcher.refresh_elo_seeds)
    elo_seeds = DATA_DIR / "elo_seeds.json"
    if elo_seeds.exists():
        try:
            raw = json.loads(elo_seeds.read_text())
            fdorg_map = {int(k): float(v) for k, v in raw.get("ratings_by_fdorg", {}).items()}
            if fdorg_map:
                return fdorg_map
        except Exception:
            pass
    # Fallback: static FIFA World Ranking points seed
    if _INITIAL_ELO_PATH.exists():
        raw = json.loads(_INITIAL_ELO_PATH.read_text())
        return {int(k): float(v) for k, v in raw.get("ratings", {}).items()}
    return {}


def _elo_update(rating: float, opp_rating: float, outcome: str, k: float = 30) -> float:
    expected = 1 / (1 + 10 ** ((opp_rating - rating) / 400))
    actual   = {"W": 1.0, "D": 0.5, "L": 0.0}[outcome]
    return rating + k * (actual - expected)


def compute_elo_ratings(matches_df: pd.DataFrame) -> dict:
    """
    Seed from live FIFA World Ranking points (already incorporate all history),
    then update ONLY from WC 2026 live matches — not historical data that FIFA
    already baked into their points.
    """
    ratings = _load_initial_elo()
    if matches_df.empty:
        return ratings
    # Only update from live WC matches (comp == "WC"), not historical backfill
    wc_only = matches_df[matches_df["comp"] == "WC"].sort_values("date")
    for _, row in wc_only.iterrows():
        tid = int(row["team_id"])
        oid = int(row["opp_id"])
        r_t = ratings.get(tid, 1500.0)
        r_o = ratings.get(oid, 1500.0)
        ratings[tid] = _elo_update(r_t, r_o, row["outcome"])
    return ratings


def rolling_features_for_team(
    team_df: pd.DataFrame, as_of_date=None, window: int = ROLLING_WINDOW
) -> dict | None:
    """Rich per-team feature dict using only matches strictly before as_of_date."""
    df = team_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if as_of_date:
        df = df[df["date"] < pd.to_datetime(as_of_date)]
    df = df.sort_values("date")

    if len(df) < 3:
        return None

    # ── Long window (8 games) ──────────────────────────────────────────────────
    recent = df.tail(window)
    w8     = np.exp(np.linspace(-1, 0, len(recent))); w8 /= w8.sum()

    outcomes_bin  = recent["outcome"].map({"W": 1, "D": 0, "L": 0}).values
    outcomes_form = recent["outcome"].map({"W": 1, "D": 0.5, "L": 0}).values
    win_rate      = float(np.mean(outcomes_bin))
    draw_rate     = float((recent["outcome"] == "D").mean())
    gf_avg        = float(recent["gf"].mean())
    ga_avg        = float(recent["ga"].mean())
    gd_avg        = float(recent["gd"].mean())
    form_score    = float(np.dot(outcomes_form, w8))
    clean_sheet   = float((recent["ga"] == 0).mean())

    # ── Short window (3 games) — momentum ─────────────────────────────────────
    short = df.tail(3)
    win_rate_3 = float(short["outcome"].map({"W": 1, "D": 0, "L": 0}).mean())
    gf_avg_3   = float(short["gf"].mean())
    ga_avg_3   = float(short["ga"].mean())

    # ── Unbeaten streak ────────────────────────────────────────────────────────
    streak = 0
    for o in reversed(df["outcome"].tolist()):
        if o in ("W", "D"):
            streak += 1
        else:
            break

    return {
        "win_rate":    win_rate,
        "draw_rate":   draw_rate,
        "gf_avg":      gf_avg,
        "ga_avg":      ga_avg,
        "gd_avg":      gd_avg,
        "form_score":  form_score,
        "clean_sheet": clean_sheet,
        "win_rate_3":  win_rate_3,
        "gf_avg_3":    gf_avg_3,
        "ga_avg_3":    ga_avg_3,
        "streak":      float(min(streak, 10)) / 10.0,   # normalise 0-1
    }


def _h2h_features(home_id: int, away_id: int, df: pd.DataFrame, as_of) -> tuple[float, float]:
    """Win-rate and draw-rate for home_id vs away_id in historical head-to-head."""
    mask = (
        (df["team_id"] == home_id) & (df["opp_id"] == away_id) &
        (df["date"] < pd.to_datetime(as_of))
    )
    h2h = df[mask]
    if len(h2h) < 2:
        return 0.5, 0.25   # no history → neutral priors
    win_r  = float((h2h["outcome"] == "W").mean())
    draw_r = float((h2h["outcome"] == "D").mean())
    return win_r, draw_r


FEATURE_COLS = [
    # long-window diffs
    "d_win_rate", "d_draw_rate", "d_gf", "d_ga", "d_gd",
    "d_form", "d_clean_sheet",
    # short-window diffs (momentum)
    "d_win_rate_3", "d_gf_3", "d_ga_3",
    # streak
    "d_streak",
    # elo signals
    "d_elo",      # current Elo gap (updated from live WC matches)
    "d_avg_elo",  # historical average Elo gap (long-run baseline strength)
    # head-to-head
    "h2h_win_rate", "h2h_draw_rate",
    # venue
    "home_adv",
]


def build_match_feature_vector(
    home_feats: dict, away_feats: dict,
    home_elo: float = 1500.0, away_elo: float = 1500.0,
    is_home_perspective: bool = True,
    h2h_win_rate: float = 0.5, h2h_draw_rate: float = 0.25,
    home_avg_elo: float = 1500.0, away_avg_elo: float = 1500.0,
) -> np.ndarray:
    s = 1 if is_home_perspective else -1
    return np.array([
        s * (home_feats["win_rate"]    - away_feats["win_rate"]),
        s * (home_feats["draw_rate"]   - away_feats["draw_rate"]),
        s * (home_feats["gf_avg"]      - away_feats["gf_avg"]),
        s * (home_feats["ga_avg"]      - away_feats["ga_avg"]),
        s * (home_feats["gd_avg"]      - away_feats["gd_avg"]),
        s * (home_feats["form_score"]  - away_feats["form_score"]),
        s * (home_feats["clean_sheet"] - away_feats["clean_sheet"]),
        s * (home_feats["win_rate_3"]  - away_feats["win_rate_3"]),
        s * (home_feats["gf_avg_3"]    - away_feats["gf_avg_3"]),
        s * (home_feats["ga_avg_3"]    - away_feats["ga_avg_3"]),
        s * (home_feats["streak"]      - away_feats["streak"]),
        s * (home_elo     - away_elo)     / 400.0,
        s * (home_avg_elo - away_avg_elo) / 400.0,
        h2h_win_rate,
        h2h_draw_rate,
        float(is_home_perspective),
    ])


def build_training_data(matches_df: pd.DataFrame, elo_ratings: dict, avg_elo_ratings: dict = None):
    """Build (X, y, weights, feature_names) from all stored match rows."""
    avg_elo = avg_elo_ratings or {}
    name_cur, _name_avg = _load_name_elo_maps()
    X_rows, y_rows, w_rows = [], [], []
    df = matches_df.sort_values("date")

    fixtures = df[["fixture_id", "date", "team_id", "opp_id",
                   "opp_name", "is_home", "outcome", "comp"]].drop_duplicates(
        subset=["fixture_id", "team_id"])

    for _, row in fixtures.iterrows():
        tid, oid    = int(row["team_id"]), int(row["opp_id"])
        as_of       = row["date"]
        opp_name    = row.get("opp_name", "") or ""

        t_feats = rolling_features_for_team(df[df["team_id"] == tid], as_of_date=as_of)
        o_feats = rolling_features_for_team(df[df["team_id"] == oid], as_of_date=as_of)
        if t_feats is None or o_feats is None:
            continue

        t_elo = elo_ratings.get(tid, 1500.0)
        if oid == 0:
            # Resolve by opponent name — fixes 58% of training rows that used 1500
            o_elo = _opp_elo_by_name(opp_name, name_cur)
        else:
            o_elo = elo_ratings.get(oid, 1500.0)

        t_avg_elo = avg_elo.get(tid, t_elo)
        o_avg_elo = avg_elo.get(oid, o_elo)  # for oid=0, falls back to name-resolved o_elo

        h2h_w, h2h_d = _h2h_features(tid, oid, df, as_of)

        x = build_match_feature_vector(
            t_feats, o_feats, t_elo, o_elo,
            bool(row["is_home"]), h2h_w, h2h_d,
            t_avg_elo, o_avg_elo,
        )
        X_rows.append(x)
        y_rows.append(row["outcome"])
        w_rows.append(match_weight(str(row["comp"])))

    if not X_rows:
        return np.empty((0, len(FEATURE_COLS))), [], np.array([]), FEATURE_COLS

    return np.array(X_rows), y_rows, np.array(w_rows), FEATURE_COLS
