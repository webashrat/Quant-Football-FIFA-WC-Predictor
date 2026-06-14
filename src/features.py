import json
import pandas as pd
import numpy as np
from src.config import ROLLING_WINDOW, DATA_DIR

_INITIAL_ELO_PATH = DATA_DIR / "initial_elo.json"

# Approximate Elo ratings for teams that appear in match history but are not
# in the FDORG system (opp_id == 0). Used by perf_delta to avoid defaulting
# every unknown opponent to 1500 regardless of their actual quality.
# Values sourced from FIFA World Ranking / World Football Elo (June 2026 approx).
_NAME_ELO_FALLBACK: dict[str, float] = {
    # Top Europeans (would dominate at 1500 default)
    "Italy": 1700, "England": 1750, "Denmark": 1660, "Poland": 1590,
    "Serbia": 1620, "Ukraine": 1580, "Wales": 1540, "Hungary": 1570,
    "Slovakia": 1510, "Georgia": 1520, "Iceland": 1530, "Greece": 1520,
    "Romania": 1510, "Slovenia": 1530, "Albania": 1490, "North Macedonia": 1460,
    "Republic of Ireland": 1490, "Bulgaria": 1440, "Montenegro": 1450,
    "Armenia": 1430, "Azerbaijan": 1420, "Kosovo": 1430, "Latvia": 1400,
    "Moldova": 1390, "Estonia": 1400, "Luxembourg": 1400, "Belarus": 1390,
    "Cyprus": 1420, "Lithuania": 1390, "Faroe Islands": 1380,
    "Northern Ireland": 1390, "Russia": 1560,
    "Israel": 1500, "Finland": 1510,
    # Near-zero nations
    "Gibraltar": 1080, "Liechtenstein": 1120, "Andorra": 1120,
    "Malta": 1180, "San Marino": 1000,
    # South America
    "Costa Rica": 1570, "Peru": 1590, "Bolivia": 1480, "Chile": 1620,
    "Venezuela": 1530,
    # CONCACAF
    "Honduras": 1460, "Jamaica": 1390, "El Salvador": 1420,
    "Guatemala": 1410, "Trinidad and Tobago": 1380, "Cuba": 1200,
    "Martinique": 1300, "Nicaragua": 1270, "Bermuda": 1220,
    "Grenada": 1250, "Suriname": 1280, "Guyana": 1220,
    "Barbados": 1100, "Saint Lucia": 1080, "Cayman Islands": 1050,
    "Dominica": 1080, "Belize": 1100, "Anguilla": 1000,
    "Saint Kitts and Nevis": 1080, "British Virgin Islands": 1000,
    "Saint Vincent and the Grenadines": 1100, "Turks and Caicos Islands": 1000,
    "Montserrat": 1050,
    "Guadeloupe": 1300, "Aruba": 1150, "Dominican Republic": 1150,
    # Africa
    "Nigeria": 1590, "Cameroon": 1580, "Burkina Faso": 1500, "Mali": 1510,
    "Angola": 1440, "Zambia": 1420, "Mozambique": 1350, "Mauritania": 1380,
    "Guinea": 1460, "Togo": 1390, "Gabon": 1400, "Eswatini": 1250,
    "Zimbabwe": 1370, "Malawi": 1350, "Namiba": 1350, "Namibia": 1350,
    "Tanzania": 1340, "Uganda": 1400, "Liberia": 1340, "Rwanda": 1360,
    "Equatorial Guinea": 1380, "Comoros": 1340, "Benin": 1400,
    "Guinea-Bissau": 1360, "Niger": 1300, "Central African Republic": 1290,
    "Sierra Leone": 1360, "Gambia": 1410, "Burundi": 1280,
    "South Sudan": 1200, "Djibouti": 1150, "Madagascar": 1320,
    "Ethiopia": 1340, "Libya": 1340, "Sudan": 1310, "Congo": 1380,
    "Kenya": 1370, "Lesotho": 1250, "Botswana": 1300,
    "São Tomé and Príncipe": 1100,
    # Asia
    "Oman": 1420, "Bahrain": 1430, "United Arab Emirates": 1450,
    "Syria": 1400, "Lebanon": 1380, "Palestine": 1360,
    "China PR": 1450, "Vietnam": 1410, "Thailand": 1400,
    "Indonesia": 1380, "Malaysia": 1360, "India": 1400,
    "Tajikistan": 1380, "Kazakhstan": 1390, "Kyrgyzstan": 1340,
    "Turkmenistan": 1310, "Myanmar": 1290, "Bangladesh": 1250,
    "Pakistan": 1220, "Nepal": 1220, "Afghanistan": 1180,
    "Hong Kong": 1310, "Singapore": 1300, "Cambodia": 1260,
    "Maldives": 1180, "Sri Lanka": 1230, "Mongolia": 1150,
    "Taiwan": 1300, "North Korea": 1380, "Yemen": 1310,
    "Philippines": 1320,
    # Oceania
    "New Caledonia": 1300, "Fiji": 1300, "Solomon Islands": 1280,
    "Papua New Guinea": 1250, "Vanuatu": 1230, "Tahiti": 1250,
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


def _load_initial_elo() -> dict:
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


def _expected_win_prob(team_elo: float, opp_elo: float) -> float:
    """Elo-implied probability of a win (on the W=1, D=0.5, L=0 scale)."""
    return 1.0 / (1.0 + 10.0 ** ((opp_elo - team_elo) / 400.0))


def _opp_elo(row, elo_ratings: dict) -> float:
    """Resolve opponent Elo: FDORG lookup → name fallback → 1500 default."""
    try:
        opp_id = int(row["opp_id"]) if row["opp_id"] and not pd.isna(row["opp_id"]) else 0
    except (ValueError, TypeError):
        opp_id = 0
    if opp_id and opp_id in elo_ratings:
        return elo_ratings[opp_id]
    # Name-based fallback for teams not in FDORG (opp_id == 0 or missing)
    name = str(row.get("opp_name", "")).strip()
    if name in _NAME_ELO_FALLBACK:
        return _NAME_ELO_FALLBACK[name]
    return 1500.0  # true unknown — neutral


def _perf_deltas(rows: pd.DataFrame, team_elo: float, elo_ratings: dict) -> list[float]:
    """
    For each match, actual_score - elo_expected_score.
    Positive = overperformed vs schedule; negative = underperformed.
    Using opp Elo accounts for opponent quality: a loss to Germany barely
    hurts; a win over Barbados barely helps.
    """
    deltas = []
    for _, r in rows.iterrows():
        o_elo  = _opp_elo(r, elo_ratings)
        exp    = _expected_win_prob(team_elo, o_elo)
        actual = {"W": 1.0, "D": 0.5, "L": 0.0}.get(str(r["outcome"]), 0.5)
        deltas.append(actual - exp)
    return deltas


def rolling_features_for_team(
    team_df: pd.DataFrame,
    as_of_date=None,
    window: int = ROLLING_WINDOW,
    elo_ratings: dict = None,
    team_elo: float = None,
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

    # ── Opponent-adjusted performance delta ───────────────────────────────────
    # actual_outcome_score minus Elo-expected_score, averaged over the window.
    # Rewards teams that beat strong opponents; doesn't punish losses to giants.
    if elo_ratings is not None and team_elo is not None:
        d8 = _perf_deltas(recent, team_elo, elo_ratings)
        d3 = _perf_deltas(df.tail(3), team_elo, elo_ratings)
        perf_delta_8 = float(np.mean(d8)) if d8 else 0.0
        perf_delta_3 = float(np.mean(d3)) if d3 else 0.0
    else:
        perf_delta_8 = 0.0
        perf_delta_3 = 0.0

    return {
        "win_rate":     win_rate,
        "draw_rate":    draw_rate,
        "gf_avg":       gf_avg,
        "ga_avg":       ga_avg,
        "gd_avg":       gd_avg,
        "form_score":   form_score,
        "clean_sheet":  clean_sheet,
        "win_rate_3":   win_rate_3,
        "gf_avg_3":     gf_avg_3,
        "ga_avg_3":     ga_avg_3,
        "streak":       float(min(streak, 10)) / 10.0,
        "perf_delta_8": perf_delta_8,
        "perf_delta_3": perf_delta_3,
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
    # opponent-adjusted performance delta (actual - elo_expected, by schedule strength)
    "d_perf_delta_8", "d_perf_delta_3",
    # elo
    "d_elo",
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
) -> np.ndarray:
    s = 1 if is_home_perspective else -1
    return np.array([
        s * (home_feats["win_rate"]      - away_feats["win_rate"]),
        s * (home_feats["draw_rate"]     - away_feats["draw_rate"]),
        s * (home_feats["gf_avg"]        - away_feats["gf_avg"]),
        s * (home_feats["ga_avg"]        - away_feats["ga_avg"]),
        s * (home_feats["gd_avg"]        - away_feats["gd_avg"]),
        s * (home_feats["form_score"]    - away_feats["form_score"]),
        s * (home_feats["clean_sheet"]   - away_feats["clean_sheet"]),
        s * (home_feats["win_rate_3"]    - away_feats["win_rate_3"]),
        s * (home_feats["gf_avg_3"]      - away_feats["gf_avg_3"]),
        s * (home_feats["ga_avg_3"]      - away_feats["ga_avg_3"]),
        s * (home_feats["streak"]        - away_feats["streak"]),
        s * (home_feats["perf_delta_8"]  - away_feats["perf_delta_8"]),
        s * (home_feats["perf_delta_3"]  - away_feats["perf_delta_3"]),
        s * (home_elo - away_elo) / 400.0,
        h2h_win_rate,
        h2h_draw_rate,
        float(is_home_perspective),
    ])


def build_training_data(matches_df: pd.DataFrame, elo_ratings: dict):
    """Build (X, y, weights, feature_names) from all stored match rows."""
    X_rows, y_rows, w_rows = [], [], []
    df = matches_df.sort_values("date")

    fixtures = df[["fixture_id", "date", "team_id", "opp_id",
                   "is_home", "outcome", "comp"]].drop_duplicates(
        subset=["fixture_id", "team_id"])

    for _, row in fixtures.iterrows():
        tid, oid = int(row["team_id"]), int(row["opp_id"])
        as_of    = row["date"]

        t_elo = elo_ratings.get(tid, 1500.0)
        o_elo = elo_ratings.get(oid, 1500.0)

        t_feats = rolling_features_for_team(
            df[df["team_id"] == tid], as_of_date=as_of,
            elo_ratings=elo_ratings, team_elo=t_elo,
        )
        o_feats = rolling_features_for_team(
            df[df["team_id"] == oid], as_of_date=as_of,
            elo_ratings=elo_ratings, team_elo=o_elo,
        )
        if t_feats is None or o_feats is None:
            continue

        h2h_w, h2h_d = _h2h_features(tid, oid, df, as_of)

        x = build_match_feature_vector(
            t_feats, o_feats, t_elo, o_elo,
            bool(row["is_home"]), h2h_w, h2h_d,
        )
        X_rows.append(x)
        y_rows.append(row["outcome"])
        w_rows.append(match_weight(str(row["comp"])))

    if not X_rows:
        return np.empty((0, len(FEATURE_COLS))), [], np.array([]), FEATURE_COLS

    return np.array(X_rows), y_rows, np.array(w_rows), FEATURE_COLS
