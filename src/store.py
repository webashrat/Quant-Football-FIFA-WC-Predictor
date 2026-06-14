import pandas as pd
from pathlib import Path
from src.config import DATA_DIR

MATCHES_PATH = DATA_DIR / "matches.parquet"
PREDICTIONS_PATH = DATA_DIR / "predictions_log.csv"


def _parse_raw_match(m: dict, team_id: int) -> dict | None:
    score = m.get("score", {})
    full = score.get("fullTime", {})
    home_id = m["homeTeam"]["id"]
    is_home = home_id == team_id

    gh = full.get("home")
    ga_raw = full.get("away")
    if gh is None or ga_raw is None:
        return None

    gf = gh if is_home else ga_raw
    ga = ga_raw if is_home else gh

    if gf > ga:
        outcome = "W"
    elif gf < ga:
        outcome = "L"
    else:
        outcome = "D"

    return {
        "fixture_id": m["id"],
        "date":       m["utcDate"][:10],
        "team_id":    team_id,
        "team_name":  m["homeTeam"]["name"] if is_home else m["awayTeam"]["name"],
        "opp_id":     m["awayTeam"]["id"] if is_home else m["homeTeam"]["id"],
        "opp_name":   m["awayTeam"]["name"] if is_home else m["homeTeam"]["name"],
        "is_home":    int(is_home),
        "gf":         gf,
        "ga":         ga,
        "gd":         gf - ga,
        "outcome":    outcome,
        "comp":       m.get("competition", {}).get("code", "UNK"),
        "stage":      m.get("stage", "UNK"),
    }


def parse_raw_matches(raw: list, team_id: int) -> pd.DataFrame:
    rows = [_parse_raw_match(m, team_id) for m in raw]
    rows = [r for r in rows if r]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_matches() -> pd.DataFrame:
    if MATCHES_PATH.exists():
        return pd.read_parquet(MATCHES_PATH)
    return pd.DataFrame()


def save_matches(df: pd.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MATCHES_PATH, index=False)


def append_matches(new_df: pd.DataFrame):
    existing = load_matches()
    if existing.empty:
        save_matches(new_df)
        return
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["fixture_id", "team_id"])
    combined = combined.sort_values("date").reset_index(drop=True)
    save_matches(combined)


def log_prediction(records: list[dict]):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(records)
    if PREDICTIONS_PATH.exists():
        df_existing = pd.read_csv(PREDICTIONS_PATH)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["fixture_id"])
    else:
        df_all = df_new
    df_all.to_csv(PREDICTIONS_PATH, index=False)


def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return pd.DataFrame()


def update_actual_outcomes(results_date: str) -> int:
    """
    Match finished results from the parquet into predictions_log.csv.
    Looks for home-team rows (is_home=1) whose fixture_id appears in predictions
    for results_date and fills actual_outcome (W/D/L from home team's perspective).
    Returns count of rows updated.
    """
    preds = load_predictions()
    if preds.empty:
        return 0

    # Find predictions without outcomes for the given date
    mask = (preds["pred_date"] == results_date) & preds["actual_outcome"].isna()
    pending = preds[mask]
    if pending.empty:
        return 0

    matches = load_matches()
    if matches.empty:
        return 0

    # Home-team rows from parquet (outcome is always from home perspective in predictions)
    home_rows = matches[matches["is_home"] == 1][["fixture_id", "outcome"]].drop_duplicates()
    outcome_map = dict(zip(home_rows["fixture_id"], home_rows["outcome"]))

    updated = 0
    for idx in pending.index:
        fid = preds.at[idx, "fixture_id"]
        try:
            fid_int = int(fid)
        except (ValueError, TypeError):
            continue
        if fid_int in outcome_map:
            preds.at[idx, "actual_outcome"] = outcome_map[fid_int]
            updated += 1

    if updated:
        preds.to_csv(PREDICTIONS_PATH, index=False)
    return updated
