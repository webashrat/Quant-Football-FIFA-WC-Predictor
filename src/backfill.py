"""
backfill.py — one-time cold-start loader from the Kaggle international results CSV.

The CSV uses team names; we map them to football-data.org team IDs so they slot
straight into our existing match store (matches.parquet).
"""
import pandas as pd
from pathlib import Path
from src.store import append_matches
from src.config import DATA_DIR

RESULTS_CSV = DATA_DIR / "results.csv"

# Map dataset name → (team_id, canonical_name) for all WC 2026 teams.
# Keys must match the exact strings in results.csv.
WC2026_NAME_TO_ID: dict[str, tuple[int, str]] = {
    "Uruguay":                   (758,  "Uruguay"),
    "Germany":                   (759,  "Germany"),
    "Spain":                     (760,  "Spain"),
    "Paraguay":                  (761,  "Paraguay"),
    "Argentina":                 (762,  "Argentina"),
    "Ghana":                     (763,  "Ghana"),
    "Brazil":                    (764,  "Brazil"),
    "Portugal":                  (765,  "Portugal"),
    "Japan":                     (766,  "Japan"),
    "Mexico":                    (769,  "Mexico"),
    "United States":             (771,  "United States"),
    "South Korea":               (772,  "South Korea"),
    "France":                    (773,  "France"),
    "South Africa":              (774,  "South Africa"),
    "Algeria":                   (778,  "Algeria"),
    "Australia":                 (779,  "Australia"),
    "New Zealand":               (783,  "New Zealand"),
    "Switzerland":               (788,  "Switzerland"),
    "Ecuador":                   (791,  "Ecuador"),
    "Sweden":                    (792,  "Sweden"),
    "Czech Republic":            (798,  "Czechia"),
    "Croatia":                   (799,  "Croatia"),
    "Saudi Arabia":              (801,  "Saudi Arabia"),
    "Tunisia":                   (802,  "Tunisia"),
    "Turkey":                    (803,  "Turkey"),
    "Senegal":                   (804,  "Senegal"),
    "Belgium":                   (805,  "Belgium"),
    "Morocco":                   (815,  "Morocco"),
    "Austria":                   (816,  "Austria"),
    "Colombia":                  (818,  "Colombia"),
    "Egypt":                     (825,  "Egypt"),
    "Canada":                    (828,  "Canada"),
    "Haiti":                     (836,  "Haiti"),
    "Iran":                      (840,  "Iran"),
    "Bosnia and Herzegovina":    (1060, "Bosnia-Herzegovina"),
    "Panama":                    (1836, "Panama"),
    "Cape Verde":                (1930, "Cape Verde Islands"),
    "DR Congo":                  (1934, "Congo DR"),
    "Ivory Coast":               (1935, "Ivory Coast"),
    "Qatar":                     (8030, "Qatar"),
    "Jordan":                    (8049, "Jordan"),
    "Iraq":                      (8062, "Iraq"),
    "Uzbekistan":                (8070, "Uzbekistan"),
    "Netherlands":               (8601, "Netherlands"),
    "Norway":                    (8872, "Norway"),
    "Scotland":                  (8873, "Scotland"),
    "Curaçao":                   (9460, "Curaçao"),
}

_WC2026_IDS = {tid for tid, _ in WC2026_NAME_TO_ID.values()}


def load_and_ingest(since: str = "2020-01-01", verbose: bool = True) -> int:
    """
    Read results.csv, filter to WC 2026 teams and matches since `since`,
    convert to our per-team row format, and append to matches.parquet.

    Returns the number of new rows stored.
    """
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found.\n"
            "Download from:\n"
            "  curl -o data/results.csv 'https://raw.githubusercontent.com/"
            "JamshedAli18/International-football-results-from-1872-to-2024/main/results.csv'"
        )

    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, format="mixed")
    df = df[df["date"] >= pd.Timestamp(since)].copy()
    df = df.dropna(subset=["home_score", "away_score"])

    wc_teams = set(WC2026_NAME_TO_ID.keys())
    df = df[df["home_team"].isin(wc_teams) | df["away_team"].isin(wc_teams)]

    rows = []
    fixture_counter = -1  # negative IDs to avoid collision with real API fixture IDs

    for _, m in df.iterrows():
        h_name = m["home_team"]
        a_name = m["away_team"]
        h_in_wc = h_name in WC2026_NAME_TO_ID
        a_in_wc = a_name in WC2026_NAME_TO_ID

        h_score = int(m["home_score"])
        a_score = int(m["away_score"])
        date_str = pd.Timestamp(m["date"]).strftime("%Y-%m-%d")
        comp = m["tournament"]
        is_neutral = bool(m["neutral"])

        for perspective in ["home", "away"]:
            if perspective == "home" and not h_in_wc:
                continue
            if perspective == "away" and not a_in_wc:
                continue

            if perspective == "home":
                tid, tname = WC2026_NAME_TO_ID[h_name]
                oid, oname = (WC2026_NAME_TO_ID[a_name] if a_in_wc else (0, a_name))
                gf, ga = h_score, a_score
                is_home = 0 if is_neutral else 1
            else:
                tid, tname = WC2026_NAME_TO_ID[a_name]
                oid, oname = (WC2026_NAME_TO_ID[h_name] if h_in_wc else (0, h_name))
                gf, ga = a_score, h_score
                is_home = 0

            outcome = "W" if gf > ga else ("L" if gf < ga else "D")

            rows.append({
                "fixture_id": fixture_counter,
                "date":       date_str,
                "team_id":    tid,
                "team_name":  tname,
                "opp_id":     oid,
                "opp_name":   oname,
                "is_home":    is_home,
                "gf":         gf,
                "ga":         ga,
                "gd":         gf - ga,
                "outcome":    outcome,
                "comp":       comp,
                "stage":      "HISTORICAL",
            })
            fixture_counter -= 1

    if not rows:
        if verbose:
            print("  No matching rows found in results.csv.")
        return 0

    new_df = pd.DataFrame(rows)
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df = new_df.sort_values("date").reset_index(drop=True)

    append_matches(new_df)

    if verbose:
        per_team = new_df.groupby("team_name").size().sort_values(ascending=False)
        print(f"  Loaded {len(new_df)} rows for {per_team.shape[0]} WC 2026 teams.")
        print(f"  Date range: {new_df['date'].min().date()} → {new_df['date'].max().date()}")
        print(f"  Matches per team (top 10):")
        for name, cnt in per_team.head(10).items():
            print(f"    {name:<28} {cnt}")

    return len(new_df)


if __name__ == "__main__":
    print("Running historical backfill from results.csv...")
    n = load_and_ingest(since="2020-01-01")
    print(f"Done — {n} rows ingested.")
