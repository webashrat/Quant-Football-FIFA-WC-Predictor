"""
daily_run.py — run once per day to:
  1. Ingest yesterday's finished WC results
  2. Cold-start backfill from results.csv (runs once)
  3. Rebuild features + retrain on all completed data
  4. Predict win/draw/loss probabilities for today's fixtures
  5. Log predictions (scored tomorrow)
  6. Evaluate yesterday's predictions (if any)
"""
import sys
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from datetime import date, timedelta

from src.client import fetch_todays_fixtures, fetch_yesterdays_results, fetch_team_matches
from src.store import parse_raw_matches, load_matches, append_matches, log_prediction
from src.features import (compute_elo_ratings, rolling_features_for_team,
                           build_training_data, build_match_feature_vector,
                           _h2h_features, FEATURE_COLS)
from src.evaluate import score_yesterday
from src.backfill import load_and_ingest, RESULTS_CSV
from src.config import BACKFILL_FROM, BACKFILL_LIMIT
from src.player_features import get_team_player_features
from src.player_update_job import (
    initialize_all_squads, update_performances_async, ALL_WC_TEAM_IDS
)
from src.player_store import all_initialized_teams, get_team_wc_summary, load_squad
from src.tm_api import get_team_club_season_summary
from src.predictor import UnifiedPredictor
from src.elo_fetcher import refresh_elo_seeds, get_avg_fdorg_elo_map

W = 58  # output width


def _bar(pct: float, width: int = 28) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _print_match_block(
    home_name: str, away_name: str,
    group: str, stage: str, kick_off: str,
    home_feats: dict | None, away_feats: dict | None,
    result: dict,                        # UnifiedPredictor.predict() output
    home_player: dict | None = None,
    away_player: dict | None = None,
):
    win_pct  = result["win_pct"]
    draw_pct = result["draw_pct"]
    loss_pct = result["loss_pct"]
    h_elo    = result["home_elo"]
    a_elo    = result["away_elo"]
    tip_team = result["tip_team"]
    conf     = result["confidence"]
    tip      = home_name if tip_team == "home" else (away_name if tip_team == "away" else "Draw")
    group_label = group.replace("_", " ").title() if group else stage.replace("_", " ").title()

    print(f"\n  {'═' * W}")
    print(f"  {home_name}  vs  {away_name}")
    print(f"  {group_label}  |  Kick-off {kick_off}")
    print(f"  {'─' * W}")

    # ── Squad & player context ────────────────────────────────
    home_p = home_player or {}
    away_p = away_player or {}
    if home_p or away_p:
        # Squad value row
        h_sv = home_p.get("squad_value_m", 0)
        a_sv = away_p.get("squad_value_m", 0)
        h_av = home_p.get("available_value_m", 0)
        a_av = away_p.get("available_value_m", 0)
        print(f"\n  {'':22} {home_name[:16]:>16}  {away_name[:16]:>16}")
        print(f"  {'':─<22} {'':─>16}  {'':─>16}")
        print(f"  {'Squad value (€M)':<22} {h_sv:>16.0f}  {a_sv:>16.0f}")
        print(f"  {'Available (€M)':<22} {h_av:>16.0f}  {a_av:>16.0f}")
        print(f"  {'Elo rating':<22} {h_elo:>16.0f}  {a_elo:>16.0f}")
        print(f"  {'Avg age':<22} {home_p.get('avg_age',0):>16.1f}  {away_p.get('avg_age',0):>16.1f}")

        # Key players
        print(f"\n  {'─' * W}")
        print(f"  Key Players  (Transfermarkt market value)")
        for side_name, side_p in [(home_name, home_p), (away_name, away_p)]:
            top = side_p.get("top_players", [])
            inj = [i["name"].lower() for i in side_p.get("injured", [])]
            sus = [s["name"].lower() for s in side_p.get("suspended", [])]
            print(f"\n  {side_name}:")
            for p in top[:5]:
                pl = p["name"].lower()
                if any(pl in n or n in pl for n in inj):
                    status, flag = "INJURED   ", "✗"
                elif any(pl in n or n in pl for n in sus):
                    status, flag = "SUSPENDED ", "✗"
                else:
                    status, flag = "FIT       ", "✓"
                print(f"    {flag} {p['name']:<26} {p['position']:<16} €{p['value_m']:.1f}M  {status}")

        # Injuries / suspensions
        any_issues = any(side_p.get("injured") or side_p.get("suspended") for side_p in [home_p, away_p])
        if any_issues:
            print(f"\n  {'─' * W}")
            for side_name, side_p in [(home_name, home_p), (away_name, away_p)]:
                for inj in side_p.get("injured", [])[:4]:
                    ret = inj.get("return_date", "?") or "?"
                    print(f"  ✗ INJURED   {side_name[:12]:<12} │ {inj['name']:<24} return: {ret}")
                for sus in side_p.get("suspended", [])[:2]:
                    print(f"  ✗ SUSPENDED {side_name[:12]:<12} │ {sus['name']:<24} until: {sus.get('until','?')}")

        # Club season form
        any_cs = any(side_p.get("club_season", {}).get("top_scorers") for side_p in [home_p, away_p])
        if any_cs:
            print(f"\n  {'─' * W}")
            print(f"  Club Season Form  25/26  (goals + 0.7×assists per 90, top scorers)")
            print(f"  {'Player':<26}  {'Club':>10}  {'Apps':>4}  {'Min':>5}  {'G':>3}  {'A':>3}  {'G+A/90':>7}")
            for side_name, side_p in [(home_name, home_p), (away_name, away_p)]:
                for p in side_p.get("club_season", {}).get("top_scorers", [])[:3]:
                    p90 = (p["goals"] + p["assists"] * 0.7) / (p["minutes"] / 90) if p["minutes"] else 0
                    print(f"  {p['name']:<26}  {side_name[:10]:>10}  {p['apps']:>4}  {p['minutes']:>5}  {p['goals']:>3}  {p['assists']:>3}  {p90:>7.2f}")

        # WC stats
        for side_name, side_p in [(home_name, home_p), (away_name, away_p)]:
            wc = side_p.get("wc_summary", {})
            if wc.get("wc_matches_played", 0) > 0:
                risk = "  ⚠ " + ", ".join(wc["suspension_risk"]) if wc.get("suspension_risk") else ""
                print(f"\n  {side_name} WC 2026: {wc['wc_matches_played']}gm  "
                      f"{wc['goals_per_match']:.1f}G/gm  "
                      f"{wc['yellow_cards_total']}Y{risk}")

    # ── Recent form table ─────────────────────────────────────
    print(f"\n  {'─' * W}")
    if home_feats and away_feats:
        rows = [
            ("Win rate",           f"{home_feats['win_rate']:.0%}",    f"{away_feats['win_rate']:.0%}",
             f"{home_feats['win_rate']-away_feats['win_rate']:+.0%}"),
            ("Goals scored/game",  f"{home_feats['gf_avg']:.2f}",      f"{away_feats['gf_avg']:.2f}",
             f"{home_feats['gf_avg']-away_feats['gf_avg']:+.2f}"),
            ("Goals conceded/game",f"{home_feats['ga_avg']:.2f}",      f"{away_feats['ga_avg']:.2f}",
             f"{home_feats['ga_avg']-away_feats['ga_avg']:+.2f}"),
            ("Goal diff/game",     f"{home_feats['gd_avg']:+.2f}",     f"{away_feats['gd_avg']:+.2f}",
             f"{home_feats['gd_avg']-away_feats['gd_avg']:+.2f}"),
            ("Form score",         f"{home_feats['form_score']:.3f}",  f"{away_feats['form_score']:.3f}",
             f"{home_feats['form_score']-away_feats['form_score']:+.3f}"),
        ]
        print(f"  Recent Form  (last 8 matches)")
        print(f"  {'Metric':<22} {home_name[:14]:>14}  {away_name[:14]:>14}  {'Diff':>8}")
        print(f"  {'':─<22} {'':─>14}  {'':─>14}  {'':─>8}")
        for label, hv, av, diff in rows:
            print(f"  {label:<22} {hv:>14}  {av:>14}  {diff:>8}")
    else:
        print(f"  Recent Form  : insufficient rolling history — Elo signal only")

    # ── Final prediction box (one unified output) ─────────────
    inner = W + 2
    h_lbl = (home_name + " win")[:20]
    a_lbl = (away_name + " win")[:20]
    print(f"\n  ┌{'─' * inner}┐")
    print(f"  │  {h_lbl:<20}  {_bar(win_pct)}  {win_pct:>5.1f}%  │")
    print(f"  │  {'Draw':<20}  {_bar(draw_pct)}  {draw_pct:>5.1f}%  │")
    print(f"  │  {a_lbl:<20}  {_bar(loss_pct)}  {loss_pct:>5.1f}%  │")
    print(f"  │{'─' * inner}│")
    print(f"  │  Prediction ▶  {tip:<20}  Confidence: {conf:<6}  {'':6}│")
    print(f"  └{'─' * inner}┘")


def _ensure_team_history(team_id: int, team_name: str, matches_df):
    if not matches_df.empty and team_id in matches_df["team_id"].values:
        return matches_df
    print(f"  Backfilling via API for {team_name} (id={team_id})...")
    raw = fetch_team_matches(team_id, date_from=BACKFILL_FROM, limit=BACKFILL_LIMIT)
    new_df = parse_raw_matches(raw, team_id)
    if new_df.empty:
        return matches_df
    append_matches(new_df)
    return load_matches()


def run(today: str = None, yesterday: str = None):
    today     = today     or date.today().isoformat()
    yesterday = yesterday or (date.today() - timedelta(days=1)).isoformat()

    print(f"\n{'═' * (W + 4)}")
    print(f"  FIFA WC 2026 Daily Predictor  |  {today}")
    print(f"{'═' * (W + 4)}\n")

    # ── Step 0: Refresh Elo seeds (no-op if already done today) ─
    print("[ Step 0 ] Refreshing Elo seeds from eloratings.net...")
    refresh_elo_seeds()

    # ── Step 1: Score yesterday ──────────────────────────────
    print("[ Step 1 ] Evaluating yesterday's predictions...")
    score_yesterday(yesterday)

    # ── Step 1b: One-time squad initialisation (runs once, ~5 min) ───────────
    n_init = len(all_initialized_teams())
    if n_init < len(ALL_WC_TEAM_IDS) // 2:
        print(f"\n[ Step 1b ] First run — initialising all WC 2026 squads from Transfermarkt...")
        print(f"  ({n_init} of {len(ALL_WC_TEAM_IDS)} teams already stored; scraping the rest in background)")
        initialize_all_squads(verbose=True)

    # ── Step 2: Ingest yesterday's results ──────────────────
    print("[ Step 2 ] Ingesting yesterday's WC results...")
    finished = fetch_yesterdays_results(yesterday)
    for m in finished:
        for tid, tname in [(m["homeTeam"]["id"], m["homeTeam"]["name"]),
                           (m["awayTeam"]["id"], m["awayTeam"]["name"])]:
            raw_df = parse_raw_matches([m], tid)
            if not raw_df.empty:
                append_matches(raw_df)
    print(f"  Stored {len(finished)} finished match(es) from {yesterday}."
          if finished else "  No finished WC matches from yesterday.")

    # ── Step 2b: Update player performances async (background) ──────────────
    if finished:
        print(f"  Launching background player performance update for {len(finished)} match(es)...")
        update_performances_async(yesterday, finished)

    # ── Step 3: Fetch today's fixtures ──────────────────────
    print(f"\n[ Step 3 ] Fetching today's fixtures ({today})...")
    todays = fetch_todays_fixtures(today)
    pending       = [m for m in todays if m["status"] in ("TIMED", "SCHEDULED")]
    finished_today = [m for m in todays if m["status"] == "FINISHED"]
    print(f"  {len(todays)} total  |  {len(pending)} pending  |  {len(finished_today)} already finished")

    if finished_today:
        print("  Ingesting today's already-finished matches...")
        for m in finished_today:
            for tid in [m["homeTeam"]["id"], m["awayTeam"]["id"]]:
                raw_df = parse_raw_matches([m], tid)
                if not raw_df.empty:
                    append_matches(raw_df)

    if not pending:
        print("\n  No pending matches to predict today.")
        return

    # ── Step 3b: Cold-start from results.csv ────────────────
    matches_df    = load_matches()
    is_cold_start = matches_df.empty or (matches_df["stage"] == "HISTORICAL").sum() == 0
    if is_cold_start and RESULTS_CSV.exists():
        print("\n[ Step 3b ] Cold-start: loading historical data from results.csv...")
        load_and_ingest(since="2020-01-01", verbose=True)
    elif is_cold_start:
        print("\n[ Step 3b ] results.csv not found — skipping historical backfill.")

    # ── Step 4: Ensure team history ──────────────────────────
    print("\n[ Step 4 ] Ensuring team history is loaded...")
    matches_df = load_matches()
    for m in pending:
        for tid, tname in [(m["homeTeam"]["id"], m["homeTeam"]["name"]),
                           (m["awayTeam"]["id"], m["awayTeam"]["name"])]:
            matches_df = _ensure_team_history(tid, tname, matches_df)

    # ── Step 5: Retrain unified predictor ────────────────────
    print("\n[ Step 5 ] Retraining model on all completed matches...")
    matches_df      = load_matches()
    elo_ratings     = compute_elo_ratings(matches_df)
    avg_elo_ratings = get_avg_fdorg_elo_map()
    X, y, weights, feat_names = build_training_data(matches_df, elo_ratings, avg_elo_ratings)
    predictor   = UnifiedPredictor()

    if len(X) >= 5:
        predictor.fit(X, y, sample_weight=weights, feature_names=feat_names)
        auc_str = (f"CV ROC-AUC: {predictor.cv_auc_mean:.3f} ± {predictor.cv_auc_std:.3f}"
                   if predictor.cv_auc_mean else "")
        print(f"  Trained on {len(X)} rows  |  {auc_str}")
    else:
        print(f"  Only {len(X)} rows — Elo-only fallback.")

    # ── Step 6: Per-match prediction blocks ─────────────────
    print(f"\n[ Step 6 ] Predictions for today's {len(pending)} match(es):")

    log_records = []
    for m in pending:
        home_id   = m["homeTeam"]["id"]
        away_id   = m["awayTeam"]["id"]
        home_name = m["homeTeam"]["name"]
        away_name = m["awayTeam"]["name"]
        kick_off  = m["utcDate"][11:16] + " UTC"
        group     = m.get("group", "")
        stage     = m.get("stage", "")

        home_hist  = matches_df[matches_df["team_id"] == home_id]
        away_hist  = matches_df[matches_df["team_id"] == away_id]
        home_feats = rolling_features_for_team(home_hist, as_of_date=today)
        away_feats = rolling_features_for_team(away_hist, as_of_date=today)

        # ── Single unified prediction (ML + Elo + players) ───
        print(f"  Predicting {home_name} vs {away_name}...")
        result = predictor.predict(home_id, away_id, today, matches_df, elo_ratings, avg_elo_ratings)

        # ── Player context for display ───────────────────────
        home_player = get_team_player_features(home_id, today)
        away_player = get_team_player_features(away_id, today)
        home_player["club_season"] = get_team_club_season_summary(load_squad(home_id))
        away_player["club_season"] = get_team_club_season_summary(load_squad(away_id))
        home_player["wc_summary"]  = get_team_wc_summary(home_id)
        away_player["wc_summary"]  = get_team_wc_summary(away_id)

        _print_match_block(
            home_name, away_name, group, stage, kick_off,
            home_feats, away_feats, result,
            home_player=home_player,
            away_player=away_player,
        )

        tip = home_name if result["tip_team"] == "home" else (
            away_name if result["tip_team"] == "away" else "Draw")
        log_records.append({
            "fixture_id":     m["id"],
            "pred_date":      today,
            "kick_off":       kick_off,
            "home":           home_name,
            "away":           away_name,
            "group":          group,
            "stage":          stage,
            "p_win":          result["win"],
            "p_draw":         result["draw"],
            "p_loss":         result["loss"],
            "tip":            tip,
            "home_elo":       result["home_elo"],
            "away_elo":       result["away_elo"],
            "actual_outcome": None,
        })

    if log_records:
        log_prediction(log_records)
        print(f"\n  Predictions saved → data/predictions_log.csv")

    print(f"\n{'═' * (W + 4)}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    run(
        today     = args[0] if len(args) > 0 else None,
        yesterday = args[1] if len(args) > 1 else None,
    )
