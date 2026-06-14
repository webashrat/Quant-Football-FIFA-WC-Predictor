"""
player_update_job.py — orchestrates all player data tasks.

Two entry points:

  1. initialize_all_squads()
     One-time setup: scrapes all 48 WC 2026 squads from Transfermarkt.
     Safe to re-run; skips teams already in the store.

  2. update_performances(yesterday, finished_matches)
     Daily async job: for each finished WC match from yesterday,
       - finds the ESPN event ID by date + team names
       - fetches full lineup + per-player stats
       - saves to player_store (wc_performances.parquet)
       - updates squad entries with ESPN player IDs (for future linking)

  3. print_player_report(fdorg_team_id)
     Pretty-print WC tournament stats for a team.

Called from daily_run.py after ingesting yesterday's results.
"""

import logging
import threading
import time
from datetime import date as date_cls

from src.player_features import FDORG_TO_TM
from src.player_scraper import scrape_all_squads, fetch_espn_event_id, fetch_match_performances
from src.player_store import (
    load_squad, save_squad, log_performances, get_player_wc_totals, get_team_wc_summary, all_initialized_teams
)
from src.tm_api import enrich_squad_with_club_stats

logger = logging.getLogger(__name__)

ALL_WC_TEAM_IDS = list(FDORG_TO_TM.keys())

# ── Squad initialisation ──────────────────────────────────────────────────────

def initialize_all_squads(force: bool = False, verbose: bool = True) -> None:
    """
    Scrape all 48 WC 2026 squads from Transfermarkt (3 threads).
    Idempotent: skips teams already in store unless force=True.
    """
    already = set(all_initialized_teams())
    pending = [tid for tid in ALL_WC_TEAM_IDS if force or tid not in already]

    if not pending:
        if verbose:
            print(f"  All {len(ALL_WC_TEAM_IDS)} squads already initialized.")
        return

    if verbose:
        print(f"  Initializing {len(pending)} squad(s) from Transfermarkt (3 parallel threads)...")

    results = scrape_all_squads(pending, force=force, max_workers=3)

    if verbose:
        ok  = sum(1 for v in results.values() if v)
        skipped = len(pending) - len(results)
        print(f"  Done: {ok} squads scraped, {skipped} skipped/failed.")
        total_players = sum(len(v) for v in results.values())
        print(f"  Total players stored: {total_players}")


# ── Daily performance update ──────────────────────────────────────────────────

def update_performances(
    yesterday: str,
    finished_matches: list[dict],
    verbose: bool = True,
) -> None:
    """
    For each finished WC match from yesterday, fetch ESPN match data and log
    per-player performance records.

    finished_matches: list of match dicts from football-data.org API:
      {id, homeTeam: {id, name}, awayTeam: {id, name}, status: "FINISHED"}
    """
    if not finished_matches:
        if verbose:
            print("  No finished matches — player performance update skipped.")
        return

    if verbose:
        print(f"  Updating player performances for {len(finished_matches)} match(es) from {yesterday}...")

    for m in finished_matches:
        home_id   = m["homeTeam"]["id"]
        away_id   = m["awayTeam"]["id"]
        home_name = m["homeTeam"]["name"]
        away_name = m["awayTeam"]["name"]
        fixture_id = m["id"]

        if verbose:
            print(f"    {home_name} vs {away_name} (fdorg_id={fixture_id}) ...")

        # Find ESPN event ID
        espn_id = fetch_espn_event_id(yesterday, home_name, away_name)
        if espn_id is None:
            if verbose:
                print(f"      ESPN event not found — skipping.")
            continue

        if verbose:
            print(f"      ESPN event_id={espn_id}")

        # Ensure squads are loaded (scrape if needed)
        for tid, tname in [(home_id, home_name), (away_id, away_name)]:
            if not load_squad(tid):
                if verbose:
                    print(f"      Fetching squad for {tname}...")
                scrape_all_squads([tid], max_workers=1)
                time.sleep(1)

        # Fetch ESPN match data
        records = fetch_match_performances(
            espn_event_id=espn_id,
            home_fdorg_id=home_id,
            away_fdorg_id=away_id,
            fdorg_fixture_id=fixture_id,
            match_date=yesterday,
        )

        if records:
            log_performances(records)
            if verbose:
                played  = sum(1 for r in records if r["minutes_est"] > 0)
                goals   = sum(r["goals"] for r in records)
                assists = sum(r["assists"] for r in records)
                cards   = sum(r["yellow_cards"] for r in records)
                print(f"      Logged {len(records)} players | {played} played | "
                      f"{int(goals)}G {int(assists)}A {int(cards)}Y")
        else:
            if verbose:
                print(f"      No player records returned from ESPN.")

        time.sleep(0.8)   # polite rate limit


def update_performances_async(
    yesterday: str,
    finished_matches: list[dict],
) -> threading.Thread:
    """
    Run update_performances in a background thread. Returns the thread.
    The caller does NOT need to join — it runs fire-and-forget.
    Logs results to logger rather than stdout to avoid interleaving.
    """
    def _run():
        try:
            update_performances(yesterday, finished_matches, verbose=False)
            logger.info(f"Background player update complete for {yesterday}")
        except Exception as e:
            logger.error(f"Background player update failed: {e}")

    t = threading.Thread(target=_run, daemon=True, name="player-update")
    t.start()
    return t


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_player_report(fdorg_team_id: int, team_name: str = "") -> None:
    """Pretty-print WC tournament stats for a team."""
    squad  = load_squad(fdorg_team_id)
    totals = get_player_wc_totals(fdorg_team_id)
    summary = get_team_wc_summary(fdorg_team_id)

    label = team_name or str(fdorg_team_id)
    print(f"\n  ── {label} — WC 2026 Tournament Stats ──────────────────")

    if squad:
        print(f"  Squad on file: {len(squad)} players")
        print(f"  Top 5 by market value:")
        top = sorted(squad, key=lambda p: p["market_value_m"], reverse=True)[:5]
        for p in top:
            print(f"    {p['name']:<26} {p['position']:<18} €{p['market_value_m']:.1f}M")

    if totals.empty:
        print(f"  No WC performance data yet.")
        return

    print(f"\n  WC Matches: {summary['wc_matches_played']}  |  "
          f"Goals/match: {summary['goals_per_match']:.2f}  |  "
          f"Assists/match: {summary['assists_per_match']:.2f}  |  "
          f"Total yellows: {summary['yellow_cards_total']}")

    if summary["suspension_risk"]:
        print(f"  Suspension risk (2+ yellows): {', '.join(summary['suspension_risk'])}")

    print(f"\n  Individual stats (sorted by minutes):")
    print(f"  {'Player':<26} {'Pos':>5}  {'M':>3}  {'Min':>4}  {'G':>3}  {'A':>3}  {'Y':>3}  {'R':>3}")
    print(f"  {'':─<26} {'':─>5}  {'':─>3}  {'':─>4}  {'':─>3}  {'':─>3}  {'':─>3}  {'':─>3}")
    for _, row in totals.sort_values("minutes_est", ascending=False).head(20).iterrows():
        print(
            f"  {row['player_name']:<26} {'':>5}  "
            f"{int(row['matches']):>3}  "
            f"{int(row['minutes_est']):>4}  "
            f"{int(row['goals']):>3}  "
            f"{int(row['assists']):>3}  "
            f"{int(row['yellow_cards']):>3}  "
            f"{int(row['red_cards']):>3}"
        )


# ── Club season enrichment ────────────────────────────────────────────────────

def enrich_team_club_stats(fdorg_team_id: int, verbose: bool = True) -> int:
    """
    Fetch current club season stats from TM API for every player in a team's squad
    and persist back to the squad JSON. Returns number of players enriched.
    """
    squad = load_squad(fdorg_team_id)
    if not squad:
        return 0
    if verbose:
        _, slug, _ = FDORG_TO_TM.get(fdorg_team_id, (0, str(fdorg_team_id), ""))
        print(f"  Enriching {len(squad)} players for {slug} ({fdorg_team_id})...")
    enriched = enrich_squad_with_club_stats(squad, max_workers=4, delay=0.4)
    n = sum(1 for p in enriched if p.get("club_season", {}).get("apps", 0) > 0)
    save_squad(fdorg_team_id, enriched)
    if verbose:
        print(f"    → {n} players with club season data")
    return n


def enrich_all_teams_club_stats(verbose: bool = True) -> None:
    """
    Enrich all 45 WC team squads with current club season stats from TM API.
    Runs sequentially per team (parallel within each team at 4 threads).
    Takes ~15-20 minutes for all 2100+ players. Run once or refresh weekly.
    """
    initialized = all_initialized_teams()
    total_enriched = 0
    for i, tid in enumerate(initialized, 1):
        if verbose:
            print(f"[{i}/{len(initialized)}] Team {tid}")
        total_enriched += enrich_team_club_stats(tid, verbose=verbose)
        time.sleep(0.5)
    if verbose:
        print(f"\nDone. {total_enriched} players enriched with club season stats.")


def enrich_all_teams_async(verbose: bool = False) -> threading.Thread:
    """Fire-and-forget: enrich all team squads with club season data."""
    def _run():
        try:
            enrich_all_teams_club_stats(verbose=verbose)
            logger.info("Background club stats enrichment complete.")
        except Exception as e:
            logger.error(f"Background club stats enrichment failed: {e}")

    t = threading.Thread(target=_run, daemon=True, name="club-enrichment")
    t.start()
    return t


# ── Entry point ───────────────────────────────────────────────────────────────

def run_full_init(verbose: bool = True) -> None:
    """
    Full one-time initialisation:
      1. Scrape all 48 WC 2026 squads from TM (names, values)
      2. Enrich all squads with current club season stats via TM API
    """
    if verbose:
        print("\n[ Player Init ] Scraping all WC 2026 squads from Transfermarkt...")
    initialize_all_squads(verbose=verbose)

    if verbose:
        print("\n[ Player Init ] Enriching squads with club season stats (TM API)...")
        print("  This fetches exact goals/assists/minutes for each player this season.")
        print("  Takes ~15-20 min for all 2100+ players. Running in background...\n")
    enrich_all_teams_async(verbose=True)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    if cmd == "init":
        run_full_init()
    elif cmd == "enrich":
        # Enrich one team or all teams
        if len(sys.argv) > 2:
            tid = int(sys.argv[2])
            enrich_team_club_stats(tid, verbose=True)
        else:
            enrich_all_teams_club_stats(verbose=True)
    elif cmd == "report":
        tid = int(sys.argv[2]) if len(sys.argv) > 2 else 771
        print_player_report(tid)
    else:
        print(f"Unknown command: {cmd}")
