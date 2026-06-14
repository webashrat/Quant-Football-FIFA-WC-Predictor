import requests
import time
from datetime import date
from src.config import BASE_URL, HEADERS


def _get(url, params=None, retries=3):
    for attempt in range(retries):
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(60)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"API failed after {retries} attempts: {url}")


def fetch_team_matches(team_id: int, date_from: str, limit: int = 50) -> list:
    from datetime import timedelta
    today = date.today()
    # API enforces max 750-day window
    earliest_allowed = (today - timedelta(days=740)).isoformat()
    safe_from = max(date_from, earliest_allowed)
    data = _get(
        f"{BASE_URL}/teams/{team_id}/matches",
        params={"dateFrom": safe_from, "dateTo": today.isoformat(), "limit": limit},
    )
    matches = data.get("matches", [])
    return [m for m in matches if m.get("score", {}).get("fullTime", {}).get("home") is not None]


def fetch_todays_fixtures(today: str = None) -> list:
    """
    Returns all WC fixtures for today, plus any next-day matches that kick off
    before 06:00 UTC (i.e. evening/late-night matches in non-UTC timezones that
    FDORG assigns to the following calendar date).
    """
    from datetime import timedelta, datetime
    d = today or date.today().isoformat()
    tomorrow = (date.fromisoformat(d) + timedelta(days=1)).isoformat()

    today_data    = _get(f"{BASE_URL}/competitions/WC/matches", params={"dateFrom": d, "dateTo": d})
    tomorrow_data = _get(f"{BASE_URL}/competitions/WC/matches", params={"dateFrom": tomorrow, "dateTo": tomorrow})

    today_matches = today_data.get("matches", [])

    # Include next-day matches that kick off before 06:00 UTC (they belong to tonight)
    for m in tomorrow_data.get("matches", []):
        utc = m.get("utcDate", "")
        try:
            kickoff_hour = int(utc[11:13])
            if kickoff_hour < 6:
                today_matches.append(m)
        except (ValueError, IndexError):
            pass

    return today_matches


def fetch_yesterdays_results(yesterday: str) -> list:
    data = _get(
        f"{BASE_URL}/competitions/WC/matches",
        params={"dateFrom": yesterday, "dateTo": yesterday, "status": "FINISHED"},
    )
    return data.get("matches", [])
