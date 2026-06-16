# Quant Football — FIFA WC 2026 Predictor

A live, data-driven match prediction system for the 2026 FIFA World Cup. Runs every match day: ingests results, retrains the model, and prints probability breakdowns for upcoming fixtures.

**Current accuracy: 4/14 (29%) | Model CV AUC: 0.607 | Training rows: 932**

> **Draw crisis**: WC 2026 draw rate is 50% (8/16 matches). Model now applies a 1.35× draw recalibration and squad-value differential nudge.

---

## How it works

Three signal groups are blended into a single probability triple (Win / Draw / Loss):

| Signal | Source | Weight |
|--------|--------|--------|
| Historical form + Elo | football-data.org API (900+ matches) | Base ML |
| Player quality | Transfermarkt TM API — squad value, club-season G+A/90 | Log-odds nudge |
| Tournament momentum | ESPN WC 2026 live data — exact minutes played | Log-odds nudge |

**Key feature**: Elo ratings sourced daily from [eloratings.net](https://eloratings.net/) (244 national teams, proper Elo methodology) rather than FIFA's arbitrary ranking points. This gives accurate relative strength for all teams including those outside the top 50 (e.g. Northern Ireland, Gibraltar, Barbados).

---

## Architecture

```
src/
  config.py           — API keys, paths, constants
  client.py           — football-data.org API client (fixtures, results)
  store.py            — parquet match store (append + dedup)
  backfill.py         — historical match backfill for all 48 WC teams
  elo_fetcher.py      — daily Elo refresh from eloratings.net (244 teams)
  features.py         — rolling features, Elo, build_training_data
  model.py            — LightGBM wrapper (_make_lgbm)
  predictor.py        — UnifiedPredictor: fit + predict + player nudge
  player_scraper.py   — ESPN WC player performance scraper (exact minutes via TM API)
  player_features.py  — Transfermarkt squad scraper + availability signals
  player_store.py     — squad JSON store + WC performance parquet helpers
  player_update_job.py — TM API club-season stat enrichment (background thread)
  tm_api.py           — Transfermarkt unofficial API client (6-hour disk cache)
  daily_run.py        — Main entry point: ingest → retrain → predict → print
  evaluate.py         — Backtesting and accuracy reporting

data/
  elo_seeds.json            — Live Elo from eloratings.net, auto-refreshed daily
  initial_elo.json          — Static FIFA WR fallback seeds (used if elo_seeds.json missing)
  predictions_log.csv       — All predictions made with actual outcomes (tracked)
  results.csv               — Scraped WC results log (tracked)
  matches.parquet           — Full match history (gitignored — rebuilt by backfill)
  players/squads/           — Per-team squad JSONs with TM player IDs (gitignored)
  players/wc_performances.parquet  — WC 2026 player minutes/goals (gitignored)
  tm_api_cache/             — Per-player TM API response cache, 6h TTL (gitignored)
  player_cache/             — TM squad page scrape cache, daily TTL (gitignored)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo>
cd "Quant Football FIFA WC Predictor"
pip install -r requirements.txt
```

### 2. API credentials

Create a `.env` file (never commit this):

```bash
# .env
API_TOKEN=your_football_data_org_token   # https://www.football-data.org/
```

Load it before running:

```bash
export $(cat .env | xargs)
```

### 3. Backfill historical data

Downloads match history for all 48 WC teams (runs once, ~5 min):

```bash
python -m src.backfill
```

### 4. Enrich squads with club-season stats

Fetches TM API club stats for all 48 squads (runs once, ~20 min in background):

```bash
python -m src.player_update_job enrich
```

---

## Daily usage

Run this each match day to ingest yesterday's results, retrain, and predict today's fixtures:

```bash
export $(cat .env | xargs)
python -m src.daily_run
```

Sample output:

```
══════════════════════════════════════════════════════════
  Brazil  vs  Morocco
  Group C  |  Kick-off 22:00 UTC

  Squad value (€M)               Brazil        Morocco
  Elo rating                       1766           1755

  Club Season Form  25/26
  Vinicius Junior    Brazil   22G  14A   0.67/90
  Raphinha           Brazil   21G   8A   1.09/90
  Ayoub El Kaabi    Morocco   21G   2A   0.71/90

  ┌────────────────────────────────────────────────────────────┐
  │  Brazil win            ██████░░░░░░░░  22.6%              │
  │  Draw                  ████████████████  57.4%              │
  │  Morocco win           ██████░░░░░░░░  19.9%              │
  │  Prediction ▶  Draw          Confidence: HIGH              │
  └────────────────────────────────────────────────────────────┘
```

---

## Data sources

| Source | What it provides | Rate limit |
|--------|-----------------|------------|
| [football-data.org](https://www.football-data.org/) | Live WC fixtures, scores, historical matches | 10 req/min (free tier) |
| [eloratings.net](https://eloratings.net/) | Real Elo ratings for 244 national teams, updated after every match | Daily refresh |
| [Transfermarkt](https://www.transfermarkt.com/) | Squad composition, market values, player profiles | ~1 req/sec (scraping) |
| [TM unofficial API](https://tmapi.transfermarkt.technology/) | Per-player career game logs, exact WC minutes, club-season G+A | 6h cache |
| ESPN WC API | Player-level match appearances, substitution minutes | Public |

---

## Results so far (WC 2026)

| Date | Match | Tip | Tip% | Result | |
|------|-------|-----|-------|--------|-|
| Jun 11 | Mexico vs South Africa | Mexico | 77.1% | 2–0 | ✓ |
| Jun 11 | South Korea vs Czechia | South Korea | 61.0% | 2–1 | ✓ |
| Jun 12 | Canada vs Bosnia-Herzegovina | Canada | 68.5% | 1–1 | ✗ |
| Jun 13 | USA vs Paraguay | USA | 38.9% | 4–1 | ✓ |
| Jun 13 | Qatar vs Switzerland | Qatar | 33.8% | 1–1 | ✗ |
| Jun 13 | Brazil vs Morocco | Morocco | 34.5% | 1–1 | ✗ |
| Jun 13 | Haiti vs Scotland | Scotland | 37.0% | 0–1 | ✓ |
| Jun 13 | Australia vs Turkey | Turkey | 39.0% | 2–0 | ✗ |
| Jun 14 | Germany vs Curaçao | Germany | 79.7% | 7–1 | ✓ |
| Jun 14 | Netherlands vs Japan | Netherlands | 62.4% | 2–2 | ✗ |
| Jun 14 | Ivory Coast vs Ecuador | Ecuador | 36.7% | 1–0 | ✗ |
| Jun 14 | Sweden vs Tunisia | Sweden | 39.5% | 5–1 | ✓ |
| Jun 15 | Spain vs Cape Verde Islands | Spain | 83.0% | 0–0 | ✗ |
| Jun 15 | Belgium vs Egypt | Belgium | 53.0% | 1–1 | ✗ |
| Jun 15 | Saudi Arabia vs Uruguay | Uruguay | 40.0% | 1–1 | ✗ |
| Jun 15 | Iran vs New Zealand | Iran | 58.0% | 2–2 | ✗ |

### June 16 Predictions (with draw-boost model)

| Match | Tip | Win% | Draw% | Loss% | Conf |
|-------|-----|------|-------|-------|------|
| France vs Senegal | **France** | 46.1% | 39.4% | 14.5% | MEDIUM |
| Iraq vs Norway | **Norway** | 35.3% | 27.6% | 37.1% | LOW |
| Argentina vs Algeria | **Argentina** | 48.3% | 31.5% | 20.2% | MEDIUM |
| Austria vs Jordan | **Austria** | 43.4% | 41.9% | 14.8% | LOW |

---

## Model details

- **Algorithm**: LightGBM (3-class: W/D/L) wrapped in `CalibratedClassifierCV(cv=5, method="isotonic")`
- **Training data**: ~932 international matches, 2022–present, weighted by competition importance (WC = 1.0, Friendly = 0.3)
- **Features** (home − away diffs, 16 total): win rate, draw rate, goals for/against, goal diff, form score (recency-weighted), clean sheet rate, 3-match momentum, unbeaten streak, current Elo gap (`d_elo`), historical average Elo gap (`d_avg_elo`), H2H win/draw rate, home advantage
- **Player nudge**: applied in log-odds space, capped at ±12pp total. Components: squad availability (TM injuries/suspensions), club-season form (G+A/90 of top 3 scorers), WC tournament momentum (goals/game), **squad value differential** (€M gap / 200, capped ±5pp)
- **WC draw recalibration**: P(draw) boosted ×1.35 post-prediction. WC 2026 empirical draw rate = 50% (8/16 matches) vs model baseline ~25%. Conservative 1.35× applied; probability redistributed proportionally from W/L.
- **Elo**: seeded daily from [eloratings.net](https://eloratings.net/) (244 teams, proper Elo methodology), updated intra-tournament from live WC matches (K=30). Falls back to FIFA WR points if fetch fails.
# Quant-Football-FIFA-WC-Predictor
