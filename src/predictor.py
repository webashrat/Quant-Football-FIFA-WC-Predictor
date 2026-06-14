"""
predictor.py — single unified match predictor for WC 2026.

Combines three signal groups into one probability:

  1. Historical form + Elo  →  LightGBM (calibrated, 900+ matches)
  2. Player quality         →  squad value, club-season form, availability (TM API)
  3. Tournament momentum    →  WC goals-per-game so far (ESPN data)

Signals 2 and 3 are blended with signal 1 in log-odds space so the
adjustment is proportional to uncertainty — a large form gap moves
a 50/50 more than it moves a 90/10.  Output is one normalized triple.
"""

import math
import logging
from typing import Optional

import numpy as np

from src.features import (
    build_match_feature_vector,
    rolling_features_for_team,
    _h2h_features,
)
from src.player_features import get_team_player_features, availability_prob_nudge
from src.player_store    import load_squad, load_performances, get_team_wc_summary
from src.tm_api          import get_team_club_season_summary

logger = logging.getLogger(__name__)

# How strongly player signals move the log-odds.
# 1.0 means 8pp player gap shifts a 50/50 match by ~0.32 log-odds (~8%).
# Increase to give player data more weight relative to the ML model.
_PLAYER_LO_SCALE = 1.0

# Hard caps on total player nudge (in probability-percentage points)
# before log-odds conversion so a huge value gap can't dominate completely.
_MAX_PLAYER_NUDGE_PP = 12.0


class UnifiedPredictor:
    """
    Fit once, call predict() per match.
    Internally blends ML + player signals; externally returns one probability.
    """

    def __init__(self):
        self._model   = None   # calibrated LightGBM wrapper (Predictor from daily_run)
        self.classes_ = None
        self.cv_auc_mean = None
        self.cv_auc_std  = None
        self.is_fitted   = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y, sample_weight=None, feature_names=None):
        from src.model import _make_lgbm
        from sklearn.calibration import CalibratedClassifierCV

        base  = _make_lgbm()
        model = CalibratedClassifierCV(base, cv=5, method="isotonic")
        model.fit(X, y, sample_weight=sample_weight)

        self._model   = model
        self.classes_ = list(model.classes_)

        # cross-val AUC for reporting
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            _make_lgbm(), X, y,
            cv=5, scoring="roc_auc_ovr", n_jobs=-1,
        )
        self.cv_auc_mean = float(scores.mean())
        self.cv_auc_std  = float(scores.std())
        self.is_fitted   = True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        home_id: int,
        away_id: int,
        today: str,
        matches_df,
        elo_ratings: dict,
        avg_elo_ratings: dict = None,
    ) -> dict:
        """
        Returns a single unified probability dict:
          win, draw, loss   — floats (sum to 1.0)
          win_pct, draw_pct, loss_pct — percentages (sum to 100)
          tip               — "home_name" | "away_name" | "Draw"
          confidence        — "HIGH" | "MEDIUM" | "LOW"
          home_elo, away_elo
          tip_team          — "home" | "away" | "draw"
        """
        avg = avg_elo_ratings or {}
        h_elo     = elo_ratings.get(home_id, 1500.0)
        a_elo     = elo_ratings.get(away_id, 1500.0)
        h_avg_elo = avg.get(home_id, h_elo)
        a_avg_elo = avg.get(away_id, a_elo)

        # ── Step 1: ML probabilities (or Elo fallback) ────────────────────────
        hh = matches_df[matches_df["team_id"] == home_id]
        ah = matches_df[matches_df["team_id"] == away_id]
        hf = rolling_features_for_team(hh, as_of_date=today)
        af = rolling_features_for_team(ah, as_of_date=today)
        h2h_w, h2h_d = _h2h_features(home_id, away_id, matches_df, today)

        if self.is_fitted and hf and af:
            xv = build_match_feature_vector(
                hf, af, h_elo, a_elo, True, h2h_w, h2h_d,
                h_avg_elo, a_avg_elo,
            )
            raw = self._ml_proba(xv)
        else:
            raw = self._elo_proba(h_elo, a_elo)

        p_w, p_d, p_l = raw["W"], raw["D"], raw["L"]

        # ── Step 2: Player nudge (pp, positive = favours home) ────────────────
        nudge_pp = self._player_nudge(home_id, away_id, today)
        nudge_pp = max(-_MAX_PLAYER_NUDGE_PP, min(_MAX_PLAYER_NUDGE_PP, nudge_pp))

        # ── Step 3: Blend in log-odds space ───────────────────────────────────
        # Only the win/loss split shifts; draw probability is held proportionally.
        p_w, p_d, p_l = self._apply_nudge(p_w, p_d, p_l, nudge_pp)

        # ── Step 4: Package result ────────────────────────────────────────────
        win_pct  = round(p_w * 100, 1)
        draw_pct = round(p_d * 100, 1)
        loss_pct = round(p_l * 100, 1)

        if p_w >= p_d and p_w >= p_l:
            tip_team = "home"
        elif p_l >= p_w and p_l >= p_d:
            tip_team = "away"
        else:
            tip_team = "draw"

        # Confidence = gap between top outcome and the next best
        sorted_pcts = sorted([win_pct, draw_pct, loss_pct], reverse=True)
        gap = sorted_pcts[0] - sorted_pcts[1]
        confidence = "HIGH" if gap > 12 else "MEDIUM" if gap > 5 else "LOW"

        return {
            "win":      round(p_w, 4),
            "draw":     round(p_d, 4),
            "loss":     round(p_l, 4),
            "win_pct":  win_pct,
            "draw_pct": draw_pct,
            "loss_pct": loss_pct,
            "tip_team": tip_team,
            "confidence": confidence,
            "home_elo": round(h_elo, 1),
            "away_elo": round(a_elo, 1),
            "elo_gap":  round(h_elo - a_elo, 1),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ml_proba(self, xv: np.ndarray) -> dict:
        proba = self._model.predict_proba(xv.reshape(1, -1))[0]
        return {c: float(p) for c, p in zip(self.classes_, proba)}

    def _elo_proba(self, h_elo: float, a_elo: float) -> dict:
        d  = (h_elo - a_elo) / 400.0
        ph = 1 / (1 + 10 ** (-d))
        return {
            "W": round(ph * 0.75, 4),
            "D": 0.25,
            "L": round((1 - ph) * 0.75, 4),
        }

    def _apply_nudge(self, p_w, p_d, p_l, nudge_pp: float):
        """Shift win/loss in log-odds space, keep draw proportional."""
        if abs(nudge_pp) < 0.01:
            return p_w, p_d, p_l

        eps = 1e-6
        p_w = max(eps, p_w)
        p_l = max(eps, p_l)

        # log-odds of win relative to loss (excluding draw)
        lo = math.log(p_w / p_l)
        lo_adj = lo + (nudge_pp / 100.0) * 4.0 * _PLAYER_LO_SCALE

        # Redistribute win+loss bucket with adjusted log-odds
        win_loss_sum = p_w + p_l
        p_w_new = win_loss_sum * (math.exp(lo_adj) / (1 + math.exp(lo_adj)))
        p_l_new = win_loss_sum - p_w_new

        # Keep draw unchanged, renormalise
        total = p_w_new + p_d + p_l_new
        return p_w_new / total, p_d / total, p_l_new / total

    def _player_nudge(self, home_id: int, away_id: int, today: str) -> float:
        """Compute total player nudge in pp (positive = favours home)."""
        nudge = 0.0

        # 1. Squad availability (injuries / suspensions)
        try:
            hp = get_team_player_features(home_id, today)
            ap = get_team_player_features(away_id, today)
            nudge += availability_prob_nudge(hp, ap)
        except Exception as e:
            logger.debug(f"availability nudge error: {e}")

        # 2. Club-season form (G+A per 90 from top 3 scorers)
        try:
            h_form = _form_score(home_id)
            a_form = _form_score(away_id)
            nudge += max(-6.0, min(6.0, (h_form - a_form) * 8.0))
        except Exception as e:
            logger.debug(f"form nudge error: {e}")

        # 3. WC tournament momentum (goals/match so far)
        try:
            h_wc = _wc_gpg(home_id)
            a_wc = _wc_gpg(away_id)
            nudge += max(-3.0, min(3.0, (h_wc - a_wc) * 2.0))
        except Exception as e:
            logger.debug(f"wc momentum nudge error: {e}")

        return nudge


# ── Module-level helpers ──────────────────────────────────────────────────────

def _form_score(fdorg_id: int) -> float:
    sq   = load_squad(fdorg_id)
    cs   = get_team_club_season_summary(sq)
    tops = cs.get("top_scorers", [])[:3]
    if not tops:
        return 0.0
    total_min = sum(p["minutes"] for p in tops)
    if not total_min:
        return 0.0
    ga = sum(p["goals"] + p["assists"] * 0.7 for p in tops)
    return ga / (total_min / 90)


def _wc_gpg(fdorg_id: int) -> float:
    try:
        perf = load_performances()
        if perf.empty:
            return 0.0
        t = perf[perf["fdorg_team_id"] == fdorg_id]
        if t.empty:
            return 0.0
        return float(t["goals"].sum()) / t["fixture_id"].nunique()
    except Exception:
        return 0.0
