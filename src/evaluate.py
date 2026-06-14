import pandas as pd
import numpy as np
from src.store import load_predictions


def score_yesterday(yesterday: str):
    preds = load_predictions()
    if preds.empty:
        print("No predictions logged yet.")
        return

    scored = preds[(preds["pred_date"] == yesterday) & preds["actual_outcome"].notna()]
    if scored.empty:
        print(f"No scored predictions for {yesterday}.")
        return

    log_losses, brier_scores = [], []
    for _, row in scored.iterrows():
        actual = row["actual_outcome"]
        p = {"W": row["p_win"], "D": row["p_draw"], "L": row["p_loss"]}
        p_actual = p.get(actual, 0.0)

        log_losses.append(-np.log(max(p_actual, 1e-7)))

        # Brier: sum of squared errors across all classes
        brier = sum((p.get(c, 0.0) - (1.0 if c == actual else 0.0)) ** 2
                    for c in ["W", "D", "L"])
        brier_scores.append(brier)

    print(f"\n--- Evaluation for {yesterday} ---")
    print(f"Matches scored  : {len(scored)}")
    print(f"Mean log-loss   : {np.mean(log_losses):.4f}")
    print(f"Mean Brier score: {np.mean(brier_scores):.4f}")
    print("(lower is better for both)\n")
