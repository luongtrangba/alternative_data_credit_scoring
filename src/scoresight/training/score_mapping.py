"""
ScoreSight · Score Mapping (probability -> credit score 300-850)

Standardizes P(bad) from calibrated models to a unified credit score
using the classic scorecard formula:

    Score = Offset + Factor * ln(odds_good),   odds_good = (1 - p) / p
    Factor = PDO / ln(2)
    Offset = BaseScore - Factor * ln(BaseOdds)

Higher score = lower P(bad) = safer borrower.
"""

from __future__ import annotations

import numpy as np

PDO = 50
BASE_SCORE = 600
BASE_ODDS = 9.0
SCORE_MIN, SCORE_MAX = 300, 850

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def prob_bad_to_score(p_bad: np.ndarray | float) -> np.ndarray:
    """Convert P(bad) to credit score [300, 850]. Higher = less risk."""
    p = np.clip(np.asarray(p_bad, dtype=float), 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    score = OFFSET + FACTOR * np.log(odds_good)
    return np.clip(np.round(score), SCORE_MIN, SCORE_MAX)


DECISION_THRESHOLDS = {"approve": 620, "review": 540}


def decision(score: float) -> str:
    """Map credit score to decision: approve / manual_review / decline."""
    if score >= DECISION_THRESHOLDS["approve"]:
        return "approve"
    if score >= DECISION_THRESHOLDS["review"]:
        return "manual_review"
    return "decline"


if __name__ == "__main__":
    for p in [0.01, 0.05, 0.10, 0.20, 0.40, 0.70]:
        s = float(prob_bad_to_score(p))
        print(f"P(bad)={p:.2f} -> score={s:.0f} -> {decision(s)}")
