"""
ScoreSight · T4 — Score Mapping (probability -> điểm 0–1000)
===========================================================

Chuẩn hóa P(bad) từ các mô hình khác nhau (thin/semi/thick) về CÙNG thang điểm
để so sánh xuyên suốt danh mục, theo công thức scorecard kinh điển:

    Score = Offset + Factor · ln(odds_good),   odds_good = (1 - p) / p
    Factor = PDO / ln(2)
    Offset = BaseScore − Factor · ln(BaseOdds)

- PDO (Points to Double the Odds): mỗi PDO điểm, odds tốt/xấu nhân đôi.
- Neo BaseScore tại BaseOdds = odds của population (default ~10% -> 9:1).

Điểm cao = P(bad) thấp = tốt. Dùng chung bởi T4 (calibration) và T5 (serving).
"""

from __future__ import annotations

import numpy as np

PDO = 50            # mỗi 50 điểm, odds good/bad nhân đôi
BASE_SCORE = 600    # điểm neo
BASE_ODDS = 9.0     # odds good:bad tại điểm neo (≈ population 10% bad)
SCORE_MIN, SCORE_MAX = 300, 850

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def prob_bad_to_score(p_bad: np.ndarray | float) -> np.ndarray:
    """P(bad) -> điểm tín dụng [300, 850]. Cao = ít rủi ro."""
    p = np.clip(np.asarray(p_bad, dtype=float), 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    score = OFFSET + FACTOR * np.log(odds_good)
    return np.clip(np.round(score), SCORE_MIN, SCORE_MAX)


# Ngưỡng quyết định (Decision Engine — dùng ở T5)
DECISION_THRESHOLDS = {"approve": 620, "review": 540}  # >=approve, [review,approve), <review


def decision(score: float) -> str:
    if score >= DECISION_THRESHOLDS["approve"]:
        return "approve"
    if score >= DECISION_THRESHOLDS["review"]:
        return "manual_review"
    return "decline"


if __name__ == "__main__":
    for p in [0.01, 0.05, 0.10, 0.20, 0.40, 0.70]:
        s = float(prob_bad_to_score(p))
        print(f"P(bad)={p:.2f} -> score={s:.0f} -> {decision(s)}")
