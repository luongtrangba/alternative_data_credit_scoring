"""Tests for scoresight.training.score_mapping."""

import numpy as np

from scoresight.training.score_mapping import (
    SCORE_MAX,
    SCORE_MIN,
    decision,
    prob_bad_to_score,
)


class TestProbBadToScore:
    def test_low_risk_gives_high_score(self):
        score = float(prob_bad_to_score(0.01))
        assert score > 700

    def test_high_risk_gives_low_score(self):
        score = float(prob_bad_to_score(0.70))
        assert score < 400

    def test_score_within_bounds(self):
        for p in [0.001, 0.01, 0.05, 0.10, 0.50, 0.90, 0.999]:
            score = float(prob_bad_to_score(p))
            assert SCORE_MIN <= score <= SCORE_MAX

    def test_monotonic_decreasing(self):
        probs = [0.01, 0.05, 0.10, 0.20, 0.40, 0.70]
        scores = [float(prob_bad_to_score(p)) for p in probs]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_array_input(self):
        probs = np.array([0.05, 0.10, 0.50])
        scores = prob_bad_to_score(probs)
        assert scores.shape == (3,)
        assert all(SCORE_MIN <= s <= SCORE_MAX for s in scores)

    def test_extreme_probabilities_clipped(self):
        score_low = float(prob_bad_to_score(0.0))
        score_high = float(prob_bad_to_score(1.0))
        assert score_low == SCORE_MAX
        assert score_high == SCORE_MIN


class TestDecision:
    def test_approve(self):
        assert decision(700) == "approve"
        assert decision(620) == "approve"

    def test_manual_review(self):
        assert decision(619) == "manual_review"
        assert decision(540) == "manual_review"

    def test_decline(self):
        assert decision(539) == "decline"
        assert decision(300) == "decline"

    def test_boundary_approve(self):
        assert decision(620) == "approve"
        assert decision(619) == "manual_review"

    def test_boundary_review(self):
        assert decision(540) == "manual_review"
        assert decision(539) == "decline"
