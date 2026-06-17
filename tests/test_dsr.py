"""Tests for DSR-related utilities in the serving layer."""

import numpy as np
import pandas as pd
import pytest


class TestComputeDsr:
    """Test the DSR computation logic (reimplemented here to avoid loading model artifacts)."""

    def _compute_dsr(self, row_dict: dict, weights: pd.Series) -> float:
        row = pd.DataFrame([row_dict])
        shared = [f for f in weights.index if f in row.columns]
        valid = row[shared].notna().astype(float).iloc[0]
        w = weights[shared]
        return float((valid * w).sum() / w.sum())

    def _assign_group(self, dsr: float, thin_max: float = 0.40, semi_max: float = 0.70) -> str:
        if dsr <= thin_max:
            return "thin"
        if dsr <= semi_max:
            return "semi"
        return "thick"

    def test_all_present_gives_dsr_1(self):
        weights = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        dsr = self._compute_dsr({"a": 10, "b": 20, "c": 30}, weights)
        assert dsr == pytest.approx(1.0)

    def test_all_missing_gives_dsr_0(self):
        weights = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        dsr = self._compute_dsr({"a": np.nan, "b": np.nan, "c": np.nan}, weights)
        assert dsr == pytest.approx(0.0)

    def test_weighted_partial(self):
        weights = pd.Series({"a": 1.0, "b": 3.0})
        dsr = self._compute_dsr({"a": 10, "b": np.nan}, weights)
        assert dsr == pytest.approx(1.0 / 4.0)

    def test_high_weight_missing_penalizes_more(self):
        weights = pd.Series({"a": 1.0, "b": 9.0})
        dsr_high_missing = self._compute_dsr({"a": 10, "b": np.nan}, weights)
        dsr_low_missing = self._compute_dsr({"a": np.nan, "b": 20}, weights)
        assert dsr_low_missing > dsr_high_missing

    def test_assign_thin(self):
        assert self._assign_group(0.20) == "thin"
        assert self._assign_group(0.40) == "thin"

    def test_assign_semi(self):
        assert self._assign_group(0.41) == "semi"
        assert self._assign_group(0.70) == "semi"

    def test_assign_thick(self):
        assert self._assign_group(0.71) == "thick"
        assert self._assign_group(1.00) == "thick"

    def test_boundary_thin_semi(self):
        assert self._assign_group(0.40) == "thin"
        assert self._assign_group(0.401) == "semi"

    def test_boundary_semi_thick(self):
        assert self._assign_group(0.70) == "semi"
        assert self._assign_group(0.701) == "thick"
