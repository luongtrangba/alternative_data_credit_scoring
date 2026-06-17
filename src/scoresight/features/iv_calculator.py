"""
ScoreSight · IV Calculator (Information Value + WoE)

Computes Information Value (IV) and Weight of Evidence (WoE) for each
feature against the good/bad target. IV measures separating power and
is used for feature selection and DSR weight refinement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "sme_altdata.parquet"
DICT = ROOT / "data" / "feature_dictionary.json"
OUT = ROOT / "t3_features" / "output"

TARGET = "default"
ID_COL = "customer_id"
DEBUG_COLS = ["_gt_latent_risk", "_gt_quality", "_dsr_raw", "_dsr_group"]
SEGMENT_COLS = ["enterprise_size", "industry", "region"]


def compute_woe_iv(
    x: pd.Series, y: pd.Series, bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """Return (IV, WoE table). y = 1 is bad/default.

    Convention: WoE = ln(%good / %bad). WoE > 0 means bin is better than average.
    Uses Laplace smoothing to avoid log(0).
    """
    d = pd.DataFrame({"x": x, "y": y}).dropna(subset=["x"])
    if len(d) == 0 or d["x"].nunique() <= 1:
        return 0.0, pd.DataFrame()

    is_numeric = pd.api.types.is_numeric_dtype(d["x"]) and d["x"].nunique() > 12
    if is_numeric:
        d["bin"] = pd.qcut(d["x"], q=bins, duplicates="drop")
    else:
        d["bin"] = d["x"].astype(str)

    g = d.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    g.columns = ["total", "bad"]
    g["good"] = g["total"] - g["bad"]
    nb = len(g)
    tot_good, tot_bad = g["good"].sum(), g["bad"].sum()
    g["dist_good"] = (g["good"] + 0.5) / (tot_good + 0.5 * nb)
    g["dist_bad"] = (g["bad"] + 0.5) / (tot_bad + 0.5 * nb)
    g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
    g["bad_rate"] = (g["bad"] / g["total"]).round(4)
    g["iv_part"] = ((g["dist_good"] - g["dist_bad"]) * g["woe"]).round(5)
    g[["woe"]] = g[["woe"]].round(4)
    return float(g["iv_part"].sum()), g.reset_index()


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "very strong (suspect leakage)"


def compute_all_iv(
    df: pd.DataFrame,
    features: list[str],
    source_of: dict,
    bins: int = 10,
    save_woe_to: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute IV for a list of features. Returns (iv_df, dict[woe tables])."""
    rows, woe_tables = [], {}
    for c in features:
        iv, tbl = compute_woe_iv(df[c], df[TARGET], bins=bins)
        woe_tables[c] = tbl
        rows.append({
            "feature": c,
            "source": source_of.get(c, "firmographic"),
            "iv": round(iv, 4),
            "strength": iv_strength(iv),
            "missing_rate": round(df[c].isna().mean(), 4),
        })
        if save_woe_to is not None and not tbl.empty:
            tbl.to_csv(save_woe_to / f"{c}.csv", index=False)
    iv_df = (
        pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    )
    return iv_df, woe_tables


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    woe_dir = OUT / "woe"
    woe_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(DATA)
    fdict = json.loads(DICT.read_text(encoding="utf-8"))
    source_of = {k: v["source"] for k, v in fdict.items()}
    cfg = json.loads((ROOT / "configs" / "dsr_config.json").read_text())

    features = [c for c in df.columns if c not in [ID_COL, TARGET] + DEBUG_COLS]

    iv_df, _ = compute_all_iv(
        df, features, source_of, bins=cfg["iv"]["bins"], save_woe_to=woe_dir
    )
    iv_df.to_csv(OUT / "iv_table.csv", index=False)

    drop_thr = cfg["iv"]["drop_threshold"]
    print("=" * 68)
    print("  T3 · INFORMATION VALUE")
    print("=" * 68)
    print(f"{'feature':32s} {'source':14s} {'IV':>6s}  strength")
    print("-" * 68)
    for _, r in iv_df.iterrows():
        print(f"{r['feature']:32s} {r['source']:14s} {r['iv']:6.3f}  {r['strength']}")

    weak = iv_df[iv_df["iv"] < drop_thr]
    print("-" * 68)
    print("Average IV by source:")
    for s, v in (
        iv_df.groupby("source")["iv"].mean().sort_values(ascending=False).items()
    ):
        print(f"  {s:18s} {v:.3f}")
    print(
        f"\nUseless features (IV < {drop_thr}): "
        f"{list(weak['feature']) if len(weak) else 'none'}"
    )
    print(
        f"\nSaved: {(OUT / 'iv_table.csv').relative_to(ROOT)} + "
        f"{len(features)} WoE tables at {woe_dir.relative_to(ROOT)}/"
    )


if __name__ == "__main__":
    main()
