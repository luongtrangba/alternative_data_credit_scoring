"""
ScoreSight · DSR Calculator (Weighted Data Sufficiency Rate)

Computes weighted DSR for each MSME and assigns thin/semi/thick groups
for model routing in the training stage.

Weight formula: w_refined = domain_weight * clip(IV / median_IV, mult_min, mult_max)
DSR formula:    DSR_wq = sum(w_i * valid_i) / sum(w_i)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scoresight.features.iv_calculator import compute_all_iv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "sme_altdata.parquet"
DICT = ROOT / "data" / "feature_dictionary.json"
CFG = ROOT / "configs" / "dsr_config.json"
OUT = ROOT / "t3_features" / "output"

TARGET = "default"


def refine_weights(
    domain_w: pd.Series, iv_map: dict, cfg: dict
) -> pd.DataFrame:
    """Refine domain weights using IV. Returns comparison table with w_refined."""
    p = cfg["weight_refinement"]
    iv = pd.Series(
        {f: min(iv_map.get(f, 0.0), p["iv_cap"]) for f in domain_w.index}
    )
    med = np.median(iv[iv > 0]) if (iv > 0).any() else 1.0
    mult = (iv / med).clip(p["mult_min"], p["mult_max"])
    w_ref = domain_w * mult
    out = pd.DataFrame({
        "domain_weight": domain_w.round(3),
        "iv": iv.round(3),
        "iv_multiplier": mult.round(2),
        "w_refined": w_ref.round(3),
    })
    return out.sort_values("w_refined", ascending=False)


def compute_dsr(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted DSR = sum(w * valid) / sum(w) over alt-data fields."""
    fields = list(weights.index)
    valid = df[fields].notna().astype(float)
    return (valid.mul(weights, axis=1).sum(axis=1) / weights.sum()).round(4)


def assign_group(dsr: pd.Series, cfg: dict) -> pd.Series:
    """Assign DSR group: thin / semi / thick."""
    t = cfg["dsr_thresholds"]
    return pd.cut(
        dsr,
        bins=[-0.01, t["thin_max"], t["semi_max"], 1.01],
        labels=["thin", "semi", "thick"],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    fdict = json.loads(DICT.read_text(encoding="utf-8"))
    cfg = json.loads(Path(CFG).read_text(encoding="utf-8"))
    source_of = {k: v["source"] for k, v in fdict.items()}

    alt_sources = set(cfg["alt_data_sources"])
    alt_fields = [f for f, v in fdict.items() if v["source"] in alt_sources]
    domain_w = pd.Series({f: fdict[f]["default_weight"] for f in alt_fields})

    iv_df, _ = compute_all_iv(df, alt_fields, source_of, bins=cfg["iv"]["bins"])
    iv_map = dict(zip(iv_df["feature"], iv_df["iv"]))
    wtbl = refine_weights(domain_w, iv_map, cfg)
    wtbl.to_csv(OUT / "weights_refined.csv")
    with open(ROOT / "configs" / "weights_refined.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: float(v) for k, v in wtbl["w_refined"].items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    w_ref = wtbl["w_refined"]
    df["dsr_weighted"] = compute_dsr(df, w_ref)
    df["dsr_group"] = assign_group(df["dsr_weighted"], cfg).astype(str)

    df.drop(
        columns=[c for c in ["_dsr_raw", "_dsr_group"] if c in df.columns]
    ).to_parquet(ROOT / "data" / "sme_scored_dsr.parquet", index=False)

    print("=" * 68)
    print("  T3 · WEIGHTED DSR")
    print("=" * 68)
    print("Refined weights (top 8 and bottom 4):")
    print(f"  {'feature':30s} {'domain':>7s} {'IV':>6s} {'mult':>6s} {'w_ref':>7s}")
    show = pd.concat([wtbl.head(8), wtbl.tail(4)])
    for f, r in show.iterrows():
        print(
            f"  {f:30s} {r['domain_weight']:7.2f} {r['iv']:6.3f} "
            f"{r['iv_multiplier']:6.2f} {r['w_refined']:7.3f}"
        )

    print("-" * 68)
    print("Weighted DSR groups:")
    g = df.groupby("dsr_group", observed=True).agg(
        n=(TARGET, "size"),
        default_rate=(TARGET, "mean"),
        avg_dsr=("dsr_weighted", "mean"),
    )
    for grp in ["thin", "semi", "thick"]:
        if grp in g.index:
            r = g.loc[grp]
            print(
                f"  {grp:6s}: n={int(r['n']):>6,} ({r['n'] / len(df):5.1%}) | "
                f"default={r['default_rate']:.1%} | avg_dsr={r['avg_dsr']:.2f}"
            )
    print("=" * 68)


if __name__ == "__main__":
    main()
