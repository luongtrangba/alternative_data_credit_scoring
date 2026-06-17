"""
ScoreSight · T3 — DSR Calculator (Weighted Data Sufficiency Rate)
================================================================

Tính DSR CÓ TRỌNG SỐ cho từng MSME và phân nhóm thin/semi/thick để routing
mô hình ở T4. Khác với DSR thô (đếm đều mọi trường), DSR có trọng số phạt nặng
hơn khi thiếu các trường QUAN TRỌNG.

Trọng số w_i = domain_weight (chuyên gia) ĐÃ ĐƯỢC IV HIỆU CHỈNH:

    w_refined = domain_weight × clip(IV / median_IV, mult_min, mult_max)

Đây là NỬA ĐẦU của feedback loop: dữ liệu thực tế (qua IV) validate lại niềm tin
ban đầu của chuyên gia. Nửa sau (Feature Importance từ T4) sẽ tiếp tục cập nhật.

Công thức DSR:

    DSR_wq = Σ_i ( w_i · 1[valid_i] ) / Σ_i w_i

với valid_i = 1 nếu trường i không missing (đã qua QC).

Chạy:  python3 t3_features/dsr_calculator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "t3_features"))
from iv_calculator import compute_all_iv  # noqa: E402

DATA = ROOT / "data" / "sme_altdata.parquet"
DICT = ROOT / "data" / "feature_dictionary.json"
CFG = ROOT / "configs" / "dsr_config.json"
OUT = ROOT / "t3_features" / "output"

TARGET = "default"


# --------------------------------------------------------------------------- #
def refine_weights(domain_w: pd.Series, iv_map: dict, cfg: dict) -> pd.DataFrame:
    """Hiệu chỉnh trọng số domain bằng IV. Trả về bảng so sánh + w_refined."""
    p = cfg["weight_refinement"]
    iv = pd.Series({f: min(iv_map.get(f, 0.0), p["iv_cap"]) for f in domain_w.index})
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
    """DSR có trọng số = Σ(w·valid) / Σw, trên các trường alt-data."""
    fields = list(weights.index)
    valid = df[fields].notna().astype(float)
    return (valid.mul(weights, axis=1).sum(axis=1) / weights.sum()).round(4)


def assign_group(dsr: pd.Series, cfg: dict) -> pd.Series:
    t = cfg["dsr_thresholds"]
    return pd.cut(dsr, bins=[-0.01, t["thin_max"], t["semi_max"], 1.01],
                 labels=["thin", "semi", "thick"])


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    fdict = json.loads(DICT.read_text(encoding="utf-8"))
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    source_of = {k: v["source"] for k, v in fdict.items()}

    # Trường alt-data = feature có source thuộc 7 nhóm nguồn (loại identity)
    alt_sources = set(cfg["alt_data_sources"])
    alt_fields = [f for f, v in fdict.items() if v["source"] in alt_sources]
    domain_w = pd.Series({f: fdict[f]["default_weight"] for f in alt_fields})

    # 1) IV trên các trường alt-data -> hiệu chỉnh trọng số
    iv_df, _ = compute_all_iv(df, alt_fields, source_of, bins=cfg["iv"]["bins"])
    iv_map = dict(zip(iv_df["feature"], iv_df["iv"]))
    wtbl = refine_weights(domain_w, iv_map, cfg)
    wtbl.to_csv(OUT / "weights_refined.csv")
    with open(ROOT / "configs" / "weights_refined.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in wtbl["w_refined"].items()}, f,
                  ensure_ascii=False, indent=2)

    # 2) DSR có trọng số + phân nhóm
    w_ref = wtbl["w_refined"]
    df["dsr_weighted"] = compute_dsr(df, w_ref)
    df["dsr_group"] = assign_group(df["dsr_weighted"], cfg).astype(str)

    # 3) Lưu dataset đã chấm DSR (canonical cho T4) — bỏ DSR thô của T1
    df.drop(columns=[c for c in ["_dsr_raw", "_dsr_group"] if c in df.columns]
            ).to_parquet(ROOT / "data" / "sme_scored_dsr.parquet", index=False)

    # --- Report ----------------------------------------------------------- #
    print("=" * 68)
    print("  T3 · WEIGHTED DSR")
    print("=" * 68)
    print("Trọng số sau hiệu chỉnh IV (top 8 ↑ và bottom 4 ↓):")
    print(f"  {'feature':30s} {'domain':>7s} {'IV':>6s} {'×mult':>6s} {'w_ref':>7s}")
    show = pd.concat([wtbl.head(8), wtbl.tail(4)])
    for f, r in show.iterrows():
        print(f"  {f:30s} {r['domain_weight']:7.2f} {r['iv']:6.3f} "
              f"{r['iv_multiplier']:6.2f} {r['w_refined']:7.3f}")

    print("-" * 68)
    print("Phân nhóm DSR có trọng số:")
    g = df.groupby("dsr_group", observed=True).agg(
        n=(TARGET, "size"), default_rate=(TARGET, "mean"),
        avg_dsr=("dsr_weighted", "mean"))
    for grp in ["thin", "semi", "thick"]:
        if grp in g.index:
            r = g.loc[grp]
            print(f"  {grp:6s}: n={int(r['n']):>6,} ({r['n']/len(df):5.1%}) | "
                  f"default={r['default_rate']:.1%} | avg_dsr={r['avg_dsr']:.2f}")

    # So sánh với DSR thô (provisional) từ T1
    raw_group = pd.cut(df.get("_dsr_raw", pd.Series(np.nan, index=df.index)),
                       bins=[-0.01, 0.40, 0.70, 1.01], labels=["thin", "semi", "thick"])
    if raw_group.notna().any():
        print("-" * 68)
        print("Di chuyển nhóm: DSR thô (T1) → DSR có trọng số (hàng = thô):")
        mig = pd.crosstab(raw_group.astype(str), df["dsr_group"])
        mig = mig.reindex(index=["thin", "semi", "thick"],
                          columns=["thin", "semi", "thick"]).fillna(0).astype(int)
        moved = (raw_group.astype(str).to_numpy() != df["dsr_group"].to_numpy()).mean()
        print("           thin    semi   thick")
        for grp in mig.index:
            print(f"    {grp:6s} {mig.loc[grp,'thin']:>6,} {mig.loc[grp,'semi']:>7,} {mig.loc[grp,'thick']:>7,}")
        print(f"  => {moved:.1%} MSME đổi nhóm sau khi áp trọng số IV.")

    print("-" * 68)
    print("Default rate theo nhóm × quy mô (kiểm tra routing hợp lý):")
    ct = df.pivot_table(index="enterprise_size", columns="dsr_group",
                        values=TARGET, aggfunc="mean", observed=True)
    ct = ct.reindex(index=["micro", "small", "medium"],
                    columns=["thin", "semi", "thick"])
    print("           thin    semi   thick")
    for sz in ct.index:
        vals = "  ".join(f"{ct.loc[sz, g]:5.1%}" if pd.notna(ct.loc[sz, g]) else "   - "
                         for g in ["thin", "semi", "thick"])
        print(f"    {sz:6s}  {vals}")

    print("=" * 68)
    print(f"Đã lưu: data/sme_scored_dsr.parquet | "
          f"configs/weights_refined.json | t3_features/output/")


if __name__ == "__main__":
    main()
