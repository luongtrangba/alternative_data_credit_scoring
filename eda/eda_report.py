"""
ScoreSight · EDA — Exploratory Data Analysis
============================================

Phân tích bộ dữ liệu MSME alternative-data đã sinh ở T1, tập trung vào những
thứ QUYẾT ĐỊNH cho bài credit scoring:

  1. Tổng quan & kiểu dữ liệu
  2. Target (default) & class imbalance
  3. Missingness theo NGUỒN — đặc thù alternative data
  4. DSR & phân khúc MSME (xác nhận: size không quyết định rủi ro)
  5. Thống kê feature số + categorical
  6. IV (Information Value) — feature nào dự báo mạnh  [preview T3]
  7. Missingness có LEAK nhãn không?  (kiểm tra DSR routing hợp lệ)
  8. Đa cộng tuyến (top tương quan feature-feature)

Xuất:
  - Báo cáo text ra stdout
  - eda/figures/*.png   (biểu đồ cho slide)
  - eda/tables/*.csv    (bảng số liệu)

Chạy:  python3 eda/eda_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # không cần display
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "t1_sources"))
try:
    from scoresight.data_generator import SOURCE_GROUPS  # noqa: E402
except ImportError:
    from generate_dataset import SOURCE_GROUPS  # noqa: E402
DATA = ROOT / "data" / "sme_altdata.parquet"
DICT = ROOT / "data" / "feature_dictionary.json"
FIG = ROOT / "eda" / "figures"
TAB = ROOT / "eda" / "tables"

# Cột KHÔNG phải feature (định danh, nhãn, ground-truth debug)
ID_COL = "customer_id"
TARGET = "default"
DEBUG_COLS = ["_gt_latent_risk", "_gt_quality", "_dsr_raw", "_dsr_group"]
SEGMENT_COLS = ["enterprise_size", "industry", "region"]


def hr(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


# --------------------------------------------------------------------------- #
# IV (Information Value) với WoE binning + Laplace smoothing
# --------------------------------------------------------------------------- #
def compute_iv(x: pd.Series, y: pd.Series, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """IV của feature x với target y (1 = bad/default).

    - Numeric nhiều giá trị: chia quantile bins.
    - Categorical / ít giá trị: mỗi giá trị một bin.
    - Laplace smoothing để tránh log(0) ở bin lệch hoàn toàn.
    - Chỉ tính trên hàng KHÔNG missing (missingness xử lý riêng ở mục 7).
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
    tot_good = g["good"].sum()
    tot_bad = g["bad"].sum()
    # Laplace smoothing
    g["dist_good"] = (g["good"] + 0.5) / (tot_good + 0.5 * nb)
    g["dist_bad"] = (g["bad"] + 0.5) / (tot_bad + 0.5 * nb)
    g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
    g["iv_part"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
    g["bad_rate"] = g["bad"] / g["total"]
    return float(g["iv_part"].sum()), g


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "vô dụng"
    if iv < 0.1:
        return "yếu"
    if iv < 0.3:
        return "trung bình"
    if iv < 0.5:
        return "mạnh"
    return "rất mạnh (nghi leak)"


# --------------------------------------------------------------------------- #
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    fdict = json.loads(DICT.read_text(encoding="utf-8"))
    source_of = {k: v["source"] for k, v in fdict.items()}

    feature_cols = [c for c in df.columns
                    if c not in [ID_COL, TARGET] + DEBUG_COLS]
    numeric_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(df[c]) and c not in SEGMENT_COLS]

    # ---- 1. Tổng quan ---------------------------------------------------- #
    hr("1 · TỔNG QUAN")
    print(f"Số dòng (MSME)      : {len(df):,}")
    print(f"Số cột              : {df.shape[1]}  "
          f"(feature: {len(feature_cols)}, numeric: {len(numeric_cols)})")
    print(f"Bộ nhớ              : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"Trùng customer_id   : {df[ID_COL].duplicated().sum()}")

    # ---- 2. Target ------------------------------------------------------- #
    hr("2 · TARGET (default) & CLASS IMBALANCE")
    vc = df[TARGET].value_counts().sort_index()
    rate = df[TARGET].mean()
    print(f"good (0): {vc.get(0,0):>7,}   |   bad (1): {vc.get(1,0):>7,}")
    print(f"Default rate: {rate:.2%}   |   imbalance ratio ≈ 1:{(1-rate)/rate:.1f}")
    print("=> Cần scale_pos_weight / SMOTE khi train (mục T4).")

    # ---- 3. Missingness theo nguồn -------------------------------------- #
    hr("3 · MISSINGNESS THEO NGUỒN (đặc thù alternative data)")
    miss = df[feature_cols].isna().mean().sort_values(ascending=False)
    src_miss = (pd.Series({c: source_of.get(c, "firmographic") for c in feature_cols})
                .to_frame("source").join(miss.to_frame("missing_pct")))
    by_src = src_miss.groupby("source")["missing_pct"].mean().sort_values(ascending=False)
    print("% missing trung bình theo nhóm nguồn:")
    for s, v in by_src.items():
        bar = "█" * int(v * 40)
        print(f"  {s:18s} {v:5.1%} {bar}")
    miss.to_frame("missing_pct").to_csv(TAB / "missingness.csv")

    # ---- 4. DSR & MSME --------------------------------------------------- #
    hr("4 · DSR & PHÂN KHÚC MSME")
    grp = df.groupby("_dsr_group", observed=True).agg(
        n=(TARGET, "size"), default_rate=(TARGET, "mean"), avg_dsr=("_dsr_raw", "mean"))
    print("Nhóm DSR:")
    for g in ["thin", "semi", "thick"]:
        if g in grp.index:
            r = grp.loc[g]
            print(f"  {g:6s}: n={int(r['n']):>6,} ({r['n']/len(df):5.1%}) | "
                  f"default={r['default_rate']:.1%} | avg_dsr={r['avg_dsr']:.2f}")
    print("\nPhân khúc MSME:")
    szg = df.groupby("enterprise_size", observed=True).agg(
        n=(TARGET, "size"), default_rate=(TARGET, "mean"), avg_dsr=("_dsr_raw", "mean"))
    for s in ["micro", "small", "medium"]:
        if s in szg.index:
            r = szg.loc[s]
            print(f"  {s:6s}: n={int(r['n']):>6,} ({r['n']/len(df):5.1%}) | "
                  f"default={r['default_rate']:.1%} | avg_dsr={r['avg_dsr']:.2f}")
    spread = szg["default_rate"].max() - szg["default_rate"].min()
    print(f"\n=> Chênh lệch default giữa các quy mô: {spread:.1%} "
          f"({'PHẲNG ✓ — quy mô không quyết định rủi ro' if spread < 0.03 else 'cần xem lại'})")

    # ---- 5. Thống kê feature -------------------------------------------- #
    hr("5 · THỐNG KÊ FEATURE")
    desc = df[numeric_cols].describe(percentiles=[.05, .5, .95]).T
    desc["skew"] = df[numeric_cols].skew()
    desc[["mean", "5%", "50%", "95%", "skew"]].to_csv(TAB / "numeric_summary.csv")
    skewed = desc["skew"].abs().sort_values(ascending=False).head(5)
    print("Top 5 feature lệch nhất (lognormal — cân nhắc log-transform khi cần):")
    for c, sk in skewed.items():
        print(f"  {c:32s} skew={sk:+.1f}")
    print("\nDefault rate theo ngành (industry):")
    ind = df.groupby("industry", observed=True)[TARGET].agg(["size", "mean"]).sort_values("mean")
    for name, r in ind.iterrows():
        print(f"  {name:14s} n={int(r['size']):>6,} | default={r['mean']:.1%}")

    # ---- 6. IV ----------------------------------------------------------- #
    hr("6 · INFORMATION VALUE — feature nào dự báo mạnh? [preview T3]")
    iv_rows = []
    for c in numeric_cols + ["industry", "region"]:
        iv, _ = compute_iv(df[c], df[TARGET])
        iv_rows.append((c, source_of.get(c, "firmographic"),
                        fdict.get(c, {}).get("signal_strength", np.nan), iv))
    iv_df = pd.DataFrame(iv_rows, columns=["feature", "source", "gen_signal", "iv"]
                         ).sort_values("iv", ascending=False)
    iv_df["strength"] = iv_df["iv"].map(iv_strength)
    iv_df.to_csv(TAB / "information_value.csv", index=False)
    print("Top 15 feature theo IV:")
    print(f"  {'feature':32s} {'source':14s} {'IV':>6s}  mức độ")
    for _, r in iv_df.head(15).iterrows():
        print(f"  {r['feature']:32s} {r['source']:14s} {r['iv']:6.3f}  {r['strength']}")
    print("\nIV trung bình theo nhóm nguồn (nguồn nào giá trị nhất):")
    src_iv = iv_df.groupby("source")["iv"].mean().sort_values(ascending=False)
    for s, v in src_iv.items():
        print(f"  {s:18s} {v:.3f}")
    # Kiểm tra generator: IV có khớp signal_strength đã thiết kế?
    valid = iv_df.dropna(subset=["gen_signal"])
    corr_design = valid["iv"].corr(valid["gen_signal"])
    print(f"\n=> Corr(IV thực tế, signal_strength thiết kế) = {corr_design:+.2f} "
          f"({'khớp tốt ✓' if corr_design > 0.5 else 'cần xem lại'})")

    # ---- 7. Missingness có leak nhãn không? ----------------------------- #
    hr("7 · MISSINGNESS CÓ LEAK NHÃN KHÔNG? (tính hợp lệ của DSR routing)")
    print("IV của chỉ báo 'có nguồn' (has_source) với default — KỲ VỌNG THẤP:")
    for grp_name, cols in SOURCE_GROUPS.items():
        has = df[cols].notna().any(axis=1).astype(int)
        iv, _ = compute_iv(has, df[TARGET])
        flag = "✓ ok" if iv < 0.05 else "⚠ leak"
        print(f"  has_{grp_name:18s} IV={iv:.3f}  {flag}")
    iv_dsr, _ = compute_iv(df["_dsr_raw"], df[TARGET])
    print(f"  DSR (raw) vs default      IV={iv_dsr:.3f}  "
          f"{'✓ DSR không leak nhãn' if iv_dsr < 0.1 else '⚠'}")

    # ---- 8. Đa cộng tuyến ----------------------------------------------- #
    hr("8 · ĐA CỘNG TUYẾN (top tương quan feature-feature)")
    corr = df[numeric_cols].corr().abs()
    pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
             .stack().sort_values(ascending=False))
    print("Top 8 cặp tương quan cao (cân nhắc khi train model tuyến tính):")
    for (a, b), v in pairs.head(8).items():
        print(f"  {v:.2f}  {a} ~ {b}")

    # ---- Plots ----------------------------------------------------------- #
    _make_plots(df, iv_df, by_src, szg, grp)
    hr("HOÀN TẤT")
    print(f"Biểu đồ : {FIG.relative_to(ROOT)}/  (6 PNG)")
    print(f"Bảng    : {TAB.relative_to(ROOT)}/  (3 CSV)")


def _make_plots(df, iv_df, by_src, szg, grp) -> None:
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

    # a) default rate theo DSR group & size
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    grp.reindex(["thin", "semi", "thick"])["default_rate"].plot.bar(
        ax=ax[0], color="#c0392b", rot=0); ax[0].set_title("Default theo nhóm DSR")
    ax[0].set_ylabel("default rate"); ax[0].axhline(df["default"].mean(), ls="--", c="gray")
    szg.reindex(["micro", "small", "medium"])["default_rate"].plot.bar(
        ax=ax[1], color="#2980b9", rot=0); ax[1].set_title("Default theo quy mô MSME")
    ax[1].axhline(df["default"].mean(), ls="--", c="gray")
    fig.tight_layout(); fig.savefig(FIG / "default_rate.png"); plt.close(fig)

    # b) DSR distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["_dsr_raw"], bins=40, color="#16a085")
    for t in (0.4, 0.7): ax.axvline(t, ls="--", c="k")
    ax.set_title("Phân phối DSR (thô)"); ax.set_xlabel("DSR")
    fig.tight_layout(); fig.savefig(FIG / "dsr_distribution.png"); plt.close(fig)

    # c) missingness theo nguồn
    fig, ax = plt.subplots(figsize=(6, 3))
    by_src.sort_values().plot.barh(ax=ax, color="#8e44ad")
    ax.set_title("% missing trung bình theo nguồn"); ax.set_xlabel("missing")
    fig.tight_layout(); fig.savefig(FIG / "missingness_by_source.png"); plt.close(fig)

    # d) IV ranking
    fig, ax = plt.subplots(figsize=(6, 5))
    top = iv_df.head(15).iloc[::-1]
    ax.barh(top["feature"], top["iv"], color="#d35400")
    for t in (0.02, 0.1, 0.3): ax.axvline(t, ls=":", c="gray")
    ax.set_title("Information Value (top 15)"); ax.set_xlabel("IV")
    fig.tight_layout(); fig.savefig(FIG / "information_value.png"); plt.close(fig)

    # e) size × DSR stacked
    ct = pd.crosstab(df["enterprise_size"], df["_dsr_group"], normalize="index")
    ct = ct.reindex(index=["micro", "small", "medium"], columns=["thin", "semi", "thick"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ct.plot.bar(stacked=True, ax=ax, rot=0,
                color=["#e74c3c", "#f39c12", "#27ae60"])
    ax.set_title("Phân bố nhóm DSR theo quy mô"); ax.set_ylabel("tỷ lệ")
    ax.legend(title="DSR", bbox_to_anchor=(1.02, 1))
    fig.tight_layout(); fig.savefig(FIG / "size_vs_dsr.png"); plt.close(fig)

    # f) phân phối 1 feature mạnh theo nhãn
    fig, ax = plt.subplots(figsize=(6, 3))
    feat = "invoice_revenue_growth"
    for lab, c in [(0, "#27ae60"), (1, "#c0392b")]:
        sub = df[df["default"] == lab][feat].dropna()
        ax.hist(sub, bins=40, alpha=0.55, color=c, density=True,
                label=("good" if lab == 0 else "bad"))
    ax.set_title(f"{feat} theo nhãn"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "feature_by_label.png"); plt.close(fig)


if __name__ == "__main__":
    main()
