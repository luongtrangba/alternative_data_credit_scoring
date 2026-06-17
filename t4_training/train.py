"""
ScoreSight · T4 — Hybrid Model Training (global discriminator + DSR heads)
=========================================================================

Kiến trúc HYBRID (đã chọn sau khi đo đạc: routing 3-model-tách KHÔNG thắng global
LightGBM về AUC, vì cây boosting tự xử lý dị biệt + pooling cứu nhóm nghèo data):

  • 1 GLOBAL LightGBM (alt-data, mọi feature)  -> bộ PHÂN BIỆT rủi ro (AUC engine)
  • 3 CALIBRATION HEAD theo nhóm DSR            -> PD chính xác TRONG từng segment
        (đây là chỗ DSR thắng, đo được bằng Brier/ECE)
  • DSR TIERING                                 -> chính sách hạn mức theo nhóm

So sánh (trung thực):
  • scorecard      — single Logistic (status-quo ngân hàng)
  • firmographic   — LightGBM chỉ dữ liệu đăng ký KD (chứng minh giá trị alt-data)
  • global         — LightGBM alt-data (model của ta)

Input :  data/sme_scored_dsr.parquet
Output:  t4_training/models/scoresight_bundle.joblib, output/metrics.csv, figures/

Chạy  :  python3 t4_training/train.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "t4_training"))
from score_mapping import prob_bad_to_score, decision  # noqa: E402

DATA = ROOT / "data" / "sme_scored_dsr.parquet"
MODELS = ROOT / "t4_training" / "models"
OUT = ROOT / "t4_training" / "output"
FIG = OUT / "figures"

TARGET = "default"
DROP = ["customer_id", TARGET, "_gt_latent_risk", "_gt_quality",
        "dsr_weighted", "dsr_group"]
CATEGORICAL = ["industry", "region", "enterprise_size"]
FIRMOGRAPHIC = ["business_age_months", "num_employees",
                "industry", "region", "enterprise_size"]
GROUPS = ["thin", "semi", "thick"]
# Hệ số hạn mức theo độ đầy đủ dữ liệu (DSR tiering)
LIMIT_FACTOR = {"thin": 0.5, "semi": 0.75, "thick": 1.0}
SEED = 42


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y, p):
    if len(np.unique(y)) < 2:
        return {"n": len(y), "auc": np.nan, "ks": np.nan, "gini": np.nan}
    auc = roc_auc_score(y, p)
    return {"n": len(y), "auc": auc, "ks": ks_stat(y, p), "gini": 2 * auc - 1}


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, bins=10):
    """Expected Calibration Error — lệch trung bình giữa PD dự báo và thực tế."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e, n = 0.0, len(y)
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return e


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def make_lgbm(y, leaves=31, n_est=500):
    # Không dùng scale_pos_weight: AUC là rank-based, reweighting imbalance thường
    # hại ranking; calibration ở bước sau lo phần xác suất. Regularize để khỏi
    # overfit (dữ liệu phần lớn monotonic, model quá phức tạp không có lợi).
    return LGBMClassifier(
        n_estimators=n_est, learning_rate=0.02, num_leaves=leaves,
        min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, random_state=SEED, verbose=-1)


def make_scorecard(numeric):
    """Baseline truyền thống: 1 Logistic scorecard (impute + scale + one-hot)."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(class_weight="balanced",
                                                max_iter=2000, random_state=SEED))])


def as_lgb(X):
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


# --------------------------------------------------------------------------- #
def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    features = [c for c in df.columns if c not in DROP]
    numeric = [c for c in features if c not in CATEGORICAL]

    strat = df["dsr_group"].astype(str) + "_" + df[TARGET].astype(str)
    trv, te = train_test_split(df, test_size=0.2, stratify=strat, random_state=SEED)
    s2 = trv["dsr_group"].astype(str) + "_" + trv[TARGET].astype(str)
    fit, cal = train_test_split(trv, test_size=0.25, stratify=s2, random_state=SEED)
    print(f"Fit: {len(fit):,} | Cal: {len(cal):,} | Test: {len(te):,} | feat: {len(features)}")

    yte = te[TARGET].to_numpy()
    Xte_l = as_lgb(te[features])

    # === DISCRIMINATION: global vs baselines =============================
    print("\n--- DISCRIMINATION (AUC) ---")
    gmodel = make_lgbm(fit[TARGET]).fit(as_lgb(fit[features]), fit[TARGET])
    p_raw = gmodel.predict_proba(Xte_l)[:, 1]

    scard = make_scorecard(numeric).fit(fit[features], fit[TARGET])
    p_scard = scard.predict_proba(te[features])[:, 1]

    fmodel = make_lgbm(fit[TARGET]).fit(as_lgb(fit[FIRMOGRAPHIC]), fit[TARGET])
    p_firmo = fmodel.predict_proba(as_lgb(te[FIRMOGRAPHIC]))[:, 1]

    rows = []
    for name, p in [("global(alt-data)", p_raw), ("scorecard(logistic)", p_scard),
                    ("firmographic-only", p_firmo)]:
        m = evaluate(yte, p); m["model"] = name; rows.append(m)
        print(f"  {name:22s} AUC={m['auc']:.3f} KS={m['ks']:.3f} Gini={m['gini']:.3f}")
    print("  global theo nhóm DSR:")
    for g in GROUPS:
        mask = (te["dsr_group"] == g).to_numpy()
        m = evaluate(yte[mask], p_raw[mask])
        print(f"    {g:6s} n={m['n']:>5,} AUC={m['auc']:.3f} KS={m['ks']:.3f}")
    lift = rows[0]["auc"] - rows[2]["auc"]
    print(f"  => ALT-DATA LIFT: firmographic {rows[2]['auc']:.3f} -> "
          f"global {rows[0]['auc']:.3f}  (+{lift:.3f} AUC)")

    # === DSR-AWARE CALIBRATION (chỗ DSR thắng) ==========================
    print("\n--- CALIBRATION: single vs DSR-per-segment (cùng global model) ---")
    Xcal_l = as_lgb(cal[features])
    cal_single = CalibratedClassifierCV(FrozenEstimator(gmodel), method="isotonic")
    cal_single.fit(Xcal_l, cal[TARGET])
    cal_seg = {}
    for g in GROUPS:
        cg = cal[cal["dsr_group"] == g]
        c = CalibratedClassifierCV(FrozenEstimator(gmodel), method="isotonic")
        c.fit(as_lgb(cg[features]), cg[TARGET])
        cal_seg[g] = c

    p_single = cal_single.predict_proba(Xte_l)[:, 1]
    p_dsr = np.empty(len(te))
    for g in GROUPS:
        mask = (te["dsr_group"] == g).to_numpy()
        p_dsr[mask] = cal_seg[g].predict_proba(Xte_l[mask])[:, 1]

    print(f"  {'segment':8s} {'Brier(single)':>14s} {'Brier(DSR)':>12s} "
          f"{'ECE(single)':>12s} {'ECE(DSR)':>10s}")
    cal_rows = []
    for g in GROUPS + ["OVERALL"]:
        mask = np.ones(len(te), bool) if g == "OVERALL" else (te["dsr_group"] == g).to_numpy()
        bs, bd = brier(yte[mask], p_single[mask]), brier(yte[mask], p_dsr[mask])
        es, ed = ece(yte[mask], p_single[mask]), ece(yte[mask], p_dsr[mask])
        cal_rows.append({"segment": g, "brier_single": bs, "brier_dsr": bd,
                         "ece_single": es, "ece_dsr": ed, "n": int(mask.sum())})
        print(f"  {g:8s} {bs:14.4f} {bd:12.4f} {es:12.4f} {ed:10.4f}")
    ece_gain = np.mean([r["ece_single"] - r["ece_dsr"] for r in cal_rows[:3]])
    print(f"  => DSR calibration giảm ECE trung bình {ece_gain:+.4f} "
          f"({'tốt hơn ✓' if ece_gain > 0 else 'không cải thiện'})")

    pd.DataFrame(rows)[["model", "n", "auc", "ks", "gini"]].to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration.csv", index=False)

    # === SCORE 0–1000 + DECISION + DSR TIERING ==========================
    te = te.assign(p=p_dsr, score=prob_bad_to_score(p_dsr))
    _score_and_tiering(te)

    # === SHAP (global model) ============================================
    _shap(gmodel, Xte_l, te[TARGET])

    # === Plots + save ===================================================
    _plots(te, cal_rows, yte, p_single, p_dsr)
    bundle = {"features": features, "numeric": numeric, "categorical": CATEGORICAL,
              "global_model": gmodel, "cal_single": cal_single, "cal_seg": cal_seg,
              "limit_factor": LIMIT_FACTOR, "groups": GROUPS}
    joblib.dump(bundle, MODELS / "scoresight_bundle.joblib")
    print(f"\nĐã lưu: models/scoresight_bundle.joblib | output/metrics.csv + "
          f"calibration.csv | {len(list(FIG.glob('*.png')))} plots")


# --------------------------------------------------------------------------- #
def _score_and_tiering(te):
    print("\n--- ĐIỂM 0–1000 + DECISION + DSR TIERING ---")
    te = te.copy()
    te["decision"] = te["score"].map(decision)
    for d in ["approve", "manual_review", "decline"]:
        sub = te[te["decision"] == d]
        if len(sub):
            print(f"  {d:14s}: {len(sub):>5,} ({len(sub)/len(te):5.1%}) | "
                  f"bad rate = {sub[TARGET].mean():.1%}")
    appr = te[te["decision"] == "approve"]
    print(f"  => Approve {len(appr)/len(te):.0%} @ bad {appr[TARGET].mean():.1%} "
          f"(base {te[TARGET].mean():.1%})")
    # DSR tiering: cùng quyết định nhưng hạn mức theo độ đầy đủ dữ liệu
    print("  DSR tiering trên nhóm APPROVE (hạn mức × theo data sufficiency):")
    for g in GROUPS:
        sub = appr[appr["dsr_group"] == g]
        if len(sub):
            print(f"    {g:6s}: {len(sub):>4,} hồ sơ | limit ×{LIMIT_FACTOR[g]:.2f} "
                  f"| bad rate {sub[TARGET].mean():.1%}")


def _shap(model, X, y):
    import shap
    print("\n--- SHAP (global model) ---")
    Xs = X.sample(min(1500, len(X)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    print("  Top 10 feature theo |SHAP|:")
    for f, v in imp.head(10).items():
        print(f"    {f:30s} {v:.4f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.head(15).iloc[::-1].plot.barh(ax=ax, color="#2c3e50")
    ax.set_title("SHAP importance — global model"); ax.set_xlabel("|SHAP| TB")
    fig.tight_layout(); fig.savefig(FIG / "shap_importance.png"); plt.close(fig)


def _plots(te, cal_rows, yte, p_single, p_dsr):
    # score distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    for lab, c in [(0, "#27ae60"), (1, "#c0392b")]:
        ax.hist(te[te[TARGET] == lab]["score"], bins=40, alpha=0.6, color=c,
                density=True, label=("good" if lab == 0 else "bad"))
    for v in (540, 620):
        ax.axvline(v, ls="--", c="k", lw=0.8)
    ax.set_title("Phân phối điểm theo nhãn"); ax.set_xlabel("score"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "score_distribution.png"); plt.close(fig)

    # ECE bar: single vs DSR theo segment
    seg = [r["segment"] for r in cal_rows[:3]]
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(seg)); w = 0.38
    ax.bar(x - w/2, [r["ece_single"] for r in cal_rows[:3]], w, label="single cal", color="#95a5a6")
    ax.bar(x + w/2, [r["ece_dsr"] for r in cal_rows[:3]], w, label="DSR per-segment", color="#16a085")
    ax.set_xticks(x); ax.set_xticklabels(seg); ax.set_ylabel("ECE (thấp = tốt)")
    ax.set_title("Calibration error: single vs DSR-aware"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "calibration_ece.png"); plt.close(fig)

    # reliability curve theo segment (DSR-calibrated)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, g in zip(axes, GROUPS):
        mask = (te["dsr_group"] == g).to_numpy()
        p = p_dsr[mask]; y = yte[mask]
        edges = np.linspace(0, p.max() + 1e-9, 9)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 7)
        xs, ys = [], []
        for b in range(8):
            mm = idx == b
            if mm.sum() > 10:
                xs.append(p[mm].mean()); ys.append(y[mm].mean())
        ax.plot([0, max(xs+[0.3])], [0, max(xs+[0.3])], ls="--", c="gray")
        ax.plot(xs, ys, "o-", color="#16a085")
        ax.set_title(f"{g}"); ax.set_xlabel("PD dự báo")
    axes[0].set_ylabel("PD thực tế")
    fig.suptitle("Reliability (DSR-calibrated) theo segment")
    fig.tight_layout(); fig.savefig(FIG / "reliability_by_segment.png"); plt.close(fig)


if __name__ == "__main__":
    main()
