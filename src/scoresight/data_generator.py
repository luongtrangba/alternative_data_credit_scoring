"""
ScoreSight · Semi-Synthetic Alternative-Data Generator

Generates a credit scoring dataset for Vietnamese MSMEs using alternative data,
anchored to real credit data from UCI German Credit for realistic risk distribution.

Pipeline:
  1. Load UCI German Credit (1000 records, real good/bad labels)
  2. Fit logistic regression (pure numpy) -> P(bad) per record
  3. Bootstrap to N MSMEs, jitter latent risk in logit space
  4. Assign labels: default ~ Bernoulli(latent_risk)
  5. Generate ~32 alternative data features correlated with quality q = 1 - risk
  6. Mask by source group -> create DSR spectrum (thin / semi / thick)

Output:
  data/sme_altdata.parquet
  data/sme_altdata_sample.csv
  data/feature_dictionary.json

Run: python -m scoresight.data_generator --n 18000 --target-default-rate 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "german.data-numeric"
OUT_DIR = ROOT / "data"


def load_real_anchor() -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(RAW)
    X = arr[:, :-1]
    y = (arr[:, -1] == 2).astype(float)
    return X, y


def fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0,
    lr: float = 0.3, iters: int = 4000,
) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(Xs), 1)), Xs])
    w = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n
        w -= lr * grad
    return 1.0 / (1.0 + np.exp(-Xb @ w))


COVERAGE_RISK_TILT = 1.0


def build_latent_risk(
    rng: np.random.Generator, n: int, target_rate: float,
    coverage: np.ndarray,
) -> np.ndarray:
    X, y = load_real_anchor()
    p_real = fit_logistic(X, y)
    idx = rng.integers(0, len(p_real), size=n)
    p = p_real[idx]
    logit = np.log(p / (1 - p))
    logit += rng.normal(0, 0.7, size=n)
    logit += COVERAGE_RISK_TILT * (0.5 - coverage)
    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = np.clip(1 / (1 + np.exp(-(logit + mid))), 0.005, 0.85).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    risk = 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))
    return np.clip(risk, 0.005, 0.85)


SIGNAL = {
    "shopee_gmv_3m": 0.40, "gmv_growth_12m": 0.48, "order_count_monthly": 0.38,
    "return_rate": 0.45, "seller_rating": 0.40,
    "momo_net_cashflow_avg": 0.45, "pos_volume_6m": 0.42,
    "supplier_payment_regularity": 0.60, "payroll_regularity": 0.52,
    "active_days_per_month": 0.40,
    "invoice_revenue_12m": 0.45, "invoice_revenue_growth": 0.58,
    "unique_buyer_count": 0.50, "vat_filing_on_time_ratio": 0.50,
    "invoice_cancel_rate": 0.45,
    "electricity_consumption_avg": 0.35, "electricity_growth": 0.42,
    "utility_payment_on_time": 0.45,
    "shipment_count_monthly": 0.35, "delivery_success_rate": 0.42,
    "logistics_return_rate": 0.42,
    "google_review_count": 0.25, "google_avg_rating": 0.28,
    "facebook_page_age_months": 0.25, "facebook_engagement_rate": 0.20,
    "buyer_diversity_score": 0.52, "supplier_diversity_score": 0.45,
    "pagerank_score": 0.38, "network_default_exposure": 0.52,
}

SIZE_MULT = {"micro": 0.30, "small": 1.0, "medium": 3.5}

FINANCIAL_BLOCK = {
    "invoice_revenue_12m", "invoice_revenue_growth", "unique_buyer_count",
    "vat_filing_on_time_ratio", "invoice_cancel_rate", "gmv_growth_12m",
    "shopee_gmv_3m", "order_count_monthly", "momo_net_cashflow_avg",
    "pos_volume_6m", "electricity_growth", "electricity_consumption_avg",
}
NETWORK_BLOCK = {
    "buyer_diversity_score", "supplier_diversity_score",
    "pagerank_score", "network_default_exposure",
}


def _signal(q: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    noise = rng.normal(0, 1, size=len(q))
    z = (q - 0.5) * 4.0
    return strength * z + (1 - strength) * noise


def _lognormal_from(
    signal: np.ndarray, median: float, spread: float, rng: np.random.Generator,
) -> np.ndarray:
    return median * np.exp(spread * signal)


def generate_features(
    rng: np.random.Generator, risk: np.ndarray,
    size: np.ndarray, coverage: np.ndarray,
) -> pd.DataFrame:
    q = 1.0 - risk
    n = len(q)
    m = np.array([SIZE_MULT[s] for s in size])
    rm = np.sqrt(m)
    fin_mult = 0.5 + 1.0 * coverage
    net_mult = 0.5 + 1.0 * (1.0 - coverage)
    df = pd.DataFrame()

    def sig(name: str) -> np.ndarray:
        s = SIGNAL[name]
        if name in FINANCIAL_BLOCK:
            s = np.clip(s * fin_mult, 0.05, 0.95)
        elif name in NETWORK_BLOCK:
            s = np.clip(s * net_mult, 0.05, 0.95)
        return _signal(q, rng, s)

    emp = np.where(
        size == "micro", rng.integers(1, 11, n),
        np.where(size == "small", rng.integers(10, 51, n), rng.integers(50, 201, n)),
    )
    df["num_employees"] = emp.astype(int)
    df["business_age_months"] = np.clip(
        (rng.gamma(2.0, 16.0, n) * (0.6 + 0.7 * q) * (0.7 + 0.5 * np.log1p(m))).round(),
        1, 360,
    ).astype(int)
    df["industry"] = rng.choice(
        ["F&B", "retail", "manufacturing", "services", "agriculture", "wholesale"],
        size=n, p=[0.22, 0.28, 0.15, 0.20, 0.08, 0.07],
    )
    df["region"] = rng.choice(
        ["HCMC", "Hanoi", "Danang", "CanTho", "HaiPhong", "other"],
        size=n, p=[0.30, 0.25, 0.10, 0.07, 0.06, 0.22],
    )

    df["shopee_gmv_3m"] = (_lognormal_from(sig("shopee_gmv_3m"), 180_000_000, 0.9, rng) * m).round(-3)
    df["gmv_growth_12m"] = np.clip(0.05 + 0.30 * sig("gmv_growth_12m") + rng.normal(0, 0.15, n), -0.6, 2.0).round(3)
    df["order_count_monthly"] = np.clip((_lognormal_from(sig("order_count_monthly"), 120, 0.8, rng) * m).round(), 0, None).astype(int)
    df["return_rate"] = np.clip(0.12 - 0.06 * sig("return_rate") + rng.normal(0, 0.04, n), 0.0, 0.95).round(3)
    df["seller_rating"] = np.clip(4.2 + 0.4 * np.tanh(sig("seller_rating")) + rng.normal(0, 0.2, n), 1.0, 5.0).round(2)

    df["momo_net_cashflow_avg"] = (_lognormal_from(sig("momo_net_cashflow_avg"), 28_000_000, 0.8, rng) * m).round(-3)
    df["pos_volume_6m"] = (_lognormal_from(sig("pos_volume_6m"), 350_000_000, 0.9, rng) * m).round(-3)
    df["supplier_payment_regularity"] = np.clip(0.78 + 0.20 * np.tanh(sig("supplier_payment_regularity")) + rng.normal(0, 0.07, n), 0.0, 1.0).round(3)
    df["payroll_regularity"] = np.clip(0.80 + 0.18 * np.tanh(sig("payroll_regularity")) + rng.normal(0, 0.08, n), 0.0, 1.0).round(3)
    df["active_days_per_month"] = np.clip((20 + 8 * np.tanh(sig("active_days_per_month")) + rng.normal(0, 4, n)).round(), 0, 31).astype(int)

    df["invoice_revenue_12m"] = (_lognormal_from(sig("invoice_revenue_12m"), 1_200_000_000, 0.95, rng) * m).round(-3)
    df["invoice_revenue_growth"] = np.clip(0.06 + 0.28 * sig("invoice_revenue_growth") + rng.normal(0, 0.12, n), -0.5, 1.8).round(3)
    df["unique_buyer_count"] = np.clip((_lognormal_from(sig("unique_buyer_count"), 25, 0.7, rng) * rm).round(), 1, None).astype(int)
    df["vat_filing_on_time_ratio"] = np.clip(0.82 + 0.16 * np.tanh(sig("vat_filing_on_time_ratio")) + rng.normal(0, 0.08, n), 0.0, 1.0).round(3)
    df["invoice_cancel_rate"] = np.clip(0.05 - 0.03 * sig("invoice_cancel_rate") + rng.normal(0, 0.025, n), 0.0, 0.6).round(3)

    df["electricity_consumption_avg"] = (_lognormal_from(sig("electricity_consumption_avg"), 1_500, 0.8, rng) * m).round(1)
    df["electricity_growth"] = np.clip(0.04 + 0.20 * sig("electricity_growth") + rng.normal(0, 0.12, n), -0.5, 1.2).round(3)
    df["utility_payment_on_time"] = np.clip(0.85 + 0.13 * np.tanh(sig("utility_payment_on_time")) + rng.normal(0, 0.09, n), 0.0, 1.0).round(3)

    df["shipment_count_monthly"] = np.clip((_lognormal_from(sig("shipment_count_monthly"), 90, 0.85, rng) * m).round(), 0, None).astype(int)
    df["delivery_success_rate"] = np.clip(0.90 + 0.08 * np.tanh(sig("delivery_success_rate")) + rng.normal(0, 0.05, n), 0.0, 1.0).round(3)
    df["logistics_return_rate"] = np.clip(0.10 - 0.05 * sig("logistics_return_rate") + rng.normal(0, 0.04, n), 0.0, 0.9).round(3)

    df["google_review_count"] = np.clip((_lognormal_from(sig("google_review_count"), 35, 0.9, rng) * rm).round(), 0, None).astype(int)
    df["google_avg_rating"] = np.clip(4.0 + 0.5 * np.tanh(sig("google_avg_rating")) + rng.normal(0, 0.3, n), 1.0, 5.0).round(2)
    df["facebook_page_age_months"] = np.clip((_lognormal_from(sig("facebook_page_age_months"), 40, 0.6, rng)).round(), 0, 240).astype(int)
    df["facebook_engagement_rate"] = np.clip(0.03 + 0.02 * np.tanh(sig("facebook_engagement_rate")) + rng.normal(0, 0.02, n), 0.0, 0.5).round(4)

    df["buyer_diversity_score"] = np.clip(0.5 + 0.4 * np.tanh(sig("buyer_diversity_score")) + rng.normal(0, 0.12, n), 0.0, 1.0).round(3)
    df["supplier_diversity_score"] = np.clip(0.5 + 0.35 * np.tanh(sig("supplier_diversity_score")) + rng.normal(0, 0.12, n), 0.0, 1.0).round(3)
    df["pagerank_score"] = np.clip(_lognormal_from(sig("pagerank_score"), 0.0015, 0.5, rng), 0, None).round(6)
    df["network_default_exposure"] = np.clip(0.12 - 0.10 * np.tanh(sig("network_default_exposure")) + rng.normal(0, 0.05, n), 0.0, 1.0).round(3)
    p_flag = np.clip(0.02 + 0.10 * risk, 0, 1)
    df["shared_device_risk_flag"] = (rng.random(n) < p_flag).astype(int)

    return df


SOURCE_GROUPS = {
    "ecommerce": ["shopee_gmv_3m", "gmv_growth_12m", "order_count_monthly", "return_rate", "seller_rating"],
    "payment": ["momo_net_cashflow_avg", "pos_volume_6m", "supplier_payment_regularity", "payroll_regularity", "active_days_per_month"],
    "einvoice": ["invoice_revenue_12m", "invoice_revenue_growth", "unique_buyer_count", "vat_filing_on_time_ratio", "invoice_cancel_rate"],
    "utility": ["electricity_consumption_avg", "electricity_growth", "utility_payment_on_time"],
    "logistics": ["shipment_count_monthly", "delivery_success_rate", "logistics_return_rate"],
    "digital_footprint": ["google_review_count", "google_avg_rating", "facebook_page_age_months", "facebook_engagement_rate"],
    "graph": ["buyer_diversity_score", "supplier_diversity_score", "pagerank_score", "network_default_exposure", "shared_device_risk_flag"],
}

GROUP_BASE_AVAIL = {
    "ecommerce": 0.55, "payment": 0.70, "einvoice": 0.62, "utility": 0.58,
    "logistics": 0.50, "digital_footprint": 0.65, "graph": 0.78,
}

SIZE_PROPORTIONS = {"micro": 0.45, "small": 0.38, "medium": 0.17}
SIZE_AVAIL_LIFT = {"micro": -0.20, "small": 0.0, "medium": 0.20}

ALT_FIELDS = [c for cols in SOURCE_GROUPS.values() for c in cols]


def draw_enterprise_size(rng: np.random.Generator, n: int) -> np.ndarray:
    sizes = list(SIZE_PROPORTIONS.keys())
    probs = list(SIZE_PROPORTIONS.values())
    return rng.choice(sizes, size=n, p=probs)


def draw_coverage(rng: np.random.Generator, n: int, size: np.ndarray) -> np.ndarray:
    size_lift = np.array([SIZE_AVAIL_LIFT[s] for s in size])
    return np.clip(rng.beta(2.4, 1.9, n) + size_lift + rng.normal(0, 0.05, n), 0.03, 0.99)


def apply_dsr_masking(
    df: pd.DataFrame, rng: np.random.Generator, coverage: np.ndarray,
) -> pd.DataFrame:
    n = len(df)
    for group, cols in SOURCE_GROUPS.items():
        p_present = np.clip(coverage * (0.92 + 0.14 * GROUP_BASE_AVAIL[group]), 0.02, 0.99)
        present = rng.random(n) < p_present
        absent_idx = np.where(~present)[0]
        if len(absent_idx) > 0:
            df.loc[df.index[absent_idx], cols] = np.nan
    return df


def provisional_dsr(df: pd.DataFrame) -> pd.Series:
    return df[ALT_FIELDS].notna().mean(axis=1)


def dsr_group(dsr: pd.Series) -> pd.Series:
    return pd.cut(dsr, bins=[-0.01, 0.40, 0.70, 1.01], labels=["thin", "semi", "thick"])


def build_feature_dictionary() -> dict:
    spec = {
        "business_age_months": ("identity", "high_good", 0.6, 0.4),
        "num_employees": ("identity", "high_good", 0.5, 0.4),
        "shopee_gmv_3m": ("ecommerce", "high_good", 0.7, 0.55),
        "gmv_growth_12m": ("ecommerce", "high_good", 0.8, 0.55),
        "order_count_monthly": ("ecommerce", "high_good", 0.6, 0.55),
        "return_rate": ("ecommerce", "high_bad", 0.7, 0.55),
        "seller_rating": ("ecommerce", "high_good", 0.5, 0.55),
        "momo_net_cashflow_avg": ("payment", "high_good", 0.8, 0.75),
        "pos_volume_6m": ("payment", "high_good", 0.7, 0.60),
        "supplier_payment_regularity": ("payment", "high_good", 1.0, 0.75),
        "payroll_regularity": ("payment", "high_good", 0.9, 0.65),
        "active_days_per_month": ("payment", "high_good", 0.5, 0.75),
        "invoice_revenue_12m": ("einvoice", "high_good", 0.9, 0.80),
        "invoice_revenue_growth": ("einvoice", "high_good", 1.0, 0.80),
        "unique_buyer_count": ("einvoice", "high_good", 0.9, 0.70),
        "vat_filing_on_time_ratio": ("einvoice", "high_good", 0.8, 0.80),
        "invoice_cancel_rate": ("einvoice", "high_bad", 0.6, 0.80),
        "electricity_consumption_avg": ("utility", "high_good", 0.5, 0.50),
        "electricity_growth": ("utility", "high_good", 0.6, 0.50),
        "utility_payment_on_time": ("utility", "high_good", 0.7, 0.50),
        "shipment_count_monthly": ("logistics", "high_good", 0.5, 0.52),
        "delivery_success_rate": ("logistics", "high_good", 0.6, 0.52),
        "logistics_return_rate": ("logistics", "high_bad", 0.6, 0.52),
        "google_review_count": ("digital_footprint", "high_good", 0.3, 0.30),
        "google_avg_rating": ("digital_footprint", "high_good", 0.3, 0.30),
        "facebook_page_age_months": ("digital_footprint", "high_good", 0.3, 0.30),
        "facebook_engagement_rate": ("digital_footprint", "high_good", 0.2, 0.30),
        "buyer_diversity_score": ("graph", "high_good", 0.8, 0.62),
        "supplier_diversity_score": ("graph", "high_good", 0.6, 0.55),
        "pagerank_score": ("graph", "high_good", 0.5, 0.62),
        "network_default_exposure": ("graph", "high_bad", 0.9, 0.62),
        "shared_device_risk_flag": ("graph", "high_bad", 0.7, 0.30),
    }
    return {
        name: {
            "source": src, "direction": d, "default_weight": w,
            "signal_strength": SIGNAL.get(name, s),
        }
        for name, (src, d, w, s) in spec.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ScoreSight semi-synthetic generator")
    ap.add_argument("--n", type=int, default=18000, help="number of MSMEs")
    ap.add_argument("--target-default-rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    size = draw_enterprise_size(rng, args.n)
    coverage = draw_coverage(rng, args.n, size)
    risk = build_latent_risk(rng, args.n, args.target_default_rate, coverage)
    default = (rng.random(args.n) < risk).astype(int)
    df = generate_features(rng, risk, size, coverage)
    df["enterprise_size"] = size
    df = apply_dsr_masking(df, rng, coverage)

    df.insert(0, "customer_id", [f"SME_{i:06d}" for i in range(args.n)])
    df["default"] = default
    df["_gt_latent_risk"] = risk.round(4)
    df["_gt_quality"] = (1 - risk).round(4)

    dsr = provisional_dsr(df)
    df["_dsr_raw"] = dsr.round(3)
    df["_dsr_group"] = dsr_group(dsr).astype(str)

    pq_path = OUT_DIR / "sme_altdata.parquet"
    df.to_parquet(pq_path, index=False)
    df.head(300).to_csv(OUT_DIR / "sme_altdata_sample.csv", index=False)
    with open(OUT_DIR / "feature_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(build_feature_dictionary(), f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print(f"  ScoreSight dataset — N = {args.n:,}")
    print("=" * 64)
    print(f"Overall default rate : {default.mean():.1%}")
    print(f"Columns              : {df.shape[1]} (alt fields: {len(ALT_FIELDS)})")
    print(f"Saved                : {pq_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
