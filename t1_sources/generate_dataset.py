"""
ScoreSight · T1 — Semi-Synthetic Alternative-Data Generator
============================================================

Sinh bộ dữ liệu chấm điểm tín dụng SME dựa trên DỮ LIỆU PHI TRUYỀN THỐNG
(alternative data), nhưng được "neo" vào một bộ dữ liệu tín dụng THẬT để
phân phối rủi ro và nhãn good/bad có cơ sở thực tế.

Quy trình:
  1. Load UCI German Credit (1000 hồ sơ, nhãn good/bad thật).
  2. Fit logistic regression (pure numpy) -> P(bad) cho từng hồ sơ thật.
     => Phân phối "rủi ro tiềm ẩn" (latent risk) realistic.
  3. Bootstrap lên N SME, jitter latent risk trong không gian logit.
  4. Gán nhãn default ~ Bernoulli(latent risk).  [ground truth]
  5. Sinh ~32 feature alternative data, TƯƠNG QUAN với chất lượng q = 1 - risk,
     với cường độ tín hiệu (signal strength) khác nhau theo từng nguồn.
  6. Mask theo NHÓM NGUỒN (mỗi SME chỉ có một số nguồn) -> tạo phổ DSR
     trải đều thin / semi / thick.

Output:
  data/sme_altdata.parquet        — dataset đầy đủ (NaN = nguồn không có)
  data/sme_altdata_sample.csv     — 300 dòng để xem nhanh
  data/feature_dictionary.json    — mô tả feature: nguồn, hướng, trọng số gợi ý

Chạy:  python3 t1_sources/generate_dataset.py --n 18000 --target-default-rate 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Đường dẫn
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "german.data-numeric"
OUT_DIR = ROOT / "data"


# --------------------------------------------------------------------------- #
# Bước 1–2: Học rủi ro tiềm ẩn từ dữ liệu THẬT
# --------------------------------------------------------------------------- #
def load_real_anchor() -> tuple[np.ndarray, np.ndarray]:
    """German credit numeric: 24 feature + 1 cột nhãn (1=good, 2=bad)."""
    arr = np.loadtxt(RAW)
    X = arr[:, :-1]
    y = (arr[:, -1] == 2).astype(float)  # 1 = bad (vỡ nợ)
    return X, y


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 lr: float = 0.3, iters: int = 4000) -> np.ndarray:
    """Logistic regression bằng gradient descent thuần numpy.

    Trả về P(bad) đã hiệu chỉnh cho từng hàng của X.
    """
    # Chuẩn hóa feature (tránh feature scale lớn lấn át)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xb = np.hstack([np.ones((len(Xs), 1)), Xs])  # thêm intercept

    w = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n  # L2 (không phạt intercept)
        w -= lr * grad

    return 1.0 / (1.0 + np.exp(-Xb @ w))


# Coupling rủi ro theo ĐỘ ĐẦY ĐỦ DỮ LIỆU (coverage), KHÔNG theo quy mô:
# thin-file rủi ro hơn chút (ít dữ liệu kiểm chứng -> giám sát kém, dễ gian lận).
# Thực tế hơn "DSR độc lập hoàn toàn" và khiến DSR có giá trị dự báo + per-segment
# calibration có ý nghĩa. Vì theo coverage (không theo size) nên "micro không rủi
# ro hơn" vẫn đúng (micro có thể thick, thin có thể medium).
COVERAGE_RISK_TILT = 1.0


def build_latent_risk(rng: np.random.Generator, n: int, target_rate: float,
                      coverage: np.ndarray) -> np.ndarray:
    """Bootstrap rủi ro thật lên n SME + tilt theo coverage, rescale base rate."""
    X, y = load_real_anchor()
    p_real = fit_logistic(X, y)  # ~1000 giá trị P(bad) realistic

    # Bootstrap (lấy mẫu có hoàn lại) -> n hồ sơ
    idx = rng.integers(0, len(p_real), size=n)
    p = p_real[idx]

    # Chuyển sang logit, jitter + tilt theo coverage (thin -> rủi ro cao hơn)
    logit = np.log(p / (1 - p))
    logit += rng.normal(0, 0.7, size=n)
    logit += COVERAGE_RISK_TILT * (0.5 - coverage)

    # Rescale base rate về target. mean(sigmoid) != sigmoid(mean) nên giải
    # shift bằng bisection để E[default] = target chính xác.
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


# --------------------------------------------------------------------------- #
# Bước 5: Sinh feature alternative data từ chất lượng q = 1 - risk
# --------------------------------------------------------------------------- #
def _signal(q: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Trộn tín hiệu chất lượng với nhiễu. strength in [0,1].

    strength cao -> feature phản ánh chất lượng rõ (IV cao).
    strength thấp -> feature nhiễu, ít giá trị dự báo (IV thấp).
    """
    noise = rng.normal(0, 1, size=len(q))
    z = (q - 0.5) * 4.0  # đưa q về thang ~[-2, 2]
    return strength * z + (1 - strength) * noise


def _lognormal_from(signal: np.ndarray, median: float, spread: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Map tín hiệu -> giá trị dương dạng lognormal (doanh thu, GMV...)."""
    return median * np.exp(spread * signal)


# Hệ số quy mô áp lên các feature ĐỘ LỚN (magnitude). Quy mô độc lập với rủi ro:
# DN medium có GMV/doanh thu/lao động lớn hơn DÙ chất lượng tín dụng thế nào.
SIZE_MULT = {"micro": 0.30, "small": 1.0, "medium": 3.5}

# Signal strength của TỪNG feature — nguồn duy nhất, dùng cho cả việc sinh dữ
# liệu và feature_dictionary. Cao = phản ánh chất lượng rõ (IV cao). Đã hiệu
# chỉnh để top IV ~0.25–0.45 (mạnh nhưng tin được, tránh "leak" IV>0.5).
SIGNAL = {
    # ecommerce
    "shopee_gmv_3m": 0.40, "gmv_growth_12m": 0.48, "order_count_monthly": 0.38,
    "return_rate": 0.45, "seller_rating": 0.40,
    # payment
    "momo_net_cashflow_avg": 0.45, "pos_volume_6m": 0.42,
    "supplier_payment_regularity": 0.60, "payroll_regularity": 0.52,
    "active_days_per_month": 0.40,
    # einvoice
    "invoice_revenue_12m": 0.45, "invoice_revenue_growth": 0.58,
    "unique_buyer_count": 0.50, "vat_filing_on_time_ratio": 0.50,
    "invoice_cancel_rate": 0.45,
    # utility
    "electricity_consumption_avg": 0.35, "electricity_growth": 0.42,
    "utility_payment_on_time": 0.45,
    # logistics
    "shipment_count_monthly": 0.35, "delivery_success_rate": 0.42,
    "logistics_return_rate": 0.42,
    # digital_footprint (yếu — nhiễu)
    "google_review_count": 0.25, "google_avg_rating": 0.28,
    "facebook_page_age_months": 0.25, "facebook_engagement_rate": 0.20,
    # graph
    "buyer_diversity_score": 0.52, "supplier_diversity_score": 0.45,
    "pagerank_score": 0.38, "network_default_exposure": 0.52,
}


# Cấu trúc tín hiệu ĐẶC THÙ THEO NHÓM (vì sao cần DSR routing):
#   - Thin-file thiếu tài chính -> rủi ro lộ qua GRAPH/mạng lưới.
#   - Thick-file đủ lịch sử -> rủi ro lộ qua TÀI CHÍNH/e-invoice.
# Mỗi firm: tín hiệu của block tài chính mạnh khi coverage cao (thick), tín hiệu
# block mạng lưới mạnh khi coverage thấp (thin). Global model dùng 1 bộ trọng số
# trung bình -> kém tối ưu mỗi regime; routed model chuyên biệt -> thắng chính đáng.
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


def generate_features(rng: np.random.Generator, risk: np.ndarray,
                      size: np.ndarray, coverage: np.ndarray) -> pd.DataFrame:
    q = 1.0 - risk  # chất lượng: cao = tốt
    n = len(q)
    m = np.array([SIZE_MULT[s] for s in size])  # hệ số độ lớn theo quy mô
    rm = np.sqrt(m)  # scale dưới tuyến tính cho count (buyer, review)
    # Hệ số regime theo coverage: thick (cov cao) -> tài chính mạnh; thin -> mạng lưới mạnh
    fin_mult = 0.5 + 1.0 * coverage          # ~1.5 cho thick, ~0.5 cho thin
    net_mult = 0.5 + 1.0 * (1.0 - coverage)  # ~1.5 cho thin, ~0.5 cho thick
    df = pd.DataFrame()

    def sig(name: str) -> np.ndarray:
        """Tín hiệu độc lập cho từng feature; cường độ thay đổi theo regime nhóm."""
        s = SIGNAL[name]
        if name in FINANCIAL_BLOCK:
            s = np.clip(s * fin_mult, 0.05, 0.95)
        elif name in NETWORK_BLOCK:
            s = np.clip(s * net_mult, 0.05, 0.95)
        return _signal(q, rng, s)

    # --- Firmographic / định danh (LUÔN có, như dữ liệu đăng ký KD) ---------
    # Số lao động: quyết định bởi QUY MÔ (không phải chất lượng)
    emp = np.where(size == "micro", rng.integers(1, 11, n),
                   np.where(size == "small", rng.integers(10, 51, n),
                            rng.integers(50, 201, n)))
    df["num_employees"] = emp.astype(int)
    # Tuổi DN: chất lượng (tốt -> lâu năm) + quy mô (lớn -> lâu năm)
    df["business_age_months"] = np.clip(
        (rng.gamma(2.0, 16.0, n) * (0.6 + 0.7 * q) * (0.7 + 0.5 * np.log1p(m))).round(),
        1, 360).astype(int)
    df["industry"] = rng.choice(
        ["F&B", "retail", "manufacturing", "services", "agriculture", "wholesale"],
        size=n, p=[0.22, 0.28, 0.15, 0.20, 0.08, 0.07])
    df["region"] = rng.choice(
        ["HCMC", "Hanoi", "Danang", "CanTho", "HaiPhong", "other"],
        size=n, p=[0.30, 0.25, 0.10, 0.07, 0.06, 0.22])

    # --- Nhóm 1: E-commerce / Marketplace ----------------------------------
    df["shopee_gmv_3m"] = (_lognormal_from(sig("shopee_gmv_3m"), 180_000_000, 0.9, rng) * m).round(-3)
    df["gmv_growth_12m"] = np.clip(0.05 + 0.30 * sig("gmv_growth_12m") + rng.normal(0, 0.15, n), -0.6, 2.0).round(3)
    df["order_count_monthly"] = np.clip((_lognormal_from(sig("order_count_monthly"), 120, 0.8, rng) * m).round(), 0, None).astype(int)
    df["return_rate"] = np.clip(0.12 - 0.06 * sig("return_rate") + rng.normal(0, 0.04, n), 0.0, 0.95).round(3)
    df["seller_rating"] = np.clip(4.2 + 0.4 * np.tanh(sig("seller_rating")) + rng.normal(0, 0.2, n), 1.0, 5.0).round(2)

    # --- Nhóm 2: Digital Payment / E-wallet --------------------------------
    df["momo_net_cashflow_avg"] = (_lognormal_from(sig("momo_net_cashflow_avg"), 28_000_000, 0.8, rng) * m).round(-3)
    df["pos_volume_6m"] = (_lognormal_from(sig("pos_volume_6m"), 350_000_000, 0.9, rng) * m).round(-3)
    df["supplier_payment_regularity"] = np.clip(0.78 + 0.20 * np.tanh(sig("supplier_payment_regularity")) + rng.normal(0, 0.07, n), 0.0, 1.0).round(3)
    df["payroll_regularity"] = np.clip(0.80 + 0.18 * np.tanh(sig("payroll_regularity")) + rng.normal(0, 0.08, n), 0.0, 1.0).round(3)
    df["active_days_per_month"] = np.clip((20 + 8 * np.tanh(sig("active_days_per_month")) + rng.normal(0, 4, n)).round(), 0, 31).astype(int)

    # --- Nhóm 3: E-invoice / Tax (đã được Thuế xác nhận) -------------------
    df["invoice_revenue_12m"] = (_lognormal_from(sig("invoice_revenue_12m"), 1_200_000_000, 0.95, rng) * m).round(-3)
    df["invoice_revenue_growth"] = np.clip(0.06 + 0.28 * sig("invoice_revenue_growth") + rng.normal(0, 0.12, n), -0.5, 1.8).round(3)
    df["unique_buyer_count"] = np.clip((_lognormal_from(sig("unique_buyer_count"), 25, 0.7, rng) * rm).round(), 1, None).astype(int)
    df["vat_filing_on_time_ratio"] = np.clip(0.82 + 0.16 * np.tanh(sig("vat_filing_on_time_ratio")) + rng.normal(0, 0.08, n), 0.0, 1.0).round(3)
    df["invoice_cancel_rate"] = np.clip(0.05 - 0.03 * sig("invoice_cancel_rate") + rng.normal(0, 0.025, n), 0.0, 0.6).round(3)

    # --- Nhóm 4: Utility / Infrastructure ----------------------------------
    df["electricity_consumption_avg"] = (_lognormal_from(sig("electricity_consumption_avg"), 1_500, 0.8, rng) * m).round(1)
    df["electricity_growth"] = np.clip(0.04 + 0.20 * sig("electricity_growth") + rng.normal(0, 0.12, n), -0.5, 1.2).round(3)
    df["utility_payment_on_time"] = np.clip(0.85 + 0.13 * np.tanh(sig("utility_payment_on_time")) + rng.normal(0, 0.09, n), 0.0, 1.0).round(3)

    # --- Nhóm 5: Logistics / Supply Chain ----------------------------------
    df["shipment_count_monthly"] = np.clip((_lognormal_from(sig("shipment_count_monthly"), 90, 0.85, rng) * m).round(), 0, None).astype(int)
    df["delivery_success_rate"] = np.clip(0.90 + 0.08 * np.tanh(sig("delivery_success_rate")) + rng.normal(0, 0.05, n), 0.0, 1.0).round(3)
    df["logistics_return_rate"] = np.clip(0.10 - 0.05 * sig("logistics_return_rate") + rng.normal(0, 0.04, n), 0.0, 0.9).round(3)

    # --- Nhóm 6: Digital Footprint / Reputation (yếu — nhiễu) --------------
    df["google_review_count"] = np.clip((_lognormal_from(sig("google_review_count"), 35, 0.9, rng) * rm).round(), 0, None).astype(int)
    df["google_avg_rating"] = np.clip(4.0 + 0.5 * np.tanh(sig("google_avg_rating")) + rng.normal(0, 0.3, n), 1.0, 5.0).round(2)
    df["facebook_page_age_months"] = np.clip((_lognormal_from(sig("facebook_page_age_months"), 40, 0.6, rng)).round(), 0, 240).astype(int)
    df["facebook_engagement_rate"] = np.clip(0.03 + 0.02 * np.tanh(sig("facebook_engagement_rate")) + rng.normal(0, 0.02, n), 0.0, 0.5).round(4)

    # --- Nhóm 7: Graph / Network -------------------------------------------
    df["buyer_diversity_score"] = np.clip(0.5 + 0.4 * np.tanh(sig("buyer_diversity_score")) + rng.normal(0, 0.12, n), 0.0, 1.0).round(3)
    df["supplier_diversity_score"] = np.clip(0.5 + 0.35 * np.tanh(sig("supplier_diversity_score")) + rng.normal(0, 0.12, n), 0.0, 1.0).round(3)
    df["pagerank_score"] = np.clip(_lognormal_from(sig("pagerank_score"), 0.0015, 0.5, rng), 0, None).round(6)
    # network_default_exposure: CAO khi chất lượng THẤP (rủi ro lan truyền)
    df["network_default_exposure"] = np.clip(0.12 - 0.10 * np.tanh(sig("network_default_exposure")) + rng.normal(0, 0.05, n), 0.0, 1.0).round(3)
    # shared_device_risk_flag: hiếm, xác suất cao hơn ở nhóm rủi ro
    p_flag = np.clip(0.02 + 0.10 * risk, 0, 1)
    df["shared_device_risk_flag"] = (rng.random(n) < p_flag).astype(int)

    return df


# --------------------------------------------------------------------------- #
# Bước 6: Mask theo nhóm nguồn -> tạo phổ DSR (thin / semi / thick)
# --------------------------------------------------------------------------- #
SOURCE_GROUPS = {
    "ecommerce": ["shopee_gmv_3m", "gmv_growth_12m", "order_count_monthly",
                  "return_rate", "seller_rating"],
    "payment": ["momo_net_cashflow_avg", "pos_volume_6m", "supplier_payment_regularity",
                "payroll_regularity", "active_days_per_month"],
    "einvoice": ["invoice_revenue_12m", "invoice_revenue_growth", "unique_buyer_count",
                 "vat_filing_on_time_ratio", "invoice_cancel_rate"],
    "utility": ["electricity_consumption_avg", "electricity_growth", "utility_payment_on_time"],
    "logistics": ["shipment_count_monthly", "delivery_success_rate", "logistics_return_rate"],
    "digital_footprint": ["google_review_count", "google_avg_rating",
                          "facebook_page_age_months", "facebook_engagement_rate"],
    "graph": ["buyer_diversity_score", "supplier_diversity_score", "pagerank_score",
              "network_default_exposure", "shared_device_risk_flag"],
}

# Tỷ lệ một SME "có" mỗi nguồn (base). Một số nguồn phổ biến hơn nguồn khác.
# graph cao hơn: ngay cả thin-file cũng thường có dữ liệu mạng lưới (từ chính
# graph giao dịch của tổ chức cho vay) -> thin-file vẫn chấm được nhờ graph.
GROUP_BASE_AVAIL = {
    "ecommerce": 0.55, "payment": 0.70, "einvoice": 0.62, "utility": 0.58,
    "logistics": 0.50, "digital_footprint": 0.65, "graph": 0.78,
}

ALWAYS_PRESENT = ["business_age_months", "num_employees", "industry", "region",
                  "enterprise_size"]


def draw_coverage(rng: np.random.Generator, n: int, size: np.ndarray) -> np.ndarray:
    """Độ sẵn có dữ liệu (coverage) per firm — quyết định nhóm DSR VÀ regime tín
    hiệu. Gắn với quy mô (micro ít data hơn), độc lập với rủi ro."""
    size_lift = np.array([SIZE_AVAIL_LIFT[s] for s in size])
    return np.clip(rng.beta(2.4, 1.9, n) + size_lift + rng.normal(0, 0.05, n),
                   0.03, 0.99)


# Phân khúc MSME theo Nghị định 80/2021/NĐ-CP. Tỷ lệ xấp xỉ thực tế VN
# (micro chiếm đa số). Quy mô được vẽ ĐỘC LẬP với rủi ro -> tránh confound:
# micro KHÔNG xấu hơn về tín dụng, chỉ ÍT DỮ LIỆU hơn (luận điểm cốt lõi của DSR).
SIZE_PROPORTIONS = {"micro": 0.45, "small": 0.38, "medium": 0.17}


def draw_enterprise_size(rng: np.random.Generator, n: int) -> np.ndarray:
    sizes = list(SIZE_PROPORTIONS.keys())
    probs = list(SIZE_PROPORTIONS.values())
    return rng.choice(sizes, size=n, p=probs)


# Quy mô càng nhỏ -> càng ít nguồn dữ liệu phi truyền thống (micro hay thin-file)
SIZE_AVAIL_LIFT = {"micro": -0.20, "small": 0.0, "medium": 0.20}


def apply_dsr_masking(df: pd.DataFrame, rng: np.random.Generator,
                      coverage: np.ndarray) -> pd.DataFrame:
    """Mỗi SME chỉ có một tập nguồn -> set NaN cho nguồn vắng mặt.

    Dùng CÙNG coverage đã quyết định regime tín hiệu (draw_coverage). KHÔNG gắn
    trực tiếp với nhãn để tránh leakage — DSR chỉ quyết định ROUTING mô hình nào.
    """
    n = len(df)
    for group, cols in SOURCE_GROUPS.items():
        # factor ~1.0 để coverage thực tế xấp xỉ coverage mục tiêu
        p_present = np.clip(coverage * (0.92 + 0.14 * GROUP_BASE_AVAIL[group]), 0.02, 0.99)
        present = rng.random(n) < p_present
        absent_idx = np.where(~present)[0]
        if len(absent_idx) > 0:
            df.loc[df.index[absent_idx], cols] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Feature dictionary (cho T3 IV/weights + demo)
# --------------------------------------------------------------------------- #
def build_feature_dictionary() -> dict:
    """direction: 'high_good' = giá trị cao -> rủi ro thấp; ngược lại 'high_bad'.

    weight = trọng số w_i khởi đầu (domain expert), strength = signal đã sinh.
    """
    spec = {
        # firmographic
        "business_age_months": ("identity", "high_good", 0.6, 0.4),
        "num_employees": ("identity", "high_good", 0.5, 0.4),
        # ecommerce
        "shopee_gmv_3m": ("ecommerce", "high_good", 0.7, 0.55),
        "gmv_growth_12m": ("ecommerce", "high_good", 0.8, 0.55),
        "order_count_monthly": ("ecommerce", "high_good", 0.6, 0.55),
        "return_rate": ("ecommerce", "high_bad", 0.7, 0.55),
        "seller_rating": ("ecommerce", "high_good", 0.5, 0.55),
        # payment
        "momo_net_cashflow_avg": ("payment", "high_good", 0.8, 0.75),
        "pos_volume_6m": ("payment", "high_good", 0.7, 0.60),
        "supplier_payment_regularity": ("payment", "high_good", 1.0, 0.75),
        "payroll_regularity": ("payment", "high_good", 0.9, 0.65),
        "active_days_per_month": ("payment", "high_good", 0.5, 0.75),
        # einvoice
        "invoice_revenue_12m": ("einvoice", "high_good", 0.9, 0.80),
        "invoice_revenue_growth": ("einvoice", "high_good", 1.0, 0.80),
        "unique_buyer_count": ("einvoice", "high_good", 0.9, 0.70),
        "vat_filing_on_time_ratio": ("einvoice", "high_good", 0.8, 0.80),
        "invoice_cancel_rate": ("einvoice", "high_bad", 0.6, 0.80),
        # utility
        "electricity_consumption_avg": ("utility", "high_good", 0.5, 0.50),
        "electricity_growth": ("utility", "high_good", 0.6, 0.50),
        "utility_payment_on_time": ("utility", "high_good", 0.7, 0.50),
        # logistics
        "shipment_count_monthly": ("logistics", "high_good", 0.5, 0.52),
        "delivery_success_rate": ("logistics", "high_good", 0.6, 0.52),
        "logistics_return_rate": ("logistics", "high_bad", 0.6, 0.52),
        # digital_footprint
        "google_review_count": ("digital_footprint", "high_good", 0.3, 0.30),
        "google_avg_rating": ("digital_footprint", "high_good", 0.3, 0.30),
        "facebook_page_age_months": ("digital_footprint", "high_good", 0.3, 0.30),
        "facebook_engagement_rate": ("digital_footprint", "high_good", 0.2, 0.30),
        # graph
        "buyer_diversity_score": ("graph", "high_good", 0.8, 0.62),
        "supplier_diversity_score": ("graph", "high_good", 0.6, 0.55),
        "pagerank_score": ("graph", "high_good", 0.5, 0.62),
        "network_default_exposure": ("graph", "high_bad", 0.9, 0.62),
        "shared_device_risk_flag": ("graph", "high_bad", 0.7, 0.30),
    }
    return {
        name: {"source": src, "direction": d, "default_weight": w,
               "signal_strength": SIGNAL.get(name, s)}  # SIGNAL = nguồn duy nhất
        for name, (src, d, w, s) in spec.items()
    }


# --------------------------------------------------------------------------- #
# Tiện ích: DSR thô (unweighted) để kiểm tra phổ phân nhóm
# --------------------------------------------------------------------------- #
ALT_FIELDS = [c for cols in SOURCE_GROUPS.values() for c in cols]


def provisional_dsr(df: pd.DataFrame) -> pd.Series:
    """DSR thô = tỷ lệ field alt-data không-missing (chưa trọng số).

    T3 sẽ tính DSR có trọng số chính xác. Đây chỉ để kiểm tra phổ phân nhóm.
    """
    return df[ALT_FIELDS].notna().mean(axis=1)


def dsr_group(dsr: pd.Series) -> pd.Series:
    return pd.cut(dsr, bins=[-0.01, 0.40, 0.70, 1.01],
                 labels=["thin", "semi", "thick"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="ScoreSight semi-synthetic generator")
    ap.add_argument("--n", type=int, default=18000, help="số lượng SME")
    ap.add_argument("--target-default-rate", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1: quy mô MSME (độc lập rủi ro) + coverage (quyết định nhóm DSR & regime)
    size = draw_enterprise_size(rng, args.n)
    coverage = draw_coverage(rng, args.n, size)

    # 2–4: rủi ro tiềm ẩn (tilt theo coverage) + nhãn
    risk = build_latent_risk(rng, args.n, args.target_default_rate, coverage)
    default = (rng.random(args.n) < risk).astype(int)

    # 5: feature alternative data theo quy mô + regime
    df = generate_features(rng, risk, size, coverage)
    df["enterprise_size"] = size

    # 6: mask theo nhóm nguồn — dùng cùng coverage (micro ít data hơn medium)
    df = apply_dsr_masking(df, rng, coverage)

    # Cột định danh + nhãn + ground-truth (đánh dấu _gt_, phải drop trước khi train)
    df.insert(0, "customer_id", [f"SME_{i:06d}" for i in range(args.n)])
    df["default"] = default                  # nhãn target (1 = vỡ nợ)
    df["_gt_latent_risk"] = risk.round(4)     # CHỈ để debug/demo — KHÔNG dùng train
    df["_gt_quality"] = (1 - risk).round(4)

    # DSR thô + phân nhóm (sanity)
    dsr = provisional_dsr(df)
    df["_dsr_raw"] = dsr.round(3)
    df["_dsr_group"] = dsr_group(dsr).astype(str)

    # --- Xuất file --------------------------------------------------------- #
    pq_path = OUT_DIR / "sme_altdata.parquet"
    df.to_parquet(pq_path, index=False)
    df.head(300).to_csv(OUT_DIR / "sme_altdata_sample.csv", index=False)
    with open(OUT_DIR / "feature_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(build_feature_dictionary(), f, ensure_ascii=False, indent=2)

    # --- Summary ----------------------------------------------------------- #
    print("=" * 64)
    print(f"  ScoreSight dataset  —  N = {args.n:,}")
    print("=" * 64)
    print(f"Overall default rate : {default.mean():.1%}")
    print(f"Columns              : {df.shape[1]} (alt fields: {len(ALT_FIELDS)})")
    print(f"Saved                : {pq_path.relative_to(ROOT)}")
    print("-" * 64)
    print("Phân nhóm DSR (thô) + default rate theo nhóm:")
    grp = df.groupby("_dsr_group", observed=True)
    summary = grp.agg(n=("default", "size"),
                      default_rate=("default", "mean"),
                      avg_dsr=("_dsr_raw", "mean"))
    summary["pct"] = (summary["n"] / args.n * 100).round(1)
    for g in ["thin", "semi", "thick"]:
        if g in summary.index:
            r = summary.loc[g]
            print(f"  {g:6s}: n={int(r['n']):>6,} ({r['pct']:>4.1f}%) "
                  f"| default={r['default_rate']:.1%} | avg_dsr={r['avg_dsr']:.2f}")
    print("-" * 64)
    print("Phân khúc MSME (Nghị định 80/2021) + default rate:")
    for sz in ["micro", "small", "medium"]:
        sub = df[df["enterprise_size"] == sz]
        if len(sub):
            print(f"  {sz:6s}: n={len(sub):>6,} ({len(sub)/args.n*100:>4.1f}%) "
                  f"| default={sub['default'].mean():.1%} "
                  f"| avg_dsr={sub['_dsr_raw'].mean():.2f}")
    print("  Cross-tab quy mô × nhóm DSR (% theo hàng):")
    ct = pd.crosstab(df["enterprise_size"], df["_dsr_group"], normalize="index")
    ct = ct.reindex(index=["micro", "small", "medium"],
                    columns=["thin", "semi", "thick"]).fillna(0)
    print("           thin    semi   thick")
    for sz in ct.index:
        print(f"    {sz:6s}  {ct.loc[sz, 'thin']:5.0%}  {ct.loc[sz, 'semi']:6.0%}  {ct.loc[sz, 'thick']:6.0%}")
    print("-" * 64)
    print("Sanity — |corr| của vài feature với default (mong đợi có dấu đúng):")
    checks = ["supplier_payment_regularity", "invoice_revenue_growth",
              "unique_buyer_count", "network_default_exposure",
              "return_rate", "facebook_engagement_rate"]
    for c in checks:
        corr = df[[c, "default"]].dropna().corr().iloc[0, 1]
        print(f"  {c:32s}: {corr:+.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
