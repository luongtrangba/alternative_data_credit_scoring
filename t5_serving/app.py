"""
ScoreSight · T5 — FastAPI Scoring Service
==========================================

Pipeline một request:
  1. Hard rule: shared_device_risk_flag=1 → decline ngay
  2. Build feature row (NaN cho mọi trường thiếu)
  3. Tính DSR có trọng số → thin/semi/thick
  4. Global LightGBM → P(bad) thô
  5. Calibration head theo DSR group → P(bad) đã hiệu chỉnh
  6. PDO scorecard → credit_score [300, 850]
  7. Decision Engine → approve / manual_review / decline
  8. Credit limit = base[size] × DSR_factor
  9. SHAP TreeExplainer → top 5 lý do

Chạy:
  cd "<project_root>"
  uvicorn t5_serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t4_training.score_mapping import (  # noqa: E402
    prob_bad_to_score,
    decision as score_to_decision,
    DECISION_THRESHOLDS,
)

# ── Load artifacts at startup ────────────────────────────────────────────────

_bundle = joblib.load(ROOT / "t4_training/models/scoresight_bundle.joblib")

FEATURES: list[str] = _bundle["features"]
NUMERIC: list[str] = _bundle["numeric"]
CATEGORICAL: list[str] = _bundle["categorical"]
GLOBAL_MODEL = _bundle["global_model"]
CAL_SEG: dict = _bundle["cal_seg"]          # {"thin": CCV, "semi": CCV, "thick": CCV}
LIMIT_FACTOR: dict = _bundle["limit_factor"]  # {"thin": 0.5, ...}
GROUPS: list[str] = _bundle["groups"]

_w_raw: dict = json.loads((ROOT / "configs/weights_refined.json").read_text())
DSR_WEIGHTS = pd.Series(_w_raw, dtype=float)

_cfg: dict = json.loads((ROOT / "configs/dsr_config.json").read_text())
DSR_THR = _cfg["dsr_thresholds"]

# SHAP — khởi tạo 1 lần (TreeExplainer nhanh với LightGBM)
_EXPLAINER = shap.TreeExplainer(GLOBAL_MODEL)

# Hạn mức cơ sở theo quy mô (VND) — điều chỉnh theo sản phẩm thực tế
BASE_LIMIT_VND: dict[str, int] = {
    "micro": 50_000_000,
    "small": 200_000_000,
    "medium": 1_000_000_000,
}


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    customer_id: str = Field(..., description="Mã khách hàng/MSME")
    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Các trường alt-data. Thiếu trường nào → mặc định NaN (ảnh hưởng DSR).",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "MSME-00001",
            "fields": {
                "enterprise_size": "small",
                "industry": "retail",
                "region": "HCM",
                "num_employees": 12,
                "business_age_months": 36,
                "shopee_gmv_3m": 85000000,
                "gmv_growth_12m": 0.18,
                "order_count_monthly": 420,
                "return_rate": 0.04,
                "seller_rating": 4.6,
                "invoice_revenue_12m": 1200000000,
                "invoice_revenue_growth": 0.22,
                "supplier_payment_regularity": 0.85,
                "payroll_regularity": 0.90,
                "momo_net_cashflow_avg": 15000000,
                "utility_payment_on_time": 1.0,
                "pagerank_score": 0.45,
                "network_default_exposure": 0.05,
                "shared_device_risk_flag": 0,
            },
        }
    }}


class Reason(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increase_risk" | "decrease_risk"
    description: str


class ScoreResponse(BaseModel):
    customer_id: str
    credit_score: int             # 300–850 (cao = ít rủi ro)
    p_bad: float                  # P(default) đã calibrate
    dsr_value: float              # 0.0–1.0
    dsr_group: str                # thin / semi / thick
    enterprise_size: str
    decision: str                 # approve / manual_review / decline
    credit_limit_vnd: int
    top_reasons: List[Reason]
    warnings: List[str]


# ── Pipeline helpers ─────────────────────────────────────────────────────────

def _as_lgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def _build_row(fields: dict) -> pd.DataFrame:
    """Build 1-row DataFrame với đúng 35 features. Thiếu → NaN."""
    row: dict[str, Any] = {f: np.nan for f in FEATURES}
    for k, v in fields.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row])


def _compute_dsr(row: pd.DataFrame) -> float:
    """DSR_wq = Σ(w_i · valid_i) / Σ(w_i) — chỉ trên các trường trong weights."""
    shared = [f for f in DSR_WEIGHTS.index if f in row.columns]
    valid = row[shared].notna().astype(float).iloc[0]
    w = DSR_WEIGHTS[shared]
    return float((valid * w).sum() / w.sum())


def _assign_group(dsr: float) -> str:
    if dsr <= DSR_THR["thin_max"]:
        return "thin"
    if dsr <= DSR_THR["semi_max"]:
        return "semi"
    return "thick"


def _shap_reasons(X_lgb: pd.DataFrame, n: int = 5) -> list[Reason]:
    sv = _EXPLAINER.shap_values(X_lgb)
    # Binary LightGBM: sv là list[2 arrays] hoặc single array
    if isinstance(sv, list):
        sv = sv[1]  # class 1 = bad
    vals = sv[0]  # row 0
    top_idx = np.argsort(np.abs(vals))[::-1][:n]
    feature_labels = {
        "invoice_revenue_growth": "Tăng trưởng doanh thu hóa đơn",
        "supplier_payment_regularity": "Độ đều thanh toán nhà cung cấp",
        "gmv_growth_12m": "Tăng trưởng GMV 12 tháng",
        "unique_buyer_count": "Số khách mua độc nhất",
        "payroll_regularity": "Độ đều trả lương",
        "vat_filing_on_time_ratio": "Tỷ lệ nộp VAT đúng hạn",
        "buyer_diversity_score": "Điểm đa dạng khách mua",
        "return_rate": "Tỷ lệ hoàn trả",
        "network_default_exposure": "Phơi nhiễm nợ xấu mạng lưới",
        "invoice_revenue_12m": "Doanh thu hóa đơn 12 tháng",
        "momo_net_cashflow_avg": "Dòng tiền ròng MoMo TB",
        "pos_volume_6m": "Khối lượng POS 6 tháng",
        "shopee_gmv_3m": "GMV Shopee 3 tháng",
        "pagerank_score": "Điểm PageRank mạng lưới",
        "seller_rating": "Đánh giá người bán",
        "delivery_success_rate": "Tỷ lệ giao hàng thành công",
        "electricity_growth": "Tăng trưởng điện tiêu thụ",
        "utility_payment_on_time": "Thanh toán tiện ích đúng hạn",
        "shared_device_risk_flag": "Cờ thiết bị dùng chung (gian lận)",
        "facebook_engagement_rate": "Tỷ lệ tương tác Facebook",
        "google_review_count": "Số đánh giá Google",
        "business_age_months": "Tuổi doanh nghiệp (tháng)",
        "num_employees": "Số nhân viên",
    }
    reasons = []
    for i in top_idx:
        v = float(vals[i])
        fname = FEATURES[i]
        reasons.append(Reason(
            feature=fname,
            shap_value=round(v, 4),
            direction="increase_risk" if v > 0 else "decrease_risk",
            description=feature_labels.get(fname, fname),
        ))
    return reasons


def _credit_limit(enterprise_size: str, dsr_group: str, dec: str) -> int:
    if dec == "decline":
        return 0
    base = BASE_LIMIT_VND.get(enterprise_size, BASE_LIMIT_VND["micro"])
    factor = LIMIT_FACTOR.get(dsr_group, 0.5)
    return int(base * factor)


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ScoreSight",
    description=(
        "Hệ thống chấm điểm tín dụng MSME dựa 100% dữ liệu phi truyền thống "
        "(TMĐT, ví điện tử, hóa đơn điện tử, tiện ích, logistics, graph). "
        "Không dùng CIC/bureau."
    ),
    version="1.0.0",
)


@app.get("/health", summary="Kiểm tra service")
def health():
    return {
        "status": "ok",
        "model": "ScoreSight v1.0 — Hybrid LightGBM + DSR Calibration",
        "features": len(FEATURES),
        "dsr_groups": GROUPS,
    }


@app.get("/model-info", summary="Thông tin model và ngưỡng")
def model_info():
    return {
        "features": FEATURES,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "dsr_weights_top10": dict(DSR_WEIGHTS.nlargest(10)),
        "dsr_thresholds": DSR_THR,
        "limit_factor": LIMIT_FACTOR,
        "decision_thresholds": DECISION_THRESHOLDS,
        "base_limit_vnd": BASE_LIMIT_VND,
    }


@app.post("/score", response_model=ScoreResponse, summary="Chấm điểm tín dụng MSME")
def score(req: ScoreRequest) -> ScoreResponse:
    warnings: list[str] = []
    fields = req.fields

    # ── Hard rule: shared_device_risk_flag ──────────────────────────────────
    if fields.get("shared_device_risk_flag") == 1:
        return ScoreResponse(
            customer_id=req.customer_id,
            credit_score=300,
            p_bad=1.0,
            dsr_value=0.0,
            dsr_group="thin",
            enterprise_size=str(fields.get("enterprise_size", "unknown")),
            decision="decline",
            credit_limit_vnd=0,
            top_reasons=[Reason(
                feature="shared_device_risk_flag",
                shap_value=99.0,
                direction="increase_risk",
                description="Thiết bị dùng chung — tín hiệu gian lận (hard decline)",
            )],
            warnings=["HARD DECLINE: shared_device_risk_flag = 1"],
        )

    # ── Build feature row ───────────────────────────────────────────────────
    row = _build_row(fields)

    # Enterprise size
    enterprise_size = str(fields.get("enterprise_size", "micro"))
    if enterprise_size not in BASE_LIMIT_VND:
        warnings.append(f"enterprise_size='{enterprise_size}' không hợp lệ → mặc định 'micro'")
        enterprise_size = "micro"

    # ── DSR ─────────────────────────────────────────────────────────────────
    dsr_value = _compute_dsr(row)
    dsr_group = _assign_group(dsr_value)

    # Cảnh báo khi quá ít dữ liệu
    n_available = int(row[FEATURES].notna().sum(axis=1).iloc[0])
    if n_available < 8:
        warnings.append(
            f"Chỉ có {n_available}/{len(FEATURES)} trường — điểm kém tin cậy, cần xem xét thủ công"
        )

    # ── Model inference ─────────────────────────────────────────────────────
    X_lgb = _as_lgb(row[FEATURES])

    if dsr_group in CAL_SEG:
        p_bad = float(CAL_SEG[dsr_group].predict_proba(X_lgb)[0, 1])
    else:
        p_bad = float(GLOBAL_MODEL.predict_proba(X_lgb)[0, 1])
        warnings.append(f"Không có calibration head cho '{dsr_group}' — dùng global model")

    # ── Score & decision ────────────────────────────────────────────────────
    credit_score = int(prob_bad_to_score(p_bad))
    dec = score_to_decision(credit_score)
    credit_limit = _credit_limit(enterprise_size, dsr_group, dec)

    # ── SHAP top reasons ────────────────────────────────────────────────────
    try:
        top_reasons = _shap_reasons(X_lgb)
    except Exception as exc:
        top_reasons = []
        warnings.append(f"SHAP không tính được: {exc}")

    return ScoreResponse(
        customer_id=req.customer_id,
        credit_score=credit_score,
        p_bad=round(p_bad, 4),
        dsr_value=round(dsr_value, 4),
        dsr_group=dsr_group,
        enterprise_size=enterprise_size,
        decision=dec,
        credit_limit_vnd=credit_limit,
        top_reasons=top_reasons,
        warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("t5_serving.app:app", host="0.0.0.0", port=8000, reload=False)
