"""
ScoreSight · FastAPI Scoring Service

Pipeline per request:
  1. Hard rule: shared_device_risk_flag=1 -> immediate decline
  2. Build feature row (NaN for missing fields)
  3. Compute weighted DSR -> thin/semi/thick
  4. Global LightGBM -> raw P(bad)
  5. Calibration head by DSR group -> calibrated P(bad)
  6. PDO scorecard -> credit_score [300, 850]
  7. Decision Engine -> approve / manual_review / decline
  8. Credit limit = base[size] * DSR_factor
  9. SHAP TreeExplainer -> top 5 reasons

Run:
  uvicorn scoresight.serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

from scoresight.training.score_mapping import (
    DECISION_THRESHOLDS,
    prob_bad_to_score,
)
from scoresight.training.score_mapping import (
    decision as score_to_decision,
)

ROOT = Path(__file__).resolve().parents[3]

_bundle = joblib.load(ROOT / "t4_training/models/scoresight_bundle.joblib")

FEATURES: list[str] = _bundle["features"]
NUMERIC: list[str] = _bundle["numeric"]
CATEGORICAL: list[str] = _bundle["categorical"]
GLOBAL_MODEL = _bundle["global_model"]
CAL_SEG: dict = _bundle["cal_seg"]
LIMIT_FACTOR: dict = _bundle["limit_factor"]
GROUPS: list[str] = _bundle["groups"]

_w_raw: dict = json.loads((ROOT / "configs/weights_refined.json").read_text())
DSR_WEIGHTS = pd.Series(_w_raw, dtype=float)

_cfg: dict = json.loads((ROOT / "configs/dsr_config.json").read_text())
DSR_THR = _cfg["dsr_thresholds"]

_EXPLAINER = shap.TreeExplainer(GLOBAL_MODEL)

BASE_LIMIT_VND: dict[str, int] = {
    "micro": 50_000_000,
    "small": 200_000_000,
    "medium": 1_000_000_000,
}


class ScoreRequest(BaseModel):
    customer_id: str = Field(..., description="Customer/MSME identifier")
    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Alt-data fields. Missing fields default to NaN (affects DSR).",
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
                "supplier_payment_regularity": 0.85,
                "shared_device_risk_flag": 0,
            },
        }
    }}


class Reason(BaseModel):
    feature: str
    shap_value: float
    direction: str
    description: str


class ScoreResponse(BaseModel):
    customer_id: str
    credit_score: int
    p_bad: float
    dsr_value: float
    dsr_group: str
    enterprise_size: str
    decision: str
    credit_limit_vnd: int
    top_reasons: List[Reason]
    warnings: List[str]


FEATURE_LABELS = {
    "invoice_revenue_growth": "E-invoice revenue growth",
    "supplier_payment_regularity": "Supplier payment regularity",
    "gmv_growth_12m": "GMV growth 12 months",
    "unique_buyer_count": "Unique buyer count",
    "payroll_regularity": "Payroll regularity",
    "vat_filing_on_time_ratio": "VAT filing on-time ratio",
    "buyer_diversity_score": "Buyer diversity score",
    "return_rate": "Return rate",
    "network_default_exposure": "Network default exposure",
    "invoice_revenue_12m": "E-invoice revenue 12 months",
    "momo_net_cashflow_avg": "Average MoMo net cashflow",
    "pos_volume_6m": "POS volume 6 months",
    "shopee_gmv_3m": "Shopee GMV 3 months",
    "pagerank_score": "Network PageRank score",
    "seller_rating": "Seller rating",
    "delivery_success_rate": "Delivery success rate",
    "electricity_growth": "Electricity consumption growth",
    "utility_payment_on_time": "Utility payment on-time ratio",
    "shared_device_risk_flag": "Shared-device fraud flag",
    "facebook_engagement_rate": "Facebook engagement rate",
    "google_review_count": "Google review count",
    "business_age_months": "Business age (months)",
    "num_employees": "Number of employees",
}


def _as_lgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def _build_row(fields: dict) -> pd.DataFrame:
    row: dict[str, Any] = {f: np.nan for f in FEATURES}
    for k, v in fields.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row])


def _compute_dsr(row: pd.DataFrame) -> float:
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
    if isinstance(sv, list):
        sv = sv[1]
    vals = sv[0]
    top_idx = np.argsort(np.abs(vals))[::-1][:n]
    reasons = []
    for i in top_idx:
        v = float(vals[i])
        fname = FEATURES[i]
        reasons.append(Reason(
            feature=fname,
            shap_value=round(v, 4),
            direction="increase_risk" if v > 0 else "decrease_risk",
            description=FEATURE_LABELS.get(fname, fname),
        ))
    return reasons


def _credit_limit(enterprise_size: str, dsr_group: str, dec: str) -> int:
    if dec == "decline":
        return 0
    base = BASE_LIMIT_VND.get(enterprise_size, BASE_LIMIT_VND["micro"])
    factor = LIMIT_FACTOR.get(dsr_group, 0.5)
    return int(base * factor)


app = FastAPI(
    title="ScoreSight",
    description=(
        "MSME credit scoring system based on 100% alternative data "
        "(e-commerce, digital wallets, e-invoices, utilities, logistics, graph). "
        "No CIC/bureau required."
    ),
    version="1.0.0",
)


@app.get("/health", summary="Health check")
def health() -> dict:
    return {
        "status": "ok",
        "model": "ScoreSight v1.0 — Hybrid LightGBM + DSR Calibration",
        "features": len(FEATURES),
        "dsr_groups": GROUPS,
    }


@app.get("/model-info", summary="Model info and thresholds")
def model_info() -> dict:
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


@app.post("/score", response_model=ScoreResponse, summary="Score an MSME borrower")
def score(req: ScoreRequest) -> ScoreResponse:
    warnings: list[str] = []
    fields = req.fields

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
                description="Shared device — fraud signal (hard decline)",
            )],
            warnings=["HARD DECLINE: shared_device_risk_flag = 1"],
        )

    row = _build_row(fields)

    enterprise_size = str(fields.get("enterprise_size", "micro"))
    if enterprise_size not in BASE_LIMIT_VND:
        warnings.append(
            f"enterprise_size='{enterprise_size}' invalid, defaulting to 'micro'"
        )
        enterprise_size = "micro"

    dsr_value = _compute_dsr(row)
    dsr_group = _assign_group(dsr_value)

    n_available = int(row[FEATURES].notna().sum(axis=1).iloc[0])
    if n_available < 8:
        warnings.append(
            f"Only {n_available}/{len(FEATURES)} fields available — "
            "low confidence, manual review recommended"
        )

    X_lgb = _as_lgb(row[FEATURES])

    if dsr_group in CAL_SEG:
        p_bad = float(CAL_SEG[dsr_group].predict_proba(X_lgb)[0, 1])
    else:
        p_bad = float(GLOBAL_MODEL.predict_proba(X_lgb)[0, 1])
        warnings.append(f"No calibration head for '{dsr_group}', using global model")

    credit_score = int(prob_bad_to_score(p_bad))
    dec = score_to_decision(credit_score)
    credit_limit = _credit_limit(enterprise_size, dsr_group, dec)

    try:
        top_reasons = _shap_reasons(X_lgb)
    except Exception as exc:
        top_reasons = []
        warnings.append(f"SHAP computation failed: {exc}")

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
    uvicorn.run("scoresight.serving.app:app", host="0.0.0.0", port=8000, reload=False)
