"""
ScoreSight · Demo Client — gọi API /score với 3 profile MSME khác nhau.

Chạy (sau khi server đã lên):
    python3 t5_serving/demo_client.py
"""

from __future__ import annotations
import json, urllib.request, urllib.error

BASE = "http://localhost:8000"


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def print_result(label: str, r: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Score      : {r['credit_score']}  ({r['decision'].upper()})")
    print(f"  P(bad)     : {r['p_bad']:.2%}")
    print(f"  DSR        : {r['dsr_value']:.2f}  [{r['dsr_group']}]")
    print(f"  Hạn mức    : {r['credit_limit_vnd']:,.0f} VND")
    print(f"  Lý do chính:")
    for i, reason in enumerate(r["top_reasons"][:3], 1):
        sign = "↑ rủi ro" if reason["direction"] == "increase_risk" else "↓ rủi ro"
        print(f"    {i}. {reason['description']} ({sign}, SHAP={reason['shap_value']:+.3f})")
    if r["warnings"]:
        for w in r["warnings"]:
            print(f"  ⚠ {w}")


# --- Profile 1: MSME tốt — thick-file, small ---
p1 = {
    "customer_id": "MSME-GOOD-001",
    "fields": {
        "enterprise_size": "small",
        "industry": "retail",
        "region": "HCM",
        "num_employees": 15,
        "business_age_months": 48,
        "shopee_gmv_3m": 120_000_000,
        "gmv_growth_12m": 0.30,
        "order_count_monthly": 650,
        "return_rate": 0.02,
        "seller_rating": 4.8,
        "momo_net_cashflow_avg": 20_000_000,
        "pos_volume_6m": 300_000_000,
        "supplier_payment_regularity": 0.92,
        "payroll_regularity": 0.95,
        "active_days_per_month": 25,
        "invoice_revenue_12m": 1_500_000_000,
        "invoice_revenue_growth": 0.28,
        "unique_buyer_count": 820,
        "vat_filing_on_time_ratio": 1.0,
        "invoice_cancel_rate": 0.01,
        "electricity_consumption_avg": 2800,
        "electricity_growth": 0.12,
        "utility_payment_on_time": 1.0,
        "shipment_count_monthly": 600,
        "delivery_success_rate": 0.97,
        "logistics_return_rate": 0.03,
        "google_review_count": 180,
        "google_avg_rating": 4.5,
        "facebook_page_age_months": 36,
        "facebook_engagement_rate": 0.05,
        "buyer_diversity_score": 0.75,
        "supplier_diversity_score": 0.65,
        "pagerank_score": 0.55,
        "network_default_exposure": 0.03,
        "shared_device_risk_flag": 0,
    },
}

# --- Profile 2: MSME trung bình — semi-file, micro ---
p2 = {
    "customer_id": "MSME-MID-002",
    "fields": {
        "enterprise_size": "micro",
        "industry": "food",
        "region": "HAN",
        "num_employees": 4,
        "business_age_months": 18,
        "shopee_gmv_3m": 22_000_000,
        "gmv_growth_12m": 0.05,
        "return_rate": 0.08,
        "seller_rating": 4.1,
        "invoice_revenue_12m": 180_000_000,
        "invoice_revenue_growth": 0.03,
        "supplier_payment_regularity": 0.60,
        "payroll_regularity": 0.70,
        "utility_payment_on_time": 0.80,
        "pagerank_score": 0.20,
        "network_default_exposure": 0.12,
        "shared_device_risk_flag": 0,
    },
}

# --- Profile 3: Hard decline — shared_device_risk_flag ---
p3 = {
    "customer_id": "MSME-FRAUD-003",
    "fields": {
        "enterprise_size": "micro",
        "industry": "retail",
        "shared_device_risk_flag": 1,
        "invoice_revenue_12m": 500_000_000,
        "seller_rating": 4.7,
    },
}

if __name__ == "__main__":
    # Health check
    with urllib.request.urlopen(BASE + "/health") as r:
        h = json.loads(r.read())
    print(f"Server: {h['model']} | {h['features']} features")

    print_result("Profile 1 — MSME tốt (thick-file, small)", post("/score", p1))
    print_result("Profile 2 — MSME trung bình (thin-file, micro)", post("/score", p2))
    print_result("Profile 3 — HARD DECLINE (fraud flag)", post("/score", p3))
    print()
