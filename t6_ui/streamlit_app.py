"""
ScoreSight · T6 UI Demo
=======================
Streamlit UI for the hackathon demo. It can run in two modes:
1) Local mode: imports t5_serving.app and calls the scoring pipeline directly.
2) API mode: calls FastAPI POST /score when SCORE_API_URL is set.

Run from project root:
  streamlit run t6_ui/streamlit_app.py

Optional API mode:
  export SCORE_API_URL=http://localhost:8000/score
  uvicorn t5_serving.app:app --host 0.0.0.0 --port 8000 --reload
  streamlit run t6_ui/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

API_URL = os.getenv("SCORE_API_URL", "").strip()
PAGE_TITLE = "ScoreSight MSME Credit Scoring"

SOURCE_LABELS = {
    "identity": "Business Profile",
    "ecommerce": "E-commerce",
    "payment": "Digital Payment",
    "einvoice": "E-invoice / Tax",
    "utility": "Utility",
    "logistics": "Logistics",
    "digital_footprint": "Digital Footprint",
    "graph": "Network Graph",
}

FEATURE_LABELS = {
    "business_age_months": "Business age (months)",
    "num_employees": "Number of employees",
    "industry": "Industry",
    "region": "Region",
    "enterprise_size": "Enterprise size",
    "shopee_gmv_3m": "Shopee GMV last 3 months",
    "gmv_growth_12m": "GMV growth 12 months",
    "order_count_monthly": "Monthly order count",
    "return_rate": "Return rate",
    "seller_rating": "Seller rating",
    "momo_net_cashflow_avg": "Average MoMo net cashflow",
    "pos_volume_6m": "POS volume last 6 months",
    "supplier_payment_regularity": "Supplier payment regularity",
    "payroll_regularity": "Payroll regularity",
    "active_days_per_month": "Active days per month",
    "invoice_revenue_12m": "E-invoice revenue 12 months",
    "invoice_revenue_growth": "E-invoice revenue growth",
    "unique_buyer_count": "Unique buyer count",
    "vat_filing_on_time_ratio": "VAT filing on-time ratio",
    "invoice_cancel_rate": "Invoice cancellation rate",
    "electricity_consumption_avg": "Average electricity consumption",
    "electricity_growth": "Electricity consumption growth",
    "utility_payment_on_time": "Utility payment on-time ratio",
    "shipment_count_monthly": "Monthly shipment count",
    "delivery_success_rate": "Delivery success rate",
    "logistics_return_rate": "Logistics return rate",
    "google_review_count": "Google review count",
    "google_avg_rating": "Google average rating",
    "facebook_page_age_months": "Facebook page age (months)",
    "facebook_engagement_rate": "Facebook engagement rate",
    "buyer_diversity_score": "Buyer diversity score",
    "supplier_diversity_score": "Supplier diversity score",
    "pagerank_score": "Network PageRank score",
    "network_default_exposure": "Network default exposure",
    "shared_device_risk_flag": "Shared-device fraud flag",
}

SOURCE_HINTS = {
    "Business Profile": "Basic MSME attributes for segmentation and eligibility checks.",
    "E-commerce": "Marketplace sales, growth and return behavior.",
    "Digital Payment": "Cashflow discipline from wallet, POS, supplier and payroll patterns.",
    "E-invoice / Tax": "Verified revenue and filing behavior.",
    "Utility": "Operating proxy from electricity and utility payment behavior.",
    "Logistics": "Fulfillment volume and delivery quality.",
    "Digital Footprint": "Online reputation and engagement signals.",
    "Network Graph": "Buyer/supplier diversity and default exposure from business network.",
}

DEFAULT_MANUAL_FIELDS: Dict[str, Any] = {
    "enterprise_size": "small",
    "industry": "retail",
    "region": "HCMC",
    "num_employees": 12,
    "business_age_months": 36,
    "shopee_gmv_3m": 85_000_000,
    "gmv_growth_12m": 0.18,
    "order_count_monthly": 420,
    "return_rate": 0.04,
    "seller_rating": 4.6,
    "invoice_revenue_12m": 1_200_000_000,
    "invoice_revenue_growth": 0.22,
    "unique_buyer_count": 180,
    "vat_filing_on_time_ratio": 0.94,
    "invoice_cancel_rate": 0.03,
    "supplier_payment_regularity": 0.85,
    "payroll_regularity": 0.90,
    "momo_net_cashflow_avg": 15_000_000,
    "pos_volume_6m": 520_000_000,
    "active_days_per_month": 24,
    "utility_payment_on_time": 1.0,
    "electricity_consumption_avg": 2300,
    "electricity_growth": 0.10,
    "shipment_count_monthly": 360,
    "delivery_success_rate": 0.96,
    "logistics_return_rate": 0.05,
    "google_review_count": 120,
    "google_avg_rating": 4.5,
    "facebook_page_age_months": 30,
    "facebook_engagement_rate": 0.08,
    "buyer_diversity_score": 0.72,
    "supplier_diversity_score": 0.64,
    "pagerank_score": 0.45,
    "network_default_exposure": 0.05,
    "shared_device_risk_flag": 0,
}


def money_vnd(value: float | int | None) -> str:
    if value is None:
        return "—"
    if abs(float(value)) >= 1_000_000_000:
        return f"{float(value) / 1_000_000_000:.1f} tỷ VND"
    return f"{int(value):,} VND".replace(",", ".")


def pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@st.cache_data
def load_feature_dictionary() -> dict:
    return json.loads((ROOT / "data/feature_dictionary.json").read_text())


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/sme_altdata_sample.csv")


def call_score(customer_id: str, fields: dict) -> dict:
    payload = {"customer_id": customer_id, "fields": fields}
    if API_URL:
        import requests
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    try:
        from scoresight.serving.app import ScoreRequest, score
    except ImportError:
        from t5_serving.app import ScoreRequest, score
    return score(ScoreRequest(**payload)).model_dump()


def feature_groups(feature_dictionary: dict) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for feature, meta in feature_dictionary.items():
        source = SOURCE_LABELS.get(meta.get("source", "identity"), meta.get("source", "Other"))
        grouped.setdefault(source, []).append(feature)
    return grouped


def source_coverage(fields: dict, grouped: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for source, features in grouped.items():
        available = sum(fields.get(f) is not None for f in features)
        total = len(features)
        rows.append({"source": source, "coverage": available / total if total else 0, "available": available, "total": total})
    return pd.DataFrame(rows).sort_values("coverage", ascending=False)


def decision_badge(decision: str) -> str:
    label = decision.replace("_", " ").title()
    if decision == "approve":
        return f"<span class='badge badge-approve'>{label}</span>"
    if decision == "manual_review":
        return f"<span class='badge badge-review'>{label}</span>"
    return f"<span class='badge badge-decline'>{label}</span>"


def lending_flow(decision: str) -> tuple[str, str, str]:
    if decision == "approve":
        return "Green flow", "Auto-precheck / straight-through recommendation", "Push to LOS/RLOS with suggested limit and minimum docs."
    if decision == "manual_review":
        return "Yellow flow", "Human-in-the-loop review", "Open CMS case for RM/thẩm định with criteria-level evidence."
    return "Red flow", "Reject or senior approval required", "Route to CMS with risk flags and audit trail."


def business_impact(result: dict, fields: dict) -> dict:
    decision = result.get("decision")
    dsr = float(result.get("dsr_value") or 0)
    base_tat_hours = 48
    if decision == "approve" and dsr >= 0.75:
        tat_hours = 4
        approval_lift = "+18-25% eligible thin-file MSMEs"
        npl_guardrail = "No auto-approval if PD/risk rules breach threshold"
    elif decision == "manual_review":
        tat_hours = 18
        approval_lift = "+8-12% via better prioritization"
        npl_guardrail = "Manual review required before disbursement"
    else:
        tat_hours = 8
        approval_lift = "Protect portfolio from risky applications"
        npl_guardrail = "Hard stop / enhanced review prevents NPL leakage"
    docs = "Reduced docs" if dsr >= 0.75 else "Additional docs needed"
    return {
        "tat_saved": f"~{max(base_tat_hours - tat_hours, 0)}h saved",
        "tat_new": f"~{tat_hours}h target TAT",
        "approval_lift": approval_lift,
        "npl_guardrail": npl_guardrail,
        "docs": docs,
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.4rem; max-width: 1320px; }
        .hero {
            padding: 28px 32px; border-radius: 30px;
            background: linear-gradient(135deg, #07111f 0%, #0e2c44 54%, #f4f8fb 54%, #ffffff 100%);
            border: 1px solid rgba(15, 23, 42, .08);
            box-shadow: 0 18px 45px rgba(15, 23, 42, .08);
            margin-bottom: 20px;
        }
        .hero h1 { color: white; font-size: 2.35rem; margin: 0 0 8px 0; line-height: 1.05; }
        .hero p { color: rgba(255,255,255,.82); max-width: 680px; margin: 0; font-size: 1rem; }
        .hero .pill { display:inline-block; margin-top:14px; padding:7px 12px; border-radius:999px; background:rgba(255,255,255,.14); color:white; font-size:.85rem; }
        .metric-card, .mini-card {
            border: 1px solid rgba(15,23,42,.08); border-radius: 22px; padding: 18px;
            background: white; box-shadow: 0 10px 30px rgba(15,23,42,.05); min-height: 128px;
        }
        .mini-card { min-height: 96px; }
        .metric-card .label, .mini-card .label { color:#64748b; font-size:.78rem; text-transform: uppercase; letter-spacing:.05em; }
        .metric-card .value { color:#0f172a; font-size:1.65rem; font-weight:850; margin-top:4px; }
        .mini-card .value { color:#0f172a; font-size:1.15rem; font-weight:800; margin-top:4px; }
        .metric-card .hint, .mini-card .hint { color:#64748b; font-size:.86rem; margin-top:6px; }
        .badge { padding: 6px 11px; border-radius: 999px; font-weight: 800; font-size: .82rem; }
        .badge-approve { background:#e8f7ef; color:#0f7a43; }
        .badge-review { background:#fff5dd; color:#9a5b00; }
        .badge-decline { background:#ffecec; color:#b42318; }
        .reason-card { border-left: 4px solid #0f172a; padding: 12px 14px; background:#f8fafc; border-radius: 12px; margin-bottom: 10px; }
        .reason-card b { color:#0f172a; }
        .reason-card span { color:#64748b; font-size:.88rem; }
        .warning-card { padding: 12px 14px; border-radius: 14px; background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; }
        .flow-card { padding: 16px; border-radius: 18px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:10px; }
        .flow-card b { font-size: 1.05rem; color:#0f172a; }
        .muted { color:#64748b; font-size:.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>ScoreSight</h1>
            <p>Bank-ready MSME credit scoring UI: role-based views, explainable risk alerts, LOS/CMS handoff, RM portal support and admin/audit governance.</p>
            <span class="pill">Hackathon Demo · FastAPI + LightGBM + DSR Calibration + SHAP + Rule Governance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_card(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def manual_fields() -> dict:
    fields = dict(DEFAULT_MANUAL_FIELDS)
    with st.sidebar.expander("Business profile", expanded=True):
        fields["enterprise_size"] = st.selectbox("Enterprise size", ["micro", "small", "medium"], index=1)
        fields["industry"] = st.selectbox("Industry", ["retail", "services", "manufacturing", "food", "logistics"], index=0)
        fields["region"] = st.selectbox("Region", ["HCMC", "HaNoi", "DaNang", "Mekong", "Other"], index=0)
        fields["num_employees"] = st.number_input("Employees", 1, 500, 12)
        fields["business_age_months"] = st.number_input("Business age months", 1, 240, 36)

    with st.sidebar.expander("Data source availability", expanded=True):
        use_ecommerce = st.checkbox("E-commerce", value=True)
        use_payment = st.checkbox("Digital payment", value=True)
        use_einvoice = st.checkbox("E-invoice / tax", value=True)
        use_utility = st.checkbox("Utility", value=True)
        use_logistics = st.checkbox("Logistics", value=True)
        use_digital = st.checkbox("Digital footprint", value=True)
        use_graph = st.checkbox("Network graph", value=True)
        fields["shared_device_risk_flag"] = int(st.checkbox("Shared-device fraud flag", value=False))

    source_features = {
        "ecommerce": ["shopee_gmv_3m", "gmv_growth_12m", "order_count_monthly", "return_rate", "seller_rating"],
        "payment": ["momo_net_cashflow_avg", "pos_volume_6m", "supplier_payment_regularity", "payroll_regularity", "active_days_per_month"],
        "einvoice": ["invoice_revenue_12m", "invoice_revenue_growth", "unique_buyer_count", "vat_filing_on_time_ratio", "invoice_cancel_rate"],
        "utility": ["electricity_consumption_avg", "electricity_growth", "utility_payment_on_time"],
        "logistics": ["shipment_count_monthly", "delivery_success_rate", "logistics_return_rate"],
        "digital": ["google_review_count", "google_avg_rating", "facebook_page_age_months", "facebook_engagement_rate"],
        "graph": ["buyer_diversity_score", "supplier_diversity_score", "pagerank_score", "network_default_exposure"],
    }
    toggles = {
        "ecommerce": use_ecommerce,
        "payment": use_payment,
        "einvoice": use_einvoice,
        "utility": use_utility,
        "logistics": use_logistics,
        "digital": use_digital,
        "graph": use_graph,
    }
    for source, features in source_features.items():
        if not toggles[source]:
            for feature in features:
                fields.pop(feature, None)
    return {k: v for k, v in fields.items() if v is not None}


def sample_selector(df: pd.DataFrame) -> tuple[str, dict]:
    st.sidebar.header("Demo input")
    st.sidebar.caption("Select a sample borrower or simulate RM entry by turning data sources on/off.")
    mode = st.sidebar.radio("Input mode", ["Sample MSME", "RM quick form"], horizontal=False)
    if mode == "Sample MSME":
        group_filter = st.sidebar.selectbox("DSR profile", ["All", "thin", "semi", "thick"])
        view = df if group_filter == "All" else df[df["_dsr_group"] == group_filter]
        if view.empty:
            view = df
        labels = [f"{row.customer_id} · {row.enterprise_size} · {row._dsr_group}" for row in view.itertuples()]
        selected = st.sidebar.selectbox("Choose borrower", labels)
        row = view.iloc[labels.index(selected)]
        customer_id = str(row["customer_id"])
        drop_cols = {"customer_id", "default", "_gt_latent_risk", "_gt_quality", "_dsr_raw", "_dsr_group"}
        fields = {col: clean_value(row[col]) for col in view.columns if col not in drop_cols}
        return customer_id, {k: v for k, v in fields.items() if v is not None}
    st.sidebar.caption("RM quick form: enter MST/basic profile, then connect available data sources.")
    return "MST-DEMO-001", manual_fields()


def render_score_result(result: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Credit score", str(result.get("credit_score")), "300-850, higher is safer")
    with c2:
        render_metric_card("Decision", decision_badge(str(result.get("decision"))), "Green / Yellow / Red flow")
    with c3:
        render_metric_card("P(default)", pct(result.get("p_bad"), 2), "Calibrated by DSR segment")
    with c4:
        render_metric_card("DSR", pct(result.get("dsr_value"), 1), f"Profile: {result.get('dsr_group')}")
    with c5:
        render_metric_card("Suggested limit", money_vnd(result.get("credit_limit_vnd")), "After DSR tiering")

    warnings = result.get("warnings", [])
    if warnings:
        st.markdown("### Risk warnings")
        for warning in warnings:
            st.markdown(f"<div class='warning-card'>{warning}</div>", unsafe_allow_html=True)


def render_customer_view(result: dict) -> None:
    st.subheader("EU 1 · Customer digital lending view")
    decision = result.get("decision")
    flow, title, next_step = lending_flow(str(decision))
    public_status = {
        "approve": "Pre-qualified",
        "manual_review": "Additional review needed",
        "decline": "Not eligible at this time",
    }.get(str(decision), "Under review")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Application status", public_status, "No model logic exposed to customer")
    with c2:
        customer_limit = money_vnd(result.get("credit_limit_vnd")) if decision != "decline" else "—"
        render_metric_card("Indicative limit", customer_limit, "Subject to bank approval")
    with c3:
        render_metric_card("Next step", flow, title)
    st.markdown(
        f"""
        <div class="flow-card">
            <b>Customer message</b><br/>
            <span class="muted">{next_step}. Customer only sees a simple status, required documents and consent/TnC flow. Score, PD, SHAP and rules are hidden.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_internal_view(result: dict, fields: dict, grouped: dict[str, list[str]]) -> None:
    st.subheader("EU 2 · RM / Thẩm định / Chuyên gia phê duyệt")
    render_score_result(result)
    impact = business_impact(result, fields)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_mini_card("SLA/TAT impact", impact["tat_saved"], impact["tat_new"])
    with c2:
        render_mini_card("Approval impact", impact["approval_lift"], "Prioritize qualified MSMEs")
    with c3:
        render_mini_card("NPL guardrail", "Active", impact["npl_guardrail"])
    with c4:
        render_mini_card("Document strategy", impact["docs"], "Based on DSR confidence")

    flow, title, next_step = lending_flow(str(result.get("decision")))
    st.markdown("### Proposed credit workflow")
    st.markdown(
        f"""
        <div class="flow-card">
            <b>{flow}: {title}</b><br/>
            <span class="muted">{next_step}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Criteria-level explanation")
    st.caption("Internal users see criteria and evidence, but not raw rule-engine logic or model coefficients.")
    reasons = result.get("top_reasons", [])
    for reason in reasons:
        direction = reason.get("direction", "")
        arrow = "increases risk" if direction == "increase_risk" else "decreases risk"
        st.markdown(
            f"""
            <div class="reason-card">
                <b>{reason.get('description', reason.get('feature'))}</b><br/>
                <span>{reason.get('feature')} · {arrow}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_coverage(fields, grouped)


def render_admin_view(result: dict, customer_id: str, fields: dict, grouped: dict[str, list[str]]) -> None:
    st.subheader("EU 3 · Admin / Audit / Model Governance")
    st.caption("Full-information screen for auditability, monitoring and rule configuration. This is not exposed to customers or frontline users.")

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("### Rule-engine config")
        st.slider("Approve score threshold", 550, 780, 660, 5)
        st.slider("Decline score threshold", 300, 650, 520, 5)
        st.slider("Maximum auto-approval PD", 0.01, 0.30, 0.08, 0.01)
        st.checkbox("Hard stop: shared-device fraud flag", value=True)
        st.checkbox("Require manual review for thin-file borrowers", value=True)
        st.info("Demo config only. Production should version rules, require maker-checker approval and write audit logs.")
    with a2:
        st.markdown("### Model monitoring matrix")
        monitoring = pd.DataFrame(
            [
                ["Risk prediction", "PD, score, bad-rate by band", "Daily / Weekly", "Active"],
                ["Explainability", "SHAP top reasons + criteria", "Per case", "Active"],
                ["Representativeness", "DSR/source coverage by segment", "Monthly", "Needs data growth"],
                ["Business value", "TAT, approval lift, NPL guardrail", "Monthly", "Pilot metric"],
                ["Integration", "LOS/RLOS/CLOS, CMS, RM Portal", "Release", "Ready for API"],
            ],
            columns=["Mentor criterion", "Metric", "Cadence", "Status"],
        )
        st.dataframe(monitoring, hide_index=True, use_container_width=True)
    with a3:
        st.markdown("### Access-control matrix")
        access = pd.DataFrame(
            [
                ["Customer", "Status, indicative limit, required docs", "No score logic / no SHAP / no rules"],
                ["RM / Approval", "Score band, criteria, DSR, next best action", "No rule formula / no model coefficients"],
                ["Admin / Audit", "Full payload, SHAP, config, logs", "Controlled by maker-checker"],
            ],
            columns=["User group", "Can see", "Restriction"],
        )
        st.dataframe(access, hide_index=True, use_container_width=True)

    st.markdown("### Integration handoff")
    st.write("Recommended production path: Digital Lending / RM Portal collects consent and inputs → ScoreSight API scores silently → LOS/RLOS/CLOS receives recommendation → CMS stores case, evidence and audit trail.")
    render_payload(customer_id, fields, result)

    st.markdown("### Raw explainability and source coverage")
    reasons_df = pd.DataFrame(result.get("top_reasons", []))
    if not reasons_df.empty:
        st.dataframe(reasons_df, hide_index=True, use_container_width=True)
    render_coverage(fields, grouped)


def render_coverage(fields: dict, grouped: dict[str, list[str]]) -> None:
    coverage = source_coverage(fields, grouped)
    st.markdown("### Data sufficiency by source")
    st.bar_chart(coverage.set_index("source")[["coverage"]], height=245)
    cols = st.columns(4)
    for idx, row in enumerate(coverage.itertuples(index=False)):
        with cols[idx % 4]:
            st.caption(row.source)
            st.progress(float(row.coverage), text=f"{row.available}/{row.total} fields")
            st.caption(SOURCE_HINTS.get(row.source, ""))


def render_payload(customer_id: str, fields: dict, result: dict) -> None:
    with st.expander("Developer handoff · API payload and response", expanded=False):
        st.markdown("**POST /score payload**")
        st.json({"customer_id": customer_id, "fields": fields})
        st.markdown("**Response**")
        st.json(result)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="💳", layout="wide")
    inject_css()
    render_hero()

    feature_dictionary = load_feature_dictionary()
    grouped = feature_groups(feature_dictionary)
    sample_df = load_sample_data()
    customer_id, fields = sample_selector(sample_df)

    st.subheader(f"Borrower / MST: {customer_id}")
    st.caption("This UI follows the mentor suggestion: real risk warning, explainability, data sufficiency, business value, integration readiness and role-based access.")

    with st.spinner("Scoring borrower..."):
        try:
            result = call_score(customer_id, fields)
        except Exception as exc:
            st.error(f"Cannot score borrower: {exc}")
            st.stop()

    tab_customer, tab_internal, tab_admin, tab_solution = st.tabs([
        "Customer view",
        "RM / Approval view",
        "Admin / Audit view",
        "Solution blueprint",
    ])
    with tab_customer:
        render_customer_view(result)
    with tab_internal:
        render_internal_view(result, fields, grouped)
    with tab_admin:
        render_admin_view(result, customer_id, fields, grouped)
    with tab_solution:
        st.subheader("Bank integration blueprint")
        st.markdown(
            """
            **1. LOS/RLOS/CLOS/CMS integration**  
            ScoreSight should expose `/score` and `/decision-audit` endpoints. LOS receives score, PD, DSR, suggested limit and routing result. CMS stores the case evidence and decision trail.

            **2. RM Portal / Sale Hub**  
            RM enters MST, revenue and minimum fields; the system calls ScoreSight and returns lead quality, risk tier, suggested limit range and next best action while RM is meeting the customer.

            **3. Digital lending platform**  
            Customer signs TnC/consent and uploads documents. ScoreSight runs in the background and routes the case to green/yellow/red flows.

            **4. SCF use case**  
            Anchor/supplier/buyer network data can improve buyer-diversity, supplier-diversity and network-default-exposure features for supply-chain financing decisions.
            """
        )
        st.markdown("### Green / Yellow / Red workflow")
        workflow = pd.DataFrame(
            [
                ["Green", "Low PD + high DSR + no hard risk flag", "Auto recommendation to LOS", "Fast TAT, lower manual workload"],
                ["Yellow", "Borderline score or thin/semi data", "CMS manual review", "More approvals with controlled NPL"],
                ["Red", "High PD or fraud/hard-stop flag", "Decline / enhanced approval", "Protect NPL and fraud loss"],
            ],
            columns=["Flow", "Trigger", "System action", "Business value"],
        )
        st.dataframe(workflow, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
