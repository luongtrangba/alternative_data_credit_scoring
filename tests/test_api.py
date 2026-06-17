"""Integration tests for the FastAPI scoring service.

These tests require the model artifacts to be present:
  - t4_training/models/scoresight_bundle.joblib
  - configs/weights_refined.json
  - configs/dsr_config.json

Run the full pipeline first if artifacts are missing:
  python -m scoresight.data_generator
  python -m scoresight.features.dsr_calculator
  python -m scoresight.training.train
"""

import pytest

try:
    from fastapi.testclient import TestClient

    from scoresight.serving.app import app
    _HAS_ARTIFACTS = True
except Exception:
    _HAS_ARTIFACTS = False

skip_no_artifacts = pytest.mark.skipif(
    not _HAS_ARTIFACTS,
    reason="Model artifacts not found — run the training pipeline first",
)


@skip_no_artifacts
class TestHealthEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health_returns_ok(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["features"] > 0

    def test_model_info(self):
        r = self.client.get("/model-info")
        assert r.status_code == 200
        data = r.json()
        assert "features" in data
        assert "dsr_thresholds" in data


@skip_no_artifacts
class TestScoreEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_score_good_borrower(self):
        payload = {
            "customer_id": "TEST-001",
            "fields": {
                "enterprise_size": "small",
                "industry": "retail",
                "region": "HCM",
                "num_employees": 15,
                "business_age_months": 48,
                "shopee_gmv_3m": 120_000_000,
                "gmv_growth_12m": 0.30,
                "supplier_payment_regularity": 0.92,
                "invoice_revenue_12m": 1_500_000_000,
                "invoice_revenue_growth": 0.28,
                "shared_device_risk_flag": 0,
            },
        }
        r = self.client.post("/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert 300 <= data["credit_score"] <= 850
        assert 0 <= data["p_bad"] <= 1
        assert data["dsr_group"] in ["thin", "semi", "thick"]
        assert data["decision"] in ["approve", "manual_review", "decline"]

    def test_hard_decline_fraud_flag(self):
        payload = {
            "customer_id": "TEST-FRAUD",
            "fields": {"shared_device_risk_flag": 1},
        }
        r = self.client.post("/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["decision"] == "decline"
        assert data["credit_score"] == 300
        assert data["credit_limit_vnd"] == 0
        assert any("HARD DECLINE" in w for w in data["warnings"])

    def test_minimal_fields_returns_warning(self):
        payload = {
            "customer_id": "TEST-MIN",
            "fields": {"enterprise_size": "micro"},
        }
        r = self.client.post("/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert any("fields available" in w for w in data["warnings"])
