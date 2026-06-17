# ScoreSight — Alternative-Data Credit Scoring for MSMEs

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/luongtrangba/alternative_data_credit_scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/luongtrangba/alternative_data_credit_scoring/actions)

> Credit scoring for **micro, small, and medium enterprises (MSMEs)** using **100% alternative data** — no CIC/bureau history required. Core idea: measure **Data Sufficiency Rate (DSR)** first, then route each application to the right model with SHAP-based explainability.

**[Vietnamese documentation (README_vi.md)](README_vi.md)** | **[Scorecard Design (scorecard_master_v2.md)](scorecard_master_v2.md)**

---

## Problem Statements

~70% of Vietnamese MSMEs cannot access formal credit — not because they are high-risk, but because they lack traditional financial data (no CIC history, no audited statements, mostly cash/e-wallet transactions). Traditional models reject them at the gate due to **missing data, not high risk**.

**Key insight from our data:** Default rates are nearly identical across micro/small/medium enterprises (10.8% / 9.4% / 8.9%), but data coverage differs dramatically (micro: 57% thin-file vs medium: 61% thick-file). **Size does not determine risk — data sufficiency does.**

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     ScoreSight Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (35 alt-data fields)                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    ┌──────────────────┐                        │
│  │ Hard Rules   │───▶│ shared_device    │──▶ DECLINE             │
│  │ (fraud gate) │    │ risk_flag = 1    │                        │
│  └──────┬──────┘    └──────────────────┘                        │
│         │ pass                                                  │
│         ▼                                                       │
│  ┌─────────────────────────────┐                                │
│  │  Weighted DSR Calculation   │                                │
│  │  DSR = Σ(wᵢ·validᵢ)/Σ(wᵢ) │                                │
│  └──────────┬──────────────────┘                                │
│             │                                                   │
│    ┌────────┼────────┐                                          │
│    ▼        ▼        ▼                                          │
│  thin    semi     thick     ◄── DSR groups                      │
│    │        │        │                                          │
│    └────────┼────────┘                                          │
│             ▼                                                   │
│  ┌───────────────────┐    ┌─────────────────────┐               │
│  │  Global LightGBM  │───▶│  Calibration Head   │               │
│  │  (AUC engine)     │    │  (per DSR segment)  │               │
│  └───────────────────┘    └─────────┬───────────┘               │
│                                     ▼                           │
│  ┌───────────────────┐    ┌─────────────────────┐               │
│  │  PDO Scorecard    │───▶│  Decision Engine    │               │
│  │  P(bad)→300-850   │    │  + DSR Tiering      │               │
│  └───────────────────┘    └─────────┬───────────┘               │
│                                     ▼                           │
│  ┌───────────────────┐    ┌─────────────────────┐               │
│  │  SHAP Explainer   │───▶│  API Response       │               │
│  │  Top 5 reasons    │    │  score + decision   │               │
│  └───────────────────┘    └─────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **7 alternative data sources**: e-commerce (Shopee), digital payment (MoMo/POS), e-invoices, utilities (EVN), logistics (GHN/GHTK), digital footprint (Google/Facebook), graph/network
- **Weighted DSR** (Data Sufficiency Rate): IV-refined weights penalize missing high-value fields more heavily
- **Hybrid model**: single global LightGBM for discrimination + 3 per-segment calibration heads for accurate PD
- **DSR tiering**: credit limits adjusted by data completeness (thin ×0.5, semi ×0.75, thick ×1.0)
- **SHAP explainability**: top reasons per decision for regulatory compliance
- **Hard fraud rules**: shared-device risk flag triggers immediate decline
- **Role-based UI**: Customer view, RM/Approval view, Admin/Audit view, Solution blueprint

## Model Performance

| Metric | Result |
| ------ | ------ |
| **Alt-data lift** | Firmographic-only 0.528 → Global alt-data **0.728 AUC** (+0.200) |
| Global vs logistic scorecard | 0.728 vs 0.698 (LightGBM wins via non-linear interactions) |
| AUC by DSR group | thin 0.678 · semi 0.802 · thick 0.687 |
| **DSR calibration** | ECE overall 0.0047 → **0.0034** (single → per-segment) |
| Decision engine | Approve **53%** @ bad 4.4% · Review 40% @ 12.4% · Decline 7% @ 36.6% |

## Tech Stack

| Layer | Technology |
| ----- | ---------- |
| Data generation | NumPy, Pandas, PyArrow |
| Feature engineering | IV/WoE binning, weighted DSR |
| Modeling | LightGBM, scikit-learn (calibration, logistic baseline) |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Pydantic, Uvicorn |
| UI | Streamlit |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions (pytest + ruff) |

## Quick Start

### Option 1: Local Installation

```bash
# Clone
git clone https://github.com/luongtrangba/alternative_data_credit_scoring.git
cd alternative_data_credit_scoring

# Install
pip install -e ".[all]"

# Generate data → compute features → train model
python -m scoresight.data_generator
python -m scoresight.features.dsr_calculator
python -m scoresight.training.train

# Start API server
uvicorn scoresight.serving.app:app --host 0.0.0.0 --port 8000 --reload

# (Optional) Start Streamlit UI in another terminal
streamlit run t6_ui/streamlit_app.py
```

### Option 2: Docker

```bash
docker compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Score a borrower
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "MSME-001",
    "fields": {
      "enterprise_size": "small",
      "industry": "retail",
      "invoice_revenue_growth": 0.22,
      "supplier_payment_regularity": 0.85,
      "shared_device_risk_flag": 0
    }
  }'
```

## API Reference

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/health` | GET | Service health check |
| `/model-info` | GET | Model metadata, features, thresholds |
| `/score` | POST | Score an MSME borrower |

### POST /score — Response

```json
{
  "customer_id": "MSME-001",
  "credit_score": 690,
  "p_bad": 0.1210,
  "dsr_value": 0.74,
  "dsr_group": "thick",
  "enterprise_size": "small",
  "decision": "approve",
  "credit_limit_vnd": 150000000,
  "top_reasons": [
    {
      "feature": "invoice_revenue_growth",
      "shap_value": -0.42,
      "direction": "decrease_risk",
      "description": "E-invoice revenue growth"
    }
  ],
  "warnings": []
}
```

## Project Structure

```text
alternative_data_credit_scoring/
├── src/scoresight/              # Python package
│   ├── data_generator.py        # Semi-synthetic alt-data generator
│   ├── features/
│   │   ├── iv_calculator.py     # IV + WoE binning
│   │   └── dsr_calculator.py    # Weighted DSR + group assignment
│   ├── training/
│   │   ├── train.py             # Hybrid LightGBM + DSR calibration
│   │   └── score_mapping.py     # P(bad) → credit score [300-850]
│   └── serving/
│       ├── app.py               # FastAPI scoring service
│       └── demo_client.py       # Demo 3 MSME profiles
├── t6_ui/streamlit_app.py       # Streamlit dashboard (4 role-based views)
├── eda/                         # Exploratory data analysis + figures
├── configs/dsr_config.json      # DSR thresholds and weight refinement params
├── data/
│   ├── raw/german.data-numeric  # UCI German Credit anchor dataset
│   ├── feature_dictionary.json  # Feature metadata (source, direction, weight)
│   └── sme_altdata_sample.csv   # 300-row sample for quick inspection
├── tests/                       # Unit + integration tests
├── Dockerfile                   # Container build
├── docker-compose.yml           # API + UI services
├── pyproject.toml               # Package config + dependencies
├── scorecard_master_v2.md       # 16-variable rule-based scorecard design
└── README_vi.md                 # Full Vietnamese documentation
```

## Alternative Data Sources (7 Groups)

| Source | Key Features | Why Trustworthy |
| ------ | ------------ | --------------- |
| **E-commerce** (Shopee/Lazada) | GMV, growth, return rate, seller rating | Third-party verified transactions |
| **Digital Payment** (MoMo/POS) | Supplier payment regularity, payroll, cashflow | Payment discipline = strongest signal |
| **E-invoice / Tax** | Revenue, growth, buyer count, VAT filing ratio | Government-verified, cannot be faked |
| **Utility** (EVN) | Electricity consumption, growth | Proxy for real operating activity |
| **Logistics** (GHN/GHTK) | Shipment count, delivery success rate | Actual goods shipped |
| **Digital Footprint** | Google rating, Facebook engagement | Online presence (weak signal, noisy) |
| **Graph / Network** | Buyer diversity, default exposure, PageRank | Partner diversity and contagion risk |

## Running Tests

```bash
# Unit tests (no model artifacts needed)
pytest tests/test_score_mapping.py tests/test_dsr.py -v

# Full test suite (requires trained model)
pytest tests/ -v
```

---

## Hackathon CX Together 2026 — Product for SHB

This project was built during **Hackathon CX Together in Banking 2026** as an alternative data credit scoring solution for **SHB (Saigon Hanoi Commercial Joint Stock Bank)**.

### Hackathon Achievements

- Designed a **16-variable rule-based scorecard** with 6 pillars (see [scorecard_master_v2.md](scorecard_master_v2.md))
- Built a **hybrid ML model** proving +0.200 AUC lift from alternative data over firmographic-only baseline
- Demonstrated **DSR-aware calibration** improving ECE by 28% vs single-calibration approach
- Created a **bank-ready UI** with role-based access: Customer, RM/Approval, Admin/Audit views
- Addressed **Vietnamese regulatory compliance**: Decree 13/2023 (data privacy), Circular 11/2021 (credit classification), Decree 123/2020 (e-invoices)

### Key Design Principles

1. **Legal entity ≠ Individual** — all data belongs to the business, not personal data
2. **Behavioral > Sentiment** — measure recurring obligations (tax, insurance, utilities), not market sentiment
3. **Cross-validation > Single source** — non-traditional operational data verifies traditional financial statements
4. **Explainable scoring** — every decision comes with reason codes and improvement recommendations

---

## Roadmap

- [ ] Model versioning with MLflow/DVC
- [ ] API authentication (OAuth2/JWT)
- [ ] Telco data integration (mobile top-up patterns)
- [ ] Real-time scoring pipeline (Kafka)
- [ ] Model monitoring with Evidently AI (PSI/KS drift detection)
- [ ] Champion-Challenger A/B framework
- [ ] Batch scoring API (CSV upload)

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.
