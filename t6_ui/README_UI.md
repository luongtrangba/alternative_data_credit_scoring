# ScoreSight UI Demo

The UI is designed around the mentor feedback: the model should not only return a score, but should also show real risk warning, explainability, data sufficiency, business impact, bank integration readiness and role-based access control.

## Main screens

### 1. Customer view

For external end users applying through a digital lending platform.

Visible information:

- Application status
- Indicative limit
- Next step / required documents

Hidden information:

- Model logic
- Rule engine
- SHAP explanations
- PD calculation
- Internal scorecard thresholds

This matches the requirement that customer-facing UI must be simple and must not reveal the scoring logic.

### 2. RM / Approval view

For RM, sales hub, thẩm định and credit approver users.

Visible information:

- Credit score
- Decision flow: green / yellow / red
- Probability of default
- Data Sufficiency Rate
- Suggested limit
- Criteria-level explanation
- Data source coverage
- Business impact: TAT/SLA saving, approval lift and NPL guardrail

Hidden information:

- Raw model coefficients
- Exact rule-engine logic
- Admin-level configuration

This screen supports RM Portal / Sale Hub usage: RM enters MST or minimum borrower info and immediately receives lead quality, risk tier, suggested limit range and next best action.

### 3. Admin / Audit view

For model governance, risk, audit and admin users.

Visible information:

- Rule-engine configuration mockup
- Hard-stop rule controls
- Monitoring matrix
- Access-control matrix
- Raw API payload and response
- SHAP details
- Source coverage

This screen is intended to demonstrate auditability and governance. In production, rule changes should use maker-checker approval, versioning and audit logs.

### 4. Solution blueprint

This tab explains how the model can be integrated into bank systems:

- LOS / RLOS / CLOS / CMS
- RM Portal / Sale Hub
- Digital lending platform
- SCF use case
- Green / yellow / red credit workflow

## Why this UI fits the current backend

The repository already has a FastAPI scoring service in `t5_serving/app.py`. The UI wraps the existing scoring logic and presents outputs that bank users care about:

1. **Credit score** - 300-850 score from calibrated PD.
2. **Risk warning** - approve, manual review or decline; plus hard fraud warning if applicable.
3. **Data Sufficiency Rate** - thin/semi/thick borrower profile.
4. **Suggested credit limit** - adjusted by MSME size and DSR confidence.
5. **SHAP top reasons** - explainability for internal users and admin/audit.
6. **Source coverage** - shows which alternative data sources are present or missing.
7. **Business impact** - estimated TAT/SLA saving, approval lift and NPL guardrail messaging.
8. **Role-based access** - different information exposure for customer, internal bank user and admin/audit.

## Run local mode

Local mode imports the FastAPI pipeline directly, so you only need one command:

```bash
pip install -r requirements.txt
streamlit run t6_ui/streamlit_app.py
```

## Run API mode

API mode is closer to production because Streamlit calls FastAPI `/score`.

Terminal 1:

```bash
uvicorn t5_serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:

```bash
export SCORE_API_URL=http://localhost:8000/score
streamlit run t6_ui/streamlit_app.py
```

On Windows PowerShell:

```powershell
$env:SCORE_API_URL="http://localhost:8000/score"
streamlit run t6_ui/streamlit_app.py
```

## Suggested demo flow

1. Open a **thick** sample borrower and show high DSR, clearer decision and higher suggested limit.
2. Switch to **Customer view** and explain that the customer only sees simple status and next step.
3. Switch to **RM / Approval view** and show score, PD, DSR, criteria-level explanation and operational impact.
4. Use **RM quick form**, turn off `E-invoice / tax` and `Digital payment`, then show DSR dropping and more manual-review behavior.
5. Turn on `Shared-device fraud flag` and show the hard decline/risk warning.
6. Switch to **Admin / Audit view** and show rule config mockup, access-control matrix, monitoring matrix and raw API handoff.
7. End with **Solution blueprint** to connect the demo to LOS/CMS/RM Portal/Digital Lending/SCF.

## Integration notes

- Keep `t6_ui/streamlit_app.py` inside the project root so imports and model paths resolve correctly.
- The UI intentionally avoids changing the model pipeline. It wraps the existing scoring logic.
- To deploy later, split the UI and API into two services: one FastAPI backend and one Streamlit frontend.
- Production rule configuration should be stored in versioned config tables, not directly edited in the UI without approval workflow.
