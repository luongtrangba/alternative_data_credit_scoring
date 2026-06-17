# ScoreSight — Chấm điểm tín dụng MSME bằng dữ liệu phi truyền thống

> Tài liệu tiếng Việt đầy đủ. Xem [README.md](README.md) cho phiên bản tiếng Anh.

Nội dung bên dưới là README gốc từ Hackathon CX Together 2026.

---

# ScoreSight — Alternative-Data Credit Scoring for MSMEs

> Chấm điểm tín dụng cho **MSME (micro / small / medium enterprises)** hoàn toàn dựa trên **dữ liệu phi truyền thống (alternative data)** — không cần lịch sử CIC/bureau. Cốt lõi: đo độ đầy đủ dữ liệu (DSR) rồi chọn mô hình phù hợp cho từng hồ sơ, kèm giải thích SHAP.

---

## 1. Vấn đề & Insight cốt lõi

**~70% MSME Việt Nam** không tiếp cận được vốn chính thức: không có lịch sử CIC, không báo cáo kiểm toán, giao dịch chủ yếu tiền mặt / ví điện tử. Mô hình tín dụng truyền thống từ chối họ ngay từ đầu vì *thiếu dữ liệu*, chứ không phải vì *rủi ro cao*.

**Insight:** Doanh nghiệp micro không xấu hơn về tín dụng — chúng chỉ **ít dữ liệu hơn**. Dữ liệu được tạo ra từ chính dự án này xác nhận điều này:

> Tỷ lệ vỡ nợ gần như **bằng nhau** giữa micro/small/medium (10.8% / 9.4% / 8.9%), nhưng độ phủ dữ liệu **chênh lệch lớn** (micro: 57% thin-file, medium: 61% thick-file).

→ Không thể dùng **một** mô hình cho mọi hồ sơ. ScoreSight đo **Data Sufficiency Rate (DSR)** trước, rồi **route** đến mô hình phù hợp với lượng dữ liệu thực có.

---

## 2. Trạng thái dự án (hackathon scope)

| Tầng | Module | Trạng thái |
|---|---|---|
| **T1** Data | Semi-synthetic alt-data generator (neo vào dữ liệu thật) | ✅ Đã build · [t1_sources/generate_dataset.py](t1_sources/generate_dataset.py) |
| **T3** Features | DSR có trọng số + phân nhóm + IV | ✅ Đã build · [iv_calculator.py](t3_features/iv_calculator.py) · [dsr_calculator.py](t3_features/dsr_calculator.py) |
| **T4** Model | Global LightGBM + DSR per-segment calibration + tiering + SHAP | ✅ Đã build · [train.py](t4_training/train.py) · [score_mapping.py](t4_training/score_mapping.py) |
| **T5** Serving | FastAPI `/score` + decision engine + SHAP top reasons | ✅ Đã build · [t5_serving/app.py](t5_serving/app.py) |
| T2 / T6 | Data lake, monitoring | 📋 Roadmap (mục 8) |

---

## 3. Dataset Card — bộ dữ liệu đã sinh

Vì **không tồn tại** public dataset alternative-data cho MSME Việt Nam, ta sinh bộ **semi-synthetic**: nhãn good/bad và phân phối rủi ro được **neo vào UCI German Credit** (bộ chấm điểm tín dụng kinh điển, 1000 hồ sơ thật), còn các tín hiệu alternative data được mô hình hóa **tương quan theo chất lượng** với cường độ khác nhau.

```bash
python3 t1_sources/generate_dataset.py --n 18000 --target-default-rate 0.10
```

**Kết quả (N = 18.000):**

```
Default rate tổng     : 10.0%
Phân nhóm DSR         : thin 39.8% | semi 31.4% | thick 28.8%
Default theo DSR      : 11.4% | 9.5% | 8.4%   ← thin-file rủi ro hơn CHÚT (realistic)
Phân khúc MSME        : micro 45.1% | small 38.2% | medium 16.8%
Default theo quy mô   : 10.8% | 9.4% | 8.9%   ← quy mô gần như KHÔNG quyết định rủi ro

Cross-tab quy mô × DSR (theo hàng):
            thin   semi  thick
   micro     57%    30%    13%
   small     32%    35%    33%
   medium    12%    27%    61%
```

> **Hai trục tách biệt:** rủi ro tilt theo **độ đầy đủ dữ liệu** (thin-file rủi ro hơn chút — realistic, ít kiểm chứng) nhưng **độc lập với quy mô** (micro có thể thick, thin có thể medium). Nên "micro không bị phạt oan" vẫn đúng, đồng thời DSR có giá trị dự báo thật.

**Cách sinh (6 bước, [generate_dataset.py](t1_sources/generate_dataset.py)):**

1. **Quy mô + coverage** — vẽ micro/small/medium theo tỷ lệ VN; vẽ `coverage` (độ đầy đủ dữ liệu), độc lập rủi ro.
2. **Neo rủi ro thật** — fit logistic (numpy) trên German Credit → P(bad); bootstrap 18k, **tilt theo coverage** (thin rủi ro hơn), rescale base rate 10% bằng bisection.
3. **Gán nhãn** `default ~ Bernoulli(latent_risk)` — ground truth nhất quán.
4. **Sinh ~30 feature alt-data** từ `q = 1 − risk`; signal strength khác theo nguồn **VÀ theo regime nhóm**: graph mạnh cho thin-file, tài chính mạnh cho thick-file (vì sao cần DSR).
5. **Hệ số độ lớn theo quy mô** cho GMV/doanh thu/lao động (độc lập rủi ro → tránh confound).
6. **Mask theo nhóm nguồn** — mỗi MSME chỉ có một số nguồn; micro ít nguồn hơn → tự nhiên tạo phổ DSR thin/semi/thick.

**Output:** `data/sme_altdata.parquet` (đầy đủ, NaN = nguồn không có) · `data/sme_altdata_sample.csv` (300 dòng) · `data/feature_dictionary.json` (nguồn, hướng, trọng số, signal của từng feature).

> ⚠️ Các cột `_gt_latent_risk`, `_gt_quality`, `_dsr_*` chỉ dùng để debug/demo — **phải drop trước khi train** (tránh leakage).

---

## 4. Nguồn dữ liệu phi truyền thống (7 nhóm)

| Nhóm | Feature tiêu biểu | Vì sao đáng tin |
|---|---|---|
| **E-commerce** (Shopee/Lazada) | `shopee_gmv_3m`, `gmv_growth_12m`, `return_rate`, `seller_rating` | Sàn TMĐT là bên thứ ba xác thực giao dịch |
| **Digital Payment** (MoMo/POS) | `supplier_payment_regularity`, `momo_net_cashflow_avg`, `payroll_regularity` | Trả NCC/lương đúng hạn = kỷ luật dòng tiền (signal mạnh nhất) |
| **E-invoice / Thuế** | `invoice_revenue_12m`, `invoice_revenue_growth`, `unique_buyer_count`, `vat_filing_on_time_ratio` | Hóa đơn điện tử được Tổng cục Thuế xác nhận — không làm giả |
| **Utility** (EVN) | `electricity_consumption_avg`, `electricity_growth` | Proxy hoạt động sản xuất thực tế |
| **Logistics** (GHN/GHTK) | `shipment_count_monthly`, `delivery_success_rate` | Đơn hàng thực được vận chuyển |
| **Digital Footprint** | `google_avg_rating`, `facebook_engagement_rate` | Hiện diện online (signal yếu, nhiều nhiễu) |
| **Graph / Network** | `buyer_diversity_score`, `network_default_exposure`, `shared_device_risk_flag` | Đa dạng đối tác & rủi ro lan truyền từ mạng lưới |

`unique_buyer_count` quan trọng: 1 khách hàng chiếm 90% doanh thu = rủi ro tập trung cao.

---

## 5. T3 · DSR — Data Sufficiency Rate ✅

DSR có trọng số đo độ đầy đủ dữ liệu *quan trọng*, không chỉ đếm số trường:

$$DSR_{wq} = \frac{\sum_i w_i \cdot \mathbb{1}[\text{valid}_i]}{\sum_i w_i} \times 100\%$$

**Trọng số $w_i$ = domain expert ĐÃ ĐƯỢC IV HIỆU CHỈNH** (nửa đầu feedback loop):

$$w_{refined} = w_{domain} \times \text{clip}\left(\frac{IV}{\text{median}(IV)},\ 0.3,\ 3.0\right)$$

→ thiếu `invoice_revenue_growth` (IV 0.48, w 3.0) phạt DSR nặng hơn nhiều so với thiếu `facebook_engagement_rate` (IV 0.02, w 0.06).

**Kết quả (18k MSME):**

- Phân nhóm: thin **39.6%** · semi **28.6%** · thick **31.8%**; default theo nhóm 11.4/9.7/8.5% (thin-file rủi ro hơn chút → DSR có giá trị dự báo).
- **15.7% MSME đổi nhóm** sau khi áp trọng số IV (so với DSR thô đếm đều).
- IV trung bình theo nguồn: e-invoice cao nhất > payment > graph > … > digital_footprint thấp nhất.

**Output:** `data/sme_scored_dsr.parquet` (canonical cho T4) · `configs/weights_refined.json` · `t3_features/output/` (IV table + 35 bảng WoE).

---

## 6. T4 · Model — Hybrid (global discriminator + DSR heads) ✅

> **Phát hiện then chốt (đo đạc thực nghiệm):** train **3 model tách rời** theo nhóm DSR **KHÔNG** thắng một global LightGBM về AUC — vì cây boosting tự xử lý dị biệt qua nhánh missing, và *pooling* cứu nhóm nghèo data (thin-file ít mẫu → model riêng nhiễu). Đây là sự thật ML, không phải bug. → Ta dùng kiến trúc **hybrid** đặt DSR đúng chỗ:

| Thành phần | Vai trò |
|---|---|
| **1 Global LightGBM** (alt-data, mọi feature) | Bộ **phân biệt** rủi ro — AUC engine |
| **3 Calibration head theo DSR** (thin/semi/thick) | **PD chính xác trong từng segment** (chỗ DSR thắng, đo được) |
| **DSR tiering** | Chính sách **hạn mức** theo độ đầy đủ dữ liệu (thin ×0.5 · semi ×0.75 · thick ×1.0) |

**Kết quả (test 3.600 MSME):**

| Chỉ số | Kết quả |
|---|---|
| **ALT-DATA LIFT** | firmographic-only **0.528** → global alt-data **0.728** AUC (**+0.200**) |
| Global vs scorecard logistic | 0.728 vs 0.698 (LightGBM thắng nhờ tương tác phi tuyến) |
| AUC theo nhóm | thin 0.678 · semi 0.802 · thick 0.687 |
| **DSR calibration** | ECE overall 0.0047 → **0.0034**; semi 0.0188 → **0.0072** (single → DSR-per-segment) |
| Decision engine | approve **53%** @ bad **4.4%** · review 40% @ 12.4% · decline 7% @ **36.6%** (base 10%) |

- **Class imbalance** (10% bad): bỏ `scale_pos_weight` (hại AUC ranking) — calibration lo phần xác suất.
- **Calibration** isotonic theo từng nhóm DSR → map sang **điểm 0–1000** ([score_mapping.py](t4_training/score_mapping.py), công thức PDO/odds).
- **SHAP** trên global model → top reasons mỗi quyết định (compliance).
- **Vì sao DSR vẫn giá trị:** không phải AUC mà là **calibration PD chuẩn từng segment** + **tiering vận hành** + **governance** (thin-file → hạn mức thấp / xin thêm data). Thin-file rủi ro hơn chút (11.4% vs thick 8.4%) — realistic.

**Output:** `t4_training/models/scoresight_bundle.joblib` · `output/metrics.csv` + `calibration.csv` · 5 plots (`figures/`).

---

## 7. T5 · Serving ✅

```bash
# Khởi động server
cd "<project_root>"
uvicorn t5_serving.app:app --host 0.0.0.0 --port 8000 --reload

# Demo 3 profiles (thick/thin/fraud)
python3 t5_serving/demo_client.py

# Swagger UI
open http://localhost:8000/docs
```

**Pipeline một request `POST /score`:**

```
Input fields (dict) → Hard rule shared_device_risk_flag
  → Build 35-feature row (NaN khi thiếu)
  → DSR có trọng số → assign thin/semi/thick
  → Global LightGBM → calibration head theo DSR group
  → PDO scorecard → credit_score [300–850]
  → Decision Engine + DSR tiering credit limit
  → SHAP TreeExplainer → top 5 reasons
```

**Response mẫu:**

```json
{
  "customer_id": "MSME-00001",
  "credit_score": 690,
  "p_bad": 0.1210,
  "dsr_value": 0.74,
  "dsr_group": "thick",
  "enterprise_size": "small",
  "decision": "approve",
  "credit_limit_vnd": 150000000,
  "top_reasons": [
    {"feature": "invoice_revenue_growth", "shap_value": -0.42, "direction": "decrease_risk", "description": "Tăng trưởng doanh thu hóa đơn"},
    {"feature": "supplier_payment_regularity", "shap_value": -0.31, "direction": "decrease_risk", "description": "Độ đều thanh toán nhà cung cấp"}
  ],
  "warnings": []
}
```

| Score (thang 300–850) | Quyết định | + DSR tiering hạn mức |
|---|---|---|
| ≥ 620 | Approve | base[size] × {thin 0.5 / semi 0.75 / thick 1.0} |
| 540–619 | Manual Review | ưu tiên thin-file |
| < 540 | Decline | credit_limit_vnd = 0, kèm top reasons (SHAP) |

**Hard rule:** `shared_device_risk_flag = 1` → decline ngay, bỏ qua model (tín hiệu gian lận).

**Endpoints:** `GET /health` · `GET /model-info` · `POST /score`

---

## 8. Roadmap production (ngoài scope hackathon)

T2 Data Lake (PySpark+Parquet) · T6 Monitoring (Evidently/Grafana, PSI/KS drift) · Champion-Challenger · Kafka streaming · Airflow orchestration · Feature Store · OAuth consent (NĐ 13/2023) · gaming/seasonal-drift detection.

---

## 9. Cấu trúc thư mục

```
scoresight/
├── t1_sources/
│   └── generate_dataset.py      # ✅ Semi-synthetic alt-data generator
├── data/
│   ├── raw/german.data-numeric  # Anchor dataset (UCI German Credit)
│   ├── sme_altdata.parquet      # 18k MSME đã sinh
│   ├── sme_altdata_sample.csv
│   └── feature_dictionary.json  # nguồn / hướng / trọng số / signal
├── eda/
│   ├── eda_report.py            # ✅ EDA: missingness, IV, leak-check, collinearity
│   ├── figures/                 #    6 biểu đồ PNG cho slide
│   └── tables/                  #    IV, missingness, numeric summary (CSV)
├── t3_features/
│   ├── iv_calculator.py         # ✅ IV + WoE binning, lọc feature yếu
│   ├── dsr_calculator.py        # ✅ Trọng số IV-refined + DSR có trọng số + phân nhóm
│   └── output/                  #    iv_table.csv, weights_refined.csv, woe/*.csv
├── t4_training/
│   ├── train.py                 # ✅ Global LightGBM + DSR calibration + tiering + SHAP
│   ├── score_mapping.py         # ✅ P(bad) → điểm 0–1000 (PDO/odds) + decision
│   ├── models/                  #    scoresight_bundle.joblib
│   └── output/                  #    metrics.csv, calibration.csv, figures/ (5 PNG)
├── configs/
│   ├── dsr_config.json          # ngưỡng DSR + tham số hiệu chỉnh trọng số
│   └── weights_refined.json     # trọng số w_i sau IV
├── t5_serving/
│   ├── app.py                   # ✅ FastAPI /score, decision engine, SHAP
│   └── demo_client.py           # ✅ Demo 3 profiles
└── README.md
```

---

## 10. Khái niệm chính

- **MSME** — Micro / Small / Medium Enterprise (Nghị định 80/2021): phân theo lao động + doanh thu.
- **Alternative Data** — dữ liệu phi truyền thống (TMĐT, ví điện tử, hóa đơn điện tử, điện, logistics).
- **DSR** — tỷ lệ nguồn dữ liệu hợp lệ có trọng số → quyết định routing mô hình.
- **IV** — sức mạnh phân tách good/bad của một feature; dùng để chọn feature & validate trọng số.
- **SHAP** — đóng góp của từng feature vào một quyết định cụ thể (giải trình compliance).

## Cài đặt

```bash
pip install --user --break-system-packages numpy pandas pyarrow   # generator
python3 t1_sources/generate_dataset.py                            # sinh dữ liệu
```
# hack-cx-version-2

---

## 10. T6 · UI Demo (Streamlit)

Repo hiện có thêm dashboard demo tại `t6_ui/streamlit_app.py`. UI này dùng để showcase output chính của ScoreSight: **credit score**, **P(default)**, **DSR thin/semi/thick**, **decision**, **credit limit**, **SHAP top reasons** và **coverage theo nguồn dữ liệu**.

Chạy nhanh local mode:

```bash
cd "<project_root>"
pip install -r requirements.txt
streamlit run t6_ui/streamlit_app.py
```

Chạy theo API mode:

```bash
uvicorn t5_serving.app:app --host 0.0.0.0 --port 8000 --reload
export SCORE_API_URL=http://localhost:8000/score
streamlit run t6_ui/streamlit_app.py
```

Demo flow gợi ý: mở borrower `thick` để cho thấy hồ sơ đủ dữ liệu, chuyển sang `thin` để thấy DSR/limit giảm, sau đó bật `shared_device_risk_flag` trong Manual form để minh họa hard-decline rule.

## T6 · UI Demo

A Streamlit UI is available in `t6_ui/streamlit_app.py`.

This version follows the mentor feedback and presents the scoring model as a bank-ready workflow, not just a score page:

- **Customer view**: simple digital-lending result, indicative limit and next step; no model logic exposed.
- **RM / Approval view**: score, PD, DSR, suggested limit, criteria-level reasons, data coverage and TAT/SLA/NPL impact.
- **Admin / Audit view**: full model handoff, SHAP details, rule-engine configuration mockup, monitoring matrix and access-control matrix.
- **Solution blueprint**: integration path for LOS/RLOS/CLOS, CMS, RM Portal/Sale Hub, digital lending platform and SCF.

Run:

```bash
streamlit run t6_ui/streamlit_app.py
```

See `t6_ui/README_UI.md` for the full demo flow.
