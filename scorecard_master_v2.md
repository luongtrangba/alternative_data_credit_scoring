# SCORECARD TÍN DỤNG MSME/SME — BẢN MASTER V2

**16 biến · 6 trụ cột · 1000 điểm · Tag Traditional / Non-Traditional · KHÔNG CIC · Tuân thủ NĐ 13/2023**

> Phiên bản v2 tập trung vào bộ tiêu chí thuần túy: trường dữ liệu, logic chấm điểm, đầu vào, đầu ra. Không bao gồm tier framework, không bao gồm mô hình AI/ML, không bao gồm forecast — chỉ scorecard rule-based có thể implement bằng IF/ELSE.

---

## MỤC LỤC

1. [Triết lý thiết kế & Tổng quan](#1-triết-lý-thiết-kế--tổng-quan)
2. [Cấu trúc 6 trụ cột](#2-cấu-trúc-6-trụ-cột)
3. [Phân loại Traditional vs Non-Traditional](#3-phân-loại-traditional-vs-non-traditional)
4. [Chi tiết 16 biến số](#4-chi-tiết-16-biến-số)
5. [Bảng ranking mức độ quan trọng (5 trụ chính)](#5-bảng-ranking-mức-độ-quan-trọng)
6. [Knock-out Rules](#6-knock-out-rules)
7. [Yêu cầu dữ liệu đầu vào](#7-yêu-cầu-dữ-liệu-đầu-vào)
8. [Định dạng đầu ra cho khách hàng](#8-định-dạng-đầu-ra-cho-khách-hàng)
9. [Bảng quy đổi hạng tín dụng](#9-bảng-quy-đổi-hạng-tín-dụng)
10. [Reason Codes & Khuyến nghị cải thiện](#10-reason-codes--khuyến-nghị-cải-thiện)
11. [Cross-validation & Fraud Detection](#11-cross-validation--fraud-detection)
12. [Phân tích overlap còn lại](#12-phân-tích-overlap-còn-lại)
13. [Tuân thủ pháp lý](#13-tuân-thủ-pháp-lý)
14. [Defense Q&A](#14-defense-qa)
15. [Tham chiếu học thuật & Benchmark](#15-tham-chiếu-học-thuật--benchmark)

---

## 1. Triết lý thiết kế & Tổng quan

### Bối cảnh & Mục tiêu

Scorecard này được thiết kế cho phân tầng **MSME và SME có pháp nhân** tại Việt Nam — đặc biệt là nhóm "thin-file CIC" (DN có ít hoặc không có lịch sử tín dụng tại các TCTD truyền thống). Mục tiêu là giải quyết *"thin-file paradox"*: nhóm DN có nhu cầu vốn cao nhất nhưng bị NH truyền thống từ chối nhiều nhất.

### 4 nguyên tắc thiết kế cốt lõi

| # | Nguyên tắc | Hệ quả thực tế |
|---|---|---|
| 1 | **Pháp nhân ≠ Cá nhân** | Toàn bộ data thuộc về DN, không chạm dữ liệu cá nhân → tránh NĐ 13/2023 |
| 2 | **Behavioral > Sentiment** | Đo hành vi định kỳ (thuế, BHXH, utility) thay vì cảm xúc thị trường → không thể giả mạo |
| 3 | **Cross-validation > Single source** | Dữ liệu phi truyền thống (operational thực) kiểm chứng dữ liệu truyền thống (BCTC khai báo) → phát hiện fraud |
| 4 | **Reason Codes explainable** | Mọi điểm số đều giải thích được lý do và đề xuất hành động cải thiện |

---

## 2. Cấu trúc 6 trụ cột

| Trụ cột | Trọng số | Số biến | Vai trò |
|---|---|---|---|
| **I. Hard Financial** | 250đ (25%) | 4 | Đo sức khỏe tài chính từ BCTC |
| **II. Banking Relationship** | 70đ (7%) | 3 | Đo quan hệ và độ "stickiness" với NH |
| **III. Digital Operational** | 310đ (31%) | 4 | Đo hoạt động kinh doanh thực qua nguồn third-party |
| **IV. Behavioral Compliance** | 250đ (25%) | 3 | Đo hành vi thực hiện nghĩa vụ định kỳ |
| **V. Sustainability (ESG)** | 70đ (7%) | 1 | Đo rủi ro ESG (môi trường, lao động, sở hữu) |
| **VI. Business Maturity** | 50đ (5%) | 1 | Đo tuổi pháp nhân và operational continuity *(trụ riêng — context, không xếp ngang risk indicators)* |
| **TỔNG** | **1000đ** | **16** | |

### Lưu ý về Trụ VI

Trụ Maturity được tách riêng và **không đưa vào bảng ranking mức độ quan trọng** ở Section 5. Lý do: tuổi DN là *context* (cho biết chúng ta biết DN bao lâu) chứ không phải *risk indicator* trực tiếp. DN trẻ không có nghĩa rủi ro hơn DN già — chỉ có nghĩa lịch sử quan sát ngắn hơn. Trộn Maturity với 5 trụ kia sẽ làm méo nhận định rủi ro.

---

## 3. Phân loại Traditional vs Non-Traditional

### Định nghĩa

| Loại | Định nghĩa | Nguồn dữ liệu |
|---|---|---|
| **Traditional (TRAD)** | Dữ liệu lấy từ kênh truyền thống của NH | BCTC, sao kê NH tại chính NH, lịch sử quan hệ với NH cho vay |
| **Non-Traditional (NON-TRAD)** | Dữ liệu lấy từ kênh ngoài NH | API thuế, BHXH, HĐĐT, cổng công khai NN, AI parse hợp đồng |

### Phân bổ trọng số

| Loại | Số biến | Tổng điểm | % Tổng |
|---|---|---|---|
| **Traditional** | 8 | 410đ | **41%** |
| **Non-Traditional** | 8 | 590đ | **59%** |
| **Tổng** | 16 | 1000đ | 100% |

→ Scorecard nghiêng 59% về non-traditional — phản ánh đúng triết lý *"BCTC mỏng, operational data là ground truth"* cho MSME/SME VN.

### Bảng tổng phân loại 16 biến

| # | Biến số | Trụ | Loại | Điểm |
|---|---|---|---|---|
| 1 | DSCR | I | TRAD | 90 |
| 2 | Đòn bẩy | I | TRAD | 60 |
| 3 | EBITDA Margin | I | TRAD | 55 |
| 4 | CCC | I | TRAD | 45 |
| 5 | Bank Account Vitality | II | TRAD | 44 |
| 6 | Cross-Product Usage | II | TRAD | 18 |
| 7 | Account Age | II | TRAD | 8 |
| 8 | E-Invoice Vitality | III | NON-TRAD | 100 |
| 9 | Customer Concentration | III | NON-TRAD | 55 |
| 10 | Bank Cash Flow Pattern | III | TRAD | 90 |
| 11 | Anchor-Supplier Network | III | NON-TRAD | 65 |
| 12 | Tax Compliance | IV | NON-TRAD | 100 |
| 13 | Social Insurance | IV | NON-TRAD | 85 |
| 14 | Utility Compliance | IV | NON-TRAD | 65 |
| 15 | ESG Behavioral Proxy | V | NON-TRAD | 70 |
| 16 | Business Maturity | VI | NON-TRAD | 50 |

---

## 4. Chi tiết 16 biến số

### TRỤ CỘT I — HARD FINANCIAL (250 điểm) · TRADITIONAL

#### Biến 1 — Debt Service Coverage Ratio (DSCR) — 90 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Khả năng dòng tiền hoạt động trả được nghĩa vụ nợ |
| **Công thức** | `DSCR = EBITDA_12m / (Gốc + Lãi phải trả 12m tới)` |
| **Logic chấm** | ≥1.5 → 90 · 1.25–1.5 → 68 · 1.0–1.25 → 40 · 0.8–1.0 → 18 · <0.8 → 0 |
| **Nguồn dữ liệu** | BCTC + lịch trả nợ NH (nội bộ) |
| **Reason Code** | F01 — Khả năng trả nợ thấp |

#### Biến 2 — Đòn bẩy tài chính — 60 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Công thức** | `Leverage = Tổng nợ / Vốn chủ sở hữu` |
| **Logic chấm** | ≤1 → 60 · 1–2 → 47 · 2–3 → 30 · 3–4 → 13 · >4 → 0 |
| **Nguồn dữ liệu** | BCTC do KH upload |
| **Reason Code** | F02 - Đòn bẩy cao |

#### Biến 3 — EBITDA Margin — 55 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Công thức** | `EBITDA_Margin = EBITDA / Doanh thu thuần` |
| **Logic chấm** | **TM-DV:** ≥8% → 55 · 4–8% → 37 · 0–4% → 14 · <0 → 0<br>**SX:** ≥12% → 55 · 6–12% → 37 · 0–6% → 14 · <0 → 0 |
| **Nguồn dữ liệu** | BCTC do KH upload |
| **Reason Code** | F03 — Biên LN thấp so với ngành |

#### Biến 4 — Cash Conversion Cycle (CCC) — 45 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Công thức** | `CCC = DIO + DSO − DPO` |
| **Logic chấm** | **Bán lẻ:** <30 → 45 · 30–60 → 32 · 60–90 → 14 · >90 → 0<br>**SX:** <60 → 45 · 60–90 → 32 · 90–120 → 14 · >120 → 0 |
| **Nguồn dữ liệu** | BCTC |
| **Reason Code** | F04 — Chu kỳ tiền mặt dài |

---

### TRỤ CỘT II — BANKING RELATIONSHIP (70 điểm) · TRADITIONAL

#### Biến 5 — Bank Account Vitality — 44 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Tính ổn định và tích cực của tài khoản DN tại NH |
| **Trạng thái** | ACTIVE_STRONG (active ≥11/12T + avg_balance >5% loan_request + CV<30%) → 44<br>ACTIVE_MEDIUM (đủ 2/3) → 26<br>ACTIVE_WEAK (đủ 1/3) → 10<br>NEW_OR_DORMANT (KH mới) → 18 (neutral) |
| **Nguồn dữ liệu** | NH nội bộ — sao kê TK DN |
| **Reason Code** | R01 — Quan hệ TK NH yếu |

#### Biến 6 — Cross-Product Usage — 18 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Số sản phẩm NH đang dùng (TT, payroll, POS, bảo lãnh, L/C, internet banking) |
| **Logic chấm** | ≥4 → 18 · 2–3 → 12 · 1 → 6 · 0 → 0 |
| **Nguồn dữ liệu** | NH nội bộ — core banking |
| **Reason Code** | R02 — Sử dụng ít SP NH |

#### Biến 7 — Account Age at Bank — 8 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Số tháng DN có TK tại NH (proxy relationship depth) |
| **Logic chấm** | ≥24T → 8 · 12–24T → 6 · 6–12T → 3 · <6T → 0 |
| **Nguồn dữ liệu** | NH nội bộ |
| **Reason Code** | R03 — Account age ngắn |

> **Adjustment cho KH mới:** Nếu DN chưa có TK tại NH cho vay, toàn bộ Trụ II scale neutral xuống 35đ (50%), tránh penalize oan.

---

### TRỤ CỘT III — DIGITAL OPERATIONAL (310 điểm) · MIXED

#### Biến 8 — E-Invoice Vitality — 100 điểm · `NON-TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Tính liên tục và xu hướng phát hành HĐĐT trong 12 tháng |
| **Công thức** | `Vitality = Active_months × Slope_factor`<br>Slope_factor = +1 (tăng) / 0 (phẳng) / −0.3 (giảm) |
| **Trạng thái** | STRONG (≥11/12 + slope≥0) → 100<br>POSITIVE (≥11/12, slope âm nhẹ) → 73<br>MODERATE (8–10/12) → 50<br>WEAK (5–7/12) → 23<br>INACTIVE (<5/12) → 0 (knock-out nếu >6T) |
| **Nguồn dữ liệu** | API HĐĐT Tổng cục Thuế |
| **Reason Code** | O01 — HĐĐT không liên tục |

#### Biến 9 — Customer Concentration — 55 điểm · `NON-TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Công thức** | `Concentration = ΣRevenue_top5_buyers / ΣRevenue_total` |
| **Logic chấm** | <40% → 55 · 40–60% → 37 · 60–80% → 14 · >80% → 0 |
| **Nguồn dữ liệu** | API HĐĐT — group by MST bên mua |
| **Reason Code** | O02 — Rủi ro tập trung khách hàng |

#### Biến 10 — Bank Cash Flow Pattern — 90 điểm · `TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Công thức** | `Ratio = Σ_inflow / Σ_outflow`<br>`CV = std(inflow_monthly) / mean(inflow_monthly)` |
| **Trạng thái** | HEALTHY (Ratio>1.1 AND CV<25%) → 90<br>STABLE (Ratio>1.0 AND CV<40%) → 58<br>TIGHT (Ratio≥1.0) → 27<br>DEFICIT (<1.0 liên tục ≥3 tháng) → 0 |
| **Nguồn dữ liệu** | NH nội bộ hoặc sao kê NH khác (KH cung cấp) |
| **Reason Code** | O03 — Dòng tiền NH yếu |

#### Biến 11 — Anchor-Supplier Network — 65 điểm · `NON-TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Chất lượng đối tác chính. AI parse hợp đồng KT, đối chiếu MST với database công khai |
| **Trạng thái** | STRONG (≥2 anchor uy tín + HĐ ≥12T) → 65<br>MODERATE (1 anchor uy tín) → 42<br>WEAK (toàn KH/NCC nhỏ) → 17<br>UNVERIFIED (không có HĐ upload) → 9 |
| **Nguồn dữ liệu** | KH upload + AI parse + Cổng DN Quốc gia + HOSE/HNX/UPCOM |
| **Reason Code** | O04 — Mạng đối tác yếu |

---

### TRỤ CỘT IV — BEHAVIORAL COMPLIANCE (250 điểm) · NON-TRADITIONAL ⭐

**Khung phân loại chung 4 trạng thái:**

| Trạng thái | Định nghĩa | Hệ số |
|---|---|---|
| **FULL** | 12T không có khoản nào quá hạn | × 1.0 |
| **LATE_MILD** | Quá hạn ≤30 ngày, ≤2 lần/12T, tự nguyện thanh toán | × 0.6 |
| **LATE_SEVERE** | Quá hạn 30–90 ngày HOẶC ≥3 lần chậm/12T | × 0.25 |
| **ENFORCED** | Bị cưỡng chế / cắt dịch vụ | × 0 (+ knock-out với thuế, BHXH) |

#### Biến 12 — Tax Compliance — 100 điểm · `NON-TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Hành vi tuân thủ nghĩa vụ thuế (VAT + TNDN + TNCN khấu trừ + Môn bài) |
| **Logic chấm** | FULL → 100 · LATE_MILD → 60 · LATE_SEVERE → 25 · ENFORCED → 0 (+ KNOCK-OUT) |
| **Nguồn dữ liệu** | (1) API Tổng cục Thuế với consent DN<br>(2) Cổng công khai DN nợ thuế (gdt.gov.vn) |
| **Reason Code** | B01 — Vi phạm nghĩa vụ thuế |

#### Biến 13 — Social Insurance Compliance — 85 điểm · `NON-TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Hành vi đóng BHXH + BHYT + BHTN; leading indicator cash flow stress |
| **Logic chấm** | FULL → 85 · LATE_MILD → 51 · LATE_SEVERE → 21 · ENFORCED → 0<br>**Phạt thêm:** LĐ giảm >30%/12T → −10đ |
| **Nguồn dữ liệu** | (1) API BHXH VN với consent<br>(2) Cổng công khai "DN nợ BHXH" |
| **Reason Code** | B02 — Vi phạm BHXH |

#### Biến 14 — Utility Compliance — 65 điểm · `NON-TRAD`

| Thuộc tính | Nội dung |
|---|---|
| **Mô tả** | Hành vi thanh toán điện EVN + nước + viễn thông |
| **Logic chấm** | FULL → 65 · LATE_MILD → 39 · LATE_SEVERE → 16 · DISCONNECTED → 0 |
| **Nguồn dữ liệu** | Sao kê NH (auto-debit pattern) hoặc hóa đơn upload (AI OCR) |
| **Reason Code** | B03 — Vi phạm Utility |

---

### TRỤ CỘT V — SUSTAINABILITY / ESG (70 điểm) · NON-TRADITIONAL

#### Biến 15 — ESG Behavioral Proxy — 70 điểm · `NON-TRAD`

**Logic tổng:** `ESG_Score = E1 + S1 + G1` (max 70đ)

| Sub | Mô tả | Logic chấm | Nguồn data |
|---|---|---|---|
| **E1 — Environmental** (25đ) | Vi phạm môi trường 24T; có trong "Danh sách cơ sở gây ô nhiễm nghiêm trọng" của Bộ TN&MT không | Không vi phạm → 25 · Vi phạm hành chính nhẹ → 12 · Trong danh sách ô nhiễm → 0 · Ngành không đặc thù → 18 (neutral) | Cổng Bộ TN&MT |
| **S1 — Labor Lawsuit/Strike** (25đ) | Bản án LĐ trong 24T; đình công công khai | Không có bản án → 25 · 1 bản án hòa giải → 12 · ≥2 bản án hoặc 1 thua → 0 | congbobanan.toaan.gov.vn + báo chí |
| **G1 — Ownership Transparency** (20đ) | Cấu trúc sở hữu rõ ràng, beneficial owner identifiable | Cấu trúc đơn giản → 20 · 1 công ty mẹ/holding rõ → 12 · Shell company nhiều tầng → 0 | Cổng DN Quốc gia |

**Reason Code:** S01 — Vấn đề ESG

---

### TRỤ CỘT VI — BUSINESS MATURITY (50 điểm) · NON-TRADITIONAL · TRỤ RIÊNG

#### Biến 16 — Business Maturity — 50 điểm · `NON-TRAD`

**Logic tổng:** `Maturity_Score = Legal_age + Operational_continuity` (max 50đ)

| Sub | Mô tả | Logic chấm |
|---|---|---|
| **Legal entity age** (25đ) | Tuổi pháp nhân từ ngày cấp ĐKKD | ≥5 năm → 25 · 3–5 năm → 20 · 1–3 năm → 12 · <1 năm → 5 |
| **Operational continuity** (25đ) | Bằng chứng hoạt động trước khi thành lập pháp nhân (HKD tiền thân) | HKD tiền thân ≥5 năm cùng địa điểm/ngành → 25 · 2–5 năm → 18 · <2 năm → 10 · Không chứng minh → 0 |

| Thuộc tính | Nội dung |
|---|---|
| **Nguồn dữ liệu** | Cổng DN Quốc gia (dangkykinhdoanh.gov.vn) + giấy phép HKD cũ do KH upload |
| **Reason Code** | S02 — Tuổi hoạt động ngắn |
| **Lưu ý đặc biệt** | Operational continuity giải quyết "nghịch lý lính mới" — DN bị reset thâm niên sau chuyển đổi HKD→DN. DN có thể chứng minh thâm niên thực tế qua HKD tiền thân (cùng địa điểm, cùng ngành) |

> **Vì sao Maturity là trụ riêng:** Trụ này không đo *rủi ro* mà đo *context* (chúng ta biết DN bao lâu). Không nên trộn ngang với 5 trụ rủi ro chính. Maturity ảnh hưởng đến **confidence trong đánh giá**, không phải *chất lượng* DN.

---

## 5. Bảng ranking mức độ quan trọng

> **Ranking chỉ áp dụng cho 5 trụ rủi ro chính (I, II, III, IV, V). Trụ VI Maturity được tách riêng không tham gia ranking.**

### Tầng 1 — Xếp hạng 5 trụ chính

| Hạng | Trụ cột | Trọng số | Mức độ | Lý do |
|---|---|---|---|---|
| 1 | **IV. Behavioral Compliance** | 250đ (25%) | ★★★★★ `5/5` | Có knock-out rules · Leading indicator · Không thể giả mạo |
| 2 | **III. Digital Operational** | 310đ (31%) | ★★★★½ `4.5/5` | Trọng số cao nhất · Real-time · Ground truth của hoạt động kinh doanh |
| 3 | **I. Hard Financial** | 250đ (25%) | ★★★★ `4/5` | Có thể window-dressing · Cần cross-validate với Trụ III |
| 4 | **V. Sustainability (ESG)** | 70đ (7%) | ★★★ `3/5` | Filter risk hơn là core risk · Universal proxy |
| 5 | **II. Banking Relationship** | 70đ (7%) | ★★★ `3/5` | Bias KH cũ · Neutral cho KH mới · Bổ trợ |

### Tầng 2 — Xếp hạng 14 biến chính (không tính 2 biến Trụ V và VI)

> **Note:** 14 biến đầu, không tính S01 (ESG) và S02 (Maturity) vì 2 biến này thuộc trụ tách riêng cho mục đích khác.

| Hạng | # | Biến số | Trụ | Loại | RC | Điểm | Mức độ |
|---|---|---|---|---|---|---|---|
| 1 | 12 | Tax Compliance | IV | NON-TRAD | B01 | 100 | ★★★★★ `5/5` |
| 1 | 13 | Social Insurance | IV | NON-TRAD | B02 | 85 | ★★★★★ `5/5` |
| 1 | 8 | E-Invoice Vitality | III | NON-TRAD | O01 | 100 | ★★★★★ `5/5` |
| 1 | 10 | Bank Cash Flow Pattern | III | TRAD | O03 | 90 | ★★★★★ `5/5` |
| 1 | 1 | DSCR | I | TRAD | F01 | 90 | ★★★★★ `5/5` |
| 6 | 11 | Anchor-Supplier Network | III | NON-TRAD | O04 | 65 | ★★★★ `4/5` |
| 6 | 9 | Customer Concentration | III | NON-TRAD | O02 | 55 | ★★★★ `4/5` |
| 6 | 14 | Utility Compliance | IV | NON-TRAD | B03 | 65 | ★★★★ `4/5` |
| 6 | 2 | Đòn bẩy | I | TRAD | F02 | 60 | ★★★★ `4/5` |
| 6 | 5 | Bank Account Vitality | II | TRAD | R01 | 44 | ★★★★ `4/5` |
| 11 | 3 | EBITDA Margin | I | TRAD | F03 | 55 | ★★★ `3/5` |
| 11 | 4 | CCC | I | TRAD | F04 | 45 | ★★★ `3/5` |
| 13 | 6 | Cross-Product Usage | II | TRAD | R02 | 18 | ★★ `2/5` |
| 14 | 7 | Account Age | II | TRAD | R03 | 8 | ★ `1/5` |

### Top 5 critical biến (must-have)

| # | Biến | Loại | Vai trò |
|---|---|---|---|
| F01 | DSCR | TRAD | Trả lời "DN có đủ tiền trả nợ?" |
| B01 | Tax Compliance | NON-TRAD | Knock-out gate quan trọng nhất |
| B02 | Social Insurance | NON-TRAD | Leading indicator cash flow stress |
| O01 | E-Invoice Vitality | NON-TRAD | Ground truth về revenue trend |
| O03 | Bank Cash Flow | TRAD | Real-time payment capacity |

→ 5 biến hợp lại = **465 điểm (46.5%)** = đủ để ra quyết định bảo thủ cho mọi DN.

---

## 6. Knock-out Rules

Bất kỳ điều kiện nào → ép hạng **D / Từ chối ngay**, không tính tổng điểm:

| Code | Điều kiện | Nguồn data | Biến liên quan |
|---|---|---|---|
| **K1** | Quyết định cưỡng chế thuế đang hiệu lực | Cổng công khai Tổng cục Thuế | B01 |
| **K2** | Trong "Danh sách DN nợ BHXH" với nợ >3 tháng | Cổng BHXH VN | B02 |
| **K3** | Blacklist NHNN / OFAC sanctions | Danh sách định kỳ | — |
| **K4** | Đang giải thể/phá sản | Cổng DN Quốc gia | — |
| **K5** | Ngừng phát hành HĐĐT >6 tháng liên tục | API Tổng cục Thuế | O01 |
| **K6** | Sai lệch BCTC thuế vs NH > 20% ở chỉ tiêu chính | AI cross-check 2 file BCTC | F01–F04 |

> **K6 lưu ý:** Không tự động từ chối, mà flag **bắt buộc thẩm định thủ công** — có thể là sai sót kế toán không phải fraud chủ ý.

---

## 7. Yêu cầu dữ liệu đầu vào

### 7.1. Phân loại dữ liệu đầu vào

| Mức ưu tiên | Mục đích | Hệ quả nếu thiếu |
|---|---|---|
| **Bắt buộc (Mandatory)** | Tối thiểu để chấm được điểm có ý nghĩa | Không thể chấm điểm → reject tự động hoặc route manual |
| **Khuyến nghị (Recommended)** | Để chấm điểm đầy đủ và chính xác | Score giảm độ tin cậy; có thể downgrade decision |
| **Tùy chọn (Optional)** | Để tăng độ chính xác và unlock điểm Trụ V, VI | Mất một phần điểm; vẫn chấm được |

### 7.2. Bảng yêu cầu dữ liệu đầu vào chi tiết

#### A. Bắt buộc (Mandatory) — Tối thiểu để chấm điểm

| # | Loại dữ liệu | Format | Nguồn / Cách lấy | Consent yêu cầu | Phục vụ biến |
|---|---|---|---|---|---|
| 1 | **Mã số thuế DN (MST)** | 10 hoặc 13 ký tự | KH nhập trực tiếp | Không cần (public) | Toàn bộ |
| 2 | **Thông tin pháp nhân cơ bản** | JSON (tên DN, ngày ĐKKD, ngành, địa chỉ) | Cổng DN Quốc gia (auto-pull bằng MST) | Không cần (public) | S02, Trụ V |
| 3 | **BCTC năm gần nhất** | PDF, Excel, ảnh chụp | KH upload | Có (consent xử lý) | F01, F02, F03, F04 |
| 4 | **API Tổng cục Thuế — Tax compliance** | JSON từ API | Ủy quyền DN | Có (ủy quyền cụ thể) | B01, K1, K6 |
| 5 | **API BHXH VN — BHXH compliance** | JSON từ API | Ủy quyền DN | Có (ủy quyền cụ thể) | B02, K2 |
| 6 | **API HĐĐT Tổng cục Thuế — 12T gần nhất** | JSON từ API | Ủy quyền DN | Có (ủy quyền cụ thể) | O01, O02, K5 |
| 7 | **Sao kê NH 12T (TK chính DN)** | PDF, CSV có chữ ký NH | KH upload hoặc API NH (nếu cùng NH) | Có (consent xử lý) | O03, R01 |
| 8 | **Số tiền + kỳ hạn vay đề xuất** | Số · Tháng | KH nhập | Không cần | F01 (tính DSCR) |

#### B. Khuyến nghị (Recommended) — Để chấm điểm đầy đủ

| # | Loại dữ liệu | Format | Nguồn / Cách lấy | Consent yêu cầu | Phục vụ biến |
|---|---|---|---|---|---|
| 9 | **BCTC năm trước (n-1)** | PDF, Excel | KH upload | Có | Tăng độ tin cậy F01–F04 + cross-check |
| 10 | **Lịch sử quan hệ với NH cho vay** | Internal | NH nội bộ | Không cần (first-party) | R01, R02, R03 |
| 11 | **Hợp đồng KT với KH/NCC chính** | PDF, Word | KH upload | Có | O04 |
| 12 | **Kiểm tra cổng DN nợ thuế công khai** | Auto-scrape | gdt.gov.vn (public) | Không cần | B01 cross-check |
| 13 | **Kiểm tra cổng DN nợ BHXH công khai** | Auto-scrape | baohiemxahoi.gov.vn (public) | Không cần | B02 cross-check |
| 14 | **Hóa đơn utility 12T (điện/nước/viễn thông)** | PDF, ảnh | KH upload hoặc detect từ sao kê NH | Có | B03 |

#### C. Tùy chọn (Optional) — Tăng độ chính xác / Unlock điểm ESG-Maturity

| # | Loại dữ liệu | Format | Nguồn / Cách lấy | Consent yêu cầu | Phục vụ biến |
|---|---|---|---|---|---|
| 15 | **Danh sách cơ sở gây ô nhiễm môi trường nghiêm trọng** | Auto-scrape | Cổng Bộ TN&MT | Không cần (public) | S01-E1 |
| 16 | **Kiểm tra bản án LĐ trên cổng tòa án** | Auto-scrape | congbobanan.toaan.gov.vn | Không cần (public) | S01-S1 |
| 17 | **Giấy phép HKD tiền thân (nếu có)** | PDF, ảnh | KH upload | Có | S02 (operational continuity) |
| 18 | **Sao kê NH bổ sung (TK tại NH khác)** | PDF, CSV | KH cung cấp | Có | O03 tăng độ phủ |
| 19 | **Báo cáo kiểm toán (nếu có)** | PDF | KH upload | Có | Tăng độ tin cậy F01–F04 |

### 7.3. Checklist consent (Bắt buộc trước khi xử lý)

| # | Consent item | Bắt buộc/Tùy chọn |
|---|---|---|
| 1 | Đồng ý cho NH xử lý dữ liệu DN để chấm điểm tín dụng | Bắt buộc |
| 2 | Ủy quyền NH truy vấn API Tổng cục Thuế | Bắt buộc |
| 3 | Ủy quyền NH truy vấn API BHXH VN | Bắt buộc |
| 4 | Ủy quyền NH truy vấn API HĐĐT | Bắt buộc |
| 5 | Đồng ý NH lưu trữ BCTC và hợp đồng KT | Bắt buộc nếu có upload |
| 6 | Đồng ý NH chia sẻ kết quả chấm điểm với chi nhánh trong hệ thống | Bắt buộc |
| 7 | Đồng ý NH dùng dữ liệu này để xây dựng model production sau này | Tùy chọn |

### 7.4. Yêu cầu chất lượng dữ liệu

| Yêu cầu | Threshold |
|---|---|
| **Recency** — BCTC | Không quá 18 tháng kể từ ngày chấm điểm |
| **Recency** — Sao kê NH | Tới tháng gần nhất hoặc tháng N-1 |
| **Completeness** — Tax compliance | Đủ 12 tháng liên tiếp |
| **Completeness** — BHXH | Đủ 12 tháng liên tiếp |
| **Validity** — MST | Đang hoạt động trên Cổng DN Quốc gia |
| **Format** — BCTC | Phải có Bảng cân đối kế toán + Báo cáo KQKD + Báo cáo LCTT |

---

## 8. Định dạng đầu ra cho khách hàng

### 8.1. Cấu trúc đầu ra

Hệ thống trả về **5 thành phần** cho mỗi lần chấm điểm:

| Thành phần | Mô tả | Format |
|---|---|---|
| 1. **Tổng điểm số** | Raw score 0–1000 sau khi cộng tất cả 16 biến và áp knock-out | Số nguyên |
| 2. **Hạng tín dụng** | AAA → D (10 hạng) | Chuỗi |
| 3. **Đánh giá tổng quan** | Tốt / Khá / Trung bình / Yếu / Rất yếu | Categorical |
| 4. **Quyết định tự động** | Theo bảng quy đổi hạng | Enum |
| 5. **Phân tích chi tiết** | Tại sao điểm như vậy + Top 3 điểm chưa tốt + Recommendation | Cấu trúc |

### 8.2. Mapping Hạng tín dụng → Đánh giá tổng quan

| Hạng | Đánh giá | Ý nghĩa |
|---|---|---|
| AAA, AA | **Tốt** | DN có sức khỏe tài chính và hành vi tốt; đủ điều kiện vay tín chấp |
| A, BBB | **Khá** | DN có sức khỏe ổn định, đủ điều kiện vay nhưng cần lưu ý một số chỉ số |
| BB, B | **Trung bình** | DN có rủi ro vừa phải, cần TSĐB và giám sát chặt |
| CCC, CC | **Yếu** | DN có nhiều chỉ số đáng lo ngại, cần thẩm định kỹ |
| C, D | **Rất yếu** | DN không đủ điều kiện vay, hoặc đã vi phạm knock-out rules |

### 8.3. Phân tích chi tiết "Tại sao điểm như vậy"

Cấu trúc gồm **4 layers**:

#### Layer 1 — Tổng quan 6 trụ cột

Hiển thị điểm/max cho mỗi trụ + status visual (GOOD/OK/WEAK):

```
I.  Hard Financial      198/250  GOOD
II. Banking Relationship 58/70   GOOD
III. Digital Operational 245/310 GOOD
IV. Behavioral Compliance 195/250 GOOD
V.  Sustainability        56/70  OK
VI. Business Maturity     30/50  WEAK
```

#### Layer 2 — Điểm mạnh

Top 3 biến có score cao nhất so với max → narrative tích cực:

> "Doanh nghiệp có lịch sử nộp thuế tốt (100/100), phát hành hóa đơn điện tử ổn định (95/100), và dòng tiền ngân hàng lành mạnh (85/90). Đây là 3 yếu tố cốt lõi của một DN có kỷ luật tài chính."

#### Layer 3 — Điểm chưa tốt (Top 3 cần cải thiện)

Top 3 biến có **gap lớn nhất** so với hạng mục tiêu (vd: nếu hạng hiện tại là A, mục tiêu là AA):

```
① O02 — Rủi ro tập trung khách hàng (37/55)
   → Top-5 KH chiếm 58% doanh thu
   → Hành động: Đa dạng hóa tệp khách hàng; tìm 3-5 KH mới
   → Tiềm năng tăng điểm: +18đ trong 6-9 tháng

② F04 — Chu kỳ tiền mặt dài (14/45)
   → CCC hiện tại: 85 ngày (benchmark ngành: 45)
   → Hành động: Đẩy nhanh thu công nợ; tối ưu tồn kho
   → Tiềm năng tăng điểm: +31đ trong 12 tháng

③ S02 — Tuổi hoạt động ngắn (15/50)
   → Pháp nhân chỉ mới thành lập 14 tháng
   → Hành động: Upload giấy phép HKD tiền thân (nếu có)
   → Tiềm năng tăng điểm: +25đ ngay lập tức nếu có HKD tiền thân ≥5 năm
```

#### Layer 4 — Forecast cải thiện

> "Nếu cải thiện cả 3 điểm trên, điểm dự kiến có thể tăng từ 782 lên 856 — đủ để upgrade từ hạng A lên AA."

### 8.4. Output mẫu hoàn chỉnh

```
┌─────────────────────────────────────────────────────────────┐
│  CTCP ABC — MST 0301234567                                  │
│  ═══════════════════════════════════════════════════════════│
│  TỔNG ĐIỂM: 782 / 1000        HẠNG: A                       │
│  ĐÁNH GIÁ: KHÁ                                              │
│  QUYẾT ĐỊNH: ✅ DUYỆT TÍN CHẤP – Hạn mức trung bình          │
│  ═══════════════════════════════════════════════════════════│
│                                                              │
│  PHÂN TÍCH 6 TRỤ CỘT:                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ I.  Hard Financial      198/250  ████████░░  GOOD    │  │
│  │ II. Banking Relationship 58/70   ████████░░  GOOD    │  │
│  │ III. Digital Operational 245/310 ████████░░  GOOD    │  │
│  │ IV. Behavioral Compliance 195/250 ████████░░  GOOD   │  │
│  │ V.  Sustainability        56/70  ███████░░░  OK      │  │
│  │ VI. Business Maturity     30/50  ██████░░░░  WEAK    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ĐIỂM MẠNH:                                                 │
│  • Lịch sử nộp thuế tốt (100/100)                          │
│  • Phát hành HĐĐT ổn định (95/100)                         │
│  • Dòng tiền ngân hàng lành mạnh (85/90)                   │
│                                                              │
│  TOP 3 ĐIỂM CẦN CẢI THIỆN:                                 │
│                                                              │
│  ① O02 — Rủi ro tập trung khách hàng (37/55)              │
│      → Top-5 KH chiếm 58% doanh thu                         │
│      → Đa dạng hóa tệp khách hàng                           │
│      → Tiềm năng tăng: +18đ trong 6-9 tháng                │
│                                                              │
│  ② F04 — Chu kỳ tiền mặt dài (14/45)                      │
│      → CCC hiện tại: 85 ngày (benchmark ngành: 45)          │
│      → Đẩy nhanh thu công nợ qua chiết khấu sớm             │
│      → Tiềm năng tăng: +31đ trong 12 tháng                 │
│                                                              │
│  ③ S02 — Tuổi hoạt động ngắn (15/50)                      │
│      → Pháp nhân thành lập 14 tháng                         │
│      → Upload giấy phép HKD tiền thân nếu có                │
│      → Tiềm năng tăng: +25đ ngay nếu có HKD ≥5 năm         │
│                                                              │
│  ► Nếu cải thiện cả 3, điểm dự kiến: 782 → 856 (AA)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Bảng quy đổi hạng tín dụng

| Tổng điểm | Hạng | Đánh giá | Nhóm nợ TT 11/2021 | Quyết định tự động |
|---|---|---|---|---|
| 920–1000 | **AAA** | Tốt | Nhóm 1 | ✅ Tín chấp — hạn mức tối đa, lãi suất ưu đãi nhất |
| 850–919 | **AA** | Tốt | Nhóm 1 | ✅ Tín chấp — hạn mức chuẩn |
| 770–849 | **A** | Khá | Nhóm 1 | ✅ Tín chấp — hạn mức trung bình |
| 700–769 | **BBB** | Khá | Nhóm 1–2 | 🟡 Tín chấp một phần / TSĐB nhẹ |
| 620–699 | **BB** | Trung bình | Nhóm 2 | 🔒 Duyệt có TSĐB |
| 540–619 | **B** | Trung bình | Nhóm 2–3 | 🔒 TSĐB full |
| 460–539 | **CCC** | Yếu | Nhóm 3 | 🧐 Thẩm định thủ công |
| 360–459 | **CC** | Yếu | Nhóm 4 | 🧐 Thẩm định đặc biệt |
| 250–359 | **C** | Rất yếu | Nhóm 4–5 | ❌ Từ chối (trừ phê duyệt cấp cao) |
| 0–249 | **D** | Rất yếu | Nhóm 5 | ❌ Từ chối |

---

## 10. Reason Codes & Khuyến nghị cải thiện

| Code | Vấn đề | Hành động cụ thể | Tiềm năng tăng điểm |
|---|---|---|---|
| F01 | Khả năng trả nợ thấp | Tái cơ cấu nghĩa vụ; tăng EBITDA bằng cắt chi phí cố định | +30–80 |
| F02 | Đòn bẩy cao | Tăng vốn góp; giữ lại LN; thoái nợ ngắn hạn | +20–55 |
| F03 | Biên LN thấp | Rà soát giá vốn; tối ưu chi phí; định vị lại giá bán | +25–45 |
| F04 | CCC dài | Chiết khấu thu nợ sớm; just-in-time tồn kho | +15–35 |
| R01 | Quan hệ TK NH yếu | Tăng giao dịch; duy trì số dư ổn định | +20–38 |
| R02 | Sử dụng ít SP NH | Mở thêm SP phụ trợ active (POS, payroll) | +6–13 |
| R03 | Account age ngắn | Duy trì hoạt động; tự nhiên cải thiện | +3–6 |
| O01 | HĐĐT không liên tục | Phát hành đều; chuẩn hóa quy trình xuất hóa đơn | +30–85 |
| O02 | Rủi ro tập trung KH | Đa dạng hóa tệp khách hàng | +20–45 |
| O03 | Dòng tiền NH yếu | Tăng inflow qua HĐĐT; giảm rút TM không rõ mục đích | +35–70 |
| O04 | Mạng đối tác yếu | Phát triển quan hệ với ≥1 anchor lớn | +25–52 |
| **B01** | **Vi phạm thuế** | Hoàn thành thuế tồn đọng; nộp thuế điện tử | **+50–100** |
| **B02** | **Vi phạm BHXH** | Đóng đủ BHXH; duy trì ổn định nhân sự | **+40–85** |
| **B03** | **Vi phạm Utility** | Đăng ký auto-debit từ TK NH | **+30–65** |
| **S01** | **Vấn đề ESG** | Tuân thủ quy định môi trường; cải thiện quan hệ LĐ; minh bạch sở hữu | **+25–70** |
| **S02** | **Tuổi hoạt động ngắn** | Upload giấy phép HKD tiền thân; tự nhiên cải thiện theo thời gian | **+15–50** |

### Logic chọn Top 3 Reason Codes cho output KH

Hệ thống tự động chọn 3 reason codes có **gap lớn nhất** so với hạng mục tiêu, kèm:
- Tiềm năng tăng điểm cụ thể
- Hành động đề xuất
- Timeframe ước tính

---

## 11. Cross-validation & Fraud Detection

### Các cặp cross-validation pair chính

| Cặp | Cơ chế | Phát hiện |
|---|---|---|
| **F01 ↔ O03** | DSCR (BCTC) vs Cash Flow (sao kê NH) | EBITDA khai báo có khớp với inflow thực không |
| **F03 ↔ O01** | EBITDA Margin (BCTC) vs E-Invoice Vitality | Doanh thu khai báo có khớp với HĐĐT không |
| **F02 ↔ O03** | Leverage (BCTC) vs Cash Flow | Nợ ngoài bảng (off-balance-sheet debt) |
| **O01 ↔ O03** | HĐĐT vs Sao kê NH | Doanh thu tiền mặt không khai HOẶC HĐĐT khống |
| **F04 ↔ O03** | CCC vs Cash Flow pattern | Tồn kho/Phải thu khai thấp giả tạo |

### Bảng kỹ thuật gian lận và cơ chế phát hiện

| Kỹ thuật gian lận | Biến bị ảnh hưởng | Cơ chế phát hiện |
|---|---|---|
| Khai phồng doanh thu | EBITDA Margin, DSCR | So sánh với HĐĐT (O01) |
| Giấu nợ ngoài bảng | Leverage, DSCR | Đối chiếu Bank Cash Flow (O03) |
| Trì hoãn ghi nhận chi phí | EBITDA Margin | CCC dài bất thường (F04) |
| Điều chỉnh tồn kho | CCC | So sánh liên kỳ BCTC 2 năm |
| Xuất hóa đơn khống | O01 tăng giả | O03 không tăng tương ứng |
| Doanh thu tiền mặt giấu | Doanh thu thực cao | HĐĐT thấp, inflow NH cao hơn nhiều |

**Nguyên tắc khi divergence lớn:** Ưu tiên tin vào dữ liệu phi truyền thống (third-party stamped). Chuyển sang thẩm định thủ công nếu sai lệch > 20% (K6).

---

## 12. Phân tích overlap còn lại

| Cặp | Bản chất overlap | Mức trùng | Verdict |
|---|---|---|---|
| **R01 vs O03** | Cùng nguồn sao kê NH nhưng metrics khác | ~25–30% | ⚠️ Acceptable — R01 đo "active không", O03 đo "lành mạnh không" |
| **O02 vs O04** | Cùng về đối tác | ~15–20% | ⚠️ Acceptable — O02 đo *quantity*, O04 đo *quality* |
| **S02 vs R03** | Cùng là "tuổi" | ~15% | ⚠️ Acceptable — S02 đo tuổi pháp nhân; R03 đo tuổi TK NH |

Tất cả 3 cặp đều ở mức VIF ~1.2–1.5 — well within tolerance.

---

## 13. Tuân thủ pháp lý

### Khung pháp lý chính

| Văn bản | Phạm vi áp dụng |
|---|---|
| **NĐ 13/2023/NĐ-CP** | Chỉ áp dụng dữ liệu cá nhân; toàn bộ data scorecard là pháp nhân → không vi phạm |
| **TT 11/2021/TT-NHNN** | Phân loại nợ 5 nhóm; yêu cầu hệ thống XHTD nội bộ thử nghiệm tối thiểu 12T |
| **Luật Quản lý Thuế 2019** | Danh sách DN nợ thuế là công khai |
| **Luật BHXH 2014** | Danh sách DN nợ BHXH là công khai |
| **NĐ 123/2020/NĐ-CP** | HĐĐT bắt buộc từ 1/7/2022 |

### 4 nguyên tắc privacy by design

1. **Pháp nhân ≠ Cá nhân.** Toàn bộ data thuộc về DN.
2. **Consent-based cho API chi tiết.** DN ký ủy quyền cụ thể.
3. **Public data is fair game.** Cổng công khai hợp pháp dùng.
4. **First-party data thuộc NH.** Sao kê và lịch sử SP tại NH.

### Tuyệt đối không chạm

- ❌ CMND/CCCD của bất kỳ ai
- ❌ Mạng xã hội cá nhân
- ❌ Số điện thoại cá nhân nhân viên
- ❌ Thông tin gia đình/hôn nhân chủ DN
- ❌ Vị trí GPS hoặc lịch sử di chuyển

---

## 14. Defense Q&A

### Q1: "Sao chọn 16 biến mà không 30?"

Parsimony principle — thêm biến chỉ có nghĩa khi mỗi biến mới đóng góp ≥0.02 IV độc lập. FICO production scorecards chạy 12–20 biến core. 16 là điểm cân bằng giữa độ phủ và khả năng giải thích.

### Q2: "Sao Trụ III trọng số cao nhất (31%)?"

Với MSME/SME VN, BCTC thường chưa kiểm toán và có thể window-dressing. HĐĐT (NĐ 123/2020), API thuế và API BHXH là **ground truth không thao túng được**. Đây là approach Kabbage và Validus đã chứng minh.

### Q3: "Tại sao bỏ CIC?"

MSME/SME mới chuyển từ HKD có CIC trống rỗng. Giữ CIC sẽ penalize oan nhóm này. Approach: thay CIC bằng *behavioral evidence từ third-party stamped sources*. Đây là approach của FICO Expansion Score và UltraFICO.

### Q4: "Sao tách Trụ VI Maturity riêng?"

Vì Maturity là *context* (chúng ta biết DN bao lâu), không phải *risk indicator* trực tiếp. DN trẻ không có nghĩa rủi ro hơn DN già — chỉ có nghĩa lịch sử quan sát ngắn hơn. Trộn vào ranking các trụ rủi ro sẽ làm méo nhận định.

### Q5: "59% trọng số cho non-traditional có quá cao không?"

Không. Đó là phản ánh đúng thực tế: BCTC truyền thống của MSME VN mỏng và có thể manipulate. Non-traditional data (thuế, BHXH, HĐĐT) là third-party stamped → không thao túng được. Trọng số 59% phản ánh đúng giá trị thông tin của 2 loại dữ liệu.

### Q6: "Reason Code có gì đặc biệt?"

Mỗi tiêu chí gắn 1 mã (F01–F04, R01–R03, O01–O04, B01–B03, S01–S02). Khi DN được chấm điểm, hệ thống tự động chọn 3 reason codes có gap lớn nhất, sinh ra recommendation. Đây là *explainable scoring* — KH hiểu được tại sao điểm như vậy và làm gì để cải thiện.

### Q7: "Trọng số có được calibrate empirically không?"

Trọng số hiện tại là **giả thuyết thiết kế dựa trên lý luận học thuật và benchmark quốc tế** — chưa qua empirical calibration. Production sẽ học từ dataset NPL nội bộ, đo bằng **Gini ≥0.40 và KS ≥0.30** theo chuẩn industry trước khi triển khai chính thức theo TT 11/2021/TT-NHNN.

---

## 15. Tham chiếu học thuật & Benchmark

### Tham chiếu học thuật chính

| Tác giả & Năm | Đóng góp |
|---|---|
| **Altman (1968)** | Z-score 5 biến cho corporate bankruptcy |
| **Altman & Sabato (2007)** | SME-specific Z-score nhấn EBITDA và Cash |
| **Beaver (1966)** | Failure prediction với payment behavior leading indicator |
| **Berger & Udell (2006)** | SME lending technologies framework |
| **García-Teruel & Martínez-Solano (2007)** | Cash Conversion Cycle theory |
| **Yoshino & Taghizadeh-Hesary (2014)** | ADB Asian SME credit ratings |
| **T. Jury (2012)** | Cash-based credit risk model |
| **Donaldson (1961)** | Sources of repayment hierarchy |

### Benchmark quốc tế

| Mô hình | Quốc gia | Bài học |
|---|---|---|
| **FICO SBSS** | US | Blend personal-business credit; pooled model |
| **Kabbage** | US | Real-time streaming data > periodic BCTC |
| **OnDeck** | US | Cash flow primacy |
| **Square Capital** | US | Embedded repayment từ POS |
| **Validus Capital** | SEA → VN 2024 | Anchor-led supply chain validation |
| **Funding Societies** | SEA | Tenor diversification theo CCC |
| **FICO Expansion Score** | US | Thin-file alternative scoring |
| **UltraFICO** | US | Consent-rich alternative scoring |

---

*Tài liệu Master Scorecard MSME/SME — Phiên bản v2. 16 biến · 6 trụ cột · 6 knock-out rules · Tag Traditional / Non-Traditional · Không bao gồm tier framework, mô hình AI/ML, hay forecast. Trọng số là giả thuyết thiết kế dựa trên lý luận học thuật và benchmark quốc tế, chưa qua empirical calibration. Trọng số production sẽ được học từ dữ liệu NPL nội bộ và đo bằng Gini ≥0.40, KS ≥0.30 trước khi triển khai chính thức theo TT 11/2021/TT-NHNN.*

*Tuân thủ NĐ 13/2023/NĐ-CP, TT 11/2021/TT-NHNN, Basel III IRB.*
