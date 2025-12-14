# ⚡ CES Operation Optimization (구역전기사업 최적 운전 예측 솔루션)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Data Analysis](https://img.shields.io/badge/Focus-Forecasting%20%26%20Simulation-green) ![Status](https://img.shields.io/badge/Status-Refined-orange)

## 📖 Project Overview
본 프로젝트는 **구역전기사업자(CES, Community Energy System)**의 핵심 딜레마인 **'Make or Buy (자가발전 vs 수전)'** 의사결정을 **미래 예측 기반**으로 최적화하기 위해 개발되었습니다.

과거 시장 데이터와 외부 거시경제 지표(환율, 유가, 천연가스)를 실시간으로 학습하고, 사용자가 입력한 **현재 설비 상태**와 **비즈니스 시나리오(연료 계약 방식 등)**에 따라 시뮬레이션을 시작하여, **수익성(Profitability)**과 **안정성(Reliability)**을 동시에 고려한 **미래 30일간의 최적 운전 스케줄링 및 정비 시점**을 예측하고 제안합니다.

---

## 🎯 Business Context & Problem Solving
### 1. The Challenge: CES Business Model
SK멀티유틸리티와 같은 구역전기사업자는 일반 발전사업자와 달리 **두 가지 공급 옵션**을 가집니다.
*   **Option A (Make):** LNG를 연료로 직접 전기를 생산하여 공급 (이익 = SMP - 발전변동비)
*   **Option B (Buy):** 한전(KPX)으로부터 전기를 매입하여 공급 (비용 = SMP)

### 2. The Solution: AI-Driven Future O&M Strategy
미래의 수익을 극대화하기 위해서는 단순한 '고장 방지'를 넘어, **미래 시장 상황 예측에 따른 전략적 정비(Strategic Maintenance)**가 필요합니다.
*   **미래 고마진 구간 예측:** 설비 리스크가 다소 있더라도, 시장 예측을 통해 고마진이 예상되는 구간에는 가동을 유지하여 전력 판매 수익 극대화.
*   **미래 역마진 구간 예측:** 자가발전이 손해인 구간이 예상되므로, 이때를 **'Golden Time'**으로 삼아 예방 정비를 수행하고 전력은 수전(Buy)으로 대체.

---

## 📊 Key Analysis Logic (미래 30일 예측)
본 프로젝트는 Python을 활용해 **실시간 외부 데이터**와 **기술 분석**, **시나리오 기반 시뮬레이션**을 통합하여 미래를 예측합니다.

### Phase 1. External Data Acquisition & Market Forecasting (외부 데이터 및 시장 예측)
*   **Data Source:**
    *   전력통계정보시스템(EPSIS) 시간별 SMP (과거 학습용)
    *   **`yfinance`**: 환율(KRW=X), 국제 유가(WTI: CL=F), 천연가스(NG=F) 데이터를 실시간으로 수집하여 학습에 활용.
*   **Model:** `RandomForestRegressor` (Scikit-learn)
*   **Process:** 과거 SMP 데이터와 수집된 거시경제 지표, 계절(월) 데이터를 학습하여 **향후 30일간의 SMP**를 예측합니다.

### Phase 2. Cost & Margin Simulation (발전비용 및 마진 시뮬레이션)
*   **Business Scenarios:** `main_integrated.py` 상단에서 연료 조달 전략(`LNG_PROCUREMENT_STRATEGY`) 및 단가(`LNG_FIXED_CONTRACT_PRICE_USD`), 탄소배출권 가격(`CARBON_CREDIT_PRICE_KRW_PER_TON`) 등 주요 비용 변수를 직접 설정 가능.
*   **Cost Calculation:**
    *   **연료비 (Fuel Cost):** 설정된 조달 전략('FIXED' 또는 'SPOT')에 따라 예측 환율과 결합하여 MWh당 연료비를 계산.
    *   **환경비 (Carbon Cost):** 설정된 탄소배출권 가격과 배출계수를 기반으로 MWh당 탄소배출권 비용을 계산.
*   **Margin Calculation:** **`최종 마진 = 예측 SMP - 예측 연료비 - 예측 환경비`** 공식을 통해 자가발전 시의 최종적인 MWh당 수익성을 도출.

### Phase 3. Predictive Maintenance (설비 고장 확률 예측)
*   **Data Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
*   **Model:** `RandomForestClassifier` (Scikit-learn)
*   **Process:** 과거 센서 데이터를 기반으로 설비 고장 모델을 학습하고, 사용자가 입력한 현재 설비 상태(`USER_INPUT_CURRENT_STATE`)에서 시뮬레이션을 시작하여 **향후 30일간의 설비 고장 확률(Failure Probability)**을 예측합니다.

### Phase 4. Integrated Decision & Recommendation (통합 의사결정 및 추천)
*   **Output:** 예측된 최종 마진(Bar Chart), 설비의 예측 위험도(Line Chart), 그리고 상세한 **텍스트 기반 예측 보고서**를 함께 제공합니다.
*   **Recommendation:** 예측된 최종 마진이 음수이면서 설비 위험도가 높은 구간을 찾아, 구체적인 정비 권장 날짜를 시각화 및 텍스트로 제시합니다.

---

## 📈 Dashboard Preview (미래 30일 예측 대시보드)
*(AI-driven 예측 로직으로 생성된 시뮬레이션 결과입니다)*

![Future Prediction Dashboard](./results/future_prediction_dashboard.png)

> **[Dashboard 해석]**
> *   **🟦/🟥 막대 그래프 (Forecasted Margin):** 미래의 예측 **최종 마진**을 나타냅니다. 파란색은 이익, 빨간색은 손실 구간을 의미합니다. (SMP - 연료비 - 탄소비)
> *   **📈 빨간색 선 그래프 (Forecasted Risk):** 미래의 예측 설비 고장 위험도를 나타냅니다.
> *   **💡 텍스트 보고서 (콘솔 출력):** 그래프와 함께, 일자별 상세 예측 수치와 종합적인 정비 권장 사항이 콘솔에 텍스트로 출력됩니다.

---

## 🛠 Tech Stack & Environment
*   **Language:** Python 3.9+
*   **Libraries:**
    *   `Pandas`, `NumPy`: 데이터 처리 및 분석
    *   `Scikit-learn`: 머신러닝 모델링 (`RandomForestRegressor`, `RandomForestClassifier`)
    *   `yfinance`: 실시간 금융/원자재 데이터 수집
    *   `Matplotlib`: 데이터 시각화

---

## 🚀 How to Run

```bash
# 1. 가상 환경 설정 (Windows, 최초 1회)
python -m venv venv
venv\Scripts\activate

# 2. 필요 라이브러리 설치
pip install -r requirements.txt

# 3. (선택) 비즈니스 시나리오 및 설비 상태 입력
# main_integrated.py 파일 상단의 변수 값을 직접 수정하여 분석 시나리오를 변경할 수 있습니다.
# 예: LNG_PROCUREMENT_STRATEGY = 'SPOT'
# 예: LNG_FIXED_CONTRACT_PRICE_USD = 12.0
# 예: USER_INPUT_CURRENT_STATE = {'Tool wear [min]': 200.0, ...}

# 4. 통합 미래 예측 분석 실행
# yfinance를 통해 실제 데이터를 가져오므로 인터넷 연결이 필요합니다.
python main_integrated.py

# 5. 결과 확인
# - 터미널에 출력되는 텍스트 예측 보고서 확인
# - results/ 폴더 내 생성된 'future_prediction_dashboard.png' 이미지 확인

# 6. 가상 환경 비활성화
deactivate
```

---
## ✨ Notable Improvements
*   **수익성 분석 고도화:** 과거 프로젝트에서는 외부 API의 불안정성으로 인해 수익성 분석에 한계가 있었습니다. 본 버전에서는 이를 개선하여, **'고정가/현물가' 등 LNG 조달 전략을 사용자가 직접 선택**하고, **탄소배출권 비용까지 변수에 포함**시키는 시뮬레이션 기반 분석으로 고도화했습니다. 이를 통해 보다 현실적이고 깊이 있는 'Make or Buy' 의사결정 지원이 가능해졌습니다. 이는 실제 **연료 도입 계약 및 발전소 운영 제도**에 대한 이해도를 반영한 결과물입니다.
