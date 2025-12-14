
-----

### 📋 README.md (최종 수정본)

````markdown
# ⚡ CES Operation Optimization (구역전기사업 최적 운전 솔루션)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Data Analysis](https://img.shields.io/badge/Focus-Spark%20Spread%20%26%20PdM-green) ![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 📖 Project Overview
본 프로젝트는 **구역전기사업자(CES, Community Energy System)**의 핵심 딜레마인 **'Make or Buy (자가발전 vs 수전)'** 의사결정을 최적화하기 위해 개발되었습니다.

전력 시장의 가격 데이터(SMP, 연료비)와 발전 설비의 상태 데이터(Sensor Data)를 통합 분석하여, **수익성(Profitability)**과 **안정성(Reliability)**을 동시에 고려한 최적의 운전 스케줄링(Operating Mode)을 제안합니다.

---

## 🎯 Business Context & Problem Solving
### 1. The Challenge: CES Business Model
SK멀티유틸리티와 같은 구역전기사업자는 일반 발전사업자와 달리 **두 가지 공급 옵션**을 가집니다.
* **Option A (Make):** LNG를 연료로 직접 전기를 생산하여 공급 (이익 = SMP - 연료비)
* **Option B (Buy):** 한전(KPX)으로부터 전기를 매입하여 공급 (비용 = SMP)

### 2. The Solution: Data-Driven O&M Strategy
수익을 극대화하기 위해서는 단순한 '고장 방지'를 넘어, **시장 상황에 따른 전략적 정비(Strategic Maintenance)**가 필요합니다.
* **High Spark Spread (고마진 구간):** 설비 리스크가 다소 있더라도, 모니터링을 강화하며 가동을 유지하여 전력 판매 수익 극대화.
* **Negative Spark Spread (역마진 구간):** 자가발전이 손해인 구간이므로, 이때를 **'Golden Time'**으로 삼아 예방 정비를 수행하고 전력은 수전(Buy)으로 대체.

---

## 📊 Key Analysis Logic
본 프로젝트는 Python을 활용해 **Financial Data**와 **Technical Data**를 하나의 대시보드로 통합했습니다.

### Phase 1. Market Profitability Analysis (재무 분석)
* **Data Source:** 전력통계정보시스템(EPSIS) 시간별 SMP 및 연료원별 정산단가
* **Metric:** `Spark Spread = SMP - (LNG Cost × Heat Rate)`
* **Insight:** 발전기 가동 시의 실시간 마진(Profit)과 손실(Loss) 구간을 시계열로 식별.

### Phase 2. Predictive Maintenance (기술 분석)
* **Data Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
* **Model:** Random Forest Classifier (Scikit-learn)
* **Insight:** 공정 온도, 회전수, 토크 등의 센서 데이터를 기반으로 설비의 **고장 확률(Failure Probability)**을 실시간 예측.

### Phase 3. Integrated Decision Dashboard (통합 의사결정)
* **Output:** 시장의 수익성(Bar Chart)과 설비의 위험도(Line Chart)를 이중축으로 시각화하여 **'최적 정비 구간'** 도출.

---

## 📈 Dashboard Preview
*(본 레포지토리의 코드로 생성된 시뮬레이션 결과입니다)*

![Dashboard](./results/integrated_dashboard.png)

> **[Dashboard 해석]**
> * **🟦 Blue Bars (Profit Zone):** 마진이 확보되는 구간 → **Max Operation (전력 생산 집중)**
> * **🟥 Red Bars (Loss Zone):** 연료비가 더 비싼 구간 → **Stop & Buy (한전 수전 전환)**
> * **📈 Red Line (Risk Trend):** 설비 고장 위험도 곡선
> * **💡 Insight:** 설비 위험도(Line)가 높아지는 시점이 **Red Bars(역마진)** 구간과 겹칠 때가 기회비용을 최소화하는 **최적의 정비 타이밍**입니다.

---

## 🛠 Tech Stack & Environment
* **Language:** Python 3.9
* **Libraries:**
    * `Pandas`, `NumPy`: 대용량 시계열 데이터 전처리
    * `Scikit-learn`: Random Forest 기반 고장 예측 모델링
    * `Matplotlib`, `Seaborn`: Dual Axis(이중축) 차트 및 데이터 시각화

---

## 📂 Directory Structure
```bash
CES-Operation-Optimization/
├── data/
│   ├── hourly_smp.csv       # EPSIS 전력 판매 가격 데이터
│   ├── fuel_cost.csv        # EPSIS LNG 연료비 데이터
│   └── ai4i2020.csv         # 설비 센서 데이터 (Predictive Maintenance)
├── results/
│   └── integrated_dashboard.png  # 최종 분석 결과 이미지
├── main_integrated.py       # 통합 분석 및 시각화 실행 코드
├── requirements.txt         # 필요 라이브러리 목록
└── README.md                # 프로젝트 설명서
````

-----

## 🚀 How to Run

```bash
# 1. 가상 환경 설정 (최초 1회)
python3 -m venv venv
source venv/bin/activate

# 2. 라이브러리 설치
pip install -r requirements.txt

# 3. 통합 분석 실행
python main_integrated.py

# 4. 결과 확인
# results 폴더 내 생성된 그래프 이미지 확인

# 5. 가상 환경 비활성화
deactivate
```

-----

## 💡 Conclusion (For Recruiter)

이 프로젝트는 **SK멀티유틸리티의 사업관리 직무**에 필수적인 **'손익 기반의 운전 최적화'** 역량을 증명하기 위해 기획되었습니다.

단순한 데이터 분석을 넘어, \*\*시장 상황(Market Condition)\*\*과 \*\*설비 상태(Asset Health)\*\*를 종합적으로 고려하여 사업의 이익을 극대화하는 **엔지니어링 기반의 의사결정 모델**입니다.

```
```