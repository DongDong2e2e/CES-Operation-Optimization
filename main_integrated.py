import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Configuration Constants ---
# File Paths
SMP_FILE = 'data/HOME_전력거래_계통한계가격_시간별SMP.csv'
FUEL_FILE = 'data/HOME_전력거래_정산단가_연료원별.csv'
PDM_FILE = 'data/ai4i2020.csv'
OUTPUT_IMAGE_PATH = 'results/integrated_dashboard.png'

# Market Data Columns
SMP_DATE_COL = '기간'
SMP_VALUE_COL = '가중평균'
FUEL_DATE_COL = '기간'
FUEL_LNG_COL = 'LNG'
SCENARIO_DATE_COL = 'Date'
SCENARIO_SMP_COL = 'SMP'
SCENARIO_MONTH_COL = 'Month'
SCENARIO_SPARK_SPREAD_COL = 'Spark_Spread'
SCENARIO_FAILURE_PROB_COL = 'Failure_Prob'

# Predictive Maintenance Model Parameters
PDM_FEATURES = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
PDM_TARGET = 'Machine failure'
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
SIMULATION_YEAR = 2025 # 시뮬레이션 대상 연도

# Plotting Parameters
PLOT_FIGURE_SIZE = (14, 8)
COLOR_PROFIT = '#4ECDC4' # 수익 구간 (청록색)
COLOR_LOSS = '#FF6B6B'   # 손실 구간 (빨간색)
LINE_RISK_COLOR = 'red'
PLOT_TITLE = 'SKMU Smart O&M Strategy: Balancing Profit vs Risk'
X_LABEL = 'Date (2025 Simulation)'
Y1_LABEL = 'Spark Spread (KRW/kWh)'
Y2_LABEL = 'Machine Failure Risk (%)'
LEGEND_PROFIT = 'High Margin (Max Operation)'
LEGEND_LOSS = 'Low Margin (Maintenance Window)'
LEGEND_RISK = 'Failure Risk Trend'


# 한글 폰트 설정
if sys.platform == 'darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
elif sys.platform == 'win32': # Windows
    plt.rcParams['font.family'] = 'Malgun Gothic'
else: # Linux
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
    except:
        plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_market_data(smp_file_path, fuel_file_path):
    print(">>> 1. 시장 데이터(Market Data) 로드 및 수익성 계산 중...")
    # 1. SMP 로드
    try:
        df_smp = pd.read_csv(smp_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_smp = pd.read_csv(smp_file_path, encoding='cp949')
    df_smp[SMP_DATE_COL] = pd.to_datetime(df_smp[SMP_DATE_COL], format='%Y/%m/%d')
    # 일별 데이터로 변환
    df_market = df_smp.sort_values(SMP_DATE_COL)[[SMP_DATE_COL, SMP_VALUE_COL]].copy()
    df_market.columns = [SCENARIO_DATE_COL, SCENARIO_SMP_COL]
    
    # 2. 연료비 로드 (월별 -> 일별 매핑)
    try:
        df_fuel = pd.read_csv(fuel_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_fuel = pd.read_csv(fuel_file_path, encoding='cp949')
    df_fuel = df_fuel.iloc[1:].copy() # 헤더 정리
    df_fuel[FUEL_DATE_COL] = pd.to_datetime(df_fuel[FUEL_DATE_COL], format='%Y/%m')
    df_fuel[FUEL_LNG_COL] = pd.to_numeric(df_fuel[FUEL_LNG_COL], errors='coerce')
    
    # SMP 데이터에 연료비 병합
    df_market[SCENARIO_MONTH_COL] = df_market[SCENARIO_DATE_COL].dt.to_period('M').dt.to_timestamp()
    df_fuel = df_fuel.rename(columns={FUEL_DATE_COL: SCENARIO_MONTH_COL})
    df_market = pd.merge(df_market, df_fuel[[SCENARIO_MONTH_COL, FUEL_LNG_COL]], on=SCENARIO_MONTH_COL, how='left').ffill()
    
    # 3. Spark Spread (마진) 계산
    df_market[SCENARIO_SPARK_SPREAD_COL] = df_market[SCENARIO_SMP_COL] - df_market[FUEL_LNG_COL]
    
    # 시나리오: 2025년 1년치 데이터만 추출해서 시뮬레이션
    df_scenario = df_market[df_market[SCENARIO_DATE_COL].dt.year == SIMULATION_YEAR].copy().reset_index(drop=True)
    
    return df_scenario

def train_pdm_model(pdm_file_path):
    print(">>> 2. 설비 데이터(Machine Data) 로드 및 고장 확률 예측 모델링...")
    # 4. 예지보전 데이터 로드 및 모델 학습
    df_pdm = pd.read_csv(pdm_file_path)
    
    # 간단한 랜덤포레스트 모델 학습
    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE)
    model.fit(df_pdm[PDM_FEATURES], df_pdm[PDM_TARGET])
    return model, PDM_FEATURES, df_pdm

def generate_simulation_data(model, features, df_pdm, df_scenario):
    # 5. 시나리오 생성: 시간이 지날수록 설비가 노후화된다고 가정 (Tool wear 증가)
    days = len(df_scenario)
    simulation = pd.DataFrame()
    # 공구 마모도(Tool wear)가 0부터 250까지 선형적으로 증가한다고 가정
    simulation['Tool wear [min]'] = np.linspace(0, 250, days)
    # 나머지 변수는 평균값 주변에서 랜덤 변동
    for col in features[:-1]:
        simulation[col] = np.random.normal(df_pdm[col].mean(), df_pdm[col].std()*0.1, days)
        
    # 고장 확률 예측
    probs = model.predict_proba(simulation[features])[:, 1] # 고장(1)일 확률
    df_scenario[SCENARIO_FAILURE_PROB_COL] = probs * 100 # 퍼센트로 변환
    return df_scenario

def create_and_save_dashboard(df_scenario, output_path=OUTPUT_IMAGE_PATH):
    print(">>> 3. 통합 시각화 (Integrated Visualization) 생성 중...")
    # 6. 시각화: 이중축 그래프 (Dual Axis)
    fig, ax1 = plt.subplots(figsize=PLOT_FIGURE_SIZE)

    # [축 1] 시장 수익성 (Bar Chart)
    bar_colors = [COLOR_PROFIT if x > 0 else COLOR_LOSS for x in df_scenario[SCENARIO_SPARK_SPREAD_COL]]
    ax1.bar(df_scenario[SCENARIO_DATE_COL], df_scenario[SCENARIO_SPARK_SPREAD_COL], color=bar_colors, alpha=0.6, label='Spark Spread (Profitability)')
    
    ax1.set_xlabel(X_LABEL, fontsize=12)
    ax1.set_ylabel(Y1_LABEL, fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # [축 2] 설비 고장 위험 (Line Chart)
    ax2 = ax1.twinx()
    ax2.plot(df_scenario[SCENARIO_DATE_COL], df_scenario[SCENARIO_FAILURE_PROB_COL], color=LINE_RISK_COLOR, linewidth=3, linestyle='-', label=LEGEND_RISK)
    
    ax2.set_ylabel(Y2_LABEL, fontsize=12, color=LINE_RISK_COLOR)
    ax2.tick_params(axis='y', labelcolor=LINE_RISK_COLOR)
    ax2.set_ylim(0, 100)
    
    # [인사이트 강조] 정비 추천 구간 표시
    # 로직: 마진이 마이너스이면서(정비 기회비용 낮음), 리스크가 상승하는 구간
    plt.title(PLOT_TITLE, fontsize=16, pad=20)
    
    # 범례 생성 (수동)
    patch_profit = mpatches.Patch(color=COLOR_PROFIT, alpha=0.6, label=LEGEND_PROFIT)
    patch_loss = mpatches.Patch(color=COLOR_LOSS, alpha=0.6, label=LEGEND_LOSS)
    line_risk = plt.Line2D([0], [0], color=LINE_RISK_COLOR, linewidth=3, label=LEGEND_RISK)
    
    plt.legend(handles=[patch_profit, patch_loss, line_risk], loc='upper left')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # 저장
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(output_path)
    print(f"완료! '{output_path}' 파일을 확인하세요.")

def run_integrated_analysis():
    df_scenario = load_and_prepare_market_data(SMP_FILE, FUEL_FILE)
    model, features, df_pdm = train_pdm_model(PDM_FILE)
    df_scenario = generate_simulation_data(model, features, df_pdm, df_scenario)
    create_and_save_dashboard(df_scenario)

if __name__ == "__main__":
    run_integrated_analysis()