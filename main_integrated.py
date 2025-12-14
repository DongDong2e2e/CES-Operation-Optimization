import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime, timedelta
import sys
import os

# --- Configuration Constants ---
# File Paths
SMP_FILE = 'data/HOME_전력거래_계통한계가격_시간별 SMP.csv'
PDM_FILE = 'data/ai4i2020.csv'
OUTPUT_IMAGE_PATH = 'results/future_prediction_dashboard.png'

# Market Model Parameters
# yfinance 데이터 수집 문제로 Oil_Price와 Natural_Gas는 학습 피처에서 임시 제외
MARKET_MODEL_FEATURES = ['Exchange_Rate', 'Month'] 
MARKET_MODEL_TARGET = 'SMP'

# Predictive Maintenance Model Parameters
PDM_FEATURES = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
PDM_TARGET = 'Machine failure'
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

# --- User-Configurable "Current State" ---
# 사용자가 현재 설비 상태를 이 곳에 입력하여 시뮬레이션 시작점을 변경할 수 있습니다.
USER_INPUT_CURRENT_STATE = {
    'Tool wear [min]': 180.0,
    'Air temperature [K]': 298.0,
    'Process temperature [K]': 309.0,
    'Rotational speed [rpm]': 1500.0,
    'Torque [Nm]': 40.0
}

# Future Scenario Parameters
FUTURE_DAYS_TO_PREDICT = 30
EXCHANGE_RATE_FUTURE_INCREASE = 20
NG_FUTURE_INCREASE = 0.5  # 천연가스 가격의 미래 상승분 가정
TOOL_WEAR_RATE_PER_DAY = 2.5

# Fallback values for macro data if yfinance fails for oil/NG
# 이 값들은 yfinance에서 실제 값을 가져오기 어려울 때 사용되는 임시 기본값입니다.
DEFAULT_CURRENT_OIL_PRICE = 80.0
DEFAULT_CURRENT_NG_PRICE = 3.0 # Natural Gas price in USD/MMBtu

MAINTENANCE_RISK_THRESHOLD = 50.0 # 고장 확률이 이 값 이상이면 '고위험'으로 간주

# Plotting Parameters
PLOT_FIGURE_SIZE = (14, 8)
COLOR_PROFIT = '#4ECDC4'
COLOR_LOSS = '#FF6B6B'
LINE_RISK_COLOR = 'red'
PLOT_TITLE = 'AI-Driven Forecast: Future 30-Day Operation Strategy'

# --- Font Setup ---
if sys.platform == 'darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif sys.platform == 'win32':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
    except:
        plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def load_data_safe(filepath):
    """Safely loads a CSV file, attempting utf-8 then cp949 encoding."""
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='cp949')

def fetch_macro_data(start_date, end_date):
    """yfinance로 환율(KRW=X), 유가(CL=F), 천연가스(NG=F) 데이터를 가져옵니다."""
    print(f">>> 외부 거시경제 지표(환율, 유가, 천연가스) 수집 중... ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
    
    df_exchange = yf.Ticker("KRW=X").history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Exchange_Rate'})
    if df_exchange.empty:
        raise ValueError(f"yfinance로부터 환율 데이터(KRW=X)를 가져오지 못했습니다. ({start_date} ~ {end_date})")
    
    df_oil = yf.Ticker("CL=F").history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Oil_Price'})
    if df_oil.empty:
        print(f"경고: yfinance로부터 유가 데이터(CL=F)를 가져오지 못했습니다. 기본값 {DEFAULT_CURRENT_OIL_PRICE}을(를) 사용합니다.")
        df_oil = pd.DataFrame(index=df_exchange.index, columns=['Oil_Price'], data=DEFAULT_CURRENT_OIL_PRICE)

    df_ng = yf.Ticker("NG=F").history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Natural_Gas'})
    if df_ng.empty:
        print(f"경고: yfinance로부터 천연가스 데이터(NG=F)를 가져오지 못했습니다. 기본값 {DEFAULT_CURRENT_NG_PRICE}을(를) 사용합니다.")
        df_ng = pd.DataFrame(index=df_exchange.index, columns=['Natural_Gas'], data=DEFAULT_CURRENT_NG_PRICE)
    
    # 모든 데이터프레임의 인덱스를 날짜(date)로 정규화 (시간 및 시간대 제거)
    df_exchange.index = df_exchange.index.normalize()
    df_oil.index = df_oil.index.normalize()
    df_ng.index = df_ng.index.normalize()

    # 모든 데이터프레임의 전체 날짜 범위 생성
    all_dates = pd.date_range(start=min(df_exchange.index.min(), df_oil.index.min(), df_ng.index.min()),
                              end=max(df_exchange.index.max(), df_oil.index.max(), df_ng.index.max()))

    # 각 데이터프레임을 전체 날짜 범위에 재인덱싱하고 결측치 채우기
    df_exchange = df_exchange.reindex(all_dates).ffill().bfill() # bfill로 초반 NaN도 처리
    df_oil = df_oil.reindex(all_dates).ffill().bfill()
    df_ng = df_ng.reindex(all_dates).ffill().bfill()

    # 병합
    df_macro = pd.concat([df_exchange, df_oil, df_ng], axis=1)

    if df_macro.empty:
        raise ValueError("yfinance로부터 거시경제 데이터를 가져오지 못했습니다. 인터넷 연결을 확인하거나 티커가 유효한지 확인하세요.")

    df_macro = df_macro.reset_index().rename(columns={'index': 'Date'})
    df_macro['Date'] = df_macro['Date'].dt.tz_localize(None) # 최종적으로 Date 컬럼의 timezone 정보를 제거
    
    last_exchange_rate = df_macro['Exchange_Rate'].dropna().iloc[-1] if not df_macro['Exchange_Rate'].dropna().empty else DEFAULT_CURRENT_EXCHANGE_RATE
    last_oil_price = df_macro['Oil_Price'].dropna().iloc[-1] if not df_macro['Oil_Price'].dropna().empty else DEFAULT_CURRENT_OIL_PRICE
    last_ng_price = df_macro['Natural_Gas'].dropna().iloc[-1] if not df_macro['Natural_Gas'].dropna().empty else DEFAULT_CURRENT_NG_PRICE

    # 마지막으로 NaN이 없는지 최종 확인
    if pd.isna(last_exchange_rate) or pd.isna(last_oil_price) or pd.isna(last_ng_price):
        raise ValueError("최종적으로 가져온 환율, 유가, 천연가스 데이터 중 일부가 NaN입니다. yfinance 데이터 수집에 문제가 있습니다.")


    return df_macro, last_exchange_rate, last_oil_price, last_ng_price

def load_and_prepare_training_data(smp_path):
    """Loads historical SMP data and fetches real macro data for training."""
    print(">>> 학습 데이터 로드 및 실제 거시경제 데이터 수집 중...")
    df_smp = load_data_safe(smp_path)
    df_smp['기간'] = pd.to_datetime(df_smp['기간'], format='%Y/%m/%d').dt.tz_localize(None)
    df_smp_daily = df_smp.sort_values('기간')[['기간', '가중평균']].rename(columns={'기간':'Date', '가중평균':'SMP'})
    
    macro_start_date = df_smp_daily['Date'].min()
    macro_end_date = datetime.today()
    
    df_macro, last_exchange_rate, last_oil_price, last_ng_price = fetch_macro_data(macro_start_date, macro_end_date)
    
    df_smp_daily = pd.merge(df_smp_daily, df_macro, on='Date', how='left').ffill().bfill()
    df_smp_daily['Month'] = df_smp_daily['Date'].dt.month

    for col in MARKET_MODEL_FEATURES:
        if col not in df_smp_daily.columns:
            # yfinance 데이터 수집 문제로 인해 피처가 누락되면 경고만 출력하고 넘어감 (임시)
            print(f"경고: 시장 모델 학습에 필요한 피처 '{col}'이(가) 데이터에 없습니다. 해당 피처 없이 학습합니다.")
            # raise ValueError(f"Required feature '{col}' not found in combined training data.") # 이제 에러 대신 경고

    return df_smp_daily, last_exchange_rate, last_oil_price, last_ng_price

def train_market_predictor(df_market):
    """Trains the market prediction model based on macro-economic features."""
    print(">>> 시장 가격 예측 모델(Market AI) 학습 중...")
    
    # 현재 MARKET_MODEL_FEATURES에 없는 컬럼이 df_market에 있을 수 있으므로 필터링
    actual_features = [col for col in MARKET_MODEL_FEATURES if col in df_market.columns]
    
    if not actual_features:
        raise ValueError("시장 예측 모델 학습에 사용할 유효한 피처가 없습니다. MARKET_MODEL_FEATURES 설정을 확인하세요.")

    X = df_market[actual_features]
    y = df_market[MARKET_MODEL_TARGET]
    
    model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE)
    model.fit(X, y)
    return model

def train_failure_model(pdm_path):
    """Trains the machine failure prediction model."""
    print(">>> 설비 고장 예측 모델(Failure AI) 학습 중...")
    df_pdm = pd.read_csv(pdm_path)
    machine_model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE)
    machine_model.fit(df_pdm[PDM_FEATURES], df_pdm[PDM_TARGET])
    return machine_model

def generate_future_predictions(market_model, machine_model, last_date, current_exchange_rate, current_oil_price, current_ng_price, user_input_state):
    """Generates a 30-day future scenario based on a user-defined current state."""
    print("\n>>> 🔮 향후 30일 미래 예측 시뮬레이션 수행 중...")
    future_dates = [last_date + timedelta(days=x) for x in range(1, FUTURE_DAYS_TO_PREDICT + 1)]
    df_future = pd.DataFrame({'Date': future_dates})
    
    # --- Market Prediction ---
    df_future['Exchange_Rate'] = np.linspace(current_exchange_rate, current_exchange_rate + EXCHANGE_RATE_FUTURE_INCREASE, FUTURE_DAYS_TO_PREDICT)
    df_future['Oil_Price'] = np.linspace(current_oil_price, current_oil_price, FUTURE_DAYS_TO_PREDICT)
    df_future['Natural_Gas'] = np.linspace(current_ng_price, current_ng_price + NG_FUTURE_INCREASE, FUTURE_DAYS_TO_PREDICT)
    df_future['Month'] = df_future['Date'].dt.month

    # 시장 모델 예측에 사용할 피처 필터링 (학습 피처와 동일하게)
    market_prediction_features = [col for col in MARKET_MODEL_FEATURES if col in df_future.columns]
    if not market_prediction_features:
        raise ValueError("미래 시장 가격 예측에 사용할 유효한 피처가 없습니다. MARKET_MODEL_FEATURES 설정을 확인하세요.")
        
    df_future['Predicted_SMP'] = market_model.predict(df_future[market_prediction_features])
    
    # Improved LNG Price Formula using Natural Gas (with check for NaN)
    if 'Natural_Gas' in df_future.columns and not df_future['Natural_Gas'].isnull().all():
        df_future['Predicted_LNG'] = (df_future['Natural_Gas'] * 350) + (df_future['Exchange_Rate'] * 0.05)
    else:
        # 천연가스 데이터가 없으면 Oil_Price 기반으로 대체 (기존 단순화 공식)
        df_future['Predicted_LNG'] = (df_future['Oil_Price'] * df_future['Exchange_Rate'] * 0.0012) + 20
        print("경고: 천연가스 데이터가 없어 유가 기반 LNG 가격 공식을 사용합니다. 정확도가 낮을 수 있습니다.")

    df_future['Predicted_Spread'] = df_future['Predicted_SMP'] - df_future['Predicted_LNG']
    
    # --- Failure Prediction ---
    initial_tool_wear = user_input_state['Tool wear [min]']
    df_future['Future_Tool_Wear'] = [initial_tool_wear + (d * TOOL_WEAR_RATE_PER_DAY) for d in range(FUTURE_DAYS_TO_PREDICT)]
    
    sim_machine = pd.DataFrame()
    sim_machine['Air temperature [K]'] = np.random.normal(user_input_state['Air temperature [K]'], 2, FUTURE_DAYS_TO_PREDICT)
    sim_machine['Process temperature [K]'] = np.random.normal(user_input_state['Process temperature [K]'], 2, FUTURE_DAYS_TO_PREDICT)
    sim_machine['Rotational speed [rpm]'] = np.random.normal(user_input_state['Rotational speed [rpm]'], 50, FUTURE_DAYS_TO_PREDICT)
    sim_machine['Torque [Nm]'] = np.random.normal(user_input_state['Torque [Nm]'], 5, FUTURE_DAYS_TO_PREDICT)
    sim_machine['Tool wear [min]'] = df_future['Future_Tool_Wear']
    
    probs = machine_model.predict_proba(sim_machine[PDM_FEATURES])[:, 1]
    df_future['Failure_Prob'] = probs * 100
    
    return df_future

def create_future_dashboard(df_future):
    """Generates and saves the future prediction dashboard."""
    fig, ax1 = plt.subplots(figsize=PLOT_FIGURE_SIZE)
    
    plt.axvline(x=df_future['Date'].min(), color='black', linestyle='--', linewidth=1.5)
    
    colors = [COLOR_PROFIT if x > 0 else COLOR_LOSS for x in df_future['Predicted_Spread']]
    ax1.bar(df_future['Date'], df_future['Predicted_Spread'], color=colors, alpha=0.7, label='Forecasted Profit')
    ax1.set_ylabel('Forecasted Spark Spread (KRW)', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(df_future['Date'], df_future['Failure_Prob'], color=LINE_RISK_COLOR, linewidth=3, marker='o', markersize=4)
    ax2.set_ylabel('Forecasted Failure Risk (%)', color=LINE_RISK_COLOR, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=LINE_RISK_COLOR)
    ax2.set_ylim(0, 100)
    
    plt.text(df_future['Date'].min(), ax1.get_ylim()[1], '  Today (Prediction Start)', va='top')
    
    recomm_days = df_future[df_future['Predicted_Spread'] < 0]
    
    title_text = PLOT_TITLE
    if not recomm_days.empty:
        best_date = recomm_days['Date'].iloc[0]
        ax2.annotate(f'Best Maintenance Date\n({best_date.strftime("%Y-%m-%d")})', 
                     xy=(best_date, 0), xytext=(best_date, 50),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))
        title_text += f'\n[Recommendation] Maintain on {best_date.strftime("%m-%d")} (Min Opportunity Cost)'
    
    plt.title(title_text, fontsize=16, pad=20)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"완료! '{OUTPUT_IMAGE_PATH}'에 미래 예측 결과가 저장되었습니다.")

def generate_textual_report(df_future):
    """Generates a detailed textual report of the prediction results and recommendations."""
    print("\n" + "="*80)
    print("                 ✨ 미래 30일 운전 및 정비 최적화 예측 보고서 ✨")
    print("="*80)

    forecast_start = df_future['Date'].min().strftime('%Y-%m-%d')
    forecast_end = df_future['Date'].max().strftime('%Y-%m-%d')
    print(f"\n▶️ 예측 기간: {forecast_start} 부터 {forecast_end} ({FUTURE_DAYS_TO_PREDICT}일간)")
    
    avg_smp = df_future['Predicted_SMP'].mean()
    avg_lng = df_future['Predicted_LNG'].mean()
    avg_spread = df_future['Predicted_Spread'].mean()
    max_risk = df_future['Failure_Prob'].max()
    max_risk_date = df_future.loc[df_future['Failure_Prob'].idxmax(), 'Date'].strftime('%Y-%m-%d')

    print("\n--- 요약 ---")
    print(f"  - 평균 예측 SMP: {avg_smp:.2f} KRW")
    print(f"  - 평균 예측 LNG 가격: {avg_lng:.2f} KRW")
    print(f"  - 평균 예측 수익성 (Spark Spread): {avg_spread:.2f} KRW")
    print(f"  - 최대 설비 고장 위험: {max_risk:.2f}% (예상일: {max_risk_date})")

    print("\n--- 일자별 상세 예측 및 권고 ---")
    print("날짜         | 예측 SMP | 예측 LNG | 예측 마진 | 고장 위험 | 비고")
    print("-----------------------------------------------------------------------")

    recomm_count = 0
    for index, row in df_future.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        smp = row['Predicted_SMP']
        lng = row['Predicted_LNG']
        spread = row['Predicted_Spread']
        risk = row['Failure_Prob']
        notes = []

        if pd.isna(spread) or pd.isna(lng): # NaN 값으로 인한 문제 방지
            notes.append("데이터 부족 (LNG/마진 계산 불가)")
        else:
            if spread < 0:
                notes.append("역마진 예상 (손실 구간)")
            if risk >= MAINTENANCE_RISK_THRESHOLD:
                notes.append(f"고위험 설비 상태 ({risk:.1f}%)")
            
            if spread < 0 and risk >= MAINTENANCE_RISK_THRESHOLD:
                notes.append("-> 최적 정비 권고!")
                recomm_count += 1
            elif spread < 0 and risk < MAINTENANCE_RISK_THRESHOLD:
                notes.append("-> 정비 고려 (저마진)")

        print(f"{date} | {smp:8.2f} | {lng:8.2f} | {spread:8.2f} | {risk:7.2f}% | {', '.join(notes)}")

    print("\n--- 종합 권고 ---")
    if recomm_count > 0:
        first_recomm_date = df_future[(df_future['Predicted_Spread'] < 0) & (df_future['Failure_Prob'] >= MAINTENANCE_RISK_THRESHOLD)]['Date'].min()
        if pd.isna(first_recomm_date):
            print("  - 현재 예측된 최적 정비 권고일은 없습니다.")
        else:
            print(f"  ✅ 예측된 최적 정비 시작일: {first_recomm_date.strftime('%Y-%m-%d')}")
            print(f"     (마진이 낮고 고장 위험이 높은 기간을 활용한 기회비용 최소화 정비)")
    else:
        print("  ❌ 현재 예측 기간 내에 특별히 정비를 권고할 만한 최적의 기간은 없습니다.")
        print("     (마진이 낮고 고장 위험이 높은 기간이 겹치지 않음)")
    
    print("\n" + "="*80)
    print("보고서 생성 완료.")
    print("="*80 + "\n")


def run_future_prediction():
    """Main function to run the full prediction pipeline."""
    # 1. 모델 학습
    df_smp_daily, last_exchange_rate, last_oil_price, last_ng_price = load_and_prepare_training_data(SMP_FILE)
    market_model = train_market_predictor(df_smp_daily)
    machine_model = train_failure_model(PDM_FILE)
    
    # 2. 미래 예측 시나리오 생성
    last_date = df_smp_daily['Date'].max()
    df_future = generate_future_predictions(market_model, machine_model, last_date, last_exchange_rate, last_oil_price, last_ng_price, USER_INPUT_CURRENT_STATE)
    
    # 3. 대시보드 생성 및 추천 로직 실행
    create_future_dashboard(df_future)
    generate_textual_report(df_future)

if __name__ == "__main__":
    run_future_prediction()