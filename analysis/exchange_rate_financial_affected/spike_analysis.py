import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# 디렉토리 생성
os.makedirs('results_spike', exist_ok=True)

# 데이터 로드
df = pd.read_csv('data/final_processed_data.csv', index_col=0, parse_dates=True)

# 2024년 11월부터 2025년 12월까지 필터링
spike_period_df = df.loc['2024-11-01':'2025-12-31']

print(f"Spike period dataset shape: {spike_period_df.shape}")

# 피쳐와 타겟 분리
X = spike_period_df.drop(columns=['FX_Ret', 'Is_Spike'])
# 이 구간에서는 '환율 변화율(FX_Ret)'을 직접 예측하는 회귀 모델로 접근하여 기여도 분석 (샘플 수가 적으므로)
y = spike_period_df['FX_Ret']

# 모델 학습 (XGBoost Regressor)
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

model.fit(X, y)

# SHAP 분석
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 시각화 1: Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot - Impact on FX Returns (Nov 2024 - Dec 2025)")
plt.savefig('results_spike/shap_summary_spike_period.png', bbox_inches='tight')
plt.close()

# 시각화 2: Feature Importance
vals = np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'importance_val'])
feature_importance.sort_values(by=['importance_val'], ascending=False, inplace=True)
feature_importance.to_csv('results_spike/feature_importance_spike_period.csv', index=False)

# 이벤트 스터디 필터링
event_df = pd.read_csv('results/event_study_results.csv', parse_dates=['release_date'])
event_spike_period = event_df[(event_df['release_date'] >= '2024-11-01') & (event_df['release_date'] <= '2025-12-31')]
event_spike_period.to_csv('results_spike/event_study_spike_period.csv', index=False)

print("Deep analysis for spike period completed.")
print("Results saved in results_spike/ directory.")
