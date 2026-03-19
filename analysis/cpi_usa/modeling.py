import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# 경로 설정 및 결과 디렉토리 생성
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# 데이터 로드
df = pd.read_csv(
    PROJECT_ROOT / 'data' / 'CPI' / 'USA' / 'CPI_components' / 'final_processed_data.csv',
    index_col=0,
    parse_dates=True
)

# 피쳐와 타겟 분리
# 타겟: FX_Ret (Regression) 또는 Is_Spike (Classification)
# 사용자 요청은 "환율 급등 구간에서 영향 분석"이므로 Classification으로 접근하여 급등 원인 분석
X = df.drop(columns=['FX_Ret', 'Is_Spike'])
y = df['Is_Spike']

# 모델 학습 (XGBoost Classifier)
# 하이퍼파라미터는 간단하게 설정
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X, y)

# SHAP 분석
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 시각화 1: Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot - Impact on Exchange Rate Spikes")
plt.savefig(RESULTS_DIR / 'shap_summary_plot.png', bbox_inches='tight')
plt.close()

# 시각화 2: Feature Importance (Mean Absolute SHAP)
vals = np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'importance_val'])
feature_importance.sort_values(by=['importance_val'], ascending=False, inplace=True)
feature_importance.to_csv(RESULTS_DIR / 'feature_importance_ranking.csv', index=False)

# 시각화 3: Dependence Plots for Core CPI and Shelter
# 핵심 변수 필터링 (Core_YoY, Shelter_YoY 등)
target_vars = ['Core_YoY', 'Shelter_YoY', 'Core_MoM', 'Shelter_MoM']
# 가장 시차가 적절한 lag 변수 찾기 (예: lag1)
for var in target_vars:
    lag_var = f'{var}_lag1'
    if lag_var in X.columns:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(lag_var, shap_values, X, show=False)
        plt.title(f"SHAP Dependence Plot: {lag_var}")
        plt.savefig(RESULTS_DIR / f'dependence_{lag_var}.png', bbox_inches='tight')
        plt.close()

print("Modeling and SHAP analysis completed.")
print("Results saved in results/ directory.")
