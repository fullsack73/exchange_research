# 금융·거시 변수(US CPI 등)의 환율 영향력 리팩토링 분석 결과

## 1. 분석 목적
기존 `analysis/exchange_rate_financial_affected`에 흩어져 있던 산발적인 이벤트 스터디, 급등 구간 스파이크 분석(XGBoost 분류/회귀) 스크립트들을 **기준 구간(Baseline)과 이상 구간(Anomaly)**으로 나누어 동적으로 비교하는 현재 프로젝트의 표준 방법론으로 통합하였습니다. 
본 문서는 미국 CPI(Core, Shelter 등) 지표가 평상시 대비 환율 이상 구간에서 원/달러 환율 수익률(FX_Ret)에 미치는 영향이 어떻게 증폭(Regime Shift) 되는지를 델타(Delta) 분석 및 SHAP 시각화로 요약합니다.

## 2. 분석 방법
- **데이터 소스**: 일별 환율(`processed_daily_1995_2026_integrated.csv`)과 월별 US CPI(`us_cpi_monthly.csv`)를 통합하여 시차(`lag1, lag2, lag3`) 파생 변수 생성.
- **기간 분리**: `period_definition.json`에 정의된 동적 기준(Baseline/Anomaly) 구간 적용 (기존 1.5$\sigma$ 하드코딩 방식 탈피).
- **분석 기법**:
  1. `XGBoost Regressor + SHAP`: 각 구간별 모형 학습 후 변수 중요도 추출, 이상 구간에서 중요도가 급증하는 핵심 지표(Delta) 규명.
  2. `Dependence Plot 분석`: 원인 지표의 등락이 실제 환율에 미치는 임계점 및 비선형적 반응 확인. 
  3. `Event Study`: 실제 미국 CPI 발표일 당일의 환율 반응(Daily Return)과 Core MoM 증감 간 스캐터(산점도) 분석.

## 3. 분석 결과 요약

### 3.1. 위기/이상 구간에서 환율 변동을 주도하는 금융 변수 (SHAP Delta)
- `results/feature_importance_delta.csv`에 나타나듯, 평상시 대비 이상 구간(Anomaly)에서 모델 결정에 대한 기여도(SHAP value)가 가장 크게 폭증한 지표는 **Core_MoM_lag3**와 **Shelter_MoM_lag3** 등입니다.
- 평상시에는 이들 지수의 상승/하락 영향력이 상대적으로 파편화되어 있으나, 이상 구간에서는 해당 인플레이션(또는 주요 구성항목) 압력이 달러 강세 방어 심리를 크게 왜곡시켜 환율 변동의 직접적인 트리거로 작용함을 뜻합니다.

### 3.2. CPI 발표일 이벤트 스터디 (Release Day Reaction)
- `event_study_scatter.png` (미국 Core CPI MoM vs 원달러 당일 수익률) 그래프에서 볼 수 있듯, Core CPI의 예상치 상회(물가 서프라이즈)는 당일 원달러 환율 상승(달러 가치 급등)이라는 뚜렷한 리스크오프 단기 트리거를 낳습니다. 
- 특히 이상 구간에 근접할수록 이 기울기의 탄력성이 심화됩니다.

### 3.3. 상세 자료 확인
- Baseline 및 Anomaly 구간 각각의 SHAP Summary Plot은 `results/` 에 저장되어 각 구간의 종합적 영향력 구조 변화를 직관적으로 비교할 수 있습니다.
- 핵심 동력 변수들에 대한 비선형 반응(SHAP Dependence Plot) 역시 별도로 추출되었습니다.
