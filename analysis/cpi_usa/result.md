### 분석 목적
미국 CPI 발표 이벤트와 CPI 세부 지표가 원/달러 환율 급등(스파이크) 구간에 어떤 영향을 주는지 점검하기

### 분석 방법
1. 이벤트 스터디: CPI 발표일 당일 환율 반응률(fx_reaction)과 변동성 비율(vol_ratio) 계산
2. 전체 구간 분류 모델: XGBoost 분류(Is_Spike) + SHAP 중요도 분석
3. 이상구간 심화 분석: 2024-11~2025-12 구간 XGBoost 회귀(FX_Ret) + SHAP 중요도 분석

### 실행 결과
- 이벤트 스터디 분석 이벤트 수: 46건
- 이상구간(2024-11~2025-12) 이벤트 수: 13건
- 발표일 평균 환율 반응률(fx_reaction): 0.000355
- 발표일 평균 절대 반응률: 0.004029
- |vol_ratio| > 1 수준의 상대적 큰 반응 이벤트: 18건

### 주요 결과 해석
1. 이벤트 스터디
- 발표일 반응률은 이벤트마다 방향/크기가 다르며, 일부 이벤트는 직전 5거래일 대비 큰 변동(고 vol_ratio)을 보임.

2. 전체 구간 스파이크 분류 모델(SHAP 상위 변수)
- 상위 변수: Energy_YoY_lag2, Food_YoY, Shelter_YoY_lag2, Durables_YoY_lag3, Headline_MoM_lag1
- 해석: 스파이크 분류에는 단일 변수보다 CPI 세부 항목의 시차(lag) 조합이 중요하게 작용함.

3. 이상구간(2024-11~2025-12) 회귀 모델(SHAP 상위 변수)
- 상위 변수: MedicalCare_YoY_lag1, Apparel_MoM, Headline_MoM, Apparel_YoY, Headline_YoY
- 해석: 이상구간만 보면 전체구간 분류 모델과 상위 변수 구성이 일부 다르게 나타나며, 국면별 민감도 차이가 존재함.

### 생성 산출물
- results/event_study_results.csv
- results/event_study_scatter.png
- results/shap_summary_plot.png
- results/feature_importance_ranking.csv
- results/dependence_Core_MoM_lag1.png
- results/dependence_Core_YoY_lag1.png
- results/dependence_Shelter_MoM_lag1.png
- results/dependence_Shelter_YoY_lag1.png
- results_spike/shap_summary_spike_period.png
- results_spike/feature_importance_spike_period.csv
- results_spike/event_study_spike_period.csv

### 한계 및 주의사항
- 이벤트 스터디는 CPI surprise(컨센서스 대비 실제치)가 아니라 CPI 지수의 월간 변화율(MoM)을 대용 변수로 사용함.
- 분류/회귀 결과는 표본 구간 및 피처 엔지니어링(시차 변수)에 민감하므로, 모델 성능 검증(예: 시계열 분할 백테스트)과 함께 해석 필요.
