# 환율 분석 프로젝트

## 개요
장기에서 환율이 어떻게 움직이는지 규명하고 최근의 이상 기간동안의 환율 급등 원인이 뭔지 파악하고, 분석해서 위기상황 및 기타 이벤트적 상황에서 환율이 어떻게 변동하는지에 대한 새로운 이론과 원리를 찾아내는 것을 목적으로 함.

## 폴더 구조
- `analysis/`: 분석 스크립트 및 결과 (anomaly, baseline, daily_threshold, m2_components, shap_ml, lstm_validation_daily, lstm_validation_monthly)
- `data/`: 거시경제 지표 및 환율 원본 데이터 (10y_bond, CPI, exchange_rate, m2, policy_rate, process_scripts, production_index)
- `reports/`: 보고서 파일 및 관련 문서 모음

## 진행상황
1. 이상구간에서 선형관계 분석 해보니까 베이스라인(장기)과 확실한 괴리를 포착

2. 이상구간에서의 환율 급등 원인을 Random Forest + SHAP로 분석, M2의 영향력을 확인. 선형관계 분석과 달라 임계점 가설 확립

3. 이상구간 기준, M2의 각 구성요소의 기여도를 분석(SHAP 분석)

4. MMF, CMA에 대하여 threshold, SHAP 분석으로 임계점 가설을 증명

5. LSTM 활용하여 MMF, MMF + CPI로 환율 예측하는 모델 구축, 설명력의 부족을 확인

6. LSTM 활용하여 단기 요구불 예금으로 환율 예측하는 모델 구축, 개선을 확인

