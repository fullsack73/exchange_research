# 환율 분석 프로젝트

## 개요
장기에서 환율이 어떻게 움직이는지 규명하고 최근의 이상 기간동안의 환율 급등 원인이 뭔지 파악하고, 분석해서 위기상황 및 기타 이벤트적 상황에서 환율이 어떻게 변동하는지에 대한 새로운 이론과 원리를 찾아내는 것을 목적으로 함.

## 폴더 구조
- `analysis/`: 분석 스크립트 및 결과 (anomaly, baseline, daily_threshold, m2_components, shap_ml, lstm_validation_daily, lstm_validation_monthly)
- `data/`: 거시경제 지표 및 환율 원본 데이터 (10y_bond, CPI, exchange_rate, m2, policy_rate, process_scripts, production_index)
- `reports/`: 최종 보고서 파일 및 관련 문서 모음

## 지금까지 한거
1. 이상구간에서 선형관계 분석 해보니까 베이스라인이랑 다르게 나오더라(금리차의 역할 역전)

2. 이상구간에서 왜 올랐나 Random Forest + SHAP 돌려보니까 M2가 범인이래

3. 근데 막상 선형관계 분석에서는 M2_KOR은 선형 관계가 없다고 하는데?

4. 선형적으로 증가하는게 아니라 M2가 임계점을 넘으면 급발진 해서 그런거 아닐까?(가설)

5. 그래서 임계점 분석 + MMF, CMA와 환율에 대해 SHAP 분석을 해보니까 M2 증가량, 특히 MMF 증가량이 특정 임계점을 넘을 때 환율이 급발진 한다는 것 까지 찾았음(가설 검증)

## 오늘 기준 한거
