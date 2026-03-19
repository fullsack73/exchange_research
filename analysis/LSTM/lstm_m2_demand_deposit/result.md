# LSTM 분석 결과: 금리차 + 수시입출식저축성예금

## 분석 목적
이전 실험과 동일한 구조로, 아래 두 모델의 환율 예측 성능을 비교했습니다.

- Model A: USD_KRW + RATE_SPREAD_KOR_USA
- Model B: USD_KRW + RATE_SPREAD_KOR_USA + M2_수시입출식저축성예금

## 데이터/설정
- 데이터셋: `daily_dataset_m2_demand_deposit.csv`
- 데이터 기간: 2015-12-01 ~ 2026-03-16
- 시퀀스 길이: 30일
- 예측 시점: 5영업일 ahead (`pred_step=5`)
- 학습/평가 분할: 각 기간별 시계열 80/20
- 평가 지표: RMSE, MAE

## 결과 요약

### 1) 전체 구간 (full_2015_2026)
- 표본 수: 2532 (train=2025, test=507)
- Model A: RMSE 43.98, MAE 37.48
- Model B: RMSE 29.54, MAE 25.20
- 우수 모델: Model B

### 2) 이상 구간 (anomaly_2024_11_to_2026_03)
- 표본 수: 333 (train=266, test=67)
- Model A: RMSE 21.21, MAE 18.13
- Model B: RMSE 17.68, MAE 14.18
- 우수 모델: Model B

## 해석
- 전체 구간과 이상 구간 모두에서 `수시입출식저축성예금`을 추가한 Model B가 개선되었습니다.
- 특히 이상 구간에서 RMSE가 21.21 -> 17.68로 낮아져, 해당 변수의 설명력이 예측 성능으로도 이어지는 결과를 확인했습니다.

## 산출물
- `analysis/lstm_m2_demand_deposit/results.json`
- `analysis/lstm_m2_demand_deposit/results.txt`
- `analysis/lstm_m2_demand_deposit/full/full_2015_2026_lstm_plot_full.png`
- `analysis/lstm_m2_demand_deposit/eval/full_2015_2026_lstm_plot_eval.png`
- `analysis/lstm_m2_demand_deposit/full/anomaly_2024_11_to_2026_03_lstm_plot_full.png`
- `analysis/lstm_m2_demand_deposit/eval/anomaly_2024_11_to_2026_03_lstm_plot_eval.png`
