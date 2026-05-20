# PPT 생성 요청 프롬프트

아래 내용을 기반으로 발표용 PPT를 만들어줘.

## 발표 목적

이 발표는 기존 `핀셋_3차발표.pdf`와 겹치는 내용을 반복하지 않고, 그 이후 새로 구현한 **FX impact 최종 파이프라인**을 설명하는 것이 목적이다.

3차 발표에서 이미 다룬 내용:
- 이상구간 정의
- M2와 환율의 Granger 인과성
- M2가 환율을 선행한다는 분석
- 환율 충격이 여러 거시경제 변수로 전이되는 lead-lag 개념

이번 PPT에서는 위 내용은 짧은 전제로만 언급하고, 아래 내용에 집중해줘:
- 기존 forecast 기반 FX impact pipeline의 한계
- event-time panel로 재구현한 이유
- local projection 방식
- actual/predicted/scenario FX 입력 구조
- response curve 해석
- target별 성능과 scenario 민감도
- 최종 산출물과 한계

디자인 방향, 색상, 폰트, 레이아웃 스타일 제안은 쓰지 말고, 슬라이드별 **내용 구성**과 **넣을 그림/표**만 반영해줘.

## 사용 자료

발표 대본:
- `reports/presentation_script_fx_impact_event_time.md`

핵심 결과 문서:
- `analysis/fx_impact/result.md`

핵심 산출물:
- `analysis/fx_impact/reports/final/event_panel/anomaly_event_panel.csv`
- `analysis/fx_impact/reports/final/event_panel/local_projection_coefficients.csv`
- `analysis/fx_impact/reports/final/event_panel/event_response_forecasts.csv`
- `analysis/fx_impact/reports/final/event_panel/local_projection_metrics.csv`
- `analysis/fx_impact/reports/final/event_panel/scenario_response_forecasts.csv`
- `analysis/fx_impact/reports/final/event_panel/event_panel_model_selection.csv`

핵심 plot:
- `analysis/fx_impact/reports/final/event_panel/plots/response_summary_top_targets.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_KOSPI.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_Import_Price_Index.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_Industrial_Production.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_Trade_Balance.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_CSI_CCSI.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_Imports.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_Foreign_Bond_Investment.png`
- `analysis/fx_impact/reports/final/event_panel/plots/response_Foreign_Stock_Investment.png`

비교용 기존 forecast plot:
- `analysis/fx_impact/reports/final/plots/forecast_KOSPI.png`
- `analysis/fx_impact/reports/final/plots/response_KOSPI.png`

## 전체 구성

총 15장으로 구성해줘.  
발표 시간은 10~13분 기준으로 잡아줘.  
각 슬라이드는 제목, 핵심 bullet, 넣을 그림/표만 포함해줘.

---

## 슬라이드 1. 3차 발표 이후, FX Impact 최종 파이프라인

핵심 내용:
- 이번 발표는 3차 발표의 반복이 아니라, 이후 구현한 최종 FX impact pipeline 설명이다.
- 최종 질문: “이상구간에서 예측된 환율 경로가 주어졌을 때 국내 거시·금융 변수의 h개월 뒤 반응을 어떻게 추정할 것인가?”
- 최종 방법론: event-time local projection.

넣을 그림/표:
- 별도 이미지 없음.
- 키워드 3개만 표시: `event-time panel`, `local projection`, `response curve`.

---

## 슬라이드 2. 이전 발표에서 이어받는 전제

핵심 내용:
- 이상구간은 이미 정의된 분석 단위로 사용한다.
- M2 -> FX 선행성은 이전 발표의 결론으로 두고 반복 설명하지 않는다.
- 이번 분석에서 환율은 최종 목적변수가 아니라 거시·금융 변수로 충격을 전달하는 입력 경로다.

넣을 그림/표:
- 별도 이미지 없음.
- 간단한 텍스트 요약만 사용:
  - `이상구간`
  - `hybrid M2 기반 FX prediction`
  - `target response`

---

## 슬라이드 3. 이번 발표의 새 질문: Forecast가 아니라 Response

핵심 내용:
- 기존 질문: 환율이나 target level을 잘 예측하는가.
- 새 질문: 환율 경로가 주어졌을 때 target이 h개월 뒤 얼마나 변하는가.
- x축은 calendar date가 아니라 horizon 1~6개월이다.
- y축은 target level이 아니라 event month 대비 변화량이다.

넣을 그림/표:
- 직접 도식으로 표현:
  - `FX path`
  - `event month t`
  - `target response at h=1,2,...,6`
- 기존 파일 이미지는 넣지 않아도 됨.

---

## 슬라이드 4. 기존 Final Pipeline의 한계

핵심 내용:
- 기존 forecast plot은 level path를 따라가는 예측선 중심이었다.
- anomaly month만 이어붙이면 실제 calendar gap이 사라져 lag가 왜곡된다.
- 최종 파생효과 추정에는 forecast보다 event-time response가 더 적합하다.

넣을 그림/표:
- 왼쪽 또는 앞부분에 기존 forecast 예시:
  - `analysis/fx_impact/reports/final/plots/forecast_KOSPI.png`
- 오른쪽 또는 뒤쪽에 새 response 예시:
  - `analysis/fx_impact/reports/final/event_panel/plots/response_KOSPI.png`

---

## 슬라이드 5. Event-Time Panel의 핵심 원칙

핵심 내용:
- X row는 anomaly event month로 제한한다.
- lag와 response는 원래 월별 calendar index에서 계산한다.
- 이상구간을 이어붙여 shift하지 않는다.
- 핵심 문장: “이상구간 X는 유지하되, lag 왜곡은 제거한다.”

넣을 그림/표:
- 직접 도식으로 표현:
  - 원래 calendar time 위에 `t-1`, `event_date=t`, `t+1`, `t+h` 표시.
  - `event_date=t`만 anomaly row로 강조.
  - lag와 response는 원래 calendar에서 가져온다는 화살표 표시.

---

## 슬라이드 6. Event Panel 데이터 구성

핵심 내용:
- row 단위: `event_date`, `target`, `horizon`.
- anomaly event month: 68개.
- horizon: 1~6개월.
- 최종 panel row 수: 2,937개.
- selected targets:
  - `CSI_CCSI`
  - `Imports`
  - `KOSPI`
  - `Industrial_Production`
  - `Import_Price_Index`
  - `Foreign_Bond_Investment`
  - `Foreign_Stock_Investment`
  - `Trade_Balance`

넣을 그림/표:
- `analysis/fx_impact/reports/final/event_panel/anomaly_event_panel.csv`에서 일부 컬럼만 표로 표시:
  - `event_date`
  - `target`
  - `horizon`
  - `actual_fx_t`
  - `pred_fx_t`
  - `actual_response`
  - `transform`
  - `unit_label`

---

## 슬라이드 7. Actual Response의 정의

핵심 내용:
- actual response는 실제 target 데이터에서 계산한다.
- log-diff target: `log(target_{t+h}) - log(target_t)`.
- diff target: `target_{t+h} - target_t`.
- y축 delta는 예측 오차가 아니라 event month 대비 h개월 뒤 변화량이다.

넣을 그림/표:
- 수식 2개를 표시:
  - `log response = log(target_{t+h}) - log(target_t)`
  - `level delta = target_{t+h} - target_t`
- 별도 이미지 없음.

---

## 슬라이드 8. FX 입력: Actual, Predicted, Scenario

핵심 내용:
- actual FX: 실제 환율 경로.
- predicted FX: `hybrid_m2_model_a`의 daily prediction을 월평균 resample 후 month-end macro data에 결합.
- predicted FX 결측 월은 actual FX로 fill.
- scenario FX: predicted FX baseline에 event date 기준 +5% shock 적용.
- scenario-baseline delta는 target의 shock 민감도 지표다.

넣을 그림/표:
- 직접 도식으로 세 경로 표시:
  - `actual FX`
  - `hybrid_m2 predicted FX baseline`
  - `+5% scenario FX`
- 필요하면 `event_response_forecasts.csv` 또는 `scenario_response_forecasts.csv`에서 한 target 예시 row를 표로 추가.

---

## 슬라이드 9. Local Projection 모델

핵심 내용:
- target별, horizon별로 별도 모델을 학습한다.
- 기본 식:
  - `response_{target,t,h} ~ fx_shock_t + fx_lag1...fx_lag6 + target_lag1`
- 사용 모델:
  - `LocalProjection_OLS`
  - `LocalProjection_Ridge`
  - `LocalProjection_ElasticNet`
- event_date 기준 time split.
- scaler는 train에만 fit.
- 미래 target 정보 누수 방지.

넣을 그림/표:
- 위 회귀식을 크게 표시.
- 모델 3개를 작은 표로 정리:
  - OLS: 기본 선형회귀
  - Ridge: 계수 안정화
  - ElasticNet: 계수 안정화 + 일부 변수 영향 축소

---

## 슬라이드 10. Response Plot 읽는 법

핵심 내용:
- x축: horizon 1~6개월.
- y축: target response.
- 선:
  - actual mean response on test events
  - predicted response using actual FX
  - predicted response using hybrid M2 predicted FX
  - predicted response using scenario FX
- predicted FX baseline과 scenario FX의 차이가 shock 민감도다.
- RMSE/NRMSE는 예측 성능 지표이고, scenario-baseline delta는 민감도 지표다.

넣을 그림/표:
- 예시 plot:
  - `analysis/fx_impact/reports/final/event_panel/plots/response_KOSPI.png`
- plot 옆 또는 아래에 선 4개의 의미를 텍스트로 설명.

---

## 슬라이드 11. 모델 성능 요약

핵심 내용:
- 전체 predicted-FX test 기준:
  - ElasticNet RMSE: 1770.129739
  - ElasticNet NRMSE: 1.2328
  - Ridge RMSE: 1779.840873
  - Ridge NRMSE: 1.2707
  - OLS RMSE: 1780.422375
  - OLS NRMSE: 1.2762
- 평균적으로 ElasticNet이 가장 안정적이다.
- 안정적 target:
  - `CSI_CCSI`
  - `Foreign_Bond_Investment`
  - `Foreign_Stock_Investment`
  - `Imports`
- 해석 주의 target:
  - `Import_Price_Index`
  - `Industrial_Production`
  - `KOSPI`
  - `Trade_Balance`

넣을 그림/표:
- `analysis/fx_impact/reports/final/event_panel/local_projection_metrics.csv` 기반 성능 요약 표.
- `analysis/fx_impact/reports/final/event_panel/event_panel_model_selection.csv` 기반 target별 best model 표.

---

## 슬라이드 12. Scenario 민감도 결과

핵심 내용:
- scenario-baseline absolute response delta가 큰 변수:
  - `Foreign_Stock_Investment`: +1023.551512, h=6
  - `Trade_Balance`: +386.788263, h=1
  - `Foreign_Bond_Investment`: -276.915493, h=1
  - `CSI_CCSI`: -2.391069, h=2
  - `Imports`: -0.021083, h=3
- 이 순위는 예측 정확도 순위가 아니라 shock 민감도 순위다.

넣을 그림/표:
- 반드시 삽입:
  - `analysis/fx_impact/reports/final/event_panel/plots/response_summary_top_targets.png`

---

## 슬라이드 13. 주요 Target 해석 시 주의점

핵심 내용:
- `KOSPI`: 금융 변수로 환율 shock과 연결되지만 NRMSE 기준 안정성은 제한적.
- `Import_Price_Index`: 환율과 직관적으로 연결되지만 유가, 원자재 가격, 전가율 영향이 커서 오차가 큼.
- `Industrial_Production`: scenario-baseline effect가 거의 0에 가까움.
- `Trade_Balance`: scenario delta는 크지만 event별 response 예측은 불확실성이 있음.

넣을 그림/표:
- 4개 plot을 함께 사용:
  - `analysis/fx_impact/reports/final/event_panel/plots/response_KOSPI.png`
  - `analysis/fx_impact/reports/final/event_panel/plots/response_Import_Price_Index.png`
  - `analysis/fx_impact/reports/final/event_panel/plots/response_Industrial_Production.png`
  - `analysis/fx_impact/reports/final/event_panel/plots/response_Trade_Balance.png`

---

## 슬라이드 14. 최종 구현 변경과 결론

핵심 내용:
- 기존 final pipeline은 forecast 중심이었다.
- 최종 pipeline은 event-time local projection 중심으로 재구현했다.
- 기존 forecast 산출물은 비교/개발 과정용으로 남기고, 최종 해석은 response curve 기준으로 한다.
- 구현상 핵심 변화:
  - anomaly event month만 X row로 사용
  - lag와 response는 원래 calendar time에서 계산
  - actual/predicted/scenario FX mode를 비교
  - target별 horizon response curve 생성

넣을 그림/표:
- 직접 전환 도식 사용:
  - `old: calendar-time forecast`
  - `new: event-time response curve`
- 별도 파일 이미지는 필수 아님.

---

## 슬라이드 15. 산출물과 한계

핵심 내용:
- 최종 산출물 위치:
  - `analysis/fx_impact/reports/final/event_panel/`
- 핵심 파일:
  - `anomaly_event_panel.csv`
  - `local_projection_coefficients.csv`
  - `event_response_forecasts.csv`
  - `local_projection_metrics.csv`
  - `scenario_response_forecasts.csv`
  - `event_panel_model_selection.csv`
- 주요 plot:
  - `analysis/fx_impact/reports/final/event_panel/plots/`
  - `analysis/fx_impact/reports/final/plots/response_{target}.png`
- 재현 커맨드:
  - `python analysis/fx_impact/run_final_fx_impact_pipeline.py`
- 한계:
  - anomaly event 표본이 작다.
  - target별·horizon별 모델 안정성에 제약이 있다.
  - +5% shock scenario는 실험 가정이다.
  - 일부 target은 환율 외 추가 설명변수가 필요하다.

넣을 그림/표:
- 산출물 파일 목록 표.
- 마지막에는 핵심 결론 문장:
  - “이상구간을 이어붙이지 않고, 원래 calendar time을 보존한 채 환율 충격의 h개월 뒤 response를 추정했다.”

---

## 부록 슬라이드. 예상 질문

필요하면 부록으로 3~5개 Q&A 슬라이드를 추가해줘.

Q&A 내용:
- OLS local projection이 무엇인지
- Ridge와 ElasticNet을 왜 썼는지
- hybrid M2 모델 결과를 어떻게 입력으로 사용했는지
- actual response 정답을 어떻게 계산했는지
- scenario-baseline delta가 왜 예측 성능이 아니라 민감도인지

넣을 그림/표:
- 별도 이미지 없음.
