# FX Impact 최종 발표 대본

이 대본은 `핀셋_3차발표.pdf`에서 이미 다룬 내용을 최대한 반복하지 않도록 재구성했다.  
3차 발표의 핵심이 `이상구간 정의`, `M2 -> FX 인과성`, `환율 충격의 변수별 시차 경로`였다면, 이번 발표는 그 이후 작업인 **최종 FX impact 파이프라인 재구현**에 초점을 둔다.

권장 발표 시간은 10~13분이다.

---

## 슬라이드 1. 제목: 3차 발표 이후, FX Impact 최종 파이프라인

**화면 가정:** 제목, `event-time local projection`, `response curve`, `scenario shock` 키워드.

**발표 대본:**

이번 발표는 앞선 3차 발표의 내용을 반복하기보다, 그 이후에 최종 파이프라인을 어떻게 바꿨는지에 집중하겠습니다.

3차 발표에서는 이상구간에서 M2가 환율보다 선행하는지, 그리고 환율 충격이 거시경제 변수로 어떤 시차를 두고 퍼지는지를 중심으로 봤습니다.

이번에는 질문을 조금 더 모델링 쪽으로 좁혔습니다. “이상구간에서 예측된 환율 경로가 주어졌을 때, 국내 거시·금융 변수들의 h개월 뒤 반응을 어떻게 추정할 것인가?”가 핵심입니다.

결론부터 말하면, 기존 forecast 방식은 최종 파생효과 추정에는 한계가 있어서, event-time local projection 방식으로 파이프라인을 새로 구현했습니다.

**전환 멘트:** 먼저 이전 발표에서 가져오는 전제를 짧게만 확인하겠습니다.

---

## 슬라이드 2. 이전 발표에서 이어받는 전제

**화면 가정:** 3차 발표 요약 2줄: `이상구간`, `M2 선행성`, `변수별 시차 반응`.

**발표 대본:**

이 슬라이드는 복습용입니다. 자세한 이상구간 정의나 Granger 인과성 검정 결과는 3차 발표에서 이미 다뤘기 때문에 여기서는 길게 설명하지 않겠습니다.

이번 최종 모델은 두 가지 전제를 이어받습니다.

첫째, 분석 대상은 정상구간 전체가 아니라 `Is_Abnormal_Period`로 표시된 이상구간입니다. 둘째, 환율은 단독 목표가 아니라 거시·금융 변수로 충격을 전달하는 중간 경로로 봅니다.

다만 이번 발표에서 새로 보는 것은 “M2가 환율을 선행하는가”가 아닙니다. 그 부분은 이전 발표의 역할이었습니다. 이번에는 예측된 환율 경로를 입력으로 넣었을 때, target response를 어떻게 계산하고 예측하는지가 핵심입니다.

**전환 멘트:** 그래서 최종 질문은 forecast가 아니라 response입니다.

---

## 슬라이드 3. 이번 발표의 새 질문: Forecast가 아니라 Response

**화면 가정:** `FX path -> target response at h=1..6` 흐름도.

**발표 대본:**

기존 질문이 “환율을 잘 예측할 수 있는가”였다면, 이번 질문은 한 단계 뒤입니다.

환율 경로가 실제값이든, hybrid M2 모델의 예측값이든, 또는 5% shock을 준 scenario든, 그 환율 경로가 주어졌을 때 target이 몇 개월 뒤 얼마나 움직이는지를 예측하고 싶었습니다.

그래서 최종 산출물도 더 이상 target level path를 따라가는 forecast plot이 아닙니다. x축은 horizon, 즉 1개월부터 6개월 뒤이고, y축은 event month 대비 target의 변화량입니다.

이 구조가 중요한 이유는 파생효과를 볼 때 우리가 궁금한 것이 “다음 달 target 값이 얼마냐”보다 “환율 충격 뒤 h개월 후 target response가 얼마나 크냐”이기 때문입니다.

**전환 멘트:** 그런데 기존 final pipeline은 이 질문에 딱 맞지 않았습니다.

---

## 슬라이드 4. 기존 Final Pipeline의 한계

**화면 가정:** 기존 `forecast_*.png`와 새 `response_*.png`를 대비하는 그림.

**발표 대본:**

기존 final pipeline은 월별 연속 시계열 forecast에 가까웠습니다. 이 방식은 일반적인 예측 문제에는 자연스럽지만, 이상구간 조건부 파생효과를 직접 보여주기에는 애매했습니다.

또 다른 문제는 anomaly month만 이어붙이는 방식입니다. 이상구간에 해당하는 월만 뽑아서 연속된 데이터처럼 만들면, 실제 달력상으로 몇 개월 떨어져 있던 사건들이 바로 옆 달처럼 붙어버립니다.

그러면 lag와 response가 왜곡됩니다. 예를 들어 실제로는 6개월 차이가 나는 두 anomaly month가 데이터 안에서는 1개월 차이처럼 처리될 수 있습니다.

따라서 최종 구현에서는 anomaly event month는 유지하되, lag와 미래 response는 반드시 원래 월별 calendar index에서 가져오도록 바꿨습니다.

**전환 멘트:** 이게 event-time panel의 핵심입니다.

---

## 슬라이드 5. Event-Time Panel의 핵심 원칙

**화면 가정:** event month `t`는 anomaly row, `t-1`, `t+h`는 원래 calendar에서 가져오는 도식.

**발표 대본:**

이번 구현에서 가장 중요한 원칙은 하나입니다.

X row는 anomaly event month로 제한합니다. 하지만 lag와 response는 원래 calendar time에서 계산합니다.

예를 들어 2022년 9월이 이상구간이라면 이 달을 event date로 씁니다. 그런데 환율 lag 1개월은 anomaly list에서 직전 row가 아니라 실제 2022년 8월입니다. target의 3개월 뒤 response도 anomaly list의 세 번째 다음 row가 아니라 실제 2022년 12월 값을 사용합니다.

이 방식은 이상구간만 분석한다는 조건은 유지하면서도, 시간 간격을 인위적으로 압축하지 않습니다. 그래서 이번 작업의 핵심을 한 문장으로 말하면 “이상구간 X는 유지하되, lag 왜곡은 제거한 것”입니다.

**전환 멘트:** 이 원칙으로 만든 데이터가 `anomaly_event_panel.csv`입니다.

---

## 슬라이드 6. 새 Event Panel 구성

**화면 가정:** `event_date`, `target`, `horizon` 단위 테이블 샘플.

**발표 대본:**

새 패널의 row 단위는 `event_date`, `target`, `horizon`입니다. 즉 특정 이상구간 월 t에서 특정 target이 h개월 뒤 얼마나 반응했는지를 한 줄로 저장합니다.

사용한 anomaly event month는 68개이고, horizon은 1개월부터 6개월까지입니다. 최종 event-time panel row 수는 2,937개입니다.

target은 8개입니다. `CSI_CCSI`, `Imports`, `KOSPI`, `Industrial_Production`, `Import_Price_Index`, `Foreign_Bond_Investment`, `Foreign_Stock_Investment`, `Trade_Balance`입니다.

여기서 target 선정 이유 자체는 이전 발표와 일부 겹치기 때문에 길게 반복하지 않겠습니다. 이번 발표에서는 이 target들을 대상으로 response를 어떻게 산출하고 예측했는지에 집중하겠습니다.

**전환 멘트:** 그럼 이 패널의 정답값, 즉 actual response는 어떻게 만들었는지 보겠습니다.

---

## 슬라이드 7. Actual Response의 정의

**화면 가정:** `actual_response = target_{t+h} - target_t` 또는 `log(target_{t+h}) - log(target_t)`.

**발표 대본:**

모델의 정답값은 실제 target 데이터에서 계산했습니다. event date가 이상구간이면, 원래 월별 데이터에서 `target_t`와 `target_{t+h}`를 가져옵니다.

로그 변화로 보는 target은 `log(target_{t+h}) - log(target_t)`를 actual response로 정의했습니다. 예를 들어 Imports처럼 비율 변화로 해석하는 변수가 여기에 들어갑니다.

차분으로 보는 target은 `target_{t+h} - target_t`를 response로 정의했습니다. 예를 들어 투자액이나 무역수지처럼 level delta 자체가 의미 있는 변수는 이 방식입니다.

따라서 response plot의 y축은 예측 오차가 아닙니다. event month와 h개월 뒤 사이에 target이 실제로 얼마나 변했는지, 또는 모델이 그 변화량을 얼마나 예측했는지를 의미합니다.

**전환 멘트:** 이제 환율 입력이 어떻게 들어갔는지 설명하겠습니다.

---

## 슬라이드 8. FX 입력: Actual, Predicted, Scenario

**화면 가정:** 세 경로: `actual FX`, `hybrid_m2 predicted FX`, `+5% scenario FX`.

**발표 대본:**

이번 모델에는 세 가지 FX mode가 들어갑니다.

첫째는 actual FX입니다. 실제 환율 변화가 주어졌을 때 target response를 얼마나 설명하는지 보는 기준입니다.

둘째는 predicted FX입니다. 여기서는 `hybrid_m2_model_a`가 만든 daily FX prediction을 월평균으로 바꾼 뒤, month-end 기준 macro dataset에 붙였습니다. 예측 FX가 없는 월은 actual FX로 채웠습니다.

셋째는 scenario FX입니다. predicted FX baseline에 event date 기준 5% shock을 준 경로입니다. 이때 scenario는 단순히 level만 올려놓는 것이 아니라 event date의 shock feature에 반영되도록 만들었습니다.

그래서 scenario-baseline delta는 “hybrid M2 예측 환율 경로 대비, 5% shock을 주면 target response 예측이 얼마나 달라지는가”를 뜻합니다.

**전환 멘트:** 이 response를 학습하는 모델은 local projection 구조입니다.

---

## 슬라이드 9. Local Projection 모델

**화면 가정:** `response_{t,h} ~ fx_shock_t + fx_lag1..fx_lag6 + target_lag1`.

**발표 대본:**

Local projection은 horizon별 response를 직접 회귀하는 방식입니다. 1개월 뒤 예측을 만든 다음 그 예측을 다시 2개월 뒤에 넣는 방식이 아니라, h개월 뒤 response를 바로 종속변수로 둡니다.

이번에는 target별, horizon별로 모델을 따로 학습했습니다. 기본 feature는 event month의 FX shock, 환율 lag 1개월부터 6개월, 그리고 target lag 1개월입니다.

모델은 세 가지를 비교했습니다. `LocalProjection_OLS`는 기본 선형회귀이고, `LocalProjection_Ridge`는 계수가 과도하게 커지는 것을 막는 회귀입니다. `LocalProjection_ElasticNet`은 Ridge의 안정화 효과에 더해 일부 feature의 영향력을 더 강하게 줄일 수 있습니다.

데이터 split은 event_date 기준 time split입니다. 마지막 20~30% anomaly events를 test로 사용했고, scaler는 train에만 fit했습니다. 즉 미래 target 정보가 feature로 새어 들어가는 leakage를 막았습니다.

**전환 멘트:** 이제 plot을 어떻게 읽어야 하는지 보겠습니다.

---

## 슬라이드 10. Response Plot 읽는 법

**화면 가정:** target별 `response_{target}.png`, x축 horizon 1~6, 선 4개.

**발표 대본:**

새 plot의 x축은 horizon입니다. 1은 event month로부터 1개월 뒤, 6은 6개월 뒤 response를 뜻합니다.

선은 네 가지입니다. 첫째, test event에서 관측된 actual mean response입니다. 둘째, actual FX를 넣었을 때 모델이 예측한 response입니다. 셋째, hybrid M2 predicted FX를 넣었을 때의 response입니다. 넷째, 5% shock scenario FX를 넣었을 때의 response입니다.

그래프에서 가장 중요한 비교는 predicted FX baseline과 scenario FX의 차이입니다. 이 차이가 크면 해당 target은 모델상 환율 shock에 민감하게 반응한다고 볼 수 있습니다.

다만 이 차이는 성능 지표가 아닙니다. 모델이 얼마나 잘 맞췄는지는 RMSE나 NRMSE로 보고, shock에 얼마나 예민한지는 scenario-baseline response delta로 봅니다.

**전환 멘트:** 먼저 예측 성능부터 짧게 보겠습니다.

---

## 슬라이드 11. 모델 성능 요약

**화면 가정:** 전체 평균 RMSE/NRMSE와 target별 best model 표.

**발표 대본:**

전체 predicted-FX test 기준으로 보면, 평균적으로는 ElasticNet이 가장 안정적이었습니다. ElasticNet의 전체 RMSE는 약 1,770.13이고 NRMSE는 약 1.23입니다.

Ridge와 OLS도 큰 차이는 아니지만, 평균 NRMSE 기준으로는 ElasticNet이 가장 낮았습니다. 이 결과 때문에 최종 plot model도 target별 성능을 기준으로 선택했습니다.

target별로 보면 `CSI_CCSI`, `Foreign_Bond_Investment`, `Foreign_Stock_Investment`, `Imports`는 predicted-FX NRMSE 기준 상대적으로 안정적인 그룹입니다.

반대로 `Import_Price_Index`, `Industrial_Production`, `KOSPI`, `Trade_Balance`는 해석에 더 주의가 필요한 그룹으로 정리했습니다. 이 말은 경제적으로 중요하지 않다는 뜻이 아니라, test 구간에서 response를 맞추는 loss 기준으로 불확실성이 더 크다는 의미입니다.

**전환 멘트:** 다음은 정확도가 아니라 민감도 결과입니다.

---

## 슬라이드 12. Scenario 민감도 결과

**화면 가정:** `response_summary_top_targets.png`, scenario-baseline peak effect bar chart.

**발표 대본:**

scenario-baseline 기준으로 절대 response delta가 큰 변수는 외국인 주식투자, 무역수지, 외국인 채권투자였습니다.

`Foreign_Stock_Investment`는 horizon 6개월에서 양의 방향으로 큰 반응을 보였고, peak delta는 약 1,023.55입니다. `Trade_Balance`는 horizon 1개월에서 약 386.79의 양의 delta가 나타났습니다. `Foreign_Bond_Investment`는 horizon 1개월에서 약 -276.92로 음의 방향 반응이 컸습니다.

`CSI_CCSI`는 horizon 2개월에서 약 -2.39, `Imports`는 horizon 3개월에서 약 -0.021의 로그 변화 반응을 보였습니다.

여기서 중요한 점은 이 순위가 “모델이 잘 맞춘 순위”가 아니라는 것입니다. 이 순위는 baseline 대비 5% shock scenario를 줬을 때 예측 response가 얼마나 달라졌는지, 즉 shock 민감도를 보여줍니다.

**전환 멘트:** 그래서 주요 target은 성능과 민감도를 같이 보고 해석해야 합니다.

---

## 슬라이드 13. 주요 Target 해석 시 주의점

**화면 가정:** KOSPI, Import Price Index, Industrial Production, Trade Balance의 response plot 4개.

**발표 대본:**

KOSPI는 환율 shock과 연결될 수 있는 금융 변수이지만, 이번 test 기준에서는 NRMSE가 약 1.24로 아주 안정적인 target은 아니었습니다. 따라서 방향성은 참고하되, 개별 event 예측력은 조심해서 봐야 합니다.

Import Price Index는 환율과 직관적으로 연결되는 변수지만, 이번 결과에서는 NRMSE가 약 1.79로 가장 높은 편입니다. 환율 외에도 유가, 원자재 가격, 계약 시차, 가격 전가율 같은 요인이 크기 때문으로 볼 수 있습니다.

Industrial Production은 best model이 ElasticNet이지만, scenario-baseline peak effect는 거의 0에 가까웠습니다. 즉 이번 horizon 1~6개월과 target 정의 안에서는 직접적 반응이 뚜렷하지 않았습니다.

Trade Balance는 horizon 1개월에서 scenario delta가 비교적 크게 나타났지만, NRMSE 기준으로는 해석 주의 그룹입니다. 반응 크기는 보이지만, event별 response를 안정적으로 맞추는 데에는 불확실성이 남아 있습니다.

**전환 멘트:** 마지막으로 이번 구현의 결론과 산출물을 정리하겠습니다.

---

## 슬라이드 14. 최종 구현 변경과 결론

**화면 가정:** `forecast pipeline -> event-time LP pipeline` 전환 요약.

**발표 대본:**

최근 커밋 기준으로 가장 큰 변경은 final FX impact pipeline을 event-time local projection 중심으로 재구현한 것입니다.

기존 forecast 기반 산출물은 개발 과정과 비교용으로 남겨두되, 최종 파생효과 해석은 event-time response curve 기준으로 바꿨습니다.

이 변경으로 세 가지가 가능해졌습니다. 첫째, 이상구간 event row만 유지할 수 있습니다. 둘째, lag와 response는 원래 calendar time에서 계산하므로 gap 왜곡을 피할 수 있습니다. 셋째, baseline FX와 scenario FX를 비교해서 target별 shock 민감도를 볼 수 있습니다.

따라서 이번 최종 파이프라인은 “환율 level path를 맞추는 모델”이 아니라, “환율 shock 이후 국내 변수의 h개월 뒤 response를 추정하는 모델”에 더 가깝습니다.

**전환 멘트:** 산출물과 한계를 말씀드리고 마무리하겠습니다.

---

## 슬라이드 15. 산출물과 한계

**화면 가정:** 산출물 파일 목록과 재현 커맨드.

**발표 대본:**

최종 산출물은 `analysis/fx_impact/reports/final/event_panel/` 아래에 정리했습니다.

핵심 파일은 `anomaly_event_panel.csv`, `local_projection_coefficients.csv`, `event_response_forecasts.csv`, `local_projection_metrics.csv`, `scenario_response_forecasts.csv`, 그리고 target별 response plot입니다. 사용자 확인용 plot은 `analysis/fx_impact/reports/final/plots/response_{target}.png`에도 배치했습니다.

재현은 `python analysis/fx_impact/run_final_fx_impact_pipeline.py` 명령으로 가능합니다.

한계도 있습니다. anomaly event 표본이 많지 않기 때문에 target별·horizon별 모델의 통계적 안정성에는 제약이 있습니다. 또 scenario는 +5% shock이라는 실험 가정이므로, 실제 정책 충격이나 시장 충격의 지속성과는 다를 수 있습니다.

그래도 이번 구현의 핵심 성과는 분명합니다. 이상구간을 단순히 이어붙이지 않고, 원래 calendar time을 보존한 채 환율 충격의 파생효과를 response curve로 볼 수 있게 만들었습니다.

**마무리 멘트:** 이상으로 발표를 마치겠습니다. 감사합니다.

---

## 부록. 예상 질문 답변

### Q1. 이전 3차 발표와 이번 발표의 차이는 무엇인가?

3차 발표는 “이상구간에서 M2가 환율을 선행하는가”와 “환율 충격이 어떤 변수로 번지는가”를 설명했습니다. 이번 발표는 그 이후 단계로, 예측된 환율 경로를 입력으로 넣어 target별 h개월 뒤 response를 추정하는 최종 파이프라인을 설명합니다.

### Q2. Forecast 방식은 폐기된 것인가?

최종 파생효과 해석 기준으로는 폐기했다고 말해도 됩니다. 기존 forecast 산출물은 비교·개발 과정의 보조 자료이고, 최종 결과는 event-time local projection의 response curve를 기준으로 해석합니다.

### Q3. hybrid M2 모델 결과는 어떻게 사용했는가?

`hybrid_m2_model_a`의 daily FX prediction을 월평균으로 바꾸고, month-end 기준 macro dataset에 붙였습니다. 이 monthly predicted FX를 baseline FX 경로로 사용했습니다. 결측 월은 actual FX로 채웠습니다.

### Q4. 정답 response는 어떻게 계산했는가?

정답은 실제 target 데이터에서 계산했습니다. event month t가 이상구간이면, 원래 calendar index에서 `target_t`와 `target_{t+h}`를 가져와 차분 또는 로그 차분으로 actual response를 만들었습니다.

### Q5. scenario-baseline delta가 크다는 것은 무엇인가?

모델이 더 잘 맞췄다는 뜻이 아니라, +5% FX shock scenario를 넣었을 때 baseline 대비 예측 response가 크게 달라졌다는 뜻입니다. 즉 target의 환율 shock 민감도에 가까운 지표입니다.
