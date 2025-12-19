# sleep_running_project

---

# Sleep and Running Performance Analysis (Garmin Data)

## 1. Project Objective

본 프로젝트의 목적은 **웨어러블 기기(Garmin)에서 수집한 수면 데이터가 일별 러닝 퍼포먼스 및 훈련 강도에 어떤 영향을 미치는지 분석**하는 것이다.
단순 예측 정확도 향상이 아닌, **수면 지표의 설명력과 그 한계를 분석적으로 규명**하는 데 초점을 둔다.

---

## 2. Data Description

### Data Source

* **Device**: Garmin Forerunner 265
* **Period**: 약 2개월
* **Matched days**: 64일 (수면–러닝 병합 후)

### Sleep Features

* 수면 점수 (`score`)
* 수면 시간 (`duration_min`)
* 수면 필요 시간 (`sleep_need_min`)
* 수면 부채 (`sleep_debt`)
* 안정시 심박수 (`resting_hr`)
* 혈중 산소 포화도 (`pulse_ox`)
* 취침 시각 (`sleep_start_hour`)
* 단기 수면 지표 (short/late sleep)
* 4주 평균 수면 점수 (`sleep_score_4w`)
* 요일 / 주말 여부

### Running Targets

* `avg_speed` : 평균 러닝 속도 (km/h)
* `tss` : Training Stress Score

---

## 3. Methodology

### Preprocessing

* 날짜 단위 정합을 위해 수면·러닝 날짜를 **day-level로 정규화**
* 하루에 여러 러닝이 있을 경우 일 단위 집계
* 수면 관련 파생 변수(feature engineering) 생성

### Models

* Linear Regression
* Random Forest Regressor

### Evaluation Metrics

* MAE
* RMSE
* R² Score

---

## 4. Results

### Performance Summary

* `avg_speed`: R² ≈ **-0.34**
* `tss`: R² ≈ **-0.38 (Random Forest)**

R²가 음수라는 것은, **수면 변수만으로는 일별 러닝 퍼포먼스를 정확히 예측하기 어렵다는 점**을 의미한다.

---

### Feature Importance Analysis

#### Average Speed

중요 변수:

1. 4주 평균 수면 점수 (`sleep_score_4w`)
2. 취침 시각
3. 안정시 심박수
4. 혈중 산소 포화도
5. 수면 시간

→ 단기 수면보다 **장기적인 수면 습관과 회복 상태**가 더 중요한 신호로 작용함을 확인.

#### Training Stress Score (TSS)

중요 변수:

1. 안정시 심박수
2. 수면 시간
3. 수면 부채
4. 혈중 산소 포화도

→ 수면 상태는 **훈련 강도 선택에 영향을 주는 제약 조건**으로 작용.

---

## 5. Interpretation

본 분석 결과는 다음을 시사한다.

* 수면 데이터는 **러닝 퍼포먼스를 직접 예측하기에는 정보가 불충분**
* 그러나 수면과 생리적 지표는 **훈련 강도 조절과 회복 상태를 반영하는 의미 있는 신호**를 제공
* 높은 TSS 또는 빠른 속도는 수면 외에도 **훈련 계획, 누적 피로, 운동 의도**에 크게 의존

즉, 낮은 예측 성능은 모델 실패가 아니라 **문제 자체의 복잡성을 반영한 결과**이다.

---

## 6. Limitations and Future Work

### Limitations

* 데이터 샘플 수가 제한적 (64일)
* 훈련 유형(인터벌/롱런 등) 정보 미포함
* 누적 훈련량 변수 부재

### Future Improvements

* 전날 및 최근 7일 TSS / 거리 누적 변수 추가
* 전날 수면 vs 당일 수면 비교 실험
* 고/저 강도 훈련 분류 문제로 확장

---

## 7. Conclusion

> **수면 데이터는 러닝 퍼포먼스의 직접적인 예측 변수라기보다는,
> 훈련 강도와 회복 상태를 설명하는 보조적·맥락적 변수로 작용한다.**

본 프로젝트는 **현실적인 데이터 한계 속에서의 해석 가능성**을 중시한 분석 사례이다.

---

## 8. Project Structure

```
sleep_fatigue_project/
├── data/
├── src/
│   └── sleep_run_garmin.py
├── outputs/
│   ├── merged_sleep_run.csv
│   ├── summary.json
│   ├── metrics__*.csv
│   └── *.png
└── README.md

