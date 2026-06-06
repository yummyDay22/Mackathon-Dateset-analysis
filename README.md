# Mackathon-Dateset-analysis
# 서울 전세 보증금 예측 모델 (Seoul Jeonse Deposit Prediction)

매커톤 6조 데이터 분석 프로젝트 — 130만 건의 서울 전세 거래 데이터로 보증금을 예측하는 회귀 모델.

## 🏆 최종 성과

| 지표 | 값 |
|---|---|
| **CV RMSE** | **6,768 만원** (log RMSE 0.5782) |
| **단독 최고 모델 대비 개선** | -32만원 (0.47%) |
| **데이터 보존율** | 97.30% (정제 후) |
| **5-Fold 변동** | < 1% (6,775 ~ 6,812만원) |

## 📊 데이터

- **train**: 1,339,272 행 × 16 컬럼 (정제 후 1,303,107 행)
- **test**: 328,254 행 × 15 컬럼
- **기간**: 2011 ~ 2022 (12년)
- **타겟**: Y (전세 보증금, 만원)

## 🛠️ 기술 스택

- **언어**: Python 3.x
- **데이터 처리**: pandas, numpy, pyarrow
- **머신러닝**: scikit-learn, LightGBM, XGBoost, CatBoost
- **시각화**: matplotlib, seaborn
- **저장 포맷**: parquet, npy

## 🔧 파이프라인

```
01_load_inspect.py        → 원본 데이터 탐색
02_clean.py               → 정제 (이상치/중복 제거)
02b_outlier_analysis.py   → 이상치 분위수 분석
03_feature_engineering.py → 35개 신규 변수 생성 (5-Fold OOF)
04_lgbm.py                → LightGBM 5-Fold
05_xgb.py                 → XGBoost 5-Fold
06_catboost.py            → CatBoost 4-Fold
07_ensemble.py            → 3모델 앙상블
07b_ensemble_2model.py    → 2모델 앙상블 백업
07c_ensemble_3model.py    → 3모델 최적 가중 탐색
08_all_ensembles.py       → 모든 조합(11개) 생성 및 비교
08_insights.py            → 인사이트 1차 추출
09_insights_deep.py       → 인사이트 심화 분석
10_insights_extra30.py    → 인사이트 21~50번 추가
11_verify_mapo.py         → 마포구 케이스 스터디 검증
```

## 🌟 핵심 차별점

### 1. 운영진 노이즈 패턴 식별
- train Y 99.5% 분위 = **정확히 58,000만원**
- train area 99.7% 분위 = **121.9㎡** (test의 area max와 소수점까지 일치)
- 분위수 기반 정밀 절단으로 무의미한 데이터 손실 방지

### 2. 도로명 단위 가격 결정 발견
- 같은 동 안에서도 도로명별 분산이 **동 전체 분산의 71%**
- LightGBM Feature Importance 1위 = `road_mean_Y_smooth`
- Bayesian smoothing (k=20) 적용: `(n·road_mean + k·global_mean) / (n+k)`

### 3. 데이터 누수 원천 차단
- 모든 target encoding은 **5-Fold OOF** 처리
- 시계열 변수는 `shift(1)` 후 rolling
- test의 unseen 카테고리(2,608행)는 global mean fallback

### 4. 카테고리별 가격 결정 메커니즘 차등
| 구분 | 면적-Y 상관 | 금리 1%p 영향 | 주요 결정 변수 |
|---|---|---|---|
| 아파트 | 0.541 | -13.1% | 면적 · 층 · 입지 |
| 오피스텔 | 0.479 | -16.3% | 면적 · 위치 · 신축 여부 |
| 연립다세대 | 0.287 | -23.1% | 건물나이 · 동 위치 |

→ 5개 교호작용 변수(age_x_type, floor_x_type 등)로 차등 학습

## 📐 앙상블 수식

```
y_log_ensemble = w_L × y_log_LGBM + w_X × y_log_XGB + w_C × y_log_CAT
y_final = exp(y_log_ensemble) - 1

제약: w_L + w_X + w_C = 1, w_m ≥ 0
최적: w_L = 0.55, w_X = 0.30, w_C = 0.15
```

## 🎯 모델별 단독 성능 (5-Fold OOF)

| 모델 | RMSE (만원) | log RMSE |
|---|---|---|
| LightGBM | 6,800 | 0.5795 |
| XGBoost | 6,826 | 0.5815 |
| CatBoost (4-fold) | 6,801 | 0.5840 |
| **3모델 앙상블 (최적)** | **6,768** | **0.5782** |

## 📁 폴더 구조

```
.
├── 01-11_*.py                # 파이프라인 스크립트
├── seoul_dong_map.py         # 서울 동→구 매핑
├── data/                     # 원본/정제 데이터 (대용량은 .gitignore)
├── outputs/                  # 모델 결과
│   ├── answer_*.csv          # 11개 제출 파일 (단독/2모델/3모델 × avg/opt)
│   ├── ensemble_comparison.csv  # 11개 조합 RMSE 비교
│   ├── insights_final.md     # 인사이트 50개
│   ├── metrics_*.json        # 각 모델 지표
│   └── fi_*.csv              # 변수 중요도
└── figures/                  # 시각화 결과
```

## 📑 인사이트

전체 **50개 인사이트**가 `outputs/insights_final.md`에 명제 + 정량 근거 형식으로 정리됨.

**주요 발견**:
- 노이즈는 분위수 절단선 (Y 99.5%, area 99.7%)
- 가격은 동이 아니라 도로명에서 결정 (71%)
- 강남 프리미엄은 1.13배에 불과 (5.8억 캡 후)
- 작은 평수가 ㎡당 1.54배 비싸다 (역원룸 효과)
- 역세권 임계점은 500m (300m 이내는 소음 디스카운트)
- 신축 프리미엄이 무너졌다 (외곽 공급 집중)
- 연립다세대가 금리에 가장 민감 (-23%/1%p)
- 마포 스타벅스 역설 (스타벅스 많을수록 가격↓, 서울 평균과 반대)

## 🏗️ 재현 방법

1. **데이터 준비**: `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` 배치
2. **순차 실행**:
   ```bash
   python 01_load_inspect.py
   python 02_clean.py
   python 03_feature_engineering.py
   python 04_lgbm.py
   python 05_xgb.py
   python 06_catboost.py
   python 08_all_ensembles.py
   ```
3. **최종 제출**: `outputs/answer_lgbm_xgb_cat_opt.csv`

## 📝 라이선스 / 출처

- 데이터: 매커톤 대회 제공
- 코드: 본 리포지토리 작성자
