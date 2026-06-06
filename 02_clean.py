"""
Phase 2: 데이터 정제 (Cleaning)
방침:
  - 트리 기반 모델은 NaN을 그대로 처리할 수 있음 → 결측치는 일단 유지
  - 중복 제거, Y 이상치 제거, 면적 이상치 제거
  - EDA 용도로 imputed 카피본도 별도 저장
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

# 1) 원본 한글 컬럼명으로 다시 로드 (변동 % 컬럼 공백 문제 정확히 처리)
train = pd.read_csv(DATA / "train.csv")
test  = pd.read_csv(DATA / "test.csv")

# 한글 → 영문 (변동 % 공백 포함하여 정확히 매핑)
ko2en = {
    "동": "dong",
    "전용면적(㎡)": "area_m2",
    "층": "floor",
    "도로명": "road_id",
    "구분": "building_type",
    "건물나이": "building_age",
    "계약년월": "contract_ym",
    "종가": "kospi_close",
    "거래량": "kospi_volume",
    "변동 %": "kospi_change_pct",   # 공백 포함
    "한국은행기준금리": "base_rate",
    "대학최소거리": "univ_min_dist",
    "역최소거리": "station_min_dist",
    "마트": "mart_count",
    "스타벅스": "starbucks_count",
    "Y": "Y",
    "id": "id",
}
train = train.rename(columns=ko2en)
test  = test.rename(columns=ko2en)
print("columns:", list(train.columns))

# 2) contract_ym → int (현재 float)
train["contract_ym"] = train["contract_ym"].astype(int)
test["contract_ym"]  = test["contract_ym"].astype(int)

# 3) 정제 통계 기록
N0 = len(train)
print(f"\n[Initial] train rows: {N0:,}")

# 4) 중복 제거 (train)
dup_n = train.duplicated().sum()
train = train.drop_duplicates().reset_index(drop=True)
print(f"[Dedupe] removed {dup_n:,} duplicates -> {len(train):,}")

# 5) Y 이상치
#    근거: train Y 99.5%=5.8억인데 99.5%~max 사이가 5.8억→287억으로 50배 점프(노이즈)
#    test가 깨끗하다는 가이드 진술. 정확히 0.5% (99.5분위) 컷.
Y_CAP = 58_000   # 5.8억 (train 99.5% 분위)
n_y_out = (train["Y"] > Y_CAP).sum()
train = train[train["Y"] <= Y_CAP].reset_index(drop=True)
print(f"[Y outlier] removed Y > {Y_CAP:,} (99.5%) : {n_y_out:,} rows ({n_y_out/N0*100:.3f}%) -> {len(train):,}")

# 낮은 Y는 노이즈가 아님(외곽 연립의 정상 보증금) → 그대로 둠
print(f"  Y range now: {train['Y'].min():.0f} ~ {train['Y'].max():.0f}")

# 6) 면적 이상치
#    근거: test max=121.9㎡ = train 99.7% 분위와 정확히 일치
#    → 운영진이 train에 0.3% 면적 노이즈 주입한 것으로 추정. 정확히 0.3% 컷.
AREA_CAP = 121.9   # train 99.7% 분위 = test max
n_area_out = (train["area_m2"] > AREA_CAP).sum()
train = train[train["area_m2"] <= AREA_CAP].reset_index(drop=True)
print(f"[Area outlier] removed area > {AREA_CAP}㎡ (99.7%) : {n_area_out:,} rows ({n_area_out/N0*100:.3f}%) -> {len(train):,}")

# 7) 결측치 현황 (정제 후)
print("\n[Missing - after clean]")
miss = train.isna().sum()
print(miss[miss > 0])

# 8) 카테고리 / 정수 타입 캐스팅
# building_type: 카테고리, road_id: int but 카테고리적 의미, dong: 카테고리
for df in [train, test]:
    df["dong"] = df["dong"].astype("category")
    df["building_type"] = df["building_type"].astype("category")
    # road_id는 int 유지 (CatBoost용으로 cat_features에 등록할 예정)

# 9) 정제 후 train 저장 (NaN 보존 - tree 모델용)
train.to_parquet("data/train_clean.parquet", index=False)
test.to_parquet("data/test_clean.parquet", index=False)
print(f"\nSaved data/train_clean.parquet ({len(train):,} rows) / data/test_clean.parquet ({len(test):,} rows)")

# 10) EDA용 imputed copy (그래프/통계 분석 시 결측 채워서 처리 편하게)
train_eda = train.copy()
test_eda  = test.copy()
for col in ["floor", "building_age", "univ_min_dist", "station_min_dist", "mart_count", "starbucks_count"]:
    if train_eda[col].isna().any():
        # dong + building_type별 median으로 채우고, 그래도 비면 글로벌 median
        med_by_group = train_eda.groupby(["dong", "building_type"], observed=True)[col].transform("median")
        train_eda[col] = train_eda[col].fillna(med_by_group)
        train_eda[col] = train_eda[col].fillna(train_eda[col].median())

print("\n[Missing - EDA copy]")
miss2 = train_eda.isna().sum()
print(miss2[miss2 > 0] if (miss2 > 0).any() else "  (none) ✅")

train_eda.to_parquet("data/train_eda.parquet", index=False)
print("Saved data/train_eda.parquet")

# 11) 최종 요약
print("\n" + "=" * 60)
print("[Clean summary]")
print(f"  train rows: {N0:,} → {len(train):,} ({len(train)/N0*100:.2f}% kept)")
print(f"  Y range: {train['Y'].min():,.0f} ~ {train['Y'].max():,.0f}")
print(f"  area range: {train['area_m2'].min():.1f} ~ {train['area_m2'].max():.1f}")
print(f"  Y mean / median: {train['Y'].mean():,.0f} / {train['Y'].median():,.0f}")
print("=" * 60)
