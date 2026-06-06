"""
Phase 1: 데이터 로딩 & 1차 점검
- train/test/sample 로드
- 컬럼명 영문화 (한글 컬럼이 라이브러리에서 종종 깨지므로)
- 데이터 타입 / 결측치 / 기본 통계 / 중복 확인
- 동/도로명 카디널리티 점검
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

# 1) 로드
train = pd.read_csv(DATA / "train.csv")
test  = pd.read_csv(DATA / "test.csv")
sub   = pd.read_csv(DATA / "sample_submission.csv")

print("=" * 60)
print("[Shape]")
print(f"  train: {train.shape}")
print(f"  test : {test.shape}")
print(f"  sub  : {sub.shape}")

print("\n[Train columns]")
print(list(train.columns))

print("\n[Test columns]")
print(list(test.columns))

print("\n[Dtypes - train]")
print(train.dtypes)

# 2) 한글 → 영문 컬럼명 매핑
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
    "변동%": "kospi_change_pct",
    "한국은행기준금리": "base_rate",
    "대학최소거리": "univ_min_dist",
    "역최소거리": "station_min_dist",
    "마트": "mart_count",
    "스타벅스": "starbucks_count",
    "Y": "Y",
    "id": "id",
}
train_e = train.rename(columns=ko2en)
test_e  = test.rename(columns=ko2en)
print("\n[English columns - train]")
print(list(train_e.columns))

# 3) 결측치 점검
print("\n[Missing - train]")
miss_tr = train_e.isna().sum()
print(miss_tr[miss_tr > 0])
print("\n[Missing - test]")
miss_te = test_e.isna().sum()
print(miss_te[miss_te > 0] if (miss_te > 0).any() else "  (none)")

# 4) 중복
dup_tr = train_e.duplicated().sum()
dup_te = test_e.duplicated().sum()
print(f"\n[Duplicated rows] train={dup_tr:,}  test={dup_te:,}")

# 5) 타겟 통계
print("\n[Y stats (만원)]")
desc = train_e["Y"].describe(percentiles=[0.001, 0.01, 0.5, 0.99, 0.995, 0.999])
print(desc)

# 6) 카디널리티
print("\n[Cardinality]")
for c in ["dong", "road_id", "building_type", "contract_ym"]:
    print(f"  {c}: train={train_e[c].nunique():,}  test={test_e[c].nunique():,}")

# 7) test에 있는데 train에 없는 카테고리 점검
print("\n[Unseen categories in test]")
for c in ["dong", "road_id", "building_type"]:
    tr_set = set(train_e[c].dropna().unique())
    te_set = set(test_e[c].dropna().unique())
    diff = te_set - tr_set
    print(f"  {c}: {len(diff):,} unseen ({len(diff)/max(len(te_set),1)*100:.2f}%)")

# 8) 면적 / 층 / 건물나이 범위
print("\n[Numeric range comparison train vs test]")
for c in ["area_m2", "floor", "building_age"]:
    print(f"  {c}: train min={train_e[c].min()}, max={train_e[c].max()}  | test min={test_e[c].min()}, max={test_e[c].max()}")

# 9) 시점 범위
print("\n[Contract YM]")
print(f"  train: {train_e['contract_ym'].min()} ~ {train_e['contract_ym'].max()}")
print(f"  test : {test_e['contract_ym'].min()} ~ {test_e['contract_ym'].max()}")

# 10) 저장 (영문 컬럼 버전)
train_e.to_parquet("data/train_raw.parquet", index=False)
test_e.to_parquet("data/test_raw.parquet", index=False)
print("\nSaved: data/train_raw.parquet, data/test_raw.parquet")
print("=" * 60)
