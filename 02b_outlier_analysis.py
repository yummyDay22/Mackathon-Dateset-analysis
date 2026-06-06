"""
이상치 결정 근거 분석
- Y / area_m2 양쪽 tail 확인
- train vs test 범위 비교
- 정확한 분위수 기반 임계값 계산
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

# 원본 다시 로드 (중복만 제거한 상태로 분석)
train = pd.read_parquet(DATA / "train_raw.parquet")
test  = pd.read_parquet(DATA / "test_raw.parquet")

# 컬럼명 영문 매핑 - raw 파일은 이미 일부 매핑돼있음. 변동 % 미매핑 컬럼 처리
train = train.rename(columns={"변동 %": "kospi_change_pct"})
test  = test.rename(columns={"변동 %": "kospi_change_pct"})

# 중복만 제거
train = train.drop_duplicates().reset_index(drop=True)
N = len(train)
print(f"[중복 제거 후 train] {N:,} rows\n")

# =========================================
# 1) Y 양쪽 tail 분석
# =========================================
print("=" * 70)
print("[Y (전세보증금, 만원) 분포 분석]")
print("=" * 70)

ys = train["Y"]
print(f"\n test에는 Y가 없으므로 train만 분석. test 면적 max=121.9㎡(작은편) 참고.")

# 양쪽 분위수
pcts = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
for p in pcts:
    v = ys.quantile(p)
    print(f"  {p*100:6.2f}% 분위 : {v:>12,.0f} 만원  ({v/10000:.2f}억)")

# 낮은쪽: 601만원이 정말 노이즈일까?
print("\n[낮은 Y 표본 — 가장 싼 100건]")
low100 = train.nsmallest(100, "Y")
print(f"  Y 분포: {low100['Y'].min()}~{low100['Y'].max()}")
print(f"  area_m2 분포: {low100['area_m2'].min():.1f} ~ {low100['area_m2'].max():.1f} (median {low100['area_m2'].median():.1f})")
print(f"  building_type 분포:")
print(low100["building_type"].value_counts().to_string())
print(f"  dong top5:")
print(low100["dong"].value_counts().head().to_string())

# 분위수 기반 임계값
y_995 = ys.quantile(0.995)
y_005 = ys.quantile(0.005)
n_high = (ys > y_995).sum()
n_low  = (ys < y_005).sum()
print(f"\n  Y > 99.5%({y_995:,.0f}) : {n_high:,} rows ({n_high/N*100:.3f}%) ← 0.5% 컷")
print(f"  Y < 0.5%({y_005:,.0f})   : {n_low:,} rows ({n_low/N*100:.3f}%)")

# =========================================
# 2) area_m2 양쪽 tail 분석
# =========================================
print("\n" + "=" * 70)
print("[area_m2 (전용면적) 분포 분석]")
print("=" * 70)

a_tr = train["area_m2"]
a_te = test["area_m2"]
print(f"\n  train: min={a_tr.min():.1f}, max={a_tr.max():.1f}, median={a_tr.median():.1f}")
print(f"  test : min={a_te.min():.1f}, max={a_te.max():.1f}, median={a_te.median():.1f}  ← 정답 기준 (깨끗함)")

# train 양쪽 분위수
print("\n  [train area_m2 분위수]")
for p in pcts:
    v = a_tr.quantile(p)
    flag = "  ← test max(121.9) 초과" if v > 121.9 else ""
    print(f"  {p*100:6.2f}% 분위 : {v:>10.2f} ㎡{flag}")

# 낮은쪽: 10㎡는 노이즈?
print("\n[낮은 area 표본 — 가장 작은 100건]")
low100a = train.nsmallest(100, "area_m2")
print(f"  area_m2 분포: {low100a['area_m2'].min():.1f}~{low100a['area_m2'].max():.1f}")
print(f"  Y 분포: {low100a['Y'].min():,.0f} ~ {low100a['Y'].max():,.0f} (median {low100a['Y'].median():,.0f})")
print(f"  building_type:")
print(low100a["building_type"].value_counts().to_string())

# test에 비슷한 작은 면적이 얼마나?
print(f"\n  test area_m2 작은 분위: 0.1%={a_te.quantile(0.001):.2f}, 1%={a_te.quantile(0.01):.2f}, 5%={a_te.quantile(0.05):.2f}")
print(f"  test에 area<=15㎡ : {(a_te<=15).sum():,} rows ({(a_te<=15).sum()/len(test)*100:.3f}%)")
print(f"  train에 area<=15㎡: {(a_tr<=15).sum():,} rows ({(a_tr<=15).sum()/N*100:.3f}%)")

# 0.3% 컷 임계값
a_997 = a_tr.quantile(0.997)
n_high_a = (a_tr > a_997).sum()
n_above_test_max = (a_tr > 121.9).sum()
print(f"\n  area > 99.7%({a_997:.2f}㎡): {n_high_a:,} rows ({n_high_a/N*100:.3f}%) ← 0.3% 컷")
print(f"  area > 121.9(test max)   : {n_above_test_max:,} rows ({n_above_test_max/N*100:.3f}%)")

# =========================================
# 3) 결론 / 권장 임계값
# =========================================
print("\n" + "=" * 70)
print("[결론]")
print("=" * 70)
print(f"""
■ 높은 값이 노이즈인 근거
  1) test가 깨끗하다고 명시 → test 범위 밖은 train에만 있는 노이즈일 가능성 ↑
     - test area max = 121.9㎡  → train의 200㎡↑는 비현실적
     - test Y는 없지만 서울 전세 287억은 물리적으로 불가능 (호텔/빌딩급)
  2) 분포가 매우 right-skewed: max가 99.5% 분위의 50배 이상으로 튐
  3) 이런 행을 학습에 넣으면 트리 모델이 극단 split을 학습해서 일반 행에 손해

■ 낮은 값이 노이즈가 아닌 근거 (확인 필요)
  - Y min(601만원) 행들: 대부분 오피스텔/연립의 작은 면적 → 월세 보증금일 수 있음(현실 가능)
  - area min(10㎡): test에도 10㎡대 있음 → 정상 (원룸/도시형생활주택)
  → 낮은쪽은 그대로 유지

■ 정확히 0.5% / 0.3% 컷 임계값
  - Y > {y_995:,.0f}만원 ({y_995/10000:.1f}억) 제거 → 약 0.5%
  - area > {a_997:.1f}㎡ 제거 → 약 0.3%
""")
