"""
마포구 두 인사이트 데이터 검증:
1) 한강변 vs 홍대상권 동의 평균 Y 차이
2) 마포구 스타벅스 밀도 → 평균 Y 낮음 (강남과 반대) — 검증
"""
import pandas as pd, numpy as np
from pathlib import Path
import sys; sys.path.insert(0, ".")
from seoul_dong_map import DONG_TO_GU

DATA = Path("data")
train = pd.read_parquet(DATA / "train_eda.parquet")
train["gu"] = train["dong"].map(DONG_TO_GU).fillna("기타")

# ============================================================
# 1) 마포 한강변 vs 홍대상권 동 평균
# ============================================================
mapo = train[train["gu"]=="마포구"]
print(f"마포구 거래 수: {len(mapo):,}")
print("\n[마포구 동별 평균 Y, 거래 100건 이상]")
mp = mapo.groupby("dong")["Y"].agg(["mean","count","median"]).sort_values("mean", ascending=False)
mp = mp[mp["count"]>=100]
for d, row in mp.iterrows():
    print(f"  {d}: 평균 {row['mean']:>7,.0f}만원 ({int(row['count']):,}건), 중앙값 {row['median']:,.0f}")

# 한강변 vs 홍대권
riverside = ["토정동", "현석동", "창전동", "용강동", "마포동", "신공덕동", "공덕동"]  # 한강·도심접근권
hongdae   = ["서교동", "동교동", "합정동", "망원동", "연남동", "성산동"]  # 홍대상권권
rs_mean = train[train["dong"].isin(riverside)]["Y"].mean()
hd_mean = train[train["dong"].isin(hongdae)]["Y"].mean()
print(f"\n한강·도심권(토정·현석·창전·용강·마포·신공덕·공덕) 평균: {rs_mean:,.0f}만원")
print(f"홍대상권권(서교·동교·합정·망원·연남·성산) 평균: {hd_mean:,.0f}만원")
print(f"비율: {rs_mean/hd_mean:.2f}배")

# ============================================================
# 2) 마포 신축 vs 구축 비교 (도심 입지 = 안전자산형)
# ============================================================
print("\n[마포 신축 vs 구축 평균 Y]")
mp_new = mapo[mapo["building_age"]<=3]["Y"].mean()
mp_old = mapo[mapo["building_age"]>=20]["Y"].mean()
print(f"신축(0-3년): {mp_new:,.0f}만원 ({(mapo['building_age']<=3).sum():,}건)")
print(f"구축(20년+): {mp_old:,.0f}만원 ({(mapo['building_age']>=20).sum():,}건)")
print(f"비율: 구축이 신축의 {mp_old/mp_new:.2f}배")

# 마포 구축 아파트 동별
print("\n[마포 구축 아파트(>=20년) 동별 평균 Y]")
core_old = mapo[(mapo["building_age"]>=20)&(mapo["building_type"]=="아파트")]
core_dong = core_old.groupby("dong")["Y"].agg(["mean","count"])
core_dong = core_dong[core_dong["count"]>=50].sort_values("mean", ascending=False)
for d, row in core_dong.head(8).iterrows():
    print(f"  {d}: {row['mean']:,.0f}만원 ({int(row['count']):,}건)")

# ============================================================
# 3) 마포 스타벅스 vs 평균 Y
# ============================================================
print("\n[마포구 스타벅스 개수별 평균 Y]")
sb_bins = pd.cut(mapo["starbucks_count"], bins=[-0.5, 0, 1, 3, 5, 100], labels=["0","1","2-3","4-5","5+"])
mp_sb = mapo.groupby(sb_bins, observed=True)["Y"].agg(["mean","count"])
for k, row in mp_sb.iterrows():
    print(f"  {k}개: 평균 {row['mean']:,.0f}만원 ({int(row['count']):,}건)")

# ============================================================
# 4) 강남 스타벅스 vs 평균 Y (대조군)
# ============================================================
gn = train[train["gu"]=="강남구"]
print("\n[강남구 스타벅스 개수별 평균 Y (대조군)]")
gn_sb = gn.groupby(sb_bins, observed=True)["Y"].agg(["mean","count"])
gn_sb_groups = pd.cut(gn["starbucks_count"], bins=[-0.5, 0, 1, 3, 5, 100], labels=["0","1","2-3","4-5","5+"])
gn_sb = gn.groupby(gn_sb_groups, observed=True)["Y"].agg(["mean","count"])
for k, row in gn_sb.iterrows():
    print(f"  {k}개: 평균 {row['mean']:,.0f}만원 ({int(row['count']):,}건)")

# 서울 전체 (참고)
print("\n[서울 전체 스타벅스 개수별 평균 Y]")
all_sb = pd.cut(train["starbucks_count"], bins=[-0.5, 0, 1, 3, 5, 100], labels=["0","1","2-3","4-5","5+"])
sl_sb = train.groupby(all_sb, observed=True)["Y"].agg(["mean","count"])
for k, row in sl_sb.iterrows():
    print(f"  {k}개: 평균 {row['mean']:,.0f}만원 ({int(row['count']):,}건)")

# ============================================================
# 5) 마포 스타벅스 5+ 지역의 카테고리 분포
# ============================================================
mp_high_sb = mapo[mapo["starbucks_count"]>=5]
print(f"\n[마포 스타벅스 5+ 지역 ({len(mp_high_sb):,}건) 분석]")
print(f"  카테고리 분포:")
print(mp_high_sb["building_type"].value_counts(normalize=True).round(3).to_string())
print(f"  평균 면적: {mp_high_sb['area_m2'].mean():.1f}㎡")
print(f"  평균 건물나이: {mp_high_sb['building_age'].mean():.1f}년")
print(f"  Top 3 동: {mp_high_sb['dong'].value_counts().head(3).to_dict()}")

mp_zero_sb = mapo[mapo["starbucks_count"]==0]
print(f"\n[마포 스타벅스 0개 지역 ({len(mp_zero_sb):,}건) 분석]")
print(f"  카테고리 분포:")
print(mp_zero_sb["building_type"].value_counts(normalize=True).round(3).to_string())
print(f"  평균 면적: {mp_zero_sb['area_m2'].mean():.1f}㎡")
print(f"  평균 건물나이: {mp_zero_sb['building_age'].mean():.1f}년")
