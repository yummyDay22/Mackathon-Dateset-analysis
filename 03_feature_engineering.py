"""
Phase 4: Feature Engineering + Encoding (35개 새 변수 + 인코딩 4개)

누수 방지 원칙
  - 모든 target encoding은 5-Fold OOF로 train 채움
  - test 매핑은 train 전체로 한 번에
  - 시간 rolling은 shift(1).rolling()으로 자기 자신 제외
  - 누락 매핑은 global_mean / 0 으로 fallback

생성 변수 (총 36개 새 변수 + 4개 인코딩)
  A. 시간/추세         9개  A1~A9
  B. 카테고리 통계     8개  B1~B8 (OOF 5종 + 전체통계 3종)
  C. 도메인 (지역/나이) 4개  C1~C4
  D. 면적/층           3개  D1~D3
  E. 인프라            4개  E1~E4
  F. 금융              3개  F1~F3
  G. 교호작용          5개  G1~G5
  H. 인코딩            4개  H1 (dong_label), H2 (3 dummies)

도로명 원본 컬럼은 drop (road_mean_Y_smooth / road_count 로 대체)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from seoul_dong_map import DONG_TO_GU, GANGNAM3, RIVERSIDE_DONGS

DATA = Path("data")
SEED = 42
N_FOLDS = 5

# ------------------------------------------------------------------
# 0) 로드
# ------------------------------------------------------------------
train = pd.read_parquet(DATA / "train_clean.parquet")
test  = pd.read_parquet(DATA / "test_clean.parquet")
print(f"Loaded: train {train.shape}, test {test.shape}")

# 일관 타입
for df in [train, test]:
    df["dong"] = df["dong"].astype(str)
    df["building_type"] = df["building_type"].astype(str)
    df["road_id"] = df["road_id"].astype("int64")
    df["contract_ym"] = df["contract_ym"].astype("int64")

# 동→구 매핑 누락 체크
all_dongs = set(train["dong"].unique()) | set(test["dong"].unique())
unmapped = sorted([d for d in all_dongs if d not in DONG_TO_GU])
print(f"\n[동→구 매핑] 매핑됨 {len(all_dongs)-len(unmapped)}/{len(all_dongs)}, 누락 {len(unmapped)}: {unmapped}")

# 5-Fold 인덱스
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_idx = np.zeros(len(train), dtype=np.int8)
for f, (_, va) in enumerate(kf.split(train)):
    fold_idx[va] = f
np.save(DATA / "folds.npy", fold_idx)
print(f"5-Fold indices saved: data/folds.npy")

GLOBAL_MEAN_Y = train["Y"].mean()
print(f"global_mean_Y = {GLOBAL_MEAN_Y:.2f}")


# ============================================================
# A. 시간/추세 (9개)
# ============================================================
print("\n[A] Time / trend features ...")

def add_time_features(df):
    df["year"]    = (df["contract_ym"] // 100).astype("int16")
    df["month"]   = (df["contract_ym"] % 100).astype("int8")
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype("int8")
    df["year_idx"]= (df["year"] - 2011).astype("int16")
    df["is_moving_season"] = df["month"].isin([3,4,9,10]).astype("int8")

add_time_features(train)
add_time_features(test)

# A6, A7: 동별 직전 3개월/12개월 평균 Y
# (동, 계약년월) 단위로 train 평균을 집계한 후 시계열 rolling
print("  computing dong rolling (3m, 12m)...")
monthly = (train.groupby(["dong", "contract_ym"])["Y"]
                .mean().reset_index(name="month_mean_Y"))
monthly = monthly.sort_values(["dong", "contract_ym"]).reset_index(drop=True)
g = monthly.groupby("dong", group_keys=False)["month_mean_Y"]
monthly["dong_3m_mean_Y"]  = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
monthly["dong_12m_mean_Y"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())

# dict 매핑 (동, ym) -> rolling
key3  = dict(zip(zip(monthly["dong"], monthly["contract_ym"]), monthly["dong_3m_mean_Y"]))
key12 = dict(zip(zip(monthly["dong"], monthly["contract_ym"]), monthly["dong_12m_mean_Y"]))

def map_rolling(df, mapping, default):
    keys = list(zip(df["dong"], df["contract_ym"]))
    vals = np.array([mapping.get(k, default) for k in keys], dtype=np.float32)
    return np.nan_to_num(vals, nan=default)

train["dong_3m_mean_Y"]  = map_rolling(train, key3,  GLOBAL_MEAN_Y)
train["dong_12m_mean_Y"] = map_rolling(train, key12, GLOBAL_MEAN_Y)
test["dong_3m_mean_Y"]   = map_rolling(test, key3,  GLOBAL_MEAN_Y)
test["dong_12m_mean_Y"]  = map_rolling(test, key12, GLOBAL_MEAN_Y)

# A8: 계약년월별 train 전체 평균 Y
print("  market_mean_Y_at_t ...")
mkt = train.groupby("contract_ym")["Y"].mean().to_dict()
train["market_mean_Y_at_t"] = train["contract_ym"].map(mkt).astype("float32")
test["market_mean_Y_at_t"]  = test["contract_ym"].map(mkt).fillna(GLOBAL_MEAN_Y).astype("float32")

# A9: policy_era
def policy(ym):
    if ym < 202007:  return 0   # 평시
    if ym < 202108:  return 1   # 임대차3법
    return 2                    # 금리인상기
train["policy_era"] = train["contract_ym"].map(policy).astype("int8")
test["policy_era"]  = test["contract_ym"].map(policy).astype("int8")

print("  A: year, month, quarter, year_idx, is_moving_season, dong_3m_mean_Y, dong_12m_mean_Y, market_mean_Y_at_t, policy_era")


# ============================================================
# B. 카테고리 통계 (8개) — 모두 OOF
# ============================================================
print("\n[B] Category target encodings (5-Fold OOF) ...")

def oof_target_stat(train_df, test_df, col, agg, smoothing=0.0, default=None):
    """OOF target encoding for given aggregation.
    agg: callable(series) -> scalar, e.g. np.mean, np.median, np.std
    smoothing: shrink toward default (global stat) for low-count groups
    """
    if default is None:
        default = agg(train_df["Y"].values)
    oof = np.full(len(train_df), default, dtype=np.float32)
    for f in range(N_FOLDS):
        tr_mask = fold_idx != f
        va_mask = fold_idx == f
        grp = train_df.loc[tr_mask].groupby(col)["Y"].agg(["mean", "median", "std", "count"])
        if agg is np.mean:    stat = grp["mean"]
        elif agg is np.median:stat = grp["median"]
        elif agg is np.std:   stat = grp["std"].fillna(default)
        else:                 raise ValueError("agg unsupported")
        if smoothing > 0:
            stat = (grp["count"] * stat + smoothing * default) / (grp["count"] + smoothing)
        mapping = stat.to_dict()
        keys = train_df.loc[va_mask, col].values
        oof[va_mask] = pd.Series(keys).map(mapping).fillna(default).astype(np.float32).values
    # full train -> test
    grp_full = train_df.groupby(col)["Y"].agg(["mean","median","std","count"])
    if agg is np.mean:    stat_full = grp_full["mean"]
    elif agg is np.median:stat_full = grp_full["median"]
    elif agg is np.std:   stat_full = grp_full["std"].fillna(default)
    if smoothing > 0:
        stat_full = (grp_full["count"] * stat_full + smoothing * default) / (grp_full["count"] + smoothing)
    mapping_full = stat_full.to_dict()
    te = test_df[col].map(mapping_full).fillna(default).astype(np.float32).values
    return oof, te

# B1 dong_mean_Y
oof, te = oof_target_stat(train, test, "dong", np.mean, smoothing=20.0)
train["dong_mean_Y"] = oof
test["dong_mean_Y"]  = te
# B2 dong_median_Y
oof, te = oof_target_stat(train, test, "dong", np.median, smoothing=20.0)
train["dong_median_Y"] = oof
test["dong_median_Y"]  = te
# B3 dong_std_Y
GLOBAL_STD_Y = float(train["Y"].std())
oof, te = oof_target_stat(train, test, "dong", np.std, smoothing=20.0, default=GLOBAL_STD_Y)
train["dong_std_Y"] = oof
test["dong_std_Y"]  = te

# B4 dong_count = log1p(거래수) — train 전체 기준 (누수 영향 미미)
dong_cnt = train["dong"].value_counts().to_dict()
train["dong_count"] = np.log1p(train["dong"].map(dong_cnt)).astype("float32")
test["dong_count"]  = np.log1p(test["dong"].map(dong_cnt).fillna(0)).astype("float32")

# B5 road_mean_Y_smooth
print("  road_mean_Y_smooth (smoothing k=20) ...")
road_grp = train.groupby("road_id")["Y"].agg(["mean","count"])
K_SMOOTH = 20.0
road_grp["smooth"] = (road_grp["count"]*road_grp["mean"] + K_SMOOTH*GLOBAL_MEAN_Y) / (road_grp["count"] + K_SMOOTH)
road_map = road_grp["smooth"].to_dict()
train["road_mean_Y_smooth"] = train["road_id"].map(road_map).fillna(GLOBAL_MEAN_Y).astype("float32")
test["road_mean_Y_smooth"]  = test["road_id"].map(road_map).fillna(GLOBAL_MEAN_Y).astype("float32")
n_road_fallback = test["road_id"].map(road_map).isna().sum()
print(f"    test road_id fallback to global_mean: {n_road_fallback:,} rows")

# B6 road_count
road_cnt = train["road_id"].value_counts().to_dict()
train["road_count"] = np.log1p(train["road_id"].map(road_cnt)).astype("float32")
test["road_count"]  = np.log1p(test["road_id"].map(road_cnt).fillna(0)).astype("float32")

# B7 type_mean_Y (OOF)
oof, te = oof_target_stat(train, test, "building_type", np.mean, smoothing=20.0)
train["type_mean_Y"] = oof
test["type_mean_Y"]  = te

# B8 dong_type_mean_Y (OOF on composite key)
train["_dt"] = train["dong"] + "|" + train["building_type"]
test["_dt"]  = test["dong"]  + "|" + test["building_type"]
oof, te = oof_target_stat(train, test, "_dt", np.mean, smoothing=20.0)
train["dong_type_mean_Y"] = oof
test["dong_type_mean_Y"]  = te
train = train.drop(columns="_dt")
test  = test.drop(columns="_dt")

print("  B: dong_mean/median/std/count_Y, road_mean_Y_smooth, road_count, type_mean_Y, dong_type_mean_Y")


# ============================================================
# C. 도메인 (4개)
# ============================================================
print("\n[C] Domain features ...")

def area_bin(a):
    if pd.isna(a): return np.nan
    if a < 20:   return 0
    if a < 40:   return 1
    if a < 85:   return 2
    if a < 135:  return 3
    return 4

def age_group(x):
    if pd.isna(x): return 1   # 결측은 일반
    if x <= 3:   return 0   # 신축
    if x <= 19:  return 1   # 일반
    if x <= 29:  return 2   # 노후
    return 3                # 재건축임박

for df in [train, test]:
    df["area_bin"]    = df["area_m2"].map(area_bin).astype("float32")
    df["age_group"]   = df["building_age"].map(age_group).astype("int8")
    gu = df["dong"].map(DONG_TO_GU)
    df["is_gangnam_3"]= gu.isin(GANGNAM3).astype("int8")
    df["is_riverside"]= df["dong"].isin(RIVERSIDE_DONGS).astype("int8")

n_gn_tr = train["is_gangnam_3"].sum()
n_rs_tr = train["is_riverside"].sum()
print(f"  is_gangnam_3: train {n_gn_tr:,} ({n_gn_tr/len(train)*100:.1f}%), "
      f"is_riverside: train {n_rs_tr:,} ({n_rs_tr/len(train)*100:.1f}%)")
print("  C: area_bin, age_group, is_gangnam_3, is_riverside")


# ============================================================
# D. 면적/층 (3개)
# ============================================================
print("\n[D] Area / floor features ...")
def floor_group(x):
    if pd.isna(x): return 2  # 결측은 중층
    if x <= 0:   return 0   # 지하
    if x <= 3:   return 1   # 저층
    if x <= 10:  return 2   # 중층
    return 3                # 고층

for df in [train, test]:
    df["log_area"]       = np.log1p(df["area_m2"]).astype("float32")
    df["floor_group"]    = df["floor"].map(floor_group).astype("int8")
    df["is_underground"] = (df["floor"].fillna(1) <= 0).astype("int8")

print("  D: log_area, floor_group, is_underground")


# ============================================================
# E. 인프라 (4개)
# ============================================================
print("\n[E] Infrastructure features ...")
for df in [train, test]:
    mart = df["mart_count"].fillna(0)
    sbux = df["starbucks_count"].fillna(0)
    df["infra_score"] = (mart + sbux).astype("float32")
    sta = df["station_min_dist"].fillna(0)
    df["station_proximity"] = (1.0 / (sta + 0.1)).astype("float32")
    df.loc[df["station_min_dist"].isna(), "station_proximity"] = 0.0
    uni = df["univ_min_dist"].fillna(0)
    df["univ_proximity"] = (1.0 / (uni + 0.1)).astype("float32")
    df.loc[df["univ_min_dist"].isna(), "univ_proximity"] = 0.0
    df["missing_infra_flag"] = (df["mart_count"].isna() | df["starbucks_count"].isna()).astype("int8")

print("  E: infra_score, station_proximity, univ_proximity, missing_infra_flag")


# ============================================================
# F. 금융 (3개)
# ============================================================
print("\n[F] Finance features ...")
# 계약년월별 unique 금리/종가
rate_map = train.groupby("contract_ym")["base_rate"].mean().to_dict()
def shift_ym(ym, months):
    y, m = ym // 100, ym % 100
    m -= months
    while m <= 0:
        m += 12; y -= 1
    return y*100 + m

def base_rate_diff(df, months):
    rate_now  = df["contract_ym"].map(rate_map)
    rate_prev = df["contract_ym"].map(lambda x: rate_map.get(shift_ym(int(x), months), np.nan))
    return (rate_now - rate_prev).fillna(0).astype("float32")

train["base_rate_yoy"]       = base_rate_diff(train, 12)
train["base_rate_3m_change"] = base_rate_diff(train, 3)
test["base_rate_yoy"]        = base_rate_diff(test, 12)
test["base_rate_3m_change"]  = base_rate_diff(test, 3)

train["kospi_close_log"] = np.log1p(train["kospi_close"]).astype("float32")
test["kospi_close_log"]  = np.log1p(test["kospi_close"]).astype("float32")

print("  F: base_rate_yoy, base_rate_3m_change, kospi_close_log")


# ============================================================
# H. 인코딩 (H1 dong_label, H2 type onehot) — G에서 type_label 사용
# ============================================================
print("\n[H] Encoding ...")
le_dong = LabelEncoder()
all_dong_list = sorted(set(train["dong"]) | set(test["dong"]))
le_dong.fit(all_dong_list)
train["dong_label"] = le_dong.transform(train["dong"]).astype("int16")
test["dong_label"]  = le_dong.transform(test["dong"]).astype("int16")

# type one-hot
TYPE_MAP = {"아파트": 0, "오피스텔": 1, "연립다세대": 2}
for df in [train, test]:
    df["type_label"] = df["building_type"].map(TYPE_MAP).astype("int8")
    df["is_apt"]     = (df["building_type"] == "아파트").astype("int8")
    df["is_office"]  = (df["building_type"] == "오피스텔").astype("int8")
    df["is_villa"]   = (df["building_type"] == "연립다세대").astype("int8")

print("  H: dong_label, type_label, is_apt, is_office, is_villa")


# ============================================================
# G. 교호작용 (5개)
# ============================================================
print("\n[G] Interaction features ...")
age_med   = train["building_age"].median()
floor_med = train["floor"].median()

for df in [train, test]:
    df["area_x_dong_meanY"]   = (df["log_area"] * df["dong_mean_Y"]).astype("float32")
    df["age_x_type"]          = (df["building_age"].fillna(age_med) * df["type_label"]).astype("float32")
    df["station_prox_x_type"] = (df["station_proximity"] * df["type_label"]).astype("float32")
    df["floor_x_type"]        = (df["floor"].fillna(floor_med) * df["type_label"]).astype("float32")
    df["gangnam_x_log_area"]  = (df["is_gangnam_3"] * df["log_area"]).astype("float32")

print("  G: area_x_dong_meanY, age_x_type, station_prox_x_type, floor_x_type, gangnam_x_log_area")


# ============================================================
# 도로명 원본 drop
# ============================================================
train = train.drop(columns=["road_id"])
test  = test.drop(columns=["road_id"])


# ============================================================
# 메모리 다운캐스트
# ============================================================
print("\n[Mem] downcast ...")
def downcast(df):
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        cmin, cmax = df[c].min(), df[c].max()
        if cmin >= -128 and cmax <= 127:        df[c] = df[c].astype("int8")
        elif cmin >= -32768 and cmax <= 32767:  df[c] = df[c].astype("int16")
        elif cmin >= -2**31 and cmax <= 2**31-1:df[c] = df[c].astype("int32")
downcast(train); downcast(test)


# ============================================================
# 검증 출력
# ============================================================
print("\n" + "="*60)
print("[검증]")
print("="*60)

# 1) train/test 컬럼 일치 (Y 제외)
tr_cols = [c for c in train.columns if c != "Y"]
te_cols = list(test.columns)
tr_only = set(tr_cols) - set(te_cols)
te_only = set(te_cols) - set(tr_cols)
print(f"\n1) 컬럼 일치 (Y 제외):")
print(f"   train-only: {tr_only}")
print(f"   test-only : {te_only}")
print(f"   ✅ 일치" if (not tr_only and not te_only) else "   ⚠ 불일치")

# 2) NaN 비율
print(f"\n2) NaN 비율 (train):")
miss = train.isna().mean() * 100
for c, p in miss[miss > 0].items():
    print(f"   {c}: {p:.2f}%")

# 3) 새 변수 min/mean/max
NEW_VARS = ["year","month","quarter","year_idx","is_moving_season",
            "dong_3m_mean_Y","dong_12m_mean_Y","market_mean_Y_at_t","policy_era",
            "dong_mean_Y","dong_median_Y","dong_std_Y","dong_count",
            "road_mean_Y_smooth","road_count","type_mean_Y","dong_type_mean_Y",
            "area_bin","age_group","is_gangnam_3","is_riverside",
            "log_area","floor_group","is_underground",
            "infra_score","station_proximity","univ_proximity","missing_infra_flag",
            "base_rate_yoy","base_rate_3m_change","kospi_close_log",
            "area_x_dong_meanY","age_x_type","station_prox_x_type","floor_x_type","gangnam_x_log_area",
            "dong_label","type_label","is_apt","is_office","is_villa"]

print(f"\n3) 새 변수 분포 (train):")
print(f"   {'name':<24} {'min':>10} {'mean':>10} {'max':>12} {'dtype':>10}")
for c in NEW_VARS:
    if c in train.columns:
        s = train[c].astype(float)
        print(f"   {c:<24} {s.min():>10.2f} {s.mean():>10.2f} {s.max():>12.2f} {str(train[c].dtype):>10}")

# 4) dong_mean_Y vs Y 상관
corr_dong = train["dong_mean_Y"].corr(train["Y"])
print(f"\n4) corr(dong_mean_Y, Y) = {corr_dong:.4f}  ({'✅ >0.5 강함' if corr_dong>0.5 else '⚠ 약함'})")

# 5) 강남3 vs 비강남 Y 평균
yg = train.loc[train["is_gangnam_3"]==1, "Y"].mean()
yn = train.loc[train["is_gangnam_3"]==0, "Y"].mean()
print(f"5) Y mean: 강남3={yg:,.0f}, 비강남={yn:,.0f}, 비율={yg/yn:.2f}배 ({'✅ >=1.5' if yg/yn>=1.5 else '⚠ <1.5'})")

# 6) log_area skewness
skew = float(((train["log_area"] - train["log_area"].mean())**3).mean() / train["log_area"].std()**3)
print(f"6) log_area skewness = {skew:.3f}  ({'✅ |s|<1' if abs(skew)<1 else '⚠'})")

# 7) test fallback 건수
print(f"7) test road_id fallback: {n_road_fallback:,}")

# 8) 메모리
mem_tr = train.memory_usage(deep=True).sum() / 1024**2
mem_te = test.memory_usage(deep=True).sum() / 1024**2
print(f"8) memory: train {mem_tr:.1f} MB, test {mem_te:.1f} MB")
print(f"   train shape: {train.shape}, test shape: {test.shape}")


# ============================================================
# 저장
# ============================================================
print("\n[Save]")
train.to_parquet(DATA / "train_fe.parquet", index=False)
test.to_parquet(DATA / "test_fe.parquet", index=False)

feature_groups = {
    "A_time": ["year","month","quarter","year_idx","is_moving_season",
               "dong_3m_mean_Y","dong_12m_mean_Y","market_mean_Y_at_t","policy_era"],
    "B_stat": ["dong_mean_Y","dong_median_Y","dong_std_Y","dong_count",
               "road_mean_Y_smooth","road_count","type_mean_Y","dong_type_mean_Y"],
    "C_domain": ["area_bin","age_group","is_gangnam_3","is_riverside"],
    "D_area_floor": ["log_area","floor_group","is_underground"],
    "E_infra": ["infra_score","station_proximity","univ_proximity","missing_infra_flag"],
    "F_finance": ["base_rate_yoy","base_rate_3m_change","kospi_close_log"],
    "G_interaction": ["area_x_dong_meanY","age_x_type","station_prox_x_type","floor_x_type","gangnam_x_log_area"],
    "H_encoding": ["dong_label","type_label","is_apt","is_office","is_villa"],
}
with open(DATA / "feature_groups.json", "w", encoding="utf-8") as f:
    json.dump(feature_groups, f, ensure_ascii=False, indent=2)
with open(DATA / "dong_to_gu.json", "w", encoding="utf-8") as f:
    json.dump(DONG_TO_GU, f, ensure_ascii=False, indent=2)
with open(DATA / "riverside_dong_list.json", "w", encoding="utf-8") as f:
    json.dump(sorted(RIVERSIDE_DONGS), f, ensure_ascii=False, indent=2)
with open(DATA / "unmapped_dongs.json", "w", encoding="utf-8") as f:
    json.dump(unmapped, f, ensure_ascii=False, indent=2)

print(f"  saved: data/train_fe.parquet, data/test_fe.parquet")
print(f"  saved: data/feature_groups.json, dong_to_gu.json, riverside_dong_list.json")
print(f"  saved: data/unmapped_dongs.json ({len(unmapped)} unmapped)")
print("=" * 60)
print("Phase 4 완료!")
