"""
3-model 앙상블 (CatBoost는 4-fold이므로 OOF valid 영역만 평가).
- 가중치 최적화는 CAT이 valid한 80% 영역에서 진행
- test 예측은 LGBM/XGB는 5-fold 평균, CAT은 4-fold 평균 (이미 npy에 저장됨)
- 최종 answer.csv 생성
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

DATA = Path("data"); OUT = Path("outputs")

train = pd.read_parquet(DATA / "train_fe.parquet")
y = train["Y"].values.astype(np.float32)
y_log = np.log1p(y)
folds = np.load(DATA / "folds.npy")

oof_l = np.load(OUT / "oof_lgbm.npy")
oof_x = np.load(OUT / "oof_xgb.npy")
oof_c = np.load(OUT / "oof_cat.npy")
te_l  = np.load(OUT / "test_lgbm.npy")
te_x  = np.load(OUT / "test_xgb.npy")
te_c  = np.load(OUT / "test_cat.npy")

# CatBoost OOF valid 마스크: fold 0,1,2,3만 사용
cat_valid_mask = np.isin(folds, [0,1,2,3])
print(f"CatBoost OOF valid 비율: {cat_valid_mask.sum()/len(folds)*100:.1f}%")

def rmse(yt, yp): return np.sqrt(mean_squared_error(yt, yp))

# 단독 점수 (각 모델별 valid 영역)
r_l = rmse(y_log, oof_l); ro_l = rmse(y, np.expm1(oof_l))
r_x = rmse(y_log, oof_x); ro_x = rmse(y, np.expm1(oof_x))
r_c = rmse(y_log[cat_valid_mask], oof_c[cat_valid_mask]); ro_c = rmse(y[cat_valid_mask], np.expm1(oof_c[cat_valid_mask]))
print(f"\n[단독 모델 OOF RMSE]")
print(f"  LGBM (5fold): log={r_l:.4f}  orig={ro_l:.2f} 만원")
print(f"  XGB  (5fold): log={r_x:.4f}  orig={ro_x:.2f} 만원")
print(f"  CAT  (4fold): log={r_c:.4f}  orig={ro_c:.2f} 만원")

# 앙상블 최적화는 cat valid 영역(80%)에서만 진행
y_log_v = y_log[cat_valid_mask]
y_v     = y[cat_valid_mask]
oof_l_v = oof_l[cat_valid_mask]
oof_x_v = oof_x[cat_valid_mask]
oof_c_v = oof_c[cat_valid_mask]

# 1) 균등 평균
oof_avg = (oof_l_v + oof_x_v + oof_c_v) / 3
r_avg = rmse(y_log_v, oof_avg); ro_avg = rmse(y_v, np.expm1(oof_avg))
print(f"\n[3모델 단순 평균]  log={r_avg:.4f}  orig={ro_avg:.2f} 만원")

# 2) 그리드 서치 (3차원)
print("\n[3차원 그리드 서치 (0.05 step)...]")
best = (None, np.inf)
for a in np.arange(0, 1.01, 0.05):
    for b in np.arange(0, 1.01-a, 0.05):
        c = 1.0 - a - b
        if c < 0: continue
        pred = a*oof_l_v + b*oof_x_v + c*oof_c_v
        r = rmse(y_log_v, pred)
        if r < best[1]:
            best = ((a, b, c), r)
opt_w, opt_log = best
opt_orig = rmse(y_v, np.expm1(opt_w[0]*oof_l_v + opt_w[1]*oof_x_v + opt_w[2]*oof_c_v))
print(f"  최적 가중: LGBM={opt_w[0]:.2f}, XGB={opt_w[1]:.2f}, CAT={opt_w[2]:.2f}")
print(f"  RMSE: log={opt_log:.4f}  orig={opt_orig:.2f} 만원")

# 2-model 비교 (CAT 빼고)
best2 = (None, np.inf)
for a in np.arange(0, 1.01, 0.01):
    pred = a*oof_l_v + (1-a)*oof_x_v
    r = rmse(y_log_v, pred)
    if r < best2[1]:
        best2 = ((a, 1-a), r)
opt2_w, opt2_log = best2
print(f"\n[2모델 (LGBM+XGB) 비교]")
print(f"  최적: LGBM={opt2_w[0]:.2f}, XGB={opt2_w[1]:.2f}, log={opt2_log:.4f}")

# 개선폭
improve = (opt2_log - opt_log) / opt2_log * 100
print(f"\n[CatBoost 추가 효과]")
print(f"  2모델 → 3모델 log RMSE 개선: {opt2_log:.4f} → {opt_log:.4f} ({improve:+.3f}%)")

# 최종 test 예측 (가중 평균)
te_final = opt_w[0]*te_l + opt_w[1]*te_x + opt_w[2]*te_c
pred = np.expm1(te_final)
pred = np.clip(pred, 600, 58000)

sub = pd.read_csv(DATA / "sample_submission.csv")
sub["Y"] = pred
sub.to_csv(OUT / "answer.csv", index=False)
print(f"\n✅ outputs/answer.csv 생성")
print(f"   rows: {len(sub):,}")
print(f"   Y min/mean/max: {sub['Y'].min():.0f} / {sub['Y'].mean():.0f} / {sub['Y'].max():.0f}")
print(f"   train Y ref:   {y.min():.0f} / {y.mean():.0f} / {y.max():.0f}")

with open(OUT / "metrics_ensemble_3model.json", "w") as f:
    json.dump({
        "single": {
            "lgbm_log": float(r_l), "lgbm_orig": float(ro_l),
            "xgb_log":  float(r_x), "xgb_orig":  float(ro_x),
            "cat_log":  float(r_c), "cat_orig":  float(ro_c),
        },
        "ensemble_2model_best_log": float(opt2_log),
        "ensemble_3model_best_log": float(opt_log),
        "ensemble_3model_best_orig": float(opt_orig),
        "weights_3model": {"lgbm": float(opt_w[0]), "xgb": float(opt_w[1]), "cat": float(opt_w[2])},
        "weights_2model": {"lgbm": float(opt2_w[0]), "xgb": float(opt2_w[1])},
    }, f, indent=2)
print("   metrics_ensemble_3model.json saved")
