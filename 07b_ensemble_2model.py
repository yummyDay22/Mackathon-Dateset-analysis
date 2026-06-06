"""
2-model ensemble (LGBM + XGB) — CatBoost 끝나기 전 중간 결과 확인용.
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

oof_lgbm = np.load(OUT / "oof_lgbm.npy")
oof_xgb  = np.load(OUT / "oof_xgb.npy")
te_lgbm  = np.load(OUT / "test_lgbm.npy")
te_xgb   = np.load(OUT / "test_xgb.npy")

def rmse(yt, yp): return np.sqrt(mean_squared_error(yt, yp))

r_l_log, r_l_orig = rmse(y_log, oof_lgbm), rmse(y, np.expm1(oof_lgbm))
r_x_log, r_x_orig = rmse(y_log, oof_xgb),  rmse(y, np.expm1(oof_xgb))
print(f"[Single OOF]")
print(f"  LGBM: log={r_l_log:.4f}  orig={r_l_orig:.2f} 만원")
print(f"  XGB : log={r_x_log:.4f}  orig={r_x_orig:.2f} 만원")

# 1) 50/50 단순 평균
oof_a = 0.5*oof_lgbm + 0.5*oof_xgb
te_a  = 0.5*te_lgbm  + 0.5*te_xgb
print(f"\n[Simple 50/50]  log={rmse(y_log,oof_a):.4f}  orig={rmse(y,np.expm1(oof_a)):.2f}")

# 2) RMSE 역수 가중
ws = np.array([1/r_l_log, 1/r_x_log]); ws /= ws.sum()
oof_w = ws[0]*oof_lgbm + ws[1]*oof_xgb
te_w  = ws[0]*te_lgbm  + ws[1]*te_xgb
print(f"[Weighted {ws[0]:.2f}/{ws[1]:.2f}]  log={rmse(y_log,oof_w):.4f}  orig={rmse(y,np.expm1(oof_w)):.2f}")

# 3) 그리드 서치 최적 가중
best = (None, np.inf)
for a in np.arange(0, 1.01, 0.02):
    b = 1.0 - a
    pred = a*oof_lgbm + b*oof_xgb
    r = rmse(y_log, pred)
    if r < best[1]:
        best = ((a, b), r)
opt, r_opt = best
print(f"[Optimal {opt[0]:.2f}/{opt[1]:.2f}]  log={r_opt:.4f}  orig={rmse(y,np.expm1(opt[0]*oof_lgbm+opt[1]*oof_xgb)):.2f}")

# 최종 answer.csv 생성 (optimal 가중)
te_final = opt[0]*te_lgbm + opt[1]*te_xgb
pred = np.expm1(te_final)
pred = np.clip(pred, 600, 58000)
sub = pd.read_csv(DATA / "sample_submission.csv")
sub["Y"] = pred
sub.to_csv(OUT / "answer_2model.csv", index=False)
print(f"\n✅ Saved outputs/answer_2model.csv (rows={len(sub):,})")
print(f"   Y min/mean/max = {sub['Y'].min():.0f} / {sub['Y'].mean():.0f} / {sub['Y'].max():.0f}")
