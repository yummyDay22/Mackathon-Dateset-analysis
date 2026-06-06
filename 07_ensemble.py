"""
Phase 5-4: 3-model 앙상블 + answer.csv 생성
  - 각 모델 OOF에서 RMSE를 계산
  - 가중치 = 1/RMSE_log (낮은 RMSE에 더 큰 가중치) 후 정규화
  - log scale에서 가중평균 → expm1 역변환
  - 단순 평균과도 비교 보고
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

DATA = Path("data")
OUT  = Path("outputs")

train = pd.read_parquet(DATA / "train_fe.parquet")
test  = pd.read_parquet(DATA / "test_fe.parquet")

y = train["Y"].values.astype(np.float32)
y_log = np.log1p(y)

oof_lgbm = np.load(OUT / "oof_lgbm.npy")
oof_xgb  = np.load(OUT / "oof_xgb.npy")
oof_cat  = np.load(OUT / "oof_cat.npy")
te_lgbm  = np.load(OUT / "test_lgbm.npy")
te_xgb   = np.load(OUT / "test_xgb.npy")
te_cat   = np.load(OUT / "test_cat.npy")

# 각 모델 OOF RMSE
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

r_lgbm_log = rmse(y_log, oof_lgbm); r_lgbm_orig = rmse(y, np.expm1(oof_lgbm))
r_xgb_log  = rmse(y_log, oof_xgb);  r_xgb_orig  = rmse(y, np.expm1(oof_xgb))
r_cat_log  = rmse(y_log, oof_cat);  r_cat_orig  = rmse(y, np.expm1(oof_cat))

print(f"[Single OOF]")
print(f"  LGBM: log={r_lgbm_log:.4f}  orig={r_lgbm_orig:.2f}")
print(f"  XGB : log={r_xgb_log:.4f}  orig={r_xgb_orig:.2f}")
print(f"  CAT : log={r_cat_log:.4f}  orig={r_cat_orig:.2f}")

# 1) 단순 평균
oof_avg = (oof_lgbm + oof_xgb + oof_cat) / 3
te_avg  = (te_lgbm + te_xgb + te_cat) / 3
r_avg_log  = rmse(y_log, oof_avg)
r_avg_orig = rmse(y, np.expm1(oof_avg))
print(f"\n[Simple avg]  log={r_avg_log:.4f}  orig={r_avg_orig:.2f}")

# 2) RMSE 역수 가중치 (log RMSE 기준)
ws = np.array([1.0/r_lgbm_log, 1.0/r_xgb_log, 1.0/r_cat_log])
ws /= ws.sum()
print(f"\nweights: LGBM={ws[0]:.3f}, XGB={ws[1]:.3f}, CAT={ws[2]:.3f}")
oof_w = ws[0]*oof_lgbm + ws[1]*oof_xgb + ws[2]*oof_cat
te_w  = ws[0]*te_lgbm  + ws[1]*te_xgb  + ws[2]*te_cat
r_w_log  = rmse(y_log, oof_w)
r_w_orig = rmse(y, np.expm1(oof_w))
print(f"[Weighted]    log={r_w_log:.4f}  orig={r_w_orig:.2f}")

# 3) 최소제곱 최적 가중 (음수 허용 X, 합=1 강제: 그리드 서치)
print(f"\n[Searching optimal non-negative weights summing to 1...]")
best = (None, np.inf)
for a in np.arange(0, 1.01, 0.05):
    for b in np.arange(0, 1.01 - a, 0.05):
        c = 1.0 - a - b
        if c < 0: continue
        pred = a*oof_lgbm + b*oof_xgb + c*oof_cat
        r = rmse(y_log, pred)
        if r < best[1]:
            best = ((a, b, c), r)
opt_w, opt_rmse = best
oof_opt = opt_w[0]*oof_lgbm + opt_w[1]*oof_xgb + opt_w[2]*oof_cat
te_opt  = opt_w[0]*te_lgbm  + opt_w[1]*te_xgb  + opt_w[2]*te_cat
r_opt_orig = rmse(y, np.expm1(oof_opt))
print(f"optimal weights: LGBM={opt_w[0]:.2f}, XGB={opt_w[1]:.2f}, CAT={opt_w[2]:.2f}")
print(f"[Optimal]     log={opt_rmse:.4f}  orig={r_opt_orig:.2f}")

# 가장 좋은 것 선택
candidates = {
    "simple_avg": (r_avg_log,  te_avg,  (1/3, 1/3, 1/3)),
    "weighted":   (r_w_log,    te_w,    tuple(ws)),
    "optimal":    (opt_rmse,   te_opt,  opt_w),
}
best_name = min(candidates, key=lambda k: candidates[k][0])
best_log, best_te, best_w = candidates[best_name]
print(f"\n선택: {best_name} (log={best_log:.4f})")

# 최종 예측
final_pred = np.expm1(best_te)
# 음수/극단치 클립 (혹시 모를)
final_pred = np.clip(final_pred, 600, 58000)

# 제출 양식 작성
sub = pd.read_csv(DATA / "sample_submission.csv")
sub["Y"] = final_pred
print(f"\n[Submission]")
print(f"  rows: {len(sub):,}")
print(f"  Y min/mean/max: {sub['Y'].min():.0f} / {sub['Y'].mean():.0f} / {sub['Y'].max():.0f}")
print(f"  Y train ref  : {y.min():.0f} / {y.mean():.0f} / {y.max():.0f}")

sub.to_csv(OUT / "answer.csv", index=False)
print(f"\n✅ Saved: outputs/answer.csv  ({len(sub):,} rows)")

# 요약 저장
metrics = {
    "single": {
        "lgbm_log": float(r_lgbm_log), "lgbm_orig": float(r_lgbm_orig),
        "xgb_log":  float(r_xgb_log),  "xgb_orig":  float(r_xgb_orig),
        "cat_log":  float(r_cat_log),  "cat_orig":  float(r_cat_orig),
    },
    "ensemble": {
        "simple_avg_log":  float(r_avg_log),  "simple_avg_orig":  float(r_avg_orig),
        "weighted_log":    float(r_w_log),    "weighted_orig":    float(r_w_orig),
        "optimal_log":     float(opt_rmse),   "optimal_orig":     float(r_opt_orig),
    },
    "weights_used": list(best_w),
    "chosen_ensemble": best_name,
}
with open(OUT / "metrics_ensemble.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"   metrics_ensemble.json saved")
