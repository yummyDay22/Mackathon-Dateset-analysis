"""
LGBM / XGB 단독 결과 비교 + 2-model 앙상블.
- 각 모델 단독 answer_<model>.csv 생성
- 50/50 단순평균 / RMSE 역수 가중 / 그리드 서치 최적 가중 비교
- 최종 best ensemble로 answer_2model.csv 생성
"""
import numpy as np, pandas as pd, json
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

# ============================================================
# 단독 모델 점수
# ============================================================
print("="*60)
print("[단독 모델 — train OOF 기준 CV 점수]")
print("="*60)
r_l_log, r_l_orig = rmse(y_log, oof_lgbm), rmse(y, np.expm1(oof_lgbm))
r_x_log, r_x_orig = rmse(y_log, oof_xgb),  rmse(y, np.expm1(oof_xgb))
print(f"  LGBM: log={r_l_log:.4f}  orig={r_l_orig:.2f} 만원")
print(f"  XGB : log={r_x_log:.4f}  orig={r_x_orig:.2f} 만원")
print(f"  차이: {(r_x_orig - r_l_orig):+.2f} 만원 ({'XGB' if r_x_orig<r_l_orig else 'LGBM'}가 더 좋음)")

# 각 모델 단독 제출 파일
def save_answer(test_log, name):
    pred = np.expm1(test_log).clip(600, 58000)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    sub["Y"] = pred
    sub.to_csv(OUT / f"answer_{name}.csv", index=False)
    print(f"  -> outputs/answer_{name}.csv (Y: {sub['Y'].min():.0f} ~ {sub['Y'].max():.0f}, mean {sub['Y'].mean():.0f})")

print("\n[단독 모델 제출 파일]")
save_answer(te_lgbm, "lgbm")
save_answer(te_xgb,  "xgb")

# ============================================================
# 앙상블 비교
# ============================================================
print("\n" + "="*60)
print("[2모델 앙상블 비교]")
print("="*60)

# 1) 단순 50/50
oof_a = 0.5*oof_lgbm + 0.5*oof_xgb
te_a  = 0.5*te_lgbm  + 0.5*te_xgb
r_a_log = rmse(y_log, oof_a); r_a_orig = rmse(y, np.expm1(oof_a))
print(f"  50/50 평균    : log={r_a_log:.4f}  orig={r_a_orig:.2f} 만원")

# 2) RMSE 역수 가중
ws = np.array([1/r_l_log, 1/r_x_log]); ws /= ws.sum()
oof_w = ws[0]*oof_lgbm + ws[1]*oof_xgb
te_w  = ws[0]*te_lgbm  + ws[1]*te_xgb
r_w_log = rmse(y_log, oof_w); r_w_orig = rmse(y, np.expm1(oof_w))
print(f"  가중 {ws[0]*100:.0f}/{ws[1]*100:.0f}      : log={r_w_log:.4f}  orig={r_w_orig:.2f} 만원")

# 3) 그리드 서치 최적
best_opt = (None, np.inf, None)
for a in np.arange(0, 1.01, 0.01):
    pred = a*oof_lgbm + (1-a)*oof_xgb
    r = rmse(y_log, pred)
    if r < best_opt[1]:
        best_opt = (a, r, rmse(y, np.expm1(pred)))
opt_a, opt_log, opt_orig = best_opt
print(f"  최적 {opt_a*100:.0f}/{(1-opt_a)*100:.0f}      : log={opt_log:.4f}  orig={opt_orig:.2f} 만원  ← BEST")

# 개선폭
improve = r_l_orig - opt_orig
print(f"\n  개선폭(LGBM 대비): {improve:+.2f} 만원 ({improve/r_l_orig*100:+.2f}%)")

# 최종 2모델 앙상블 제출
te_final = opt_a*te_lgbm + (1-opt_a)*te_xgb
pred_final = np.expm1(te_final).clip(600, 58000)
sub = pd.read_csv(DATA / "sample_submission.csv")
sub["Y"] = pred_final
sub.to_csv(OUT / "answer_2model.csv", index=False)
print(f"\n✅ outputs/answer_2model.csv  (Y: {sub['Y'].min():.0f} ~ {sub['Y'].max():.0f}, mean {sub['Y'].mean():.0f})")

# 메트릭 저장
with open(OUT / "metrics_compare_2model.json", "w") as f:
    json.dump({
        "lgbm":  {"log": float(r_l_log), "orig": float(r_l_orig)},
        "xgb":   {"log": float(r_x_log), "orig": float(r_x_orig)},
        "simple_50_50": {"log": float(r_a_log), "orig": float(r_a_orig)},
        "weighted":     {"log": float(r_w_log), "orig": float(r_w_orig), "w_lgbm": float(ws[0])},
        "optimal":      {"log": float(opt_log), "orig": float(opt_orig), "w_lgbm": float(opt_a)},
    }, f, indent=2)
print("   metrics_compare_2model.json saved")
