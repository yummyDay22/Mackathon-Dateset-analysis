"""
3모델(LGBM/XGB/CAT) 모든 조합으로 제출 파일 생성 + CV RMSE 비교표.

생성 파일 (총 11개):
  단독 (3):
    answer_lgbm.csv
    answer_xgb.csv
    answer_cat.csv
  2모델 (6 = 3조합 × 단순/최적):
    answer_lgbm_xgb_avg.csv  / answer_lgbm_xgb_opt.csv
    answer_lgbm_cat_avg.csv  / answer_lgbm_cat_opt.csv
    answer_xgb_cat_avg.csv   / answer_xgb_cat_opt.csv
  3모델 (2):
    answer_lgbm_xgb_cat_avg.csv
    answer_lgbm_xgb_cat_opt.csv

기존 answer.csv, answer_2model.csv는 삭제 (이름이 모호함).
CAT은 4-fold이므로 OOF valid 영역(80%)에서만 RMSE 계산 → 가중치 최적화도 같은 영역.
test 예측은 LGBM/XGB는 5-fold 평균, CAT은 4-fold 평균 (모두 정상 사용 가능).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from itertools import combinations

DATA = Path("data"); OUT = Path("outputs")

train = pd.read_parquet(DATA / "train_fe.parquet")
y     = train["Y"].values.astype(np.float32)
y_log = np.log1p(y)
folds = np.load(DATA / "folds.npy")
sub   = pd.read_csv(DATA / "sample_submission.csv")

# 모델별 OOF / test 예측 (log scale)
oof = {
    "lgbm": np.load(OUT / "oof_lgbm.npy"),
    "xgb":  np.load(OUT / "oof_xgb.npy"),
    "cat":  np.load(OUT / "oof_cat.npy"),
}
te = {
    "lgbm": np.load(OUT / "test_lgbm.npy"),
    "xgb":  np.load(OUT / "test_xgb.npy"),
    "cat":  np.load(OUT / "test_cat.npy"),
}

# CAT은 fold 4 OOF 비어있음 (4-fold 사용)
cat_mask = np.isin(folds, [0, 1, 2, 3])
print(f"CAT OOF valid: {cat_mask.sum()/len(folds)*100:.1f}%")

def rmse(yt, yp): return np.sqrt(mean_squared_error(yt, yp))

def save_answer(test_log, filename):
    pred = np.expm1(test_log).clip(600, 58000)
    s = sub.copy()
    s["Y"] = pred
    s.to_csv(OUT / filename, index=False)
    return pred.min(), pred.mean(), pred.max()

# 기존 모호한 파일 삭제
for f in ["answer.csv", "answer_2model.csv"]:
    p = OUT / f
    if p.exists():
        p.unlink()
        print(f"삭제: {f}")

# 결과 누적
results = []   # (조합명, 파일명, RMSE_log, RMSE_orig, 가중치 설명)

# ============================================================
# 단독 (3개)
# ============================================================
print("\n[단독 모델]")
for name in ["lgbm", "xgb", "cat"]:
    mask = cat_mask if name == "cat" else np.ones(len(folds), dtype=bool)
    r_log = rmse(y_log[mask], oof[name][mask])
    r_orig = rmse(y[mask], np.expm1(oof[name][mask]))
    fn = f"answer_{name}.csv"
    save_answer(te[name], fn)
    print(f"  {name:<5}: log={r_log:.4f}  orig={r_orig:.2f}  → {fn}")
    results.append({"combo": name, "file": fn, "rmse_log": r_log, "rmse_orig": r_orig, "weights": "—"})

# ============================================================
# 2모델 (3조합 × avg/opt = 6개)
# ============================================================
print("\n[2모델 앙상블]")
pairs = [("lgbm","xgb"), ("lgbm","cat"), ("xgb","cat")]
for m1, m2 in pairs:
    # 평가용 mask: cat이 들어가면 cat_mask, 아니면 전체
    mask = cat_mask if (m1=="cat" or m2=="cat") else np.ones(len(folds), dtype=bool)
    yL = y_log[mask]; yR = y[mask]
    o1 = oof[m1][mask]; o2 = oof[m2][mask]

    # 1) 단순평균
    avg_oof = 0.5*o1 + 0.5*o2
    avg_te  = 0.5*te[m1] + 0.5*te[m2]
    r_log = rmse(yL, avg_oof); r_orig = rmse(yR, np.expm1(avg_oof))
    fn = f"answer_{m1}_{m2}_avg.csv"
    save_answer(avg_te, fn)
    print(f"  {m1}+{m2:<3} avg: log={r_log:.4f}  orig={r_orig:.2f}  → {fn}")
    results.append({"combo": f"{m1}+{m2}", "file": fn, "rmse_log": r_log, "rmse_orig": r_orig, "weights": "50/50"})

    # 2) 그리드 최적
    best = (None, np.inf)
    for a in np.arange(0, 1.01, 0.01):
        pred = a*o1 + (1-a)*o2
        r = rmse(yL, pred)
        if r < best[1]:
            best = ((a, 1-a), r)
    wA, wB = best[0]
    opt_te = wA*te[m1] + wB*te[m2]
    r_log = best[1]; r_orig = rmse(yR, np.expm1(wA*o1 + wB*o2))
    fn = f"answer_{m1}_{m2}_opt.csv"
    save_answer(opt_te, fn)
    print(f"  {m1}+{m2:<3} opt: log={r_log:.4f}  orig={r_orig:.2f}  → {fn}  ({m1}={wA:.2f}, {m2}={wB:.2f})")
    results.append({"combo": f"{m1}+{m2}", "file": fn, "rmse_log": r_log, "rmse_orig": r_orig,
                    "weights": f"{m1}={wA:.2f}, {m2}={wB:.2f}"})

# ============================================================
# 3모델 (2개: avg, opt)
# ============================================================
print("\n[3모델 앙상블]")
yL = y_log[cat_mask]; yR = y[cat_mask]
oL = oof["lgbm"][cat_mask]
oX = oof["xgb"][cat_mask]
oC = oof["cat"][cat_mask]

# 1) 단순평균
avg_oof = (oL + oX + oC) / 3
avg_te  = (te["lgbm"] + te["xgb"] + te["cat"]) / 3
r_log = rmse(yL, avg_oof); r_orig = rmse(yR, np.expm1(avg_oof))
fn = "answer_lgbm_xgb_cat_avg.csv"
save_answer(avg_te, fn)
print(f"  3모델 avg: log={r_log:.4f}  orig={r_orig:.2f}  → {fn}")
results.append({"combo": "lgbm+xgb+cat", "file": fn, "rmse_log": r_log, "rmse_orig": r_orig,
                "weights": "33/33/33"})

# 2) 그리드 최적
best = (None, np.inf)
for a in np.arange(0, 1.01, 0.05):
    for b in np.arange(0, 1.01-a, 0.05):
        c = 1.0 - a - b
        if c < -1e-9: continue
        c = max(c, 0)
        pred = a*oL + b*oX + c*oC
        r = rmse(yL, pred)
        if r < best[1]:
            best = ((a, b, c), r)
wL, wX, wC = best[0]
opt_te = wL*te["lgbm"] + wX*te["xgb"] + wC*te["cat"]
r_log = best[1]; r_orig = rmse(yR, np.expm1(wL*oL + wX*oX + wC*oC))
fn = "answer_lgbm_xgb_cat_opt.csv"
save_answer(opt_te, fn)
print(f"  3모델 opt: log={r_log:.4f}  orig={r_orig:.2f}  → {fn}  (L={wL:.2f}, X={wX:.2f}, C={wC:.2f})")
results.append({"combo": "lgbm+xgb+cat", "file": fn, "rmse_log": r_log, "rmse_orig": r_orig,
                "weights": f"lgbm={wL:.2f}, xgb={wX:.2f}, cat={wC:.2f}"})

# ============================================================
# 비교표 저장 + 출력
# ============================================================
df = pd.DataFrame(results)
df = df.sort_values("rmse_log").reset_index(drop=True)
df.insert(0, "rank", range(1, len(df)+1))

print("\n" + "="*80)
print("[전체 비교표 (RMSE log 오름차순)]")
print("="*80)
print(df.to_string(index=False))

df.to_csv(OUT / "ensemble_comparison.csv", index=False)
print(f"\n✅ outputs/ensemble_comparison.csv 저장 ({len(df)}개 조합)")
print(f"✅ 최고: {df.iloc[0]['file']} (RMSE log={df.iloc[0]['rmse_log']:.4f}, 원본={df.iloc[0]['rmse_orig']:.0f}만원)")
