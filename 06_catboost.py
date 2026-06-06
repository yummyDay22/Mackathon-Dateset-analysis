"""
Phase 5-3: CatBoost 5-Fold 학습
  - dong / building_type을 cat_features로 직접 처리 (CatBoost 강점)
  - dong_label, type_label, is_apt 등 인코딩본은 drop (중복 정보)
"""
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error

DATA = Path("data")
OUT  = Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 42

train = pd.read_parquet(DATA / "train_fe.parquet")
test  = pd.read_parquet(DATA / "test_fe.parquet")
folds = np.load(DATA / "folds.npy")
print(f"train {train.shape}, test {test.shape}")

# CatBoost는 카테고리 처리에 강하므로 dong/building_type 원본 사용
# 중복인 dong_label / 더미들은 drop
DROP_COLS = ["dong_label", "type_label", "is_apt", "is_office", "is_villa"]
FEATURES = [c for c in train.columns if c not in ["Y"] + DROP_COLS]
CAT_FEATURES = ["dong", "building_type"]

print(f"#features = {len(FEATURES)}  (cat: {CAT_FEATURES})")

# CatBoost는 NaN 그대로 OK
# string이어야 카테고리 인식
for col in CAT_FEATURES:
    train[col] = train[col].astype(str)
    test[col]  = test[col].astype(str)

y = train["Y"].values.astype(np.float32)
y_log = np.log1p(y)

oof_log = np.zeros(len(train), dtype=np.float32)
test_log_preds = np.zeros(len(test), dtype=np.float32)

t0 = time.time()
fi_total = pd.Series(0.0, index=FEATURES)
USE_FOLDS = [0, 1, 2, 3]   # 디스크 절약: 4폴드만 사용 (fold 4 skip)
N_USE = len(USE_FOLDS)

for fold in USE_FOLDS:
    tr_mask = folds != fold
    va_mask = folds == fold

    X_tr = train.loc[tr_mask, FEATURES]
    X_va = train.loc[va_mask, FEATURES]
    y_tr = y_log[tr_mask]; y_va = y_log[va_mask]

    cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]
    tr_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
    va_pool = Pool(X_va, label=y_va, cat_features=cat_idx)
    te_pool = Pool(test[FEATURES], cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        loss_function="RMSE",
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=100,
        verbose=200,
        thread_count=-1,
        allow_writing_files=False,  # 디스크 안 쓰게
    )
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

    pred_va = model.predict(va_pool)
    oof_log[va_mask] = pred_va.astype(np.float32)
    pred_te = model.predict(te_pool)
    test_log_preds += pred_te.astype(np.float32) / N_USE

    fi = pd.Series(model.get_feature_importance(tr_pool), index=FEATURES)
    fi_total = fi_total.add(fi, fill_value=0)

    rmse_log  = np.sqrt(mean_squared_error(y_va, pred_va))
    rmse_orig = np.sqrt(mean_squared_error(y[va_mask], np.expm1(pred_va)))
    print(f"  [fold {fold}] best_iter={model.get_best_iteration()}, "
          f"rmse_log={rmse_log:.4f}, rmse_orig={rmse_orig:.2f}, elapsed={time.time()-t0:.1f}s")

# 4-fold: fold 4 위치는 OOF가 0이므로 평가에서 제외
oof_mask = np.isin(folds, USE_FOLDS)
oof_pred = np.expm1(oof_log)
test_pred = np.expm1(test_log_preds)
overall_rmse = np.sqrt(mean_squared_error(y[oof_mask], oof_pred[oof_mask]))
overall_log_rmse = np.sqrt(mean_squared_error(y_log[oof_mask], oof_log[oof_mask]))
print(f"\n[CAT CV] rmse_log={overall_log_rmse:.4f}  rmse_orig={overall_rmse:.2f} 만원")
print(f"  pred test min/mean/max = {test_pred.min():.0f} / {test_pred.mean():.0f} / {test_pred.max():.0f}")

np.save(OUT / "oof_cat.npy",  oof_log)
np.save(OUT / "test_cat.npy", test_log_preds)
fi_total /= 5
fi_total.sort_values(ascending=False).to_csv(OUT / "fi_cat.csv")
print(f"\n[Top 15 CatBoost features]")
print(fi_total.sort_values(ascending=False).head(15).to_string())

with open(OUT / "metrics_cat.json", "w") as f:
    json.dump({"rmse_log": float(overall_log_rmse), "rmse_orig": float(overall_rmse), "n_features": len(FEATURES)}, f, indent=2)
print(f"\nSaved: outputs/oof_cat.npy, test_cat.npy")
