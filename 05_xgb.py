"""
Phase 5-2: XGBoost 5-Fold 학습
  - XGBoost는 카테고리 자체 처리 약하므로 dong_label, type_label 사용
  - dong, building_type 원본 컬럼은 drop
"""
import json, time
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error

DATA = Path("data")
OUT  = Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 42

train = pd.read_parquet(DATA / "train_fe.parquet")
test  = pd.read_parquet(DATA / "test_fe.parquet")
folds = np.load(DATA / "folds.npy")
print(f"train {train.shape}, test {test.shape}")

# XGB는 string category 직접 처리 X → 원본 dong/building_type drop, label 사용
DROP_COLS = ["dong", "building_type"]
FEATURES = [c for c in train.columns if c not in ["Y"] + DROP_COLS]
print(f"#features = {len(FEATURES)}")

y = train["Y"].values.astype(np.float32)
y_log = np.log1p(y)

oof_log = np.zeros(len(train), dtype=np.float32)
test_log_preds = np.zeros(len(test), dtype=np.float32)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "tree_method": "hist",
    "seed": SEED,
}

fi_total = pd.Series(0.0, index=FEATURES)
t0 = time.time()

for fold in range(5):
    tr_mask = folds != fold
    va_mask = folds == fold

    X_tr = train.loc[tr_mask, FEATURES].astype(np.float32).values
    X_va = train.loc[va_mask, FEATURES].astype(np.float32).values
    y_tr = y_log[tr_mask]; y_va = y_log[va_mask]

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURES)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=FEATURES)
    dte = xgb.DMatrix(test[FEATURES].astype(np.float32).values, feature_names=FEATURES)

    model = xgb.train(
        params, dtr,
        num_boost_round=3000,
        evals=[(dva, "valid")],
        early_stopping_rounds=100,
        verbose_eval=200,
    )

    pred_va = model.predict(dva, iteration_range=(0, model.best_iteration+1))
    oof_log[va_mask] = pred_va.astype(np.float32)
    pred_te = model.predict(dte, iteration_range=(0, model.best_iteration+1))
    test_log_preds += pred_te.astype(np.float32) / 5

    fi = model.get_score(importance_type="gain")
    fi_s = pd.Series({f: fi.get(f, 0.0) for f in FEATURES})
    fi_total = fi_total.add(fi_s, fill_value=0)

    rmse_log  = np.sqrt(mean_squared_error(y_va, pred_va))
    rmse_orig = np.sqrt(mean_squared_error(y[va_mask], np.expm1(pred_va)))
    print(f"  [fold {fold}] best_iter={model.best_iteration}, "
          f"rmse_log={rmse_log:.4f}, rmse_orig={rmse_orig:.2f}, elapsed={time.time()-t0:.1f}s")

oof_pred = np.expm1(oof_log)
test_pred = np.expm1(test_log_preds)

overall_rmse = np.sqrt(mean_squared_error(y, oof_pred))
overall_log_rmse = np.sqrt(mean_squared_error(y_log, oof_log))
print(f"\n[XGB CV] rmse_log={overall_log_rmse:.4f}  rmse_orig={overall_rmse:.2f} 만원")
print(f"  pred test min/mean/max = {test_pred.min():.0f} / {test_pred.mean():.0f} / {test_pred.max():.0f}")

np.save(OUT / "oof_xgb.npy",  oof_log)
np.save(OUT / "test_xgb.npy", test_log_preds)
fi_total /= 5
fi_total.sort_values(ascending=False).to_csv(OUT / "fi_xgb.csv")
print(f"\n[Top 15 XGB features by gain]")
print(fi_total.sort_values(ascending=False).head(15).to_string())

with open(OUT / "metrics_xgb.json", "w") as f:
    json.dump({"rmse_log": float(overall_log_rmse), "rmse_orig": float(overall_rmse), "n_features": len(FEATURES)}, f, indent=2)
print(f"\nSaved: outputs/oof_xgb.npy, test_xgb.npy")
