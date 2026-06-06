"""
Phase 5-1: LightGBM 5-Fold 학습
  - Target: log1p(Y) (RMSE 안정화)
  - 카테고리: dong, building_type을 category dtype으로 전달
  - 같은 folds.npy 사용 (다른 모델과 OOF 일치)
  - 출력: outputs/oof_lgbm.npy, outputs/test_lgbm.npy
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

DATA = Path("data")
OUT  = Path("outputs"); OUT.mkdir(exist_ok=True)
SEED = 42

# ------------------------------------------------------------------
train = pd.read_parquet(DATA / "train_fe.parquet")
test  = pd.read_parquet(DATA / "test_fe.parquet")
folds = np.load(DATA / "folds.npy")
print(f"train {train.shape}, test {test.shape}, folds {folds.shape}")

# 학습용 특성 (drop: dong/building_type 원본은 dong_label/type_label/is_apt 등으로 대체됨)
# LGBM은 category dtype을 직접 처리할 수 있으므로 dong/building_type 그대로 두는 게 좋음
DROP_COLS = []
FEATURES = [c for c in train.columns if c not in ["Y"] + DROP_COLS]

# dong, building_type을 category로 cast
for col in ["dong", "building_type"]:
    train[col] = train[col].astype("category")
    test[col]  = test[col].astype("category")

print(f"#features = {len(FEATURES)}")

# 타겟 변환
y = train["Y"].values.astype(np.float32)
y_log = np.log1p(y)

# OOF 저장 배열
oof_log = np.zeros(len(train), dtype=np.float32)
test_log_preds = np.zeros(len(test), dtype=np.float32)

# 하이퍼파라미터 (보수적이고 안정적인 세팅)
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_data_in_leaf": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": SEED,
}

cat_cols = ["dong", "building_type"]
fi_total = pd.Series(0.0, index=FEATURES)

t0 = time.time()
for fold in range(5):
    tr_mask = folds != fold
    va_mask = folds == fold
    X_tr = train.loc[tr_mask, FEATURES]
    y_tr = y_log[tr_mask]
    X_va = train.loc[va_mask, FEATURES]
    y_va = y_log[va_mask]

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=dtr, free_raw_data=False)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    pred_va = model.predict(X_va, num_iteration=model.best_iteration)
    oof_log[va_mask] = pred_va.astype(np.float32)

    pred_te = model.predict(test[FEATURES], num_iteration=model.best_iteration)
    test_log_preds += pred_te.astype(np.float32) / 5

    # feature importance
    fi = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURES)
    fi_total = fi_total.add(fi, fill_value=0)

    # fold RMSE on original scale
    pred_va_orig = np.expm1(pred_va)
    y_va_orig    = y[va_mask]
    rmse_orig = np.sqrt(mean_squared_error(y_va_orig, pred_va_orig))
    rmse_log  = np.sqrt(mean_squared_error(y_va, pred_va))
    print(f"  [fold {fold}] best_iter={model.best_iteration}, "
          f"rmse_log={rmse_log:.4f}, rmse_orig={rmse_orig:.2f}, elapsed={time.time()-t0:.1f}s")

# ------------------------------------------------------------------
oof_pred = np.expm1(oof_log)
test_pred = np.expm1(test_log_preds)

overall_rmse = np.sqrt(mean_squared_error(y, oof_pred))
overall_log_rmse = np.sqrt(mean_squared_error(y_log, oof_log))

print(f"\n[LGBM CV]")
print(f"  RMSE (log)  = {overall_log_rmse:.4f}")
print(f"  RMSE (orig) = {overall_rmse:.2f} (만원)")
print(f"  pred test min/mean/max = {test_pred.min():.0f} / {test_pred.mean():.0f} / {test_pred.max():.0f}")

# 저장
np.save(OUT / "oof_lgbm.npy",  oof_log)     # log scale로 저장 (앙상블 시 가중 후 expm1)
np.save(OUT / "test_lgbm.npy", test_log_preds)
fi_total /= 5
fi_sorted = fi_total.sort_values(ascending=False)
print(f"\n[Top 20 features by gain]")
print(fi_sorted.head(20).to_string())
fi_sorted.to_csv(OUT / "fi_lgbm.csv")

with open(OUT / "metrics_lgbm.json", "w") as f:
    json.dump({
        "rmse_log": float(overall_log_rmse),
        "rmse_orig": float(overall_rmse),
        "n_features": len(FEATURES),
    }, f, indent=2)

print(f"\nSaved: outputs/oof_lgbm.npy, test_lgbm.npy, fi_lgbm.csv, metrics_lgbm.json")
