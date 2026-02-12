"""Feature ablation for Japan AI Cup v2."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features_v2 import build_feature_artifacts
from train_lgbm_v2 import DATA_DIR, build_time_folds


OUT_DIR = Path("competitions/japan-ai-cup/predictions")
NOTE_DIR = Path("competitions/japan-ai-cup/notes")

REFUND_FEATURES = {
    "refund_txn_count",
    "refund_total_spent",
    "refund_txn_ratio",
    "refund_amount_ratio",
}

RECENCY_WINDOW_FEATURES = {
    "txns_30d",
    "txns_60d",
    "txns_90d",
    "spent_30d",
    "spent_60d",
    "spent_90d",
    "unique_cat1_30d",
    "unique_cat1_90d",
    "txn_share_30d",
    "txn_share_90d",
    "spent_share_30d",
    "spent_share_90d",
    "cat1_diversity_change_30_90",
    "cat1_recent_ratio",
}


def run_time_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    n_bins: int = 5,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 60,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    folds = build_time_folds(train_df["last_purchase_date"], n_bins=n_bins)
    X = train_df[feature_cols]
    y = train_df["target"]

    oof = np.zeros(len(train_df), dtype=float)
    used = np.zeros(len(train_df), dtype=bool)
    scores: list[float] = []

    for val_bucket in range(1, folds.max() + 1):
        train_idx = np.where(folds < val_bucket)[0]
        val_idx = np.where(folds == val_bucket)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        if y.iloc[val_idx].nunique() < 2:
            continue

        dtrain = lgb.Dataset(
            X.iloc[train_idx],
            label=y.iloc[train_idx],
            categorical_feature=[c for c in categorical_cols if c in feature_cols],
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X.iloc[val_idx],
            label=y.iloc[val_idx],
            categorical_feature=[c for c in categorical_cols if c in feature_cols],
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ],
        )

        pred = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        oof[val_idx] = pred
        used[val_idx] = True
        scores.append(roc_auc_score(y.iloc[val_idx], pred))

    return oof, used, scores


def fit_full_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    num_boost_round: int = 400,
) -> np.ndarray:
    dtrain = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["target"],
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
        free_raw_data=False,
    )
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model.predict(test_df[feature_cols], num_iteration=model.best_iteration)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = build_feature_artifacts(
        data_csv=str(DATA_DIR / "data.csv"),
        train_flag_csv=str(DATA_DIR / "train_flag.csv"),
        sample_submission_csv=str(DATA_DIR / "sample_submission.csv"),
    )

    train_df = artifacts.train_df
    test_df = artifacts.test_df
    all_feature_cols = artifacts.feature_cols
    categorical_cols = artifacts.categorical_cols

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 120,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_depth": -1,
        "verbose": -1,
        "seed": 42,
    }

    settings = [
        ("full", set()),
        ("no_refund", REFUND_FEATURES),
        ("no_recency_window", RECENCY_WINDOW_FEATURES),
    ]

    rows: list[dict] = []

    for name, drop_set in settings:
        feature_cols = [c for c in all_feature_cols if c not in drop_set]
        print(f"\\n=== ablation: {name} (features={len(feature_cols)}) ===")

        oof, used, scores = run_time_cv(
            train_df=train_df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            params=params,
        )
        if used.sum() == 0:
            continue

        oof_auc = roc_auc_score(train_df.loc[used, "target"], oof[used])
        mean_auc = float(np.mean(scores))

        pred = fit_full_predict(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            params=params,
        )

        sub_path = OUT_DIR / f"submission_v2_ablation_{name}.csv"
        pd.DataFrame({"user_id": test_df["user_id"], "pred": pred}).to_csv(sub_path, index=False)

        rows.append(
            {
                "name": name,
                "feature_count": len(feature_cols),
                "timecv_mean_auc": mean_auc,
                "timecv_std_auc": float(np.std(scores)),
                "timecv_oof_auc": float(oof_auc),
                "submission_path": str(sub_path),
            }
        )
        print(f"timecv_mean_auc={mean_auc:.6f}, timecv_oof_auc={oof_auc:.6f}")

    result_df = pd.DataFrame(rows).sort_values("timecv_mean_auc", ascending=False)
    out_csv = NOTE_DIR / "ablation_results_v2.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\\nSaved: {out_csv}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
