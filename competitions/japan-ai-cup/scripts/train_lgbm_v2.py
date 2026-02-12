"""Train LightGBM with time-aware CV for Japan AI Cup."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features_v2 import build_feature_artifacts


DATA_DIR = Path("competitions/japan-ai-cup/data")
OUTPUT_DIR = Path("competitions/japan-ai-cup/predictions")


def build_time_folds(last_purchase_date: pd.Series, n_bins: int = 5) -> np.ndarray:
    """Create monotonic time buckets by date rank."""

    ranked = last_purchase_date.rank(method="first")
    folds = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
    return folds.astype(int).to_numpy()


def run_time_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    n_bins: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[pd.Series], np.ndarray]:
    """Run expanding-window CV: train on older buckets, validate on next bucket."""

    folds = build_time_folds(train_df["last_purchase_date"], n_bins=n_bins)

    X = train_df[feature_cols]
    y = train_df["target"]

    oof = np.zeros(len(train_df), dtype=float)
    used = np.zeros(len(train_df), dtype=bool)
    scores: list[float] = []
    importances: list[pd.Series] = []

    for val_bucket in range(1, folds.max() + 1):
        train_idx = np.where(folds < val_bucket)[0]
        val_idx = np.where(folds == val_bucket)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        y_val = y.iloc[val_idx]
        if y_val.nunique() < 2:
            continue

        print(f"[TimeCV] Fold {val_bucket}: train={len(train_idx):,}, valid={len(val_idx):,}")

        dtrain = lgb.Dataset(
            X.iloc[train_idx],
            label=y.iloc[train_idx],
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X.iloc[val_idx],
            label=y.iloc[val_idx],
            categorical_feature=categorical_cols,
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

        fold_auc = roc_auc_score(y_val, pred)
        scores.append(fold_auc)
        print(f"[TimeCV] Fold {val_bucket} AUC={fold_auc:.6f}")

        importances.append(
            pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
        )

    return oof, used, scores, importances, folds


def run_random_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    n_splits: int = 5,
) -> list[float]:
    """Reference CV for comparison with historical baseline."""

    X = train_df[feature_cols]
    y = train_df["target"]
    cv_scores: list[float] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        dtrain = lgb.Dataset(
            X.iloc[train_idx],
            label=y.iloc[train_idx],
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X.iloc[val_idx],
            label=y.iloc[val_idx],
            categorical_feature=categorical_cols,
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
        fold_auc = roc_auc_score(y.iloc[val_idx], pred)
        cv_scores.append(fold_auc)

    return cv_scores


def fit_full_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    num_boost_round: int,
) -> np.ndarray:
    dtrain = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["target"],
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model.predict(test_df[feature_cols], num_iteration=model.best_iteration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Japan AI Cup LGBM v2")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--skip-random-cv", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=60)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-l1", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--output-suffix", type=str, default="v2")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = build_feature_artifacts(
        data_csv=str(DATA_DIR / "data.csv"),
        train_flag_csv=str(DATA_DIR / "train_flag.csv"),
        sample_submission_csv=str(DATA_DIR / "sample_submission.csv"),
    )

    train_df = artifacts.train_df
    test_df = artifacts.test_df
    feature_cols = artifacts.feature_cols
    categorical_cols = artifacts.categorical_cols

    print("=" * 60)
    print("Japan AI Cup - LGBM v2")
    print("=" * 60)
    print(f"Reference date: {artifacts.reference_date}")
    print(f"Train rows: {len(train_df):,}, Test rows: {len(test_df):,}")
    print(f"Features: {len(feature_cols)}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 5,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "max_depth": -1,
        "verbose": -1,
        "seed": 42,
    }

    oof, used_mask, time_scores, importances, fold_bucket = run_time_cv(
        train_df=train_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        params=params,
        n_bins=args.n_bins,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    if used_mask.sum() == 0:
        raise RuntimeError("No valid time-CV folds produced predictions.")

    oof_auc = roc_auc_score(train_df.loc[used_mask, "target"], oof[used_mask])
    print(f"[TimeCV] Fold AUCs: {[round(v, 6) for v in time_scores]}")
    print(f"[TimeCV] Mean AUC={np.mean(time_scores):.6f}, Std={np.std(time_scores):.6f}")
    print(f"[TimeCV] OOF AUC={oof_auc:.6f} (evaluated on {used_mask.sum():,} rows)")

    if not args.skip_random_cv:
        random_scores = run_random_cv(
            train_df=train_df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            params=params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            n_splits=5,
        )
        print(f"[RandomCV] Fold AUCs: {[round(v, 6) for v in random_scores]}")
        print(f"[RandomCV] Mean AUC={np.mean(random_scores):.6f}, Std={np.std(random_scores):.6f}")

    test_pred = fit_full_and_predict(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        params=params,
        num_boost_round=args.num_boost_round,
    )

    submission = pd.DataFrame({"user_id": test_df["user_id"], "pred": test_pred})
    submission_path = OUTPUT_DIR / f"submission_{args.output_suffix}.csv"
    submission.to_csv(submission_path, index=False)

    oof_df = pd.DataFrame(
        {
            "user_id": train_df["user_id"],
            "target": train_df["target"],
            "oof_pred": oof,
            "time_bucket": fold_bucket,
        }
    )
    oof_path = OUTPUT_DIR / f"oof_{args.output_suffix}.csv"
    oof_df.to_csv(oof_path, index=False)

    if importances:
        fi = pd.concat(importances, axis=1).fillna(0.0)
        fi_mean = fi.mean(axis=1).sort_values(ascending=False)
        fi_df = fi_mean.rename("importance_gain_mean").reset_index()
        fi_df.columns = ["feature", "importance_gain_mean"]
        fi_path = OUTPUT_DIR / f"feature_importance_{args.output_suffix}.csv"
        fi_df.to_csv(fi_path, index=False)
        print("Top 10 features:")
        print(fi_df.head(10).to_string(index=False))

    print(f"Saved: {submission_path}")
    print(f"Saved: {oof_path}")
    print(
        "Prediction stats: "
        f"min={submission['pred'].min():.6f}, max={submission['pred'].max():.6f}, "
        f"mean={submission['pred'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
