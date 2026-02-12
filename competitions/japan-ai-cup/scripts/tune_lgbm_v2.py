"""Light parameter sweep for Japan AI Cup v2 (TimeCV-first)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features_v2 import build_feature_artifacts
from train_lgbm_v2 import DATA_DIR, run_random_cv, run_time_cv


OUT_CSV = Path("competitions/japan-ai-cup/notes/tuning_results_v2.csv")


def main() -> None:
    artifacts = build_feature_artifacts(
        data_csv=str(DATA_DIR / "data.csv"),
        train_flag_csv=str(DATA_DIR / "train_flag.csv"),
        sample_submission_csv=str(DATA_DIR / "sample_submission.csv"),
    )

    train_df = artifacts.train_df
    feature_cols = artifacts.feature_cols
    categorical_cols = artifacts.categorical_cols

    candidates = [
        {"name": "base", "num_leaves": 63, "min_child_samples": 60, "feature_fraction": 0.8, "lambda_l1": 0.1, "lambda_l2": 1.0},
        {"name": "leaf47_child80", "num_leaves": 47, "min_child_samples": 80, "feature_fraction": 0.8, "lambda_l1": 0.1, "lambda_l2": 1.0},
        {"name": "leaf95_child80", "num_leaves": 95, "min_child_samples": 80, "feature_fraction": 0.8, "lambda_l1": 0.1, "lambda_l2": 1.0},
        {"name": "leaf63_child120", "num_leaves": 63, "min_child_samples": 120, "feature_fraction": 0.8, "lambda_l1": 0.1, "lambda_l2": 1.0},
        {"name": "ff07_l2_2", "num_leaves": 63, "min_child_samples": 60, "feature_fraction": 0.7, "lambda_l1": 0.1, "lambda_l2": 2.0},
        {"name": "ff09_l2_2", "num_leaves": 63, "min_child_samples": 60, "feature_fraction": 0.9, "lambda_l1": 0.1, "lambda_l2": 2.0},
        {"name": "l1_05_l2_3", "num_leaves": 63, "min_child_samples": 60, "feature_fraction": 0.8, "lambda_l1": 0.5, "lambda_l2": 3.0},
        {"name": "leaf47_child120_l1_05", "num_leaves": 47, "min_child_samples": 120, "feature_fraction": 0.8, "lambda_l1": 0.5, "lambda_l2": 3.0},
    ]

    rows: list[dict] = []

    for cand in candidates:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": cand["num_leaves"],
            "min_child_samples": cand["min_child_samples"],
            "feature_fraction": cand["feature_fraction"],
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": cand["lambda_l1"],
            "lambda_l2": cand["lambda_l2"],
            "max_depth": -1,
            "verbose": -1,
            "seed": 42,
        }

        print(f"\\n=== candidate: {cand['name']} ===")
        oof, used_mask, time_scores, _, _ = run_time_cv(
            train_df=train_df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            params=params,
            n_bins=5,
            num_boost_round=400,
            early_stopping_rounds=60,
        )

        if used_mask.sum() == 0:
            continue

        time_oof_auc = roc_auc_score(train_df.loc[used_mask, "target"], oof[used_mask])
        time_mean = float(np.mean(time_scores))
        time_std = float(np.std(time_scores))

        rows.append(
            {
                "name": cand["name"],
                "num_leaves": cand["num_leaves"],
                "min_child_samples": cand["min_child_samples"],
                "feature_fraction": cand["feature_fraction"],
                "lambda_l1": cand["lambda_l1"],
                "lambda_l2": cand["lambda_l2"],
                "timecv_mean_auc": time_mean,
                "timecv_std_auc": time_std,
                "timecv_oof_auc": float(time_oof_auc),
                "used_rows": int(used_mask.sum()),
            }
        )

    if not rows:
        raise RuntimeError("No candidate produced valid TimeCV results")

    result_df = pd.DataFrame(rows).sort_values(
        ["timecv_mean_auc", "timecv_oof_auc"], ascending=False
    )

    best = result_df.iloc[0].to_dict()
    print("\\n=== best by TimeCV mean ===")
    for k in [
        "name",
        "timecv_mean_auc",
        "timecv_std_auc",
        "timecv_oof_auc",
        "num_leaves",
        "min_child_samples",
        "feature_fraction",
        "lambda_l1",
        "lambda_l2",
    ]:
        print(f"{k}: {best[k]}")

    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": int(best["num_leaves"]),
        "min_child_samples": int(best["min_child_samples"]),
        "feature_fraction": float(best["feature_fraction"]),
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": float(best["lambda_l1"]),
        "lambda_l2": float(best["lambda_l2"]),
        "max_depth": -1,
        "verbose": -1,
        "seed": 42,
    }
    random_scores = run_random_cv(
        train_df=train_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        params=best_params,
        num_boost_round=400,
        early_stopping_rounds=60,
        n_splits=5,
    )

    result_df["best_randomcv_mean_auc"] = np.nan
    result_df.loc[result_df["name"] == best["name"], "best_randomcv_mean_auc"] = float(np.mean(random_scores))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"\\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
