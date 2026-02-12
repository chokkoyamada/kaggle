"""Stepwise migration experiments from baseline toward v2 components."""

from __future__ import annotations

from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


DATA_DIR = Path("competitions/japan-ai-cup/data")
OUTPUT_DIR = Path("competitions/japan-ai-cup/predictions")
NOTE_DIR = Path("competitions/japan-ai-cup/notes")
REFERENCE_DATE = 20250203


def build_features(*, deterministic_profile: bool, drop_null_user: bool, add_refund: bool, add_recency_window: bool) -> pd.DataFrame:
    con = duckdb.connect()

    where_clause = "WHERE user_id IS NOT NULL" if drop_null_user else ""

    profile_select = """
        arg_max(age_category, date) AS age_category,
        arg_max(sex, date) AS sex,
        arg_max(user_stage, date) AS user_stage,
        arg_max(membership_start_ym, date) AS membership_start_ym,
        arg_max(user_flag_ec, date) AS user_flag_ec,
        arg_max(user_flag_1, date) AS user_flag_1,
        arg_max(user_flag_2, date) AS user_flag_2,
        arg_max(user_flag_3, date) AS user_flag_3,
        arg_max(user_flag_4, date) AS user_flag_4,
        arg_max(user_flag_5, date) AS user_flag_5,
        arg_max(user_flag_6, date) AS user_flag_6
    """ if deterministic_profile else """
        first(age_category) AS age_category,
        first(sex) AS sex,
        first(user_stage) AS user_stage,
        first(membership_start_ym) AS membership_start_ym,
        first(user_flag_ec) AS user_flag_ec,
        first(user_flag_1) AS user_flag_1,
        first(user_flag_2) AS user_flag_2,
        first(user_flag_3) AS user_flag_3,
        first(user_flag_4) AS user_flag_4,
        first(user_flag_5) AS user_flag_5,
        first(user_flag_6) AS user_flag_6
    """

    extra_agg_cols = []
    extra_derived_cols = []

    if add_refund:
        extra_agg_cols.extend(
            [
                "SUM(CASE WHEN total_price < 0 THEN 1 ELSE 0 END) AS refund_txn_count",
                "SUM(CASE WHEN total_price < 0 THEN total_price ELSE 0 END) AS refund_total_spent",
            ]
        )
        extra_derived_cols.extend(
            [
                "f.refund_txn_count * 1.0 / NULLIF(f.purchase_count, 0) AS refund_txn_ratio",
                "ABS(f.refund_total_spent) * 1.0 / NULLIF(ABS(f.total_spent), 0) AS refund_amount_ratio",
            ]
        )

    if add_recency_window:
        extra_agg_cols.extend(
            [
                f"SUM(CASE WHEN date >= {REFERENCE_DATE - 30} THEN 1 ELSE 0 END) AS txns_30d",
                f"SUM(CASE WHEN date >= {REFERENCE_DATE - 90} THEN 1 ELSE 0 END) AS txns_90d",
                f"SUM(CASE WHEN date >= {REFERENCE_DATE - 30} THEN total_price ELSE 0 END) AS spent_30d",
                f"SUM(CASE WHEN date >= {REFERENCE_DATE - 90} THEN total_price ELSE 0 END) AS spent_90d",
            ]
        )
        extra_derived_cols.extend(
            [
                "f.txns_30d * 1.0 / NULLIF(f.purchase_count, 0) AS txn_share_30d",
                "f.txns_90d * 1.0 / NULLIF(f.purchase_count, 0) AS txn_share_90d",
                "f.spent_30d * 1.0 / NULLIF(f.total_spent, 0) AS spent_share_30d",
                "f.spent_90d * 1.0 / NULLIF(f.total_spent, 0) AS spent_share_90d",
            ]
        )

    extra_agg_sql = ",\n        " + ",\n        ".join(extra_agg_cols) if extra_agg_cols else ""
    extra_derived_sql = ",\n    " + ",\n    ".join(extra_derived_cols) if extra_derived_cols else ""

    query = f"""
    WITH source AS (
        SELECT *
        FROM read_csv_auto('{DATA_DIR / 'data.csv'}')
        {where_clause}
    ),
    user_features AS (
        SELECT
            user_id,
            COUNT(*) AS purchase_count,
            SUM(total_price) AS total_spent,
            AVG(total_price) AS avg_spent,
            MAX(total_price) AS max_spent,
            MIN(total_price) AS min_spent,
            STDDEV(total_price) AS std_spent,
            SUM(amount) AS total_items,
            AVG(amount) AS avg_items,
            COUNT(DISTINCT date) AS visit_days,
            MIN(date) AS first_purchase_date,
            MAX(date) AS last_purchase_date,
            {REFERENCE_DATE} - MAX(date) AS recency,
            COUNT(DISTINCT item_category_cd_1) AS unique_cat1,
            COUNT(DISTINCT item_category_cd_2) AS unique_cat2,
            COUNT(DISTINCT item_category_cd_3) AS unique_cat3,
            COUNT(DISTINCT jan_cd) AS unique_products,
            {profile_select}
            {extra_agg_sql}
        FROM source
        GROUP BY user_id
    )
    SELECT
        f.*,
        f.total_spent / NULLIF(f.visit_days, 0) AS avg_spent_per_visit,
        f.purchase_count / NULLIF(f.visit_days, 0) AS avg_purchases_per_visit,
        f.last_purchase_date - f.first_purchase_date AS purchase_span,
        CASE
            WHEN f.last_purchase_date - f.first_purchase_date > 0
            THEN f.visit_days * 1.0 / (f.last_purchase_date - f.first_purchase_date)
            ELSE 0
        END AS visit_frequency
        {extra_derived_sql}
    FROM user_features f
    """

    df = con.execute(query).fetchdf()
    con.close()
    return df


def evaluate_variant(name: str, features_df: pd.DataFrame, submission_name: str) -> dict:
    con = duckdb.connect()
    train_labels = con.execute(
        f"SELECT user_id, churn AS target FROM read_csv_auto('{DATA_DIR / 'train_flag.csv'}')"
    ).fetchdf()
    test_users = con.execute(
        f"SELECT user_id FROM read_csv_auto('{DATA_DIR / 'sample_submission.csv'}')"
    ).fetchdf()
    con.close()

    train_df = features_df.merge(train_labels, on="user_id", how="inner")
    test_df = features_df.merge(test_users, on="user_id", how="inner")

    feature_cols = [c for c in features_df.columns if c != "user_id"]
    categorical_cols = [c for c in ["age_category", "sex", "user_stage"] if c in feature_cols]

    for col in categorical_cols:
        train_df[col] = train_df[col].fillna("Unknown").astype("category")
        test_df[col] = test_df[col].fillna("Unknown").astype("category")

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    for col in numeric_cols:
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        test_df[col] = test_df[col].fillna(med)

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))
    scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train), start=1):
        dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx], categorical_feature=categorical_cols)
        dvalid = lgb.Dataset(X_train.iloc[va_idx], label=y_train.iloc[va_idx], categorical_feature=categorical_cols)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        val_pred = model.predict(X_train.iloc[va_idx], num_iteration=model.best_iteration)
        oof[va_idx] = val_pred
        preds += model.predict(X_test, num_iteration=model.best_iteration) / 5
        scores.append(roc_auc_score(y_train.iloc[va_idx], val_pred))
        print(f"[{name}] fold{fold} auc={scores[-1]:.6f}")

    oof_auc = roc_auc_score(y_train, oof)
    mean_auc = float(np.mean(scores))
    std_auc = float(np.std(scores))

    submission_path = OUTPUT_DIR / submission_name
    pd.DataFrame({"user_id": test_df["user_id"], "pred": preds}).to_csv(submission_path, index=False)

    return {
        "name": name,
        "feature_count": len(feature_cols),
        "cv_mean_auc": mean_auc,
        "cv_std_auc": std_auc,
        "oof_auc": float(oof_auc),
        "submission_path": str(submission_path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "name": "baseline_compat",
            "deterministic_profile": False,
            "drop_null_user": False,
            "add_refund": False,
            "add_recency_window": False,
            "submission_name": "submission_stepwise_baseline_compat.csv",
        },
        {
            "name": "det_profile",
            "deterministic_profile": True,
            "drop_null_user": False,
            "add_refund": False,
            "add_recency_window": False,
            "submission_name": "submission_stepwise_det_profile.csv",
        },
        {
            "name": "det_profile_drop_null",
            "deterministic_profile": True,
            "drop_null_user": True,
            "add_refund": False,
            "add_recency_window": False,
            "submission_name": "submission_stepwise_det_profile_drop_null.csv",
        },
        {
            "name": "det_profile_plus_refund",
            "deterministic_profile": True,
            "drop_null_user": True,
            "add_refund": True,
            "add_recency_window": False,
            "submission_name": "submission_stepwise_det_profile_plus_refund.csv",
        },
        {
            "name": "det_profile_plus_refund_recency",
            "deterministic_profile": True,
            "drop_null_user": True,
            "add_refund": True,
            "add_recency_window": True,
            "submission_name": "submission_stepwise_det_profile_plus_refund_recency.csv",
        },
    ]

    rows = []
    for v in variants:
        print("=" * 72)
        print(f"Running variant: {v['name']}")
        features_df = build_features(
            deterministic_profile=v["deterministic_profile"],
            drop_null_user=v["drop_null_user"],
            add_refund=v["add_refund"],
            add_recency_window=v["add_recency_window"],
        )
        result = evaluate_variant(v["name"], features_df, v["submission_name"])
        rows.append(result)
        print(
            f"[{v['name']}] cv_mean={result['cv_mean_auc']:.6f}, "
            f"oof={result['oof_auc']:.6f}, features={result['feature_count']}"
        )

    df = pd.DataFrame(rows).sort_values(["cv_mean_auc", "oof_auc"], ascending=False)
    out_path = NOTE_DIR / "stepwise_migration_results.csv"
    df.to_csv(out_path, index=False)
    print("=" * 72)
    print(f"Saved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
