"""Feature engineering utilities for Japan AI Cup v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import duckdb
import pandas as pd


@dataclass(frozen=True)
class FeatureArtifacts:
    """Container for train/test matrices and metadata."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    categorical_cols: list[str]
    reference_date: int


def _compute_reference_date(con: duckdb.DuckDBPyConnection, data_path: str) -> int:
    max_date = con.execute(
        f"SELECT MAX(date) FROM read_csv_auto('{data_path}') WHERE user_id IS NOT NULL"
    ).fetchone()[0]
    return int(max_date) + 1


def build_feature_artifacts(
    data_csv: str,
    train_flag_csv: str,
    sample_submission_csv: str,
    categorical_cols: Sequence[str] | None = None,
) -> FeatureArtifacts:
    """Build train/test feature tables with deterministic customer attributes."""

    categorical_cols = list(categorical_cols or ["age_category", "sex", "user_stage"])

    con = duckdb.connect()
    reference_date = _compute_reference_date(con, data_csv)

    feature_query = f"""
    WITH base AS (
        SELECT *
        FROM read_csv_auto('{data_csv}')
        WHERE user_id IS NOT NULL
    ),
    profile_latest AS (
        SELECT
            user_id,
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
        FROM base
        GROUP BY user_id
    ),
    agg AS (
        SELECT
            user_id,
            COUNT(*) AS purchase_count,
            SUM(total_price) AS total_spent,
            AVG(total_price) AS avg_spent,
            MAX(total_price) AS max_spent,
            MIN(total_price) AS min_spent,
            STDDEV_SAMP(total_price) AS std_spent,
            SUM(amount) AS total_items,
            AVG(amount) AS avg_items,
            COUNT(DISTINCT date) AS visit_days,
            MIN(date) AS first_purchase_date,
            MAX(date) AS last_purchase_date,
            {reference_date} - MAX(date) AS recency,
            COUNT(DISTINCT item_category_cd_1) AS unique_cat1,
            COUNT(DISTINCT item_category_cd_2) AS unique_cat2,
            COUNT(DISTINCT item_category_cd_3) AS unique_cat3,
            COUNT(DISTINCT jan_cd) AS unique_products,
            SUM(CASE WHEN total_price < 0 THEN 1 ELSE 0 END) AS refund_txn_count,
            SUM(CASE WHEN total_price < 0 THEN total_price ELSE 0 END) AS refund_total_spent,
            SUM(CASE WHEN date >= {reference_date - 30} THEN 1 ELSE 0 END) AS txns_30d,
            SUM(CASE WHEN date >= {reference_date - 60} THEN 1 ELSE 0 END) AS txns_60d,
            SUM(CASE WHEN date >= {reference_date - 90} THEN 1 ELSE 0 END) AS txns_90d,
            SUM(CASE WHEN date >= {reference_date - 30} THEN total_price ELSE 0 END) AS spent_30d,
            SUM(CASE WHEN date >= {reference_date - 60} THEN total_price ELSE 0 END) AS spent_60d,
            SUM(CASE WHEN date >= {reference_date - 90} THEN total_price ELSE 0 END) AS spent_90d,
            COUNT(DISTINCT CASE WHEN date >= {reference_date - 30} THEN item_category_cd_1 END) AS unique_cat1_30d,
            COUNT(DISTINCT CASE WHEN date >= {reference_date - 90} THEN item_category_cd_1 END) AS unique_cat1_90d
        FROM base
        GROUP BY user_id
    )
    SELECT
        a.*, p.age_category, p.sex, p.user_stage,
        p.membership_start_ym,
        p.user_flag_ec, p.user_flag_1, p.user_flag_2, p.user_flag_3,
        p.user_flag_4, p.user_flag_5, p.user_flag_6,
        a.total_spent / NULLIF(a.visit_days, 0) AS avg_spent_per_visit,
        a.purchase_count / NULLIF(a.visit_days, 0) AS avg_purchases_per_visit,
        a.last_purchase_date - a.first_purchase_date AS purchase_span,
        a.visit_days * 1.0 / NULLIF((a.last_purchase_date - a.first_purchase_date + 1), 0) AS visit_frequency,
        a.refund_txn_count * 1.0 / NULLIF(a.purchase_count, 0) AS refund_txn_ratio,
        ABS(a.refund_total_spent) * 1.0 / NULLIF(ABS(a.total_spent), 0) AS refund_amount_ratio,
        a.txns_30d * 1.0 / NULLIF(a.purchase_count, 0) AS txn_share_30d,
        a.txns_90d * 1.0 / NULLIF(a.purchase_count, 0) AS txn_share_90d,
        a.spent_30d * 1.0 / NULLIF(a.total_spent, 0) AS spent_share_30d,
        a.spent_90d * 1.0 / NULLIF(a.total_spent, 0) AS spent_share_90d,
        a.unique_cat1_30d - a.unique_cat1_90d AS cat1_diversity_change_30_90,
        a.unique_cat1_30d * 1.0 / NULLIF(a.unique_cat1, 0) AS cat1_recent_ratio
    FROM agg a
    LEFT JOIN profile_latest p USING (user_id)
    """

    features_df = con.execute(feature_query).fetchdf()

    train_labels = con.execute(
        f"SELECT user_id, churn AS target FROM read_csv_auto('{train_flag_csv}')"
    ).fetchdf()
    test_users = con.execute(
        f"SELECT user_id FROM read_csv_auto('{sample_submission_csv}')"
    ).fetchdf()

    con.close()

    train_df = features_df.merge(train_labels, on="user_id", how="inner")
    test_df = features_df.merge(test_users, on="user_id", how="inner")

    feature_cols = [c for c in features_df.columns if c != "user_id"]

    for col in categorical_cols:
        train_df[col] = train_df[col].fillna("Unknown").astype("category")
        test_df[col] = test_df[col].fillna("Unknown").astype("category")

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    for col in numeric_cols:
        median_value = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    return FeatureArtifacts(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        reference_date=reference_date,
    )
