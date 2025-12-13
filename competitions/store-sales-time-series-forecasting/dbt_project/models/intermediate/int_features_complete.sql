{{
    config(
        materialized='view'
    )
}}

-- 最終的な特徴量セット
-- 店舗情報を統合し、カテゴリカル変数をエンコード

with base as (
    select
        id,
        date,
        store_nbr,
        family,
        sales,
        onpromotion,
        dataset_type,
        year,
        month,
        day_of_month,
        day_of_week,
        week_of_year,
        quarter,
        is_weekend,
        is_month_start,
        is_month_end,
        is_payday,
        days_in_month,
        oil_price,
        is_holiday,
        transactions,
        sales_lag_7,
        sales_lag_14,
        sales_lag_21,
        sales_lag_28,
        sales_rolling_mean_7,
        sales_rolling_mean_14,
        sales_rolling_mean_28,
        sales_rolling_std_28
    from {{ ref('int_lag_features') }}
),

stores as (
    select
        store_nbr,
        city,
        state,
        store_type,
        cluster
    from {{ ref('stg_stores') }}
),

-- familyのカテゴリコード化
family_mapping as (
    select distinct
        family,
        row_number() over (order by family) - 1 as family_encoded
    from base
),

-- store_typeのカテゴリコード化
store_type_mapping as (
    select distinct
        store_type,
        row_number() over (order by store_type) - 1 as store_type_encoded
    from stores
),

with_all_features as (
    select
        base.id,
        base.date,
        base.store_nbr,
        base.family,
        base.sales,
        base.dataset_type,

        -- カテゴリカル変数（エンコード済み）
        family_mapping.family_encoded,
        stores.cluster,
        store_type_mapping.store_type_encoded,
        base.onpromotion,

        -- 日付特徴
        base.year,
        base.month,
        base.day_of_month,
        base.day_of_week,
        base.week_of_year,
        base.quarter,
        base.is_weekend,
        base.is_month_start,
        base.is_month_end,
        base.is_payday,

        -- 外部データ特徴
        base.oil_price,
        base.is_holiday,
        base.transactions,

        -- ラグ特徴量（欠損値は0で埋める）
        coalesce(base.sales_lag_7, 0.0) as sales_lag_7,
        coalesce(base.sales_lag_14, 0.0) as sales_lag_14,
        coalesce(base.sales_lag_21, 0.0) as sales_lag_21,
        coalesce(base.sales_lag_28, 0.0) as sales_lag_28,

        -- 移動平均（欠損値は0で埋める）
        coalesce(base.sales_rolling_mean_7, 0.0) as sales_rolling_mean_7,
        coalesce(base.sales_rolling_mean_14, 0.0) as sales_rolling_mean_14,
        coalesce(base.sales_rolling_mean_28, 0.0) as sales_rolling_mean_28,
        coalesce(base.sales_rolling_std_28, 0.0) as sales_rolling_std_28

    from base
    left join stores on base.store_nbr = stores.store_nbr
    left join family_mapping on base.family = family_mapping.family
    left join store_type_mapping on stores.store_type = store_type_mapping.store_type
)

select * from with_all_features
