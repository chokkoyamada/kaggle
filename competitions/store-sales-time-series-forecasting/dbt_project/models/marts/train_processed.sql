{{
    config(
        materialized='table'
    )
}}

-- 機械学習用の訓練データ
-- モデルが直接使用する特徴量のみを選択

with features as (
    select * from {{ ref('int_features_complete') }}
    where dataset_type = 'train'
)

select
    -- ID
    cast(id as bigint) as id,

    -- ターゲット変数
    cast(sales as double) as sales,

    -- カテゴリカル特徴量
    cast(store_nbr as integer) as store_nbr,
    cast(family_encoded as integer) as family_encoded,
    cast(cluster as integer) as cluster,
    cast(store_type_encoded as integer) as store_type_encoded,
    cast(onpromotion as integer) as onpromotion,

    -- 日付特徴量
    cast(year as integer) as year,
    cast(month as integer) as month,
    cast(day_of_month as integer) as day_of_month,
    cast(day_of_week as integer) as day_of_week,
    cast(week_of_year as integer) as week_of_year,
    cast(quarter as integer) as quarter,
    cast(is_weekend as integer) as is_weekend,
    cast(is_month_start as integer) as is_month_start,
    cast(is_month_end as integer) as is_month_end,
    cast(is_payday as integer) as is_payday,

    -- 外部データ特徴量
    cast(oil_price as double) as oil_price,
    cast(is_holiday as integer) as is_holiday,
    cast(transactions as integer) as transactions,

    -- ラグ特徴量
    cast(sales_lag_7 as double) as sales_lag_7,
    cast(sales_lag_14 as double) as sales_lag_14,
    cast(sales_lag_21 as double) as sales_lag_21,
    cast(sales_lag_28 as double) as sales_lag_28,

    -- 移動平均特徴量
    cast(sales_rolling_mean_7 as double) as sales_rolling_mean_7,
    cast(sales_rolling_mean_14 as double) as sales_rolling_mean_14,
    cast(sales_rolling_mean_28 as double) as sales_rolling_mean_28,
    cast(sales_rolling_std_28 as double) as sales_rolling_std_28

from features
order by id
