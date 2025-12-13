{{
    config(
        materialized='view'
    )
}}

-- ラグ特徴量と移動平均を生成
-- DuckDBのwindow関数を使って、店舗×商品ファミリー別に時系列特徴量を計算

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
        transactions
    from {{ ref('int_external_data') }}
),

with_lag_features as (
    select
        -- すべての既存カラムを明示的に列挙（SELECT *回避）
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

        -- ラグ特徴量（7, 14, 21, 28日前の売上）
        lag(sales, 7) over w as sales_lag_7,
        lag(sales, 14) over w as sales_lag_14,
        lag(sales, 21) over w as sales_lag_21,
        lag(sales, 28) over w as sales_lag_28,

        -- 移動平均（7日、14日、28日）
        avg(sales) over (
            partition by store_nbr, family
            order by date
            rows between 7 preceding and 1 preceding
        ) as sales_rolling_mean_7,

        avg(sales) over (
            partition by store_nbr, family
            order by date
            rows between 14 preceding and 1 preceding
        ) as sales_rolling_mean_14,

        avg(sales) over (
            partition by store_nbr, family
            order by date
            rows between 28 preceding and 1 preceding
        ) as sales_rolling_mean_28,

        -- 標準偏差（売上の変動を捉える）
        stddev(sales) over (
            partition by store_nbr, family
            order by date
            rows between 28 preceding and 1 preceding
        ) as sales_rolling_std_28

    from base
    window w as (
        partition by store_nbr, family
        order by date
    )
)

select * from with_lag_features
