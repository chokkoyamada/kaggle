{{
    config(
        materialized='view'
    )
}}

-- 外部データ（石油価格、休日、トランザクション）を統合

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
        days_in_month
    from {{ ref('int_date_features') }}
),

-- 石油価格を補完（前方埋め）
oil_filled as (
    select
        date,
        oil_price,
        -- DuckDBのLAST_VALUE window関数で前方埋め
        last_value(oil_price ignore nulls) over (
            order by date
            rows between unbounded preceding and current row
        ) as oil_price_filled
    from {{ ref('stg_oil') }}
),

-- 休日フラグを作成
holidays as (
    select
        date,
        1 as is_holiday
    from {{ ref('stg_holidays') }}
    where locale = 'National'  -- 全国的な休日のみ
    group by date
),

-- トランザクション数
transactions as (
    select
        date,
        store_nbr,
        transactions
    from {{ ref('stg_transactions') }}
),

-- すべてを結合
with_external as (
    select
        base.id,
        base.date,
        base.store_nbr,
        base.family,
        base.sales,
        base.onpromotion,
        base.dataset_type,
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
        base.days_in_month,

        -- 石油価格（補完済み）
        coalesce(oil_filled.oil_price_filled, 0.0) as oil_price,

        -- 休日フラグ
        coalesce(holidays.is_holiday, 0) as is_holiday,

        -- トランザクション数（欠損の場合は0）
        coalesce(transactions.transactions, 0) as transactions

    from base
    left join oil_filled on base.date = oil_filled.date
    left join holidays on base.date = holidays.date
    left join transactions on base.date = transactions.date and base.store_nbr = transactions.store_nbr
)

select * from with_external
