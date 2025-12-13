{{
    config(
        materialized='view'
    )
}}

-- 日付から豊富な特徴量を生成
-- 時系列予測では日付特徴が重要な役割を果たす

with base as (
    select
        id,
        date,
        store_nbr,
        family,
        sales,
        onpromotion,
        dataset_type
    from {{ ref('int_all_data') }}
),

with_date_features as (
    select
        id,
        date,
        store_nbr,
        family,
        sales,
        onpromotion,
        dataset_type,

        -- 基本的な日付特徴
        cast(extract(year from date) as integer) as year,
        cast(extract(month from date) as integer) as month,
        cast(extract(day from date) as integer) as day_of_month,
        cast(extract(dayofweek from date) as integer) as day_of_week,  -- 0=日曜, 6=土曜
        cast(extract(week from date) as integer) as week_of_year,
        cast(extract(quarter from date) as integer) as quarter,

        -- 週末フラグ
        case
            when extract(dayofweek from date) in (0, 6) then 1
            else 0
        end as is_weekend,

        -- 月初・月末フラグ
        case when extract(day from date) <= 7 then 1 else 0 end as is_month_start,
        case when extract(day from date) >= 23 then 1 else 0 end as is_month_end,

        -- 給料日フラグ（15日と月末近く）
        case
            when extract(day from date) in (15, 30, 31) then 1
            else 0
        end as is_payday,

        -- 月の日数（2月対応）
        cast(extract(day from (date + interval '1 month' - interval '1 day')) as integer) as days_in_month

    from base
)

select * from with_date_features
