{{
    config(
        materialized='view'
    )
}}

-- 運賃の欠損値補完と運賃帯の作成

with base as (
    select
        passenger_id,
        survived,
        dataset_type,
        pclass,
        name,
        sex,
        age,
        sibsp,
        parch,
        ticket,
        fare,
        cabin,
        embarked,
        title,
        title_encoded,
        family_size,
        is_alone,
        age_filled,
        age_band
    from {{ ref('int_features_age') }}
),

-- 運賃の中央値を計算
fare_stats as (
    select
        median(fare) as fare_median
    from base
    where fare is not null
),

-- 欠損値を補完
with_fare_filled as (
    select
        base.passenger_id,
        base.survived,
        base.dataset_type,
        base.pclass,
        base.name,
        base.sex,
        base.age,
        base.sibsp,
        base.parch,
        base.ticket,
        base.fare,
        base.cabin,
        base.embarked,
        base.title,
        base.title_encoded,
        base.family_size,
        base.is_alone,
        base.age_filled,
        base.age_band,
        coalesce(base.fare, fare_stats.fare_median) as fare_filled
    from base
    cross join fare_stats
),

-- 運賃を4分位数で区間に分割
-- DuckDBのntile関数を使用
with_fare_band as (
    select
        passenger_id,
        survived,
        dataset_type,
        pclass,
        name,
        sex,
        age,
        sibsp,
        parch,
        ticket,
        fare,
        cabin,
        embarked,
        title,
        title_encoded,
        family_size,
        is_alone,
        age_filled,
        age_band,
        fare_filled,
        ntile(4) over (order by fare_filled) - 1 as fare_band
    from with_fare_filled
)

select * from with_fare_band
