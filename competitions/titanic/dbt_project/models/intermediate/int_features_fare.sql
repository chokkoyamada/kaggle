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
        family_size_category,
        age_filled,
        age_is_child,
        age_band
    from {{ ref('int_features_age') }}
),

-- グローバル中央値とグループ中央値（pclass, embarked）を計算
fare_stats_global as (
    select median(fare) as fare_median from base where fare is not null
),

fare_medians_by_group as (
    select
        pclass,
        embarked,
        median(fare) as fare_median
    from base
    where fare is not null
    group by pclass, embarked
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
        base.family_size_category,
        base.age_filled,
        base.age_is_child,
        base.age_band,
        coalesce(
            base.fare,
            fare_medians_by_group.fare_median,
            fare_stats_global.fare_median
        ) as fare_filled
    from base
    left join fare_medians_by_group
        on base.pclass = fare_medians_by_group.pclass
       and base.embarked = fare_medians_by_group.embarked
    cross join fare_stats_global
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
        family_size_category,
        age_filled,
        age_is_child,
        age_band,
        fare_filled,
        ntile(4) over (order by fare_filled) - 1 as fare_band
    from with_fare_filled
)

select * from with_fare_band
