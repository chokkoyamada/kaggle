{{
    config(
        materialized='view'
    )
}}

-- 年齢の欠損値補完と年齢帯の作成

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
        is_alone
    from {{ ref('int_features_family') }}
),

-- 称号ごとの年齢中央値を計算
age_medians as (
    select
        title,
        median(age) as age_median
    from base
    where age is not null
    group by title
),

-- 欠損値を補完
with_age_filled as (
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
        coalesce(base.age, age_medians.age_median) as age_filled
    from base
    left join age_medians on base.title = age_medians.title
),

-- 年齢を区間に分割
with_age_band as (
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
        case
            when age_filled <= 12 then 0  -- 子供
            when age_filled <= 18 then 1  -- 青少年
            when age_filled <= 35 then 2  -- 成人
            when age_filled <= 60 then 3  -- 中年
            else 4                         -- 高齢
        end as age_band
    from with_age_filled
)

select * from with_age_band
