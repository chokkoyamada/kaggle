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
        is_alone,
        family_size_category
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

-- 追加のフォールバック: sex と pclass ごとの年齢中央値、全体中央値
age_medians_sex_pclass as (
    select
        sex,
        pclass,
        median(age) as age_median
    from base
    where age is not null
    group by sex, pclass
),

age_median_global as (
    select median(age) as age_median from base where age is not null
),

-- 欠損値を補完（段階的フォールバック）
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
        base.family_size_category,
        coalesce(
            base.age,
            age_medians.age_median,
            age_medians_sex_pclass.age_median,
            age_median_global.age_median
        ) as age_filled
    from base
    left join age_medians on base.title = age_medians.title
    left join age_medians_sex_pclass on base.sex = age_medians_sex_pclass.sex and base.pclass = age_medians_sex_pclass.pclass
    cross join age_median_global
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
        family_size_category,
        age_filled,
        case when age_filled <= 12 then 1 else 0 end as age_is_child,
        case
            when age_filled <= 12 then 0  -- 子供
            when age_filled <= 18 then 1  -- 青少年
            when age_filled <= 35 then 2  -- 若年成人
            when age_filled <= 60 then 3  -- 中年
            else 4                         -- 高齢
        end as age_band
    from with_age_filled
)

select * from with_age_band
