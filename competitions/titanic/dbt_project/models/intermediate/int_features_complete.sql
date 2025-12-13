{{
    config(
        materialized='view'
    )
}}

-- すべての特徴量を統合し、最終的なエンコーディングを行う

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
        age_band,
        fare_filled,
        fare_band
    from {{ ref('int_features_fare') }}
),

-- Embarkedの欠損値を最頻値で補完
embarked_mode as (
    select
        embarked,
        count(*) as cnt
    from base
    where embarked is not null
    group by embarked
    order by cnt desc
    limit 1
),

with_embarked_filled as (
    select
        -- 必要な列だけを明示的に選択
        base.passenger_id,
        base.survived,
        base.dataset_type,
        base.name,
        base.sex,
        base.cabin,
        base.pclass,
        base.age_filled,
        base.fare_filled,
        coalesce(base.embarked, embarked_mode.embarked) as embarked_filled,
        base.title,
        base.title_encoded,
        base.family_size,
        base.is_alone,
        base.age_band,
        base.fare_band
    from base
    cross join embarked_mode
),

-- すべてのカテゴリカル変数をエンコード
with_all_encoded as (
    select
        passenger_id,
        survived,
        dataset_type,

        -- 元の列（参照用）
        name,
        sex,
        cabin,
        age_filled as age,
        fare_filled as fare,
        embarked_filled as embarked,
        title,

        -- エンコードされた特徴量
        cast(pclass as integer) as pclass,

        case sex
            when 'female' then 1
            when 'male' then 0
        end as sex_encoded,

        case embarked_filled
            when 'C' then 0
            when 'Q' then 1
            when 'S' then 2
        end as embarked_encoded,

        title_encoded,
        family_size,
        is_alone,
        age_band,
        fare_band,

        case when cabin is not null then 1 else 0 end as has_cabin

    from with_embarked_filled
)

select * from with_all_encoded
