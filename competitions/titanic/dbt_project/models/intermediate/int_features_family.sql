{{
    config(
        materialized='view'
    )
}}

-- 家族サイズ関連の特徴量を作成

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
        title_encoded
    from {{ ref('int_features_title') }}
),

with_family_features as (
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

        -- 家族サイズ = 兄弟姉妹・配偶者 + 親・子供 + 本人
        cast(sibsp as integer) + cast(parch as integer) + 1 as family_size,

        -- 単独者フラグ
        case when (cast(sibsp as integer) + cast(parch as integer) + 1) = 1 then 1 else 0 end as is_alone
    from base
)

select * from with_family_features
