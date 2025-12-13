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
        family_size_category,
        age_filled,
        age_is_child,
        age_band,
        fare_filled,
        fare_band
    from {{ ref('int_features_fare') }}
),

-- Embarkedの欠損値を最頻値で補完
embarked_mode as (
    -- train の分布から最頻値を算出（テスト影響排除）
    select
        embarked,
        count(*) as cnt
    from base
    where embarked is not null and dataset_type = 'train'
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
        base.ticket,
        base.pclass,
        base.age_filled,
        base.fare_filled,
        coalesce(base.embarked, embarked_mode.embarked) as embarked_filled,
        base.title,
        base.title_encoded,
        base.family_size,
        base.is_alone,
        base.family_size_category,
        base.age_band,
        base.fare_band,
        base.age_is_child
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
        ticket,
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
        family_size_category,
        age_band,
        age_is_child,
        fare_band,

        case when cabin is not null then 1 else 0 end as has_cabin,

        -- デッキ（Cabinの先頭文字）
        case
            when cabin is null then 'U'
            else substr(cabin, 1, 1)
        end as deck,

        -- 交互作用特徴量1: 性別 × 客室クラス（1等女性は生存率が非常に高い）
        case
            when sex = 'female' and pclass = 1 then 1
            when sex = 'female' and pclass in (2, 3) then 2
            when sex = 'male' and pclass = 1 then 3
            else 4
        end as sex_pclass_interaction,

        -- 交互作用特徴量2: 子供 × 客室クラス（クラスによって子供の生存率が異なる可能性）
        case
            when age_is_child = 1 and pclass = 1 then 1
            when age_is_child = 1 and pclass in (2, 3) then 2
            else 0
        end as child_pclass_interaction

    from with_embarked_filled
),

-- チケット単位のグループサイズを算出して付与
ticket_counts as (
    select ticket, count(*) as ticket_group_size
    from (
        select ticket from {{ ref('int_features_fare') }}
    ) t
    group by ticket
),

final as (
    select
        ae.passenger_id,
        ae.survived,
        ae.dataset_type,
        ae.name,
        ae.sex,
        ae.cabin,
        ae.ticket,
        ae.age,
        ae.fare,
        ae.embarked,
        ae.title,
        ae.pclass,
        ae.sex_encoded,
        ae.embarked_encoded,
        ae.title_encoded,
        ae.family_size,
        ae.is_alone,
        ae.family_size_category,
        ae.age_band,
        ae.age_is_child,
        ae.fare_band,
        ae.has_cabin,
        ae.deck,
        ae.sex_pclass_interaction,
        ae.child_pclass_interaction,
        coalesce(tc.ticket_group_size, 1) as ticket_group_size
    from with_all_encoded ae
    left join ticket_counts tc on ae.ticket = tc.ticket
)
select * from final
