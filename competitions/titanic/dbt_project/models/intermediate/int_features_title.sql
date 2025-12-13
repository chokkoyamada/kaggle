{{
    config(
        materialized='view'
    )
}}

-- 名前から称号（Title）を抽出し、統合する

with base as (
    select * from {{ ref('int_all_passengers') }}
),

with_title as (
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
        -- 名前から称号を抽出
        regexp_extract(name, ' ([A-Za-z]+)\.', 1) as title_raw
    from base
),

title_mapped as (
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
        title_raw,
        case
            when title_raw in ('Mr') then 'Mr'
            when title_raw in ('Miss', 'Mlle', 'Ms') then 'Miss'
            when title_raw in ('Mrs', 'Mme') then 'Mrs'
            when title_raw in ('Master') then 'Master'
            else 'Rare'
        end as title
    from with_title
),

title_encoded as (
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
        case title
            when 'Mr' then 0
            when 'Miss' then 1
            when 'Mrs' then 2
            when 'Master' then 3
            when 'Rare' then 4
        end as title_encoded
    from title_mapped
)

select * from title_encoded
