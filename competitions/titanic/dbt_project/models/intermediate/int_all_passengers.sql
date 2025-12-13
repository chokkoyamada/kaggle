{{
    config(
        materialized='view'
    )
}}

-- 訓練データとテストデータを統合
-- stg層で型が固定されているので、downstream は型を信頼できる

with train as (
    select
        passenger_id,
        survived,
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
        'train' as dataset_type
    from {{ ref('stg_train') }}
),

test as (
    select
        passenger_id,
        null as survived,
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
        'test' as dataset_type
    from {{ ref('stg_test') }}
),

combined as (
    select * from train
    union all
    select * from test
)

select * from combined
