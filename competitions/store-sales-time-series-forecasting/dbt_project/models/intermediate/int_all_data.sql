{{
    config(
        materialized='view'
    )
}}

-- trainとtestを統合
-- ラグ特徴量計算のため、時系列として連続したデータセットを作成

with train_data as (
    select
        id,
        date,
        store_nbr,
        family,
        sales,
        onpromotion,
        dataset_type
    from {{ ref('stg_train') }}
),

test_data as (
    select
        id,
        date,
        store_nbr,
        family,
        sales,
        onpromotion,
        dataset_type
    from {{ ref('stg_test') }}
),

combined as (
    select * from train_data
    union all
    select * from test_data
)

select * from combined
order by store_nbr, family, date
