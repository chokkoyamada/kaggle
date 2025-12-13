{{
    config(
        materialized='view'
    )
}}

-- 訓練データの取り込みと型固定
-- Titanicで学んだベストプラクティス: stagingで全カラムに明示的なCASTを適用

with source as (
    select * from {{ ref('train') }}
),

renamed_and_typed as (
    select
        cast(id as bigint) as id,
        cast(date as date) as date,
        cast(store_nbr as integer) as store_nbr,
        cast(family as varchar) as family,
        cast(sales as double) as sales,
        cast(onpromotion as integer) as onpromotion,
        'train' as dataset_type
    from source
)

select * from renamed_and_typed
