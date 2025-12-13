{{
    config(
        materialized='view'
    )
}}

-- テストデータの取り込みと型固定
-- salesカラムは存在しないため、後で予測値を入れるためにNULLを設定

with source as (
    select * from {{ ref('test') }}
),

renamed_and_typed as (
    select
        cast(id as bigint) as id,
        cast(date as date) as date,
        cast(store_nbr as integer) as store_nbr,
        cast(family as varchar) as family,
        cast(null as double) as sales,  -- テストデータには存在しないが、統一的に扱うため
        cast(onpromotion as integer) as onpromotion,
        'test' as dataset_type
    from source
)

select * from renamed_and_typed
