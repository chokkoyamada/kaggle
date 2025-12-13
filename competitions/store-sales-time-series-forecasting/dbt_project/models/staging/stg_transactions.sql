{{
    config(
        materialized='view'
    )
}}

-- トランザクション数データの取り込みと型固定
-- EDAで見たように、salesと強い相関（0.834）があるため重要な特徴量

with source as (
    select * from {{ ref('transactions') }}
),

renamed_and_typed as (
    select
        cast(date as date) as date,
        cast(store_nbr as integer) as store_nbr,
        cast(transactions as integer) as transactions
    from source
)

select * from renamed_and_typed
