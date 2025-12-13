{{
    config(
        materialized='view'
    )
}}

-- 石油価格データの取り込みと型固定
-- 3.5%の欠損値があるため、補完が必要（intermediateレイヤーで実施）

with source as (
    select * from {{ ref('oil') }}
),

renamed_and_typed as (
    select
        cast(date as date) as date,
        cast(dcoilwtico as double) as oil_price
    from source
)

select * from renamed_and_typed
