{{
    config(
        materialized='view'
    )
}}

-- 店舗マスターデータの取り込みと型固定

with source as (
    select * from {{ ref('stores') }}
),

renamed_and_typed as (
    select
        cast(store_nbr as integer) as store_nbr,
        cast(city as varchar) as city,
        cast(state as varchar) as state,
        cast(type as varchar) as store_type,
        cast(cluster as integer) as cluster
    from source
)

select * from renamed_and_typed
