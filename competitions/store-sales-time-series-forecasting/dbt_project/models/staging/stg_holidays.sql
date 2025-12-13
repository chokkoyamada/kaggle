{{
    config(
        materialized='view'
    )
}}

-- 休日・イベントデータの取り込みと型固定

with source as (
    select * from {{ ref('holidays_events') }}
),

renamed_and_typed as (
    select
        cast(date as date) as date,
        cast(type as varchar) as event_type,
        cast(locale as varchar) as locale,
        cast(locale_name as varchar) as locale_name,
        cast(description as varchar) as description,
        cast(transferred as boolean) as transferred
    from source
)

select * from renamed_and_typed
