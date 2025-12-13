{{
    config(
        materialized='view'
    )
}}

-- 訓練データのステージングモデル
-- 型を明示的に固定し、downstream で型の心配をなくす

with source as (
    select * from {{ ref('train') }}
),

renamed_and_typed as (
    select
        -- 整数型
        cast("PassengerId" as integer) as passenger_id,
        cast("Survived" as integer) as survived,
        cast("Pclass" as integer) as pclass,
        cast("SibSp" as integer) as sibsp,
        cast("Parch" as integer) as parch,

        -- 浮動小数点型
        cast("Age" as double) as age,
        cast("Fare" as double) as fare,

        -- 文字列型
        cast("Name" as varchar) as name,
        cast("Sex" as varchar) as sex,
        cast("Ticket" as varchar) as ticket,
        cast("Cabin" as varchar) as cabin,
        cast("Embarked" as varchar) as embarked
    from source
)

select * from renamed_and_typed
