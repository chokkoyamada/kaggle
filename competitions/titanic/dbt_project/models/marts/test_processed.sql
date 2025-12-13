{{
    config(
        materialized='table'
    )
}}

-- 機械学習用のテストデータ
-- モデルが直接使用する特徴量のみを選択

with features as (
    select * from {{ ref('int_features_complete') }}
    where dataset_type = 'test'
)

select
    passenger_id,
    cast(pclass as integer) as pclass,
    sex_encoded,
    embarked_encoded,
    title_encoded,
    family_size,
    is_alone,
    age_band,
    cast(fare_band as integer) as fare_band,
    has_cabin
from features
order by passenger_id
