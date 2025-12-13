"""
dbtで生成した特徴量をDuckDBからCSVにエクスポート
martsレイヤーの代替として、int_features_completeから直接エクスポート
"""
import duckdb
from pathlib import Path

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DUCKDB_PATH = PROJECT_ROOT / '../data/store_sales.duckdb'
OUTPUT_DIR = PROJECT_ROOT / '../data/processed'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("dbt生成特徴量をDuckDBからエクスポート")
print("=" * 80)

# DuckDBに接続
conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

print("\nDuckDBからデータをエクスポート中...")

# 訓練データをエクスポート（int_features_completeから直接）
print("\n訓練データを処理中...")
train_query = """
SELECT
    id,
    sales,
    store_nbr,
    family_encoded,
    cluster,
    store_type_encoded,
    onpromotion,
    year,
    month,
    day_of_month,
    day_of_week,
    week_of_year,
    quarter,
    is_weekend,
    is_month_start,
    is_month_end,
    is_payday,
    oil_price,
    is_holiday,
    transactions,
    sales_lag_7,
    sales_lag_14,
    sales_lag_21,
    sales_lag_28,
    sales_rolling_mean_7,
    sales_rolling_mean_14,
    sales_rolling_mean_28,
    sales_rolling_std_28
FROM main_intermediate.int_features_complete
WHERE dataset_type = 'train'
ORDER BY id
"""

train_df = conn.execute(train_query).fetchdf()
train_path = OUTPUT_DIR / 'train_processed.csv'
train_df.to_csv(train_path, index=False)
print(f"訓練データ保存: {train_path}")
print(f"  レコード数: {len(train_df):,}")
print(f"  特徴量数: {len(train_df.columns)}")

# テストデータをエクスポート
print("\nテストデータを処理中...")
test_query = """
SELECT
    id,
    store_nbr,
    family_encoded,
    cluster,
    store_type_encoded,
    onpromotion,
    year,
    month,
    day_of_month,
    day_of_week,
    week_of_year,
    quarter,
    is_weekend,
    is_month_start,
    is_month_end,
    is_payday,
    oil_price,
    is_holiday,
    transactions,
    sales_lag_7,
    sales_lag_14,
    sales_lag_21,
    sales_lag_28,
    sales_rolling_mean_7,
    sales_rolling_mean_14,
    sales_rolling_mean_28,
    sales_rolling_std_28
FROM main_intermediate.int_features_complete
WHERE dataset_type = 'test'
ORDER BY id
"""

test_df = conn.execute(test_query).fetchdf()
test_path = OUTPUT_DIR / 'test_processed.csv'
test_df.to_csv(test_path, index=False)
print(f"テストデータ保存: {test_path}")
print(f"  レコード数: {len(test_df):,}")
print(f"  特徴量数: {len(test_df.columns)}")

# データの概要を表示
print("\n" + "=" * 80)
print("データ概要")
print("=" * 80)

print("\n訓練データの先頭5行:")
print(train_df.head())

print("\nテストデータの先頭5行:")
print(test_df.head())

print("\n訓練データの基本統計:")
print(train_df.describe())

# 接続を閉じる
conn.close()

print("\n" + "=" * 80)
print("エクスポート完了！")
print("=" * 80)
