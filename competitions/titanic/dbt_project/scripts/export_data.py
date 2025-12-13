"""
DuckDBからdbt生成データをエクスポートするスクリプト
"""
import duckdb
import pandas as pd
from pathlib import Path

# dbtプロジェクトのルートから相対パスで設定
PROJECT_ROOT = Path(__file__).parent.parent
DUCKDB_PATH = PROJECT_ROOT / '../data/titanic.duckdb'
OUTPUT_DIR = PROJECT_ROOT / '../data/processed'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DuckDBに接続
conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

print("DuckDBからデータをエクスポート中...")

# 訓練データをエクスポート
train_df = conn.execute("SELECT * FROM main.train_processed").fetchdf()
print(f"訓練データ: {train_df.shape}")
print(f"カラム: {list(train_df.columns)}")
print(f"生存率: {train_df['survived'].mean():.2%}")

# テストデータをエクスポート
test_df = conn.execute("SELECT * FROM main.test_processed").fetchdf()
print(f"\nテストデータ: {test_df.shape}")
print(f"カラム: {list(test_df.columns)}")

# CSVとして保存
train_df.to_csv(OUTPUT_DIR / 'train_processed.csv', index=False)
test_df.to_csv(OUTPUT_DIR / 'test_processed.csv', index=False)

print(f"\nエクスポート完了:")
print(f"  - {OUTPUT_DIR / 'train_processed.csv'}")
print(f"  - {OUTPUT_DIR / 'test_processed.csv'}")

# データの概要を表示
print("\n訓練データの先頭5行:")
print(train_df.head())

print("\nテストデータの先頭5行:")
print(test_df.head())

# 接続を閉じる
conn.close()
