"""
Store Sales - 改良モデル（dbt + DuckDB特徴量を使用）

ベースラインモデルに以下を追加:
- より多くのラグ特徴（21日、28日）
- 移動平均特徴（7日、14日、28日）
- 外部データ（石油価格、休日、トランザクション）
- 店舗情報（cluster, store_type）
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / 'data/raw'
OUTPUT_DIR = PROJECT_ROOT / 'predictions'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Store Sales - 改良モデル（豊富な特徴量）")
print("=" * 80)

# データの読み込み
print("\nデータを読み込み中...")
train_df = pd.read_csv(DATA_RAW_DIR / 'train.csv', parse_dates=['date'])
test_df = pd.read_csv(DATA_RAW_DIR / 'test.csv', parse_dates=['date'])
stores_df = pd.read_csv(DATA_RAW_DIR / 'stores.csv')
oil_df = pd.read_csv(DATA_RAW_DIR / 'oil.csv', parse_dates=['date'])
holidays_df = pd.read_csv(DATA_RAW_DIR / 'holidays_events.csv', parse_dates=['date'])
transactions_df = pd.read_csv(DATA_RAW_DIR / 'transactions.csv', parse_dates=['date'])

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")

# 特徴量生成関数
def create_features(df, stores, oil, holidays, transactions, is_train=True):
    """豊富な特徴量を生成"""
    df = df.copy()

    # 日付特徴
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 23).astype(int)

    # 店舗情報を結合
    df = df.merge(stores[['store_nbr', 'cluster', 'type']], on='store_nbr', how='left')
    df['store_type'] = df['type'].astype('category').cat.codes
    df = df.drop('type', axis=1)

    # 石油価格を結合（前方埋め）
    oil_filled = oil.set_index('date')['dcoilwtico'].fillna(method='ffill').reset_index()
    df = df.merge(oil_filled, on='date', how='left')
    df['dcoilwtico'] = df['dcoilwtico'].fillna(0)

    # 休日フラグ
    national_holidays = holidays[holidays['locale'] == 'National']['date'].unique()
    df['is_holiday'] = df['date'].isin(national_holidays).astype(int)

    # トランザクション数
    df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
    df['transactions'] = df['transactions'].fillna(0)

    # familyをカテゴリコード化
    df['family'] = df['family'].astype('category').cat.codes

    return df

print("\n特徴量を生成中...")
train_features = create_features(train_df, stores_df, oil_df, holidays_df, transactions_df)
test_features = create_features(test_df, stores_df, oil_df, holidays_df, transactions_df, is_train=False)

# 訓練とテストを結合してラグ特徴量を計算
print("\nラグ特徴量を生成中...")
all_data = pd.concat([
    train_features.assign(dataset='train'),
    test_features.assign(dataset='test')
], ignore_index=True).sort_values(['store_nbr', 'family', 'date'])

# ラグ特徴量と移動平均
for lag in [7, 14, 21, 28]:
    all_data[f'sales_lag_{lag}'] = all_data.groupby(['store_nbr', 'family'])['sales'].shift(lag)

for window in [7, 14, 28]:
    all_data[f'sales_rolling_mean_{window}'] = all_data.groupby(['store_nbr', 'family'])['sales'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# 訓練とテストに再分割
train_final = all_data[all_data['dataset'] == 'train'].copy()
test_final = all_data[all_data['dataset'] == 'test'].copy()

# 特徴量リスト
feature_cols = [
    'store_nbr', 'family', 'onpromotion', 'cluster', 'store_type',
    'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_month_end',
    'dcoilwtico', 'is_holiday', 'transactions',
    'sales_lag_7', 'sales_lag_14', 'sales_lag_21', 'sales_lag_28',
    'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_28'
]

# 欠損値を0で埋める
train_final[feature_cols] = train_final[feature_cols].fillna(0)
test_final[feature_cols] = test_final[feature_cols].fillna(0)

# 検証用データ分割
validation_start_date = train_final['date'].max() - pd.Timedelta(days=30)
train_data = train_final[train_final['date'] < validation_start_date]
valid_data = train_final[train_final['date'] >= validation_start_date]

X_train = train_data[feature_cols]
y_train = train_data['sales']
X_valid = valid_data[feature_cols]
y_valid = valid_data['sales']
X_test = test_final[feature_cols]

print(f"\n訓練データ: {X_train.shape}")
print(f"検証データ: {X_valid.shape}")
print(f"テストデータ: {X_test.shape}")
print(f"特徴量数: {len(feature_cols)}")

# LightGBMモデルの訓練
print("\n" + "=" * 80)
print("LightGBMモデルを訓練中...")
print("=" * 80)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# 検証データで評価
print("\n" + "=" * 80)
print("検証データで評価")
print("=" * 80)

y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
y_pred_valid = np.maximum(y_pred_valid, 0)

rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred_valid))
print(f"\n検証データRMSLE: {rmsle:.4f}")

# 特徴量の重要度
print("\n特徴量の重要度（Top 10）:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False).head(10)
print(feature_importance.to_string(index=False))

# テストデータで予測
print("\n" + "=" * 80)
print("テストデータで予測を生成中...")
print("=" * 80)

y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_test = np.maximum(y_pred_test, 0)

# 提出用ファイルを作成
submission_df = pd.DataFrame({
    'id': test_final['id'],
    'sales': y_pred_test
})

output_path = OUTPUT_DIR / 'submission_improved.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n予測結果を保存しました: {output_path}")
print(f"予測売上統計:")
print(f"  平均: {y_pred_test.mean():.2f}")
print(f"  中央値: {np.median(y_pred_test):.2f}")
print(f"  最小値: {y_pred_test.min():.2f}")
print(f"  最大値: {y_pred_test.max():.2f}")

print("\n" + "=" * 80)
print("完了！")
print("=" * 80)
print(f"\nベースラインとの比較:")
print(f"  ベースラインRMSLE: 0.6851")
print(f"  改良モデルRMSLE: {rmsle:.4f}")
print(f"  改善: {((0.6851 - rmsle) / 0.6851 * 100):.2f}%")
