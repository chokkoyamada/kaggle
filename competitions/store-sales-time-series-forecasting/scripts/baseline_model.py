"""
Store Sales - シンプルなベースラインモデル

最小限の特徴量でLightGBMモデルを訓練し、予測を生成する。
複雑な特徴量エンジニアリング前のベンチマークを確立する。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data/raw'
OUTPUT_DIR = PROJECT_ROOT / 'predictions'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Store Sales - シンプルなベースラインモデル")
print("=" * 80)

# データの読み込み
print("\nデータを読み込み中...")
train_df = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'])
test_df = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['date'])

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")
print(f"訓練期間: {train_df['date'].min()} ～ {train_df['date'].max()}")
print(f"テスト期間: {test_df['date'].min()} ～ {test_df['date'].max()}")

# 基本的な日付特徴量を作成
def create_date_features(df):
    """日付から基本的な特徴量を生成"""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df

print("\n日付特徴量を生成中...")
train_df = create_date_features(train_df)
test_df = create_date_features(test_df)

# ラグ特徴量の作成（シンプルに7日前と14日前のみ）
print("\nラグ特徴量を生成中...")

# 訓練データとテストデータを結合してラグを計算
all_data = pd.concat([train_df, test_df], ignore_index=True)
all_data = all_data.sort_values(['store_nbr', 'family', 'date'])

# 7日前と14日前の売上
for lag in [7, 14]:
    all_data[f'sales_lag_{lag}'] = all_data.groupby(['store_nbr', 'family'])['sales'].shift(lag)

# 訓練とテストに再分割
train_with_lags = all_data[all_data['date'] <= train_df['date'].max()].copy()
test_with_lags = all_data[all_data['date'] > train_df['date'].max()].copy()

print(f"ラグ特徴量作成後の訓練データ: {train_with_lags.shape}")
print(f"ラグ特徴量作成後のテストデータ: {test_with_lags.shape}")

# 特徴量リスト
feature_cols = [
    'store_nbr', 'family', 'onpromotion',
    'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year',
    'sales_lag_7', 'sales_lag_14'
]

# familyをカテゴリコード化
print("\nカテゴリカル変数をエンコード中...")
family_mapping = {family: idx for idx, family in enumerate(train_with_lags['family'].unique())}
train_with_lags['family'] = train_with_lags['family'].map(family_mapping)
test_with_lags['family'] = test_with_lags['family'].map(family_mapping)

# ラグ特徴量の欠損値を0で埋める（期間の最初のデータ）
train_with_lags[['sales_lag_7', 'sales_lag_14']] = train_with_lags[['sales_lag_7', 'sales_lag_14']].fillna(0)
test_with_lags[['sales_lag_7', 'sales_lag_14']] = test_with_lags[['sales_lag_7', 'sales_lag_14']].fillna(0)

# 訓練データの分割（最後の1ヶ月を検証用に）
validation_start_date = train_with_lags['date'].max() - pd.Timedelta(days=30)
train_data = train_with_lags[train_with_lags['date'] < validation_start_date].copy()
valid_data = train_with_lags[train_with_lags['date'] >= validation_start_date].copy()

print(f"\n訓練データ: {train_data.shape} ({train_data['date'].min()} ～ {train_data['date'].max()})")
print(f"検証データ: {valid_data.shape} ({valid_data['date'].min()} ～ {valid_data['date'].max()})")

# 特徴量とターゲットに分割
X_train = train_data[feature_cols]
y_train = train_data['sales']
X_valid = valid_data[feature_cols]
y_valid = valid_data['sales']
X_test = test_with_lags[feature_cols]

print(f"\n特徴量数: {len(feature_cols)}")
print(f"特徴量: {feature_cols}")

# LightGBMモデルの訓練
print("\n" + "=" * 80)
print("LightGBMモデルを訓練中...")
print("=" * 80)

# LightGBMのデータセット作成
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# ハイパーパラメータ（シンプルな設定）
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

# モデル訓練
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
y_pred_valid = np.maximum(y_pred_valid, 0)  # 負の予測値を0にクリップ

# RMSLE計算（売上が0の場合も考慮）
# RMSLEは log(pred + 1) と log(actual + 1) の差の二乗平均平方根
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred_valid))
print(f"\n検証データRMSLE: {rmsle:.4f}")

# 特徴量の重要度
print("\n特徴量の重要度:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# テストデータで予測
print("\n" + "=" * 80)
print("テストデータで予測を生成中...")
print("=" * 80)

y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_test = np.maximum(y_pred_test, 0)  # 負の予測値を0にクリップ

# 提出用ファイルを作成
submission_df = pd.DataFrame({
    'id': test_with_lags['id'],
    'sales': y_pred_test
})

# CSVとして保存
output_path = OUTPUT_DIR / 'submission_baseline.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n予測結果を保存しました: {output_path}")
print(f"予測売上統計:")
print(f"  平均: {y_pred_test.mean():.2f}")
print(f"  中央値: {np.median(y_pred_test):.2f}")
print(f"  最小値: {y_pred_test.min():.2f}")
print(f"  最大値: {y_pred_test.max():.2f}")
print(f"  ゼロ予測数: {(y_pred_test == 0).sum()} / {len(y_pred_test)} ({(y_pred_test == 0).sum()/len(y_pred_test):.2%})")

print("\n" + "=" * 80)
print("完了！")
print("=" * 80)
print(f"\nKaggleに提出するには:")
print(f"  cd {PROJECT_ROOT}")
print(f"  kaggle competitions submit -c store-sales-time-series-forecasting -f {output_path} -m 'Simple baseline model'")
