"""
Store Sales - ベースラインモデル with 時系列交差検証

単一の検証期間ではなく、複数の時期で検証して真の性能を測定する。
これにより、テストデータでのスコアをより正確に予測できる。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
from datetime import datetime, timedelta

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data/raw'
OUTPUT_DIR = PROJECT_ROOT / 'predictions'

# 出力ディレクトリを作成
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Store Sales - ベースラインモデル with 時系列交差検証")
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

# ラグ特徴量の欠損値を0で埋める
train_with_lags[['sales_lag_7', 'sales_lag_14']] = train_with_lags[['sales_lag_7', 'sales_lag_14']].fillna(0)
test_with_lags[['sales_lag_7', 'sales_lag_14']] = test_with_lags[['sales_lag_7', 'sales_lag_14']].fillna(0)

# 時系列交差検証の設定
print("\n" + "=" * 80)
print("時系列交差検証の設定")
print("=" * 80)

# 複数の検証期間を定義（各期間は15日間）
validation_periods = [
    ('2017-06-16', '2017-06-30'),  # Fold 1
    ('2017-07-01', '2017-07-15'),  # Fold 2
    ('2017-07-16', '2017-07-31'),  # Fold 3
    ('2017-08-01', '2017-08-15'),  # Fold 4 (テスト期間の直前)
]

print(f"\n検証期間数: {len(validation_periods)}")
for i, (start, end) in enumerate(validation_periods, 1):
    print(f"  Fold {i}: {start} ～ {end}")

# LightGBMのハイパーパラメータ
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

# 各Foldで訓練・評価
print("\n" + "=" * 80)
print("各Foldでモデルを訓練・評価")
print("=" * 80)

fold_scores = []
fold_models = []

for fold_idx, (val_start, val_end) in enumerate(validation_periods, 1):
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx}: 検証期間 {val_start} ～ {val_end}")
    print(f"{'='*80}")

    # 訓練データと検証データに分割
    val_start_date = pd.to_datetime(val_start)
    val_end_date = pd.to_datetime(val_end)

    train_fold = train_with_lags[train_with_lags['date'] < val_start_date].copy()
    valid_fold = train_with_lags[
        (train_with_lags['date'] >= val_start_date) &
        (train_with_lags['date'] <= val_end_date)
    ].copy()

    print(f"訓練データ: {len(train_fold):,} レコード ({train_fold['date'].min()} ～ {train_fold['date'].max()})")
    print(f"検証データ: {len(valid_fold):,} レコード ({valid_fold['date'].min()} ～ {valid_fold['date'].max()})")

    # 特徴量とターゲットに分割
    X_train_fold = train_fold[feature_cols]
    y_train_fold = train_fold['sales']
    X_valid_fold = valid_fold[feature_cols]
    y_valid_fold = valid_fold['sales']

    # LightGBMデータセット作成
    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_valid = lgb.Dataset(X_valid_fold, y_valid_fold, reference=lgb_train)

    # モデル訓練
    print("\nモデル訓練中...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=200)
        ]
    )

    # 検証データで予測
    y_pred_valid = model.predict(X_valid_fold, num_iteration=model.best_iteration)
    y_pred_valid = np.maximum(y_pred_valid, 0)

    # RMSLE計算
    rmsle = np.sqrt(mean_squared_log_error(y_valid_fold, y_pred_valid))
    fold_scores.append(rmsle)
    fold_models.append(model)

    print(f"\nFold {fold_idx} RMSLE: {rmsle:.4f}")

# 全Foldの結果をまとめる
print("\n" + "=" * 80)
print("時系列交差検証の結果")
print("=" * 80)

for i, (period, score) in enumerate(zip(validation_periods, fold_scores), 1):
    print(f"Fold {i} ({period[0]} ～ {period[1]}): RMSLE = {score:.4f}")

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print(f"\n{'='*80}")
print(f"平均RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")
print(f"{'='*80}")

# 最後のFold（テスト期間に最も近い）のスコアを重視
last_fold_score = fold_scores[-1]
print(f"\n最後のFold（テスト期間直前）のスコア: {last_fold_score:.4f}")
print(f"→ このスコアがテストデータでのスコアに最も近いと予想されます")

# 全訓練データで最終モデルを訓練
print("\n" + "=" * 80)
print("全訓練データで最終モデルを訓練")
print("=" * 80)

X_train_full = train_with_lags[feature_cols]
y_train_full = train_with_lags['sales']

lgb_train_full = lgb.Dataset(X_train_full, y_train_full)

final_model = lgb.train(
    params,
    lgb_train_full,
    num_boost_round=int(np.mean([m.best_iteration for m in fold_models])),  # 各Foldの平均反復回数
    valid_sets=[lgb_train_full],
    valid_names=['train'],
    callbacks=[lgb.log_evaluation(period=200)]
)

# 特徴量の重要度
print("\n特徴量の重要度:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# テストデータで予測
print("\n" + "=" * 80)
print("テストデータで予測を生成")
print("=" * 80)

X_test = test_with_lags[feature_cols]
y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_pred_test = np.maximum(y_pred_test, 0)

# 提出用ファイルを作成
submission_df = pd.DataFrame({
    'id': test_with_lags['id'],
    'sales': y_pred_test
})

output_path = OUTPUT_DIR / 'submission_baseline_timeseries_cv.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n予測結果を保存しました: {output_path}")
print(f"予測売上統計:")
print(f"  平均: {y_pred_test.mean():.2f}")
print(f"  中央値: {np.median(y_pred_test):.2f}")
print(f"  最小値: {y_pred_test.min():.2f}")
print(f"  最大値: {y_pred_test.max():.2f}")

# まとめ
print("\n" + "=" * 80)
print("まとめ")
print("=" * 80)
print(f"時系列CV平均RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")
print(f"最終Fold RMSLE: {last_fold_score:.4f} (テストスコアの予測値)")
print(f"\n提出ファイル: {output_path}")
print(f"\n前回のベースライン:")
print(f"  検証RMSLE: 0.6851")
print(f"  Kaggle Public Score: 1.10346")
print(f"\n今回の予測:")
print(f"  時系列CV RMSLE: {mean_score:.4f}")
print(f"  予想 Public Score: ~{mean_score * 1.6:.2f} (経験的な変換)")
